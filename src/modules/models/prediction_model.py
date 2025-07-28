import torch
import torch.nn as nn


class AttentionModel(nn.Module):
    def __init__(self, hparams: dict, latent_variables: torch.Tensor):
        super().__init__()
        self.embed_dim = latent_variables.size(1)

        # Latent variables
        self.latent_variables = latent_variables

        # Trained matrices
        self.att_mat_diag = nn.Parameter(torch.ones(self.embed_dim))
        self.tok_score_mat_diag = nn.Parameter(torch.ones(self.embed_dim))

        self.use_pos_embeds = hparams["use_pos_embeds"]
        if self.use_pos_embeds:
            # Trained positional embeddings: we need
            #   2*half_context + 1 for positions of words +
            #   1 for out of context words +
            #   1 for CLS token +
            #   1 for padding tokens
            self.half_ctx_window_size = hparams["half_context"]
            self.cls_pos = hparams["half_context"] + 1
            self.out_of_context_pos = hparams["half_context"] + 2
            self.padding_pos = hparams["half_context"] + 3

            self.positional_embeddings = nn.Embedding(
                2 * hparams["half_context"] + 4,
                self.embed_dim,
                padding_idx=self.padding_pos,  # add for no grads for padding
            )

        # Dropouts
        self.local_ctx_dr = nn.Dropout(p=hparams["dropout_scores"])
        self.posit_embeds_dr = nn.Dropout(p=hparams["dropout_pos_embeds"])
        self.classes_dr = nn.Dropout(p=hparams["dropout_classes"])

        # Embeddings and their masks
        self.current_embeds = None
        self.current_masks = None

        # Token multiplier
        if "tok_multiplier" in hparams:
            if all(k in hparams["tok_multiplier"].keys() for k in ["train", "value"]):
                print(
                    f"-> Token_multiplier: train {hparams['tok_multiplier']['train']}, value {hparams['tok_multiplier']['value']}"
                )

                if hparams["tok_multiplier"]["train"]:
                    self.token_multiplier = nn.Parameter(
                        torch.tensor(float(hparams["tok_multiplier"]["value"]))
                    )
                else:
                    self.token_multiplier = float(hparams["tok_multiplier"]["value"])
            else:
                raise ValueError("Cannnot configure token_multiplier")
        else:
            self.token_multiplier = 1.0
            print("-> Token_multiplier: DISABLED")

    def init_att_module(self, embeds, masks):
        batch_size, seq_len, _ = embeds.size()
        self.current_embeds = embeds  # .clone()  # (batch_size, seq_len, embed_dim)
        self.current_masks = masks  # (batch_size, seq_len)
        self.pos_ids = (
            torch.arange(seq_len, dtype=torch.int, device=embeds.device)
            .unsqueeze(0)
            .repeat_interleave(batch_size, dim=0)
        )  # (batch_size, seq_len)

    def update_embeds_and_get_att(self, to_add: torch.Tensor, pos: torch.Tensor):
        """Adds `to_add` with positional information given by `pos` to `current_embeds`
        and computes the attention scores.

        Inputs:
            to_add: tensor that will be added to self.current_embeds,
                    size (batch_size, embed_dim)
            pos: absolute position of the word that will contribute to the next
                 prediction, size (batch_size, )"""

        if self.use_pos_embeds:
            # Relative positions of all words, relative to the centered word
            # (which has pos_id = 0)
            pos_ids = self.pos_ids - pos.unsqueeze(-1)

            # Fix the position of out-of-context words
            pos_ids[
                (pos_ids < -self.half_ctx_window_size)
                | (pos_ids > self.half_ctx_window_size) ^ (pos_ids == self.cls_pos)
            ] = self.out_of_context_pos

            # Fix the position of the CLS
            pos_ids[:, 0] = self.cls_pos
            # Fix the position of padding words
            pos_ids[self.current_masks == 0] = self.padding_pos
        else:
            pos_ids = None

        batch_size = self.current_embeds.size(0)

        # Mask used to flag the centered word
        mask = torch.ones_like(self.current_embeds)  # (batch_size, seq_len, embed_dim)
        mask[torch.arange(batch_size), pos] = 0

        # Sum to_add to self.current_embeds in the indices of the centered words
        self.current_embeds = self.current_embeds - (mask - 1) * to_add.unsqueeze(1)
        self.current_embeds = self.current_masks.unsqueeze(-1) * self.current_embeds

        # Compute and return attention
        att = self.forward(
            embeds=self.current_embeds,
            embeds_mask=self.current_masks.to(self.current_embeds.dtype),
            rel_pos_ids=pos_ids,
        )
        return att

    def forward(
        self, embeds: torch.Tensor, embeds_mask: torch.Tensor, rel_pos_ids: torch.Tensor
    ):
        batch_size, seq_len, embed_dim = embeds.size()

        if self.use_pos_embeds:
            # Get positional embeddings
            absolute_pos = (
                rel_pos_ids + self.half_ctx_window_size
            )  # (batch_size, seq_len)
            pos_embeds = self.positional_embeddings(absolute_pos)

            # Dropout in pos_embeds
            pos_embeds = self.posit_embeds_dr(pos_embeds)

            # Sum positional info to the input embeds
            merged_embeds = (
                self.token_multiplier * embeds + pos_embeds
            )  # (batch_size, seq_len, embed_dim)
        else:
            merged_embeds = embeds

        # Dropout in the latent variables
        # (nr_latent_variables, embed_dim)
        latent_variables = self.classes_dr(self.latent_variables)

        # Get a score for each tuple (batch element, latent variable , embedding)
        ent_tok_att_scores = torch.bmm(
            (latent_variables * self.att_mat_diag)
            .unsqueeze(0)
            .expand(batch_size, -1, -1),
            merged_embeds.permute(0, 2, 1),
        )  # (batch_size, nr_latent_variables, seq_len)

        # Zero out the scores corresponding to masked elements in merged_embeds
        ent_tok_att_scores = (ent_tok_att_scores * embeds_mask.unsqueeze(1)).add_(
            (embeds_mask.unsqueeze(1) - 1).mul_(1e12)
        )

        # To each merged_embeds element, attribute the highest score across
        # all latent variables.
        # An high value in this tensor means that there is at least one latent variable
        # that is highly related to the corresponding merged_embeds element
        # (batch_size, seq_length)
        top_tok_att_scores, _ = torch.max(ent_tok_att_scores, dim=1)

        # Dropout on these scores
        top_tok_att_scores = self.local_ctx_dr(top_tok_att_scores)

        # Normalize these weights.
        # An high value in att_probs means that there is at least one latent variable
        # that is highly related to the corresponding
        # merged_embeds element
        # (batch_size, seq_len)
        att_probs = nn.functional.softmax(top_tok_att_scores, dim=1)

        # Get a weighted combination of the embeds (so no positional info),
        # weigthed by its latent variable scores and a parameter matrix

        # (batch_size, seq_len, embed_dim)
        to_sum = (embeds * self.tok_score_mat_diag) * att_probs.unsqueeze(-1)

        # (batch_size, embed_dim)
        ctx_vecs = torch.sum(
            to_sum,
            dim=1,
        )

        return ctx_vecs


class PredictionModel(nn.Module):
    def __init__(self, hparams, embed_dim, actions_vocab, multitask_mode, ablat):
        super().__init__()
        self.actions_vocab = actions_vocab
        self.multitask_mode = multitask_mode

        # Certain needed scalars
        self.embed_dim = embed_dim
        self.num_actions = self.actions_vocab.max_v
        self.num_latent_variables = (
            hparams["attention_module"]["action_space_dim"] * self.num_actions
        )
        self.projection_coef = torch.tensor(hparams["self_training"]["project_coeff"])
        self.multiplier_numb_actions_training = hparams["control"][
            "multiplier_numb_actions_training"
        ]
        self.multiplier_numb_actions_inference = hparams["control"][
            "multiplier_numb_actions_inference"
        ]

        # Set up trained embeddings
        self.actions_embeddings = nn.Embedding(self.num_actions, self.embed_dim)
        self.start_action = nn.Parameter(torch.zeros(self.embed_dim))
        # Attention module
        bypass_attention_model = (
            ablat is not None and not ablat.pred_model.parser_state.use_attention
        )
        bypass_pos_embeds = (
            ablat is not None
            and ablat.pred_model.parser_state.use_attention
            and (not ablat.pred_model.attention_module.use_pos_embeds)
        )

        if not bypass_attention_model:
            self.latent_variables = nn.Parameter(
                torch.zeros(self.num_latent_variables, self.embed_dim)
            )

            hparams_att_module = {
                p: val for p, val in hparams["attention_module"].items()
            }
            hparams_att_module["use_pos_embeds"] = not (bypass_pos_embeds)

            self.action_att = AttentionModel(
                hparams=hparams_att_module,
                latent_variables=self.latent_variables,
            )

        # MLP
        bypass_next_word = (
            ablat is not None and not ablat.pred_model.parser_state.use_next_word
        )
        if not bypass_attention_model and not bypass_next_word:
            self.mlp_features = {"next_word", "attention"}
        elif bypass_attention_model and not bypass_next_word:
            self.mlp_features = {"next_word"}
        elif not bypass_attention_model and bypass_next_word:
            self.mlp_features = {"attention"}
        else:
            ValueError("Ablat: can't configure MLP")

        if len(self.mlp_features) == 2:
            self.mlp = nn.Sequential(
                nn.Dropout(p=hparams["parser_state"]["dropout"]),
                nn.Linear(
                    in_features=2 * self.embed_dim, out_features=2 * self.embed_dim
                ),
                nn.Tanh(),
                nn.Linear(
                    in_features=2 * self.embed_dim, out_features=self.num_actions
                ),
            )
        elif len(self.mlp_features) == 1:
            self.mlp = nn.Sequential(
                nn.Dropout(p=hparams["parser_state"]["dropout"]),
                nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim),
                nn.Tanh(),
                nn.Linear(in_features=self.embed_dim, out_features=self.num_actions),
            )
        print("Initialized prediction model")
        print()

    def get_embeds_for_att(
        self, logits: torch.Tensor, nonzero_score_mask: torch.Tensor
    ):
        """Given float `logits` (batch_size, num_actions) and bool `nonzero_score_mask`
        (batch_size, num_actions), it returns a weighted representation
        (batch_size, embed_dim) of the actions, to be fed into attention"""

        scores = nn.functional.softmax(logits, dim=1) * nonzero_score_mask
        weighted_embed = torch.sum(
            scores.unsqueeze(-1) * self.actions_embeddings.weight.unsqueeze(0), dim=1
        )
        return weighted_embed

    def modify_gold(
        self,
        current_gold: torch.Tensor,
        current_probs: torch.Tensor,
        allowed_for_pred: torch.Tensor,
        allow_shift_to_pass: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Given `current_gold` (batch_size, num_actions), `current_probs`
        (batch_size, num_actions), a mask for actions that can be passed to the
        new gold `allowed_for_pred` (batch_size, num_actions), and another mask
        `force_keep_shift` (batch, ) specific for the SHIFT action, returns a tensor
        (batch_size, num_actions) of new gold actions."""
        shift_probs = current_probs[:, self.actions_vocab.shift_ix]

        # We allow the model to make loss-less predictions on TR/RE actions of other
        # datasets (if it has not reached the EOA)
        probs_other_TR_RE = (
            (current_gold[:, self.actions_vocab.eoa_ix] == 0).unsqueeze(-1)
            * allowed_for_pred
            * current_probs
        )

        # If the gold action is a SHIFT, and the `probs_other_TR_RE` have probability
        # higher than the SHIFT, this means the model is trying to annotate a new span
        # with a tag from other dataset. In this case, we let the model create
        # such an action, and we move the gold shift to the next timestep.
        # Otherwise, we keep the SHIFT value from the current gold.
        should_keep_shift = torch.logical_or(
            allow_shift_to_pass,
            (probs_other_TR_RE <= shift_probs.unsqueeze(-1)).all(dim=-1),
        )

        model_decisions = self.projection_coef * probs_other_TR_RE
        model_decisions[model_decisions > 1.0] = 1.0

        # Fill in the new_gold
        new_gold = model_decisions.clone().detach() + current_gold
        new_gold[:, self.actions_vocab.shift_ix] *= should_keep_shift

        # If we remove a gold SHIFT, we pass its model probability instead of a strong 0
        removed_gold_shift = (
            current_gold[:, self.actions_vocab.shift_ix] == 1
        ) * torch.logical_not(should_keep_shift)
        new_gold[:, self.actions_vocab.shift_ix] += removed_gold_shift * shift_probs

        # Return the new gold and if we removed a gold SHIFT
        return (new_gold, removed_gold_shift)

    def start_memory(self, embeds: torch.Tensor, att_mask: torch.Tensor):
        """Gives the first memory_pointer, attention embedding and word_on_memory
        embedding."""
        batch_size = embeds.size(0)
        # Start variables
        memory_pointer = torch.zeros(batch_size, dtype=torch.int, device=embeds.device)

        if "attention" in self.mlp_features:
            # Initialize attention: we add the SOA on top of the CLS
            self.action_att.init_att_module(embeds=embeds, masks=att_mask)
            attention = self.action_att.update_embeds_and_get_att(
                to_add=self.start_action.unsqueeze(0).repeat_interleave(
                    batch_size, dim=0
                ),
                pos=memory_pointer,
            )
        else:
            attention = None

        # The first token contributing to the parser state is the first non-CLS token
        if "next_word" in self.mlp_features:
            word_on_memory = embeds[:, 1]
        else:
            word_on_memory = None

        # Move the memory pointer since we already have SOA in the action memory
        memory_pointer[:] = 1

        return memory_pointer, attention, word_on_memory

    def fw_train(self, embeds, att_mask, in_gold_actions, allowed_for_pred):

        # The model functions like a transition-based parser.
        # The parsing state is modelled by two objects:
        # 1) A read-write 'action' memory (A, a*), which encompasses a data structure
        # A,and a pointer a* to an element of the memory.
        # 2) A read-only 'word' memory (W, w*), where W is a data structure, and
        # w* is a pointer to an element of the memory.

        # In the data structures of the memories, we store an embedding.
        # Having a pointer associated with each memory enables us to think of
        # each slot of the memories as representing a distinct time
        # step, and the pointer indicates the current time, allowing sequential
        # movement.

        # We build up the action sequence encoding the spans of the sentence by
        # selecting actions sequentially. At each time step, the choice of action is
        # made with the following features:
        # 1) The data stored in W, at the slot pointed to by w*.
        # 2) An embedding of the of the action memory at that time, which depends on the
        # data stored in A, and the position of the pointer a*.

        # The slots of the word memory (i.e. W) are initialized with the embedings of
        # the tokens, and we acess each token embedding via increments to the word
        # pointer w*. w* initially points to the first non-CLS token.

        # In contrast with the word memory, we read and write to the action memory.
        # The slots of the action memory (i.e. A) are also initialized with the
        # embeddings of the tokens, but, in addition, we add a trained start of actions
        # embedding to the slot where the CLS lies. a* initially points to this slot.

        # When a decision is predicted:
        # 1) If SH is predicted, the w* is incremented.
        # 2) If a* points to a slot with the SOA or the SH action,
        #    we increment the pointer a*.
        # 3) The chosen action is added to the slot to which a* points.

        batch_size, tok_seq_len, _ = embeds.size()

        # Start memory structures
        memory_pointer, attention, word_on_memory = self.start_memory(
            embeds=embeds, att_mask=att_mask
        )

        # Number of parsed actions
        pred_count = 0

        # ---------------------
        # Specific for training
        ac_seq_len = in_gold_actions.size(1)

        timesteps_logits = torch.zeros(
            size=(
                batch_size,
                ac_seq_len * self.multiplier_numb_actions_training,
                self.num_actions,
            ),
            dtype=torch.float,
            device=embeds.device,
        )

        # Modified gold actions so certain actions don't incur in loss
        new_gold_actions = torch.zeros_like(timesteps_logits)

        # To know where we are in the `in_gold_actions`
        original_count = torch.zeros(
            size=(batch_size,), dtype=torch.int, device=embeds.device
        )

        # To control if we keep the gold SHIFT values
        removed_gold_shift = torch.zeros(
            size=(batch_size,), dtype=torch.bool, device=embeds.device
        )
        # ---------------------

        while pred_count < ac_seq_len * self.multiplier_numb_actions_training:

            if len(self.mlp_features) == 2:
                # Use as features the embedding of the word in memory,
                # and the attention of actions so far
                parser_state = torch.cat((word_on_memory, attention), dim=1)
            elif "next_word" in self.mlp_features:
                parser_state = word_on_memory
            elif "attention" in self.mlp_features:
                parser_state = attention

            # Get logits of actions
            logits = self.mlp(parser_state)  # (batch_size, num_actions)
            # Get probability of selecting each action
            probs = logits.sigmoid()

            # Get modified gold actions at this timestep
            # We force place a SHIFT at this timestep if we have removed it before
            if self.multitask_mode:
                modified_gold_this_timestep, removed_gold_shift = self.modify_gold(
                    current_gold=in_gold_actions[
                        torch.arange(batch_size), original_count
                    ],
                    # current_probs=probs.clone().detach(),
                    current_probs=probs,
                    allowed_for_pred=allowed_for_pred,
                    allow_shift_to_pass=removed_gold_shift,
                )
            else:
                modified_gold_this_timestep = in_gold_actions[:, pred_count]

            new_gold_actions[:, pred_count] = modified_gold_this_timestep

            # Add logits to the history
            timesteps_logits[:, pred_count] = logits

            # Get features for the next prediction

            if "attention" in self.mlp_features:
                # Compute input embeds for next attention
                # The actions that contribute to the embeds are the ones that were just
                # predicted and that have score >= than the SHIFT
                condition_nonzero_score = probs >= probs[
                    :, self.actions_vocab.shift_ix
                ].unsqueeze(1)
                attention_inputs = self.get_embeds_for_att(
                    logits=logits, nonzero_score_mask=condition_nonzero_score
                )

                # Get attention
                attention = self.action_att.update_embeds_and_get_att(
                    to_add=attention_inputs, pos=memory_pointer
                )

            # For the next prediction step, the original action can be the same
            # as the one in this timestep if we have removed the SHIFT
            if self.multitask_mode:
                if original_count.min() == ac_seq_len - 1:
                    break
                original_count += torch.logical_not(removed_gold_shift).int()
                original_count[original_count > (ac_seq_len - 1)] = ac_seq_len - 1
            else:
                if pred_count == ac_seq_len - 1:
                    break

            # Update the memory_pointer
            # We move on to the next token if a SH/EOA is selected as gold
            memory_pointer += torch.logical_or(
                new_gold_actions[:, pred_count, self.actions_vocab.shift_ix] == 1,
                new_gold_actions[:, pred_count, self.actions_vocab.eoa_ix] == 1,
            ).int()
            memory_pointer[memory_pointer > tok_seq_len - 1] = tok_seq_len - 1

            # Update the word_on_memory
            if "next_word" in self.mlp_features:
                word_on_memory = embeds[
                    torch.arange(batch_size), memory_pointer.clone()
                ]

            # We increment the prediction counter
            pred_count += 1

        return {
            "timesteps_logits": timesteps_logits,
            "new_gold_actions": new_gold_actions,
        }

    def fw_decode(self, embeds, att_mask):

        batch_size, tok_seq_len, _ = embeds.size()

        # Start memory structures
        memory_pointer, attention, word_on_memory = self.start_memory(
            embeds=embeds, att_mask=att_mask
        )

        # Number of parsed actions
        pred_count = 0

        # ---------------------
        # Specific for inference
        timesteps_probs = torch.zeros(
            size=(batch_size, 1, self.num_actions),
            dtype=torch.float,
            device=embeds.device,
        )
        reached_eoa = torch.zeros(batch_size, dtype=torch.int, device=embeds.device)
        # ---------------------

        while pred_count < tok_seq_len * self.multiplier_numb_actions_inference:
            if len(self.mlp_features) == 2:
                # Use as features the embedding of the word in memory,
                # and the attention of actions so far
                parser_state = torch.cat((word_on_memory, attention), dim=1)
            elif "next_word" in self.mlp_features:
                parser_state = word_on_memory
            elif "attention" in self.mlp_features:
                parser_state = attention

            # Get logits of actions
            logits = self.mlp(parser_state)  # (batch_size, num_actions)

            # Get probability of selecting each action
            probs = logits.sigmoid()
            # Add probs to the history
            timesteps_probs = torch.cat((timesteps_probs, probs.unsqueeze(1)), dim=1)

            # If the probability of the EOA is > 0.5, then we are done
            reached_eoa |= probs[:, self.actions_vocab.eoa_ix] > 0.5
            if reached_eoa.all().item():
                break

            # Get features for the next prediction

            if "attention" in self.mlp_features:
                # Compute input embeds for next attention
                # The actions that contribute to the embeds are the ones that were just
                # predicted and that have score >= than the SHIFT
                condition_nonzero_score = probs >= probs[
                    :, self.actions_vocab.shift_ix
                ].unsqueeze(1)
                attention_inputs = self.get_embeds_for_att(
                    logits=logits, nonzero_score_mask=condition_nonzero_score
                )

                # Get attention
                attention = self.action_att.update_embeds_and_get_att(
                    to_add=attention_inputs, pos=memory_pointer
                )

            # Update the memory_pointer
            # We move on to the next token if a SH/EOA is selected as the best action
            _, best_action = logits.max(axis=1)
            memory_pointer += torch.logical_or(
                best_action == self.actions_vocab.shift_ix,
                best_action == self.actions_vocab.eoa_ix,
            ).int()
            memory_pointer[memory_pointer > tok_seq_len - 1] = tok_seq_len - 1

            # Update the word_on_memory
            if "next_word" in self.mlp_features:
                word_on_memory = embeds[
                    torch.arange(batch_size), memory_pointer.clone()
                ]

            # We increment the prediction counter
            pred_count += 1

        # Return the timestep probabilities
        timesteps_probs = timesteps_probs[:, 1:]
        return {"timesteps_probs": timesteps_probs}
