from transformers import AutoModel, AutoConfig
import os, shutil, torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
    def __init__(self, model_name: str, loading_from_ckpt: bool):
        super().__init__()

        print(f"Using embedding model {model_name}")

        self.model = self._initialize_model(
            model_name=model_name, loading_from_ckpt=loading_from_ckpt
        )
        self.embed_dim = self.model.config.hidden_size

    def _initialize_model(
        self,
        model_name: str,
        loading_from_ckpt: bool,
        cache_dir: str = ".transformers_models/",
    ):
        # If we load from a checkpoint, we use AutoModel.from_config to not load the
        # original weights. The checkpoint weights will be filled by Lightning's
        # load_from_checkpoint procedure.

        if loading_from_ckpt:
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModel.from_config(config)
            print("-> Loaded embed model from checkpoint")
        else:
            model_cache_folder = os.path.join(cache_dir, model_name)
            try:
                # if cache folder already exists, just load locally
                model = AutoModel.from_pretrained(
                    model_cache_folder, local_files_only=True
                )
                print("-> Loaded base embed model from cache")
            except:
                # otherwise download the model hashed files into a temporary folder
                print("-> Downloading base embed model from huggingface")
                tmp_folder = os.path.join(cache_dir, model_name, "tmp")
                model = AutoModel.from_pretrained(model_name, cache_dir=tmp_folder)
                shutil.rmtree(tmp_folder)
                # save model files with proper formats and names to be reused later
                model.save_pretrained(model_cache_folder)
                print(f"-> Saved it in {tmp_folder}")

        return model

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Returns the last hidden state (batch_size, seq_len, embed_dim)"""
        last_hid_state = self.model(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state

        return last_hid_state
