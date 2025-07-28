import pandas as pd
import numpy as np
from src.modules.utils.action_utils import LabeledSubstring


def flatten_dictionary(nested_dict: dict, path: str = "") -> dict:
    result = {}

    for k, v in nested_dict.items():
        new_path = f"{path}-{k}" if path else str(k)

        if isinstance(v, dict):
            result.update(flatten_dictionary(v, new_path))
        else:
            result[new_path] = v

    return result


def set_value_in_nested_dict(
    nested_dict: dict, list_of_keys: list, value: float
) -> None:
    current = nested_dict
    for key in list_of_keys[:-1]:
        current = current.setdefault(key, {})
    current[list_of_keys[-1]] = value


def add_p_r_f1(df: pd.DataFrame):
    """Given a `df` with columnns 'TP', 'FN' e 'FP', concats three columns at the
    right with precision, recall and F1"""

    df["P"] = df["TP"] / ((df["TP"] + df["FP"]).replace(0, np.nan))
    df["R"] = df["TP"] / ((df["TP"] + df["FN"]).replace(0, np.nan))
    df["F1"] = 2 * (df["P"] * df["R"]) / ((df["P"] + df["R"]).replace(0, np.nan))
    df[["P", "R", "F1"]] = (df[["P", "R", "F1"]].fillna(0) * 100).round(3)

    return df


class NERMetrics:
    def __init__(self, tag_to_id: dict, dataset_to_tag_ids: dict):
        """Resposible for computing F1 metrics with different levels of granularity.
            `tag_to_id` should be a dictionary mappping a tag name to a number.
            `dataset_to_tag_ids` should be a dictionary mapping a dataset name to the
        set of tags ids of that dataset, so that span predictions of tags not associated
        with the dataset are ignored"""

        # Initialize the DataFrame containing gold/predicted labels
        self.data = pd.DataFrame(
            {
                "sent_id": pd.Series(dtype="int"),
                "start_idx": pd.Series(dtype="int"),
                "end_idx": pd.Series(dtype="int"),
                "gold_tag": pd.Series(dtype="object"),
                "pred_tag": pd.Series(dtype="object"),
                "matches_tokenization": pd.Series(dtype="bool"),
                "source_dataset": pd.Series(dtype="str"),
            }
        )
        self.tag_to_id = tag_to_id
        self.id_to_tag = {v: k for (k, v) in self.tag_to_id.items()}

        self.dataset_gold_tags = {
            dataset_name: [
                self.id_to_tag[id] for id in dataset_to_tag_ids[dataset_name]
            ]
            for dataset_name in dataset_to_tag_ids
        }

        # Initialize dictionary is all metrics
        # We track precision (key 'P'), recall (key 'R') and F1 (key 'F1')
        # For each metric, we track each dataset and the simple average of all datasets
        # Inside each dataset, we track all metrics per tag, and on average
        # (micro-average)
        # The structure of the dicionary is like
        # {
        #   'P':
        #       {
        #           {dataset1: {tag1: 0.0, tag2: 0.0, ..., 'average': 0.0},
        #           {dataset2: {tag1: 0.0, tag2: 0.0, ..., 'average': 0.0},
        #           ...,
        #           'average': 0.0,
        #       },
        #   'R':
        #       {
        #           {dataset1: {tag1: 0.0, tag2: 0.0, ..., 'average': 0.0},
        #           {dataset2: {tag1: 0.0, tag2: 0.0, ..., 'average': 0.0},
        #           ...,
        #           'average': 0.0,
        #       },
        #   'F1':
        #       {
        #           {dataset1: {tag1: 0.0, tag2: 0.0, ..., 'average': 0.0},
        #           {dataset2: {tag1: 0.0, tag2: 0.0, ..., 'average': 0.0},
        #           ...,
        #           'average': 0.0,
        #       },
        # }

        # For logging purposes, the sequence of keys are flattened and separated with
        # with an hypen
        # We initialize the dicionary here to have acess to the hyphaned keys for the
        # checkpoint callback, stored in self.metric_names
        self.metric_dict = {}
        metrics = {"P", "R", "F1"}
        for metric in metrics:
            for dataset in self.dataset_gold_tags:
                for tag in self.dataset_gold_tags[dataset]:
                    set_value_in_nested_dict(
                        self.metric_dict, [metric, dataset, tag], 0.0
                    )
                set_value_in_nested_dict(
                    self.metric_dict, [metric, dataset, "average"], 0.0
                )
            set_value_in_nested_dict(self.metric_dict, [metric, "average"], 0.0)

        self.metric_names = list(flatten_dictionary(self.metric_dict).keys())

        """
        # Tag id to assign to missing predictions
        self.missing_pred_id = max(tag_to_id.values()) + 1
        pd.set_option("future.no_silent_downcasting", True) """

    def reset(self):
        """Resets the `self.data` DataFrame so that it can ingest new data.
        Also cleans the `self.metric_dict`."""
        self.data = self.data[0:0].reset_index(drop=True)

        for metric_name in self.metric_names:
            path = metric_name.split("-")
            set_value_in_nested_dict(self.metric_dict, path, 0.0)

    def append_batch(self, data_dict: dict):
        """Given a dictionary `data_dict` with the key `sent_idx` and values
        dictionaries with keys/values:
                `gold_entities_matching_tokens`: list of LabeledSubtring,
                `gold_entities_not_matching_tokens`: list of LabeledSubtring,
                `predicted_entities`: list of LabeledSubtring
                `source_dataset`: str

        updates the `self.data` Dataframe, adding a row for each span.

        NOTE: since the spans in `gold_entities_not_matching_tokens` do not fit into
        the wordpieces of the sentences, an exact match is never possible. Hence, we
        create a new row for each such span.
        """

        for sent_idx, spans_in_sent in data_dict.items():
            source = data_dict[sent_idx]["source_dataset"]

            # Add a row for each span in `gold_entities_not_matching_tokens`
            for span in spans_in_sent["gold_entities_not_matching_tokens"]:
                row = {
                    "sent_id": sent_idx,
                    "source_dataset": source,
                    "start_idx": -1,
                    "end_idx": -1,
                    "pred_tag": None,
                    "gold_tag": span.tag,
                    "matches_tokenization": False
                }
                self.data = pd.concat(
                    [self.data, pd.DataFrame([row])], ignore_index=True
                )

            # Get the (start, end) for the entities that fit the tokens,
            # and add a row for each
            start_end_spans = set(
                LabeledSubstring(start=span.start, end=span.end)
                for span in (
                    spans_in_sent["gold_entities_matching_tokens"]
                    + spans_in_sent["predicted_entities"]
                )
            )

            # For each (start, end) compile gold/pred tags
            for s_e_span in start_end_spans:

                row = {
                    "sent_id": sent_idx,
                    "source_dataset": source,
                    "start_idx": s_e_span.start,
                    "end_idx": s_e_span.end,
                }

                # Gather all predicted tags for this span
                pred_tags = [
                    span.tag
                    for span in spans_in_sent["predicted_entities"]
                    if span.same_spans(s_e_span)
                ]
                if pred_tags:
                    row["pred_tag"] = sorted(pred_tags)
                else:
                    row["pred_tag"] = None

                # Check for gold tags
                # A span can have multiple gold tags in certain datasets, so we add
                # format the gold_tag as a list

                # We deal with the gold_matching_tokens separately to the
                # gold_not_matching_tokens
                gold_matching_tokens = s_e_span.get_from_list_with_the_same_span(
                    spans_in_sent["gold_entities_matching_tokens"]
                )

                if gold_matching_tokens:
                    row["gold_tag"] = [span.tag for span in gold_matching_tokens]
                    row["matches_tokenization"] = True
                else:
                    row["gold_tag"] = None
                    # Case where the span only has predicted tags, so obviously
                    # matches tokenization
                    row["matches_tokenization"] = True

                self.data = pd.concat(
                    [self.data, pd.DataFrame([row])], ignore_index=True
                )

    def filter_pred_tags(self, row, keep_mode):
        """Modifies row['pred_tag'] to have only tags from the row['source_dataset']
        (if keep_mode == 'source') or from other datasets if (if keep_mode == 'others')
        """
        source_tags = self.dataset_gold_tags.get(row["source_dataset"], set())
        pred_tags = row["pred_tag"]

        if pred_tags is None:
            return []

        if keep_mode == "source":
            return [tag for tag in pred_tags if tag in source_tags]
        elif keep_mode == "others":
            return [tag for tag in pred_tags if tag not in source_tags]
        else:
            raise ValueError("Wrong keep_mode.")

    def filter_df(
        self, df: pd.DataFrame, source: str, gold_tag_str: str
    ) -> pd.DataFrame:
        """ Filters `df` so that `gold_tag_str` is in the list df['gold_tag'] """
        df_filtered = df[df["source_dataset"] == source]
        df_filtered = df_filtered[
            df_filtered["gold_tag"].apply(
                lambda x: (gold_tag_str in x) if x is not None else False
            )
        ]

        if df_filtered.empty:
            df_filtered = pd.DataFrame(columns=df.columns)

        return df_filtered

    def get_tp_fn_fp(
        self, data: pd.DataFrame, filter_non_matching_tokenization: bool
    ) -> pd.DataFrame:
        """ Given a DataFrame `data`, with rows corresponding to a span of text, with 
        'gold_tag', 'pred_tag', 'matches_tokenization' and 'source_dataset' in
        its columns, returns a DataFrame with columns
        'source_dataset', 'gold_tag', 'TP', 'FP' and 'FN'. """

        if filter_non_matching_tokenization:
            data = data[~(data["matches_tokenization"] == False)]

        # Initilize counter of TR/FP/FN
        true_positives = {
            (dataset_name, tag): 0
            for dataset_name in self.dataset_gold_tags
            for tag in self.dataset_gold_tags[dataset_name]
        }
        false_positives = {
            (dataset_name, tag): 0
            for dataset_name in self.dataset_gold_tags
            for tag in self.dataset_gold_tags[dataset_name]
        }
        false_negatives = {
            (dataset_name, tag): 0
            for dataset_name in self.dataset_gold_tags
            for tag in self.dataset_gold_tags[dataset_name]
        }

        for dataset, gold in true_positives:
            # Get the rows from a specific dataset and that have a specific gold tag
            # string in the list data['gold_tag']
            group = self.filter_df(df=data, source=dataset, gold_tag_str=gold)

            # TP: number of spans where `gold_tag` is in `pred_tag`
            tp_count = (
                group["pred_tag"]
                .apply(lambda tags: (gold in tags) if tags is not None else False)
                .sum()
            )

            # FN: number of spans where `gold_tag` is not in `pred_tag`
            fn_count = (
                group["pred_tag"]
                .apply(lambda tags: (gold not in tags) if tags is not None else True)
                .sum()
            )

            # Store counts in dictionaries
            true_positives[(dataset, gold)] = (
                tp_count.item() if isinstance(tp_count, np.integer) else tp_count
            )
            false_negatives[(dataset, gold)] = (
                fn_count.item() if isinstance(fn_count, np.integer) else fn_count
            )

        # FP: number of times a tag from the source dataset is predicted without
        # being the gold
        for id, row in self.data.iterrows():
            source_dataset = row["source_dataset"]
            all_preds = row["pred_tag"]
            preds_from_source_dataset = (
                [
                    tag
                    for tag in all_preds
                    if tag in self.dataset_gold_tags[source_dataset]
                ]
                if all_preds is not None
                else []
            )
            gold_tags = row["gold_tag"] if row["gold_tag"] is not None else []
            for pred_tag in preds_from_source_dataset:
                if pred_tag not in gold_tags:
                    false_positives[(source_dataset, pred_tag)] += 1

        results = pd.DataFrame(
            {"TP": true_positives, "FN": false_negatives, "FP": false_positives}
        ).reset_index()
        results.columns = ["source_dataset", "gold_tag", "TP", "FN", "FP"]

        return results

    def compute_metrics(self):
        """On spans which have a gold label, compute metrics on several granularities,
        where predictions of tags not associated with the datasets are discarded
        in certain cases. Returns a dict.
        """

        data = self.data.copy()
        # Step 1: for each (dataset, tag) pair, compute TP, FP and FN
        counts_no_filter = self.get_tp_fn_fp(
            data=data, filter_non_matching_tokenization=False
        )
        counts_with_filter = self.get_tp_fn_fp(
            data=data, filter_non_matching_tokenization=True
        )

        # Step 2: for each dataset, get TP, FP and FN by summing across tags
        counts_per_dataset_no_filter = (
            counts_no_filter.groupby("source_dataset")[["TP", "FN", "FP"]]
            .sum()
            .reset_index()
        )
        counts_per_dataset_with_filter = (
            counts_with_filter.groupby("source_dataset")[["TP", "FN", "FP"]]
            .sum()
            .reset_index()
        )

        # Step 3: add precision, recall and F1 to these tables
        for table in [
            counts_no_filter,
            counts_with_filter,
            counts_per_dataset_no_filter,
            counts_per_dataset_with_filter,
        ]:
            table = add_p_r_f1(table)

        # Step 4: report per dataset and per tag
        grouped = counts_no_filter.groupby(["source_dataset"])

        for dataset, group in grouped:
            print()
            print(f"Results on {dataset[0]} (per tag):")
            results = (
                group.drop(columns=["source_dataset"])
                .reset_index(drop=True)
                .set_index("gold_tag")
            )
            print(results)
            for metric in ["P", "R", "F1"]:
                for _, row in results.iterrows():
                    self.metric_dict[metric][dataset[0]][row.name] = row[metric]

        # Report per dataset (micro-average)
        counts_per_dataset_no_filter = counts_per_dataset_no_filter.set_index(
            "source_dataset"
        )
        counts_per_dataset_with_filter = counts_per_dataset_with_filter.set_index(
            "source_dataset"
        )
        print()
        print("Results per dataset (across all tags):")
        print(counts_per_dataset_no_filter)
        for metric in ["P", "R", "F1"]:
            for _, row in counts_per_dataset_no_filter.iterrows():
                self.metric_dict[metric][row.name]["average"] = row[metric].item()
        print()
        print("Results per dataset, filtering spans not compatible with tokenization:")
        print(counts_per_dataset_with_filter)

        # Report simple average of datasets
        averages = counts_per_dataset_no_filter[["P", "R", "F1"]].mean(axis=0)
        print()
        print("Simple average across datasets:")
        print(averages.to_string(dtype=False))
        print()
        for metric in ["P", "R", "F1"]:
            self.metric_dict[metric]["average"] = averages[metric].item()

        return self.metric_dict
