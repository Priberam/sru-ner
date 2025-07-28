""" Script used to filter the predictions to specific entity types. """
import autorootcwd
import pandas as pd
import pickle
from src.modules.utils.action_utils import LabeledSubstring
import argparse
import yaml


def main(tag_info_path, pred_data_path, save_path):
    # Get tag specification
    with open(tag_info_path, 'r') as f:
        tag_info = yaml.safe_load(f)

    # Get the prediction data
    with open(pred_data_path, 'rb') as f:
        aux = pickle.load(f)
    
    word_data = aux['data']
    pred_df = aux['pred_df'][aux['pred_df']['pred_tag'].apply(lambda x: x is not None)]

    predictions = {}

    for row_id, row in pred_df.iterrows():
        sent_id = row['sent_id']
        wordpieces = word_data['wordpieces'][sent_id]
        source_dts = row['source_dataset']
        start_wp_idx = row['start_idx']
        end_wp_idx = row['end_idx']
        text_span = wordpieces[start_wp_idx : end_wp_idx]
        mentions_of_accept_types = [y for y in row['pred_tag'] if y in tag_info[source_dts]['relevant_types']]

        if len(mentions_of_accept_types):
            predictions[sent_id] = {'wordpieces': wordpieces,
                                    'pred_mentions': [],
                                    'gold_mentions': [],
                                    'source_dts': source_dts}
            
            for m in mentions_of_accept_types:
                span = LabeledSubstring(start=start_wp_idx, end=end_wp_idx,
                                        tag=m, text=text_span)
                predictions[sent_id]['pred_mentions'].append(span)

            predictions[sent_id]['pred_mentions'].sort(key= lambda x: (x.start, -x.end, x.tag))

            if row['gold_tag'] is not None:
                for g in row['gold_tag']:
                    span = LabeledSubstring(start=start_wp_idx, end=end_wp_idx,
                                            tag=g, text=text_span)
                    predictions[sent_id]['gold_mentions'].append(span)

                predictions[sent_id]['gold_mentions'].sort(key= lambda x: (x.start, -x.end, x.tag))

    with open(save_path, 'w') as f:

        all_source_dts = set(x['source_dts'] for x in list(predictions.values()))

        for source_dts in all_source_dts:
            sub = {sent_id : value for sent_id, value in predictions.items() if value['source_dts'] == source_dts}

            f.write("###############\n")
            f.write(f"PREDICTIONS IN DATASET {source_dts} OF TYPES {sorted(tag_info[source_dts]['relevant_types'])}\n")
            f.write("###############\n\n")

            for sent_id, value in sub.items():
                f.write(f"sent_idx={sent_id}\n")
                f.write(str(value['wordpieces']) + '\n\n')
                if len(value['gold_mentions']):
                    f.write('GOLD MENTIONS: \n')
                    for g in value['gold_mentions']:
                        f.write(str(g) + '\n')
                else:
                    f.write('GOLD MENTIONS: None.\n')

                f.write('PRED MENTIONS:\n')
                for p in value['pred_mentions']:
                    f.write(str(p) + '\n')

                f.write('\n----------\n\n')

    print('\n\nWrote output to ', save_path)
    print('DONE')
    print()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter predictions to specific entity types."
    )

    # Make sure that each entry of the yaml has the following structure
    # {source_dataset_of_the_sentence}:
    #   relevant_types:
    #       - {type_name_1}
    #       - {type_name_2}
    #       - ...
    parser.add_argument(
        "--tag_specification_yaml",
        type=str,
        required=True,
        help="Path to the yaml.",
    )

    parser.add_argument(
        "--prediction_pickle",
        type=str,
        required=True,
        help="Path to the prediction pickle.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output txt file.",
    )

    args = parser.parse_args()

    main(
        tag_info_path=args.tag_specification_yaml,
        pred_data_path=args.prediction_pickle,
        save_path=args.output_path
    )
