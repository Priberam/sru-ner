""" Used to create the synthetic datasets from the BC5CDR corpus """

import os
import argparse

def main(BC5_train_path, BC5_dev_path, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    with open(BC5_train_path, 'r') as f:
        BC5_train = f.readlines()
    with open(BC5_dev_path, 'r') as f:
        BC5_dev = f.readlines()

    # Create disjoint subsets
    break_train = [id for id, text in enumerate(BC5_train) if text == 'Pretreatment\tO\n'][-1]
    break_dev = [id for id, text in enumerate(BC5_dev) if text == 'Metformin\tB-Chemical\n'][-1]

    train_sp1 = BC5_train[:break_train]
    train_sp2 = BC5_train[break_train:]

    dev_sp1 = BC5_dev[:break_dev]
    dev_sp2 = BC5_dev[break_dev:]

    # Specialize each subset in an entity type
    splits = (([train_sp1, dev_sp1], 'Disease', 'Chemical'), ([train_sp2, dev_sp2], 'Chemical', 'Disease'))

    for split in splits:
        accept_tag = split[1]
        remove_tag = split[-1]

        for sp, type in zip(split[0], ['train', 'dev']):
            new_sp = []

            for line in sp:
                if line.endswith(f'B-{remove_tag}\n') or line.endswith(f'I-{remove_tag}\n'):
                    word = line.split('\t')[0]
                    new_line = word + '\t' + 'O\n'
                    new_sp.append(new_line)
                else:
                    new_sp.append(line)

            if new_sp[-1] == '\n':
                new_sp = new_sp[:-1]

            # Count number of entities and sentences
            numb_ents = sum([1 for l in new_sp if l.endswith(f'B-{accept_tag}\n')])
            numb_sents = new_sp.count('\n') + 1

            save_path = os.path.join(save_dir, f'{type}_{accept_tag}.tsv')
            with open(save_path, 'w') as f:
                f.writelines(new_sp)


            print()
            print(f'Create split {save_path}')
            print('Numb sentences', numb_sents)
            print('Numb entities', numb_ents)
            print()

    print('-- DONE --')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Creates disjoint splits from the BC5 corpus")
    parser.add_argument(
        "--BC5_train_path",
        type=str,
        required=True,
        help="Path to original BC5 train dataset.",
    )
    parser.add_argument(
        "--BC5_dev_path",
        type=str,
        required=True,
        help="Path to original BC5 dev dataset.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path of output folder.",
    )
    args = parser.parse_args()

    main(BC5_train_path=args.BC5_train_path,
         BC5_dev_path=args.BC5_dev_path,
         save_dir=args.save_dir)
