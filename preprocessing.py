import os
import glob

import logging
import argparse

import random
import pandas as pd
import itertools


def data_processing(input_dir):
    """ Creating positive and negative pairs from raw data-set """

    print(input_dir + '/*/*.jpg')
    images_dirs = glob.glob(input_dir + '/*/*.jpg')

    person_names = [str(s).split('/')[-2] for s in images_dirs]
    person_df = pd.DataFrame({'dir_path': images_dirs, 'person_name': person_names})
    persons_dict = {}
    for p in person_names:
        persons_dict.setdefault(p, 0)
        persons_dict[p] += 1

    positive_persons = [person for person, val in persons_dict.items() if val > 1]
    negative_persons = [person for person, val in persons_dict.items() if val == 1]

    print(len(positive_persons))

    positive_list = []
    for person in positive_persons:
        temporary_persons = list(person_df.loc[person_df['person_name'] == person, 'dir_path'])
        assert len(temporary_persons) > 1
        temporary_combos = itertools.combinations(temporary_persons, 2)
        positive_list.extend(list(temporary_combos))

    negative_imgs = list(person_df.loc[person_df['person_name'].isin(negative_persons), 'dir_path'])
    positive_imgs = list(person_df.loc[person_df['person_name'].isin(positive_persons), 'dir_path'])

    negative_list = []
    while True:
        negative_list.append((random.sample(positive_imgs, 1)[0], random.sample(negative_imgs, 1)[0]))
        if len(negative_list) > 500000:
            break
    assert len(negative_list) > 500000

    neg_frame = pd.DataFrame(negative_list, columns=['person_1', 'person_2'])
    pos_frame = pd.DataFrame(positive_list, columns=['person_1', 'person_2'])
    neg_frame['label'] = 0
    pos_frame['label'] = 1

    print(len(pos_frame))
    print(len(neg_frame))

    final_frame = pos_frame.append(neg_frame)
    final_frame.to_csv(os.path.join(input_dir, 'final.csv'), index=False)


if __name__ == '__main__':
    logging.basicConfig(filename='deepnet.log', level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input_dir', type=str, action='store', dest='input_dir', help='Input path of data to train on')

    args = parser.parse_args()
    data_processing(input_dir=args.input_dir)