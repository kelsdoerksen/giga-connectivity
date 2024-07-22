"""
Script to ensure that varying feature spaces
"""

import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Splitting data into training and testing sets',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root_dir", help='Root directory that contains geodataframe files for regional split')
    parser.add_argument("--save_dir", help="Directory to save split data")
    parser.add_argument("--aoi", help='Country/Region generating data for')
    parser.add_argument("--buffer", help='Buffer extent')
    return parser.parse_args()


def get_data_split(split_type, features, id):
    '''
    :param split_type: Train/Test/Val
    :param features: Feature df
    :param id: ID df
    :return:
    '''
    if split_type == 'Training':
        split = 'Train'
    if split_type == 'Testing':
        split = 'Test'
    if split_type == 'Val':
        split = split_type

    combined = features.merge(id, on=['giga_id_school', 'lat', 'lon'], how='outer')
    split_df = combined[combined['split'] == split]
    split_df = split_df.reset_index().drop(columns=['index'])

    return split_df


if __name__ == '__main__':
    args = get_args()
    root = args.root_dir
    save = args.save_dir
    aoi = args.aoi
    buffer = args.buffer

    feature_df = pd.read_csv('{}/{}/{}m_buffer/uncorrelated_feature_space.csv'.format(root, aoi, buffer))
    id_df = pd.read_csv('{}/{}/{}_id_info.csv'.format(root, aoi, aoi))
    splits = ['Training', 'Testing', 'Val']
    for split in splits:
        df = get_data_split(split, feature_df, id_df)
        df.to_csv('{}/{}Data_uncorrelated.csv'.format(save, split))
