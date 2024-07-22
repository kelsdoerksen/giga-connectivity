"""
Script to ensure that varying feature spaces
"""

import pandas as pd
import argparse

def get_args():

    parser = argparse.ArgumentParser(description='Splitting data into training and testing sets',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root_dir", help='Root directory that contains geodataframe files for regional split')
    parser.add_argument("--data_dir", help="Directory of full/uncorrelated feature space to split")
    parser.add_argument("--save_dir", help="Directory to save split data")
    parser.add_argument("--target", help="Model target, must be one of connectivity or schools")
    parser.add_argument("--split_type", help="How to split the train and test set, either split it "
                                             "percentage or geography")
    parser.add_argument("--aoi", help='Country/Region generating data for')
    return parser.parse_args()