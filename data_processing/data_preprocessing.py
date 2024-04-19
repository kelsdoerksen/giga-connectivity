"""
Preprocessing scripts to split feature space into 70% train, 30% test for
deterministic results
"""

import random
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Splitting data into training and testing sets',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--aoi", help="Country of interest")
parser.add_argument("--buffer", help="Buffer extent around school to extract data from")
parser.add_argument("--target", help="Model target, must be one of connectivity or schools")
args = parser.parse_args()

aoi = args.aoi
buffer = args.buffer
target = args.target

if target == 'connectivity':
    base_filepath = '/Users/kelseydoerksen/Desktop/Giga/Connectivity'
    label = 'connectivity'
if target == 'schools':
    base_filepath = '/Users/kelseydoerksen/Desktop/Giga/SchoolMapping'
    label = 'class'

seed = random.seed(46)

# Read in dataset
print('Processing features for {}'.format(aoi))
dataset = pd.read_csv('{}/{}/{}m_buffer/uncorrelated_feature_space.csv'.format(base_filepath, aoi, buffer))
if target == 'schools':
    dataset = dataset.drop(columns=['Unnamed: 0'])

X_train, X_test, y_train, y_test = train_test_split(dataset, dataset[label],
                                                    test_size=0.3, random_state = seed)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train.to_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(base_filepath, aoi, buffer))
test.to_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(base_filepath, aoi, buffer))






























