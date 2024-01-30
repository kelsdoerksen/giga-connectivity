"""
Pipeline script to train and test ML classifier
for connectivity prediction
"""

import argparse
from classifiers import lr, rf, svm, mlp, gb
import wandb
import pandas as pd
from sklearn.utils import shuffle
import os


def get_args():
    parser = argparse.ArgumentParser(description='Running ML Pipeline for Connectivity Prediction')
    parser.add_argument('--model', help='ML Model. Must be one of rf, svm, lr, xgb, mlp.')
    parser.add_argument('--aoi', help='Country of interest')
    parser.add_argument('--buffer', help='Buffer extent for data')
    parser.add_argument('--root_dir', help='Root directory of project')

    return parser.parse_args()


def load_data(country, buffer_extent):
    """
    Loading data
    :param aoi:
    :param buffer_extent:
    :return:
    """
    train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
    test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))

    training_data = train_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'connectivity.1'])
    testing_data = test_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'connectivity.1'])

    training_data = shuffle(training_data)
    testing_data = shuffle(testing_data)

    # Drop rows if lat or lon are NaN
    training_dataset = training_data[training_data['lat'].notna()]
    training_dataset = training_data[training_data['lon'].notna()]
    testing_dataset = testing_data[testing_data['lat'].notna()]
    testing_dataset = testing_data[testing_data['lon'].notna()]

    return training_dataset, testing_dataset


def get_class_balance(train_df, test_df, savedir):
    """
    Calculate class balance for total, train and test set to save
    :param: train_df: training data
    :param: test_df: testing data
    :param: savedir: savedirectory
    :return:
    """
    positive_samples_total = sum(train_df['connectivity']) + sum(test_df['connectivity'])
    negative_sampels_total = len(train_df) + len(test_df) - positive_samples_total

    with open('{}/classbalance.txt'.format(savedir), 'w') as f:
        f.write('Number of positive samples total is: {}'.format(positive_samples_total))
        f.write('Number of negative samples total is: {}'.format(negative_sampels_total))
        f.write('Number of positive training samples is: {}'.format(sum(train_df['connectivity'])))
        f.write('Number of negative training samples is: {}'.format(len(train_df) - sum(train_df['connectivity'])))
        f.write('Number of positive testing samples is: {}'.format(sum(test_df['connectivity'])))
        f.write('Number of negative testing samples is: {}'.format(len(test_df) - sum(test_df['connectivity'])))


def preprocess_samples(train_df, test_df):
    """
    Light preprocessing to data before running through models
    """
    cols_to_drop = ['connectivity', 'lat', 'lon', 'school_locations', 'giga_id_school']
    X_test = test_df.drop(columns=cols_to_drop)
    X_train = train_df.drop(columns=cols_to_drop)

    return X_train, X_test


if __name__ == '__main__':
    args = get_args()
    model = args.model
    aoi = args.aoi
    buffer = args.buffer
    root_dir = args.root_dir

    # Set up experiment
    experiment = wandb.init(project='giga-research', resume='allow', anonymous='must')
    experiment.config.update(
        dict(aoi=aoi, buffer=buffer, model=model)
    )

    # Make results directory
    results = '{}/{}/results_{}m/{}'.format(root_dir, aoi, buffer, experiment.name)
    os.makedirs(results)

    # Load data
    train_data, test_data = load_data(aoi, buffer)

    # Save data class balance breakdown
    get_class_balance(train_data, test_data, results)

    # Save lat, lon for X test and then drop label and location data
    test_latitudes = test_data['lat']
    test_longitudes = test_data['lon']

    # Some light preprocessing on data before running through models
    train_df = train_data.dropna()
    test_df = test_data.dropna()

    ytrain = train_df['connectivity']
    ytest = test_df['connectivity']
    Xtrain, Xtest = preprocess_samples(train_df, test_df)

    # Run model
    if model == 'rf':
        rf.run_rf(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            test_latitudes,
            test_longitudes,
            experiment,
            results)

    if model == 'lr':
        lr.run_lr(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            test_latitudes,
            test_longitudes,
            experiment,
            results)

    if model == 'svm':
        svm.run_svm(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            test_latitudes,
            test_longitudes,
            experiment,
            results)

    if model == 'mlp':
        mlp.run_mlp(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            test_latitudes,
            test_longitudes,
            experiment,
            results)

    if model == 'xgb':
        gb.run_xgb(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            test_latitudes,
            test_longitudes,
            experiment,
            results)








