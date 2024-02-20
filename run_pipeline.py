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
from sklearn.preprocessing import MinMaxScaler
import ast


def get_args():
    parser = argparse.ArgumentParser(description='Running ML Pipeline for Connectivity Prediction')
    parser.add_argument('--model', help='ML Model. Must be one of rf, svm, lr, gb, mlp.')
    parser.add_argument('--aoi', help='Country of interest')
    parser.add_argument('--buffer', help='Buffer extent for data')
    parser.add_argument('--root_dir', help='Root directory of project')
    parser.add_argument('--experiment_type', help='Experiment type')
    parser.add_argument('--features', help='Type of feature space. Must be one of engineer (hand crafted features),'
                                           'satclip-resnet18-l10, satclip-resnet18-l40, satclip-resnet50-l10,'
                                           'satclip-resnet50-l40, satclip-vit16-l10, satclip-vit16-l40,'
                                           'precursor-geofoundation_e011, combined')

    return parser.parse_args()


def load_data(country, buffer_extent, feature_space):
    """
    Loading data
    :param aoi:
    :param buffer_extent:
    :param feature_space: refers to the features to be loaded, engineered or embeddings
    :return:
    """

    if feature_space == 'engineer':
        train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated_fixed.csv'.format(root_dir, country, buffer_extent))
        test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated_fixed.csv'.format(root_dir, country, buffer_extent))

        training_data = train_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'connectivity.1'])
        testing_data = test_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'connectivity.1'])

        # Drop rows if lat or lon are NaN
        training_data = training_data[training_data['lat'].notna()]
        training_data = training_data[training_data['lon'].notna()]
        testing_data = testing_data[testing_data['lat'].notna()]
        testing_data = testing_data[testing_data['lon'].notna()]

    if feature_space == 'combined':
        eng_train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        eng_test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        emb_train_df = pd.read_csv('{}/{}/embeddings/TrainingData_precursor-geofoundation_e011_s1_z18_embeddings.csv'.
                                   format(root_dir, country))
        emb_test_df = pd.read_csv('{}/{}/embeddings/TestingData_precursor-geofoundation_e011_s1_z18_embeddings.csv'.
                                   format(root_dir, country))

        emb_train_df = emb_train_df.drop(columns=['connectivity', 'Unnamed: 0'])
        emb_test_df = emb_test_df.drop(columns=['connectivity', 'Unnamed: 0'])

        # sort the dataframes by location so we can match the features accordingly
        eng_train_df = eng_train_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
        eng_test_df = eng_test_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

        emb_train_df = emb_train_df.sort_values(by=['location']).reset_index()
        emb_test_df = emb_test_df.sort_values(by=['location']).reset_index()

        eng_train_df = eng_train_df.sort_values(by=['school_locations']).reset_index()
        eng_test_df = eng_test_df.sort_values(by=['school_locations']).reset_index()

        training_data = pd.concat([eng_train_df, emb_train_df], axis=1)
        testing_data = pd.concat([eng_test_df, emb_test_df], axis=1)

        training_data = training_data.drop(columns=['index'])
        testing_data = testing_data.drop(columns=['index'])

    if feature_space in ['precursor-geofoundation_v04_e008_z18', "precursor-geofoundation_v04_e008_z17", 'GeoCLIP',
                         'CSPfMoW', 'satclip-resnet18-l10', 'satclip-resnet18-l40', 'satclip-resnet50-l10',
                         'satclip-resnet50-l40', 'satclip-vit16-l10', 'satclip-vit16-l40']:

        training_data = pd.read_csv('{}/{}/embeddings/TrainingData_{}_embeddings_fixed.csv'.format(root_dir, country,
                                                                                             feature_space))
        testing_data = pd.read_csv('{}/{}/embeddings/TestingData_{}_embeddings_fixed.csv'.format(root_dir, country,
                                                                                           feature_space))

        training_data = training_data.drop(columns=['Unnamed: 0'])
        testing_data = testing_data.drop(columns=['Unnamed: 0'])

    if feature_space in ['embeddings_precursor-geofoundation_v04_e025_z17',
                         'embeddings_precursor-geofoundation_v04_e025_z18',
                         'embeddings_school-predictor_botswana_v01_e025_z17',
                         'embeddings_school-predictor_botswana_v01_e025_z18']:
        training_data = pd.read_csv('{}/{}/embeddings/TrainingData_{}_embeddings.csv'.format(root_dir, country,
                                                                                             feature_space))
        testing_data = pd.read_csv('{}/{}/embeddings/TestingData_{}_embeddings.csv'.format(root_dir, country,
                                                                                           feature_space))
        training_data = training_data.drop(columns=['Unnamed: 0', 'giga_id_school', 'fid'])
        testing_data = testing_data.drop(columns=['Unnamed: 0', 'giga_id_school', 'fid'])

    training_dataset = shuffle(training_data)
    testing_dataset = shuffle(testing_data)

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


def preprocess_samples(train_df, test_df, model_type, feature_set):
    """
    Light preprocessing to data before running through models
    """
    cols_to_drop = []
    if feature_set == 'engineer':
        cols_to_drop = ['connectivity', 'lat', 'lon', 'school_locations', 'giga_id_school']
    elif feature_set == 'combined':
        cols_to_drop = ['connectivity', 'lat', 'lon', 'school_locations', 'giga_id_school', 'location', 'connectivity.1']
    else:
        cols_to_drop = ['connectivity', 'location']
    X_test = test_df.drop(columns=cols_to_drop)
    X_train = train_df.drop(columns=cols_to_drop)

    models_with_scaling = ['svm', 'lr', 'mlp']
    if model_type in models_with_scaling:
        cols_list = X_train.columns.tolist()
        print('Scaling data...')
        # scale the data if we are not using trees/GB model
        train_scaler = MinMaxScaler()
        train_scaler.fit(X_train)
        X_train_scaled = train_scaler.transform(X_train)

        test_scaler = MinMaxScaler()
        test_scaler.fit(X_test)
        X_test_scaled = test_scaler.transform(X_test)

        # Return as dfs
        X_train_scaled_df = pd.DataFrame(data=X_train_scaled, columns=cols_list)
        X_test_scaled_df = pd.DataFrame(data=X_test_scaled, columns=cols_list)
        return X_train_scaled_df, X_test_scaled_df

    return X_train, X_test


if __name__ == '__main__':
    args = get_args()
    model = args.model
    aoi = args.aoi
    buffer = args.buffer
    root_dir = args.root_dir
    experiment_type = args.experiment_type
    features = args.features

    # Set up experiment
    experiment = wandb.init(project='giga-research',
                            resume='allow', anonymous='must')
    experiment.config.update(
        dict(aoi=aoi, buffer=buffer, model=model)
    )

    # Make results directory
    if experiment_type == 'offline':
        results = '{}/{}/results_{}m/{}'.format(root_dir, aoi, buffer, wandb.run.id)
    else:
        results = '{}/{}/results_{}m/{}'.format(root_dir, aoi, buffer, experiment.name)
    os.mkdir(results)

    # Load data
    train_data, test_data = load_data(aoi, buffer, features)

    # Save data class balance breakdown
    get_class_balance(train_data, test_data, results)

    # Save lat, lon for X test and then drop label and location data
    if features == 'engineer':
        test_latitudes = test_data['lat']
        test_longitudes = test_data['lon']
    else:
        # A little more processing to get the location info
        test_longitudes = []
        test_latitudes = []
        for i in range(len(test_data)):
            lon = ast.literal_eval(test_data['location'].loc[i])[0]
            lat = ast.literal_eval(test_data['location'].loc[i])[1]
            test_longitudes.append(lon)
            test_longitudes.append(lat)
        test_longitudes = pd.Series(test_longitudes)
        test_latitudes = pd.Series(test_latitudes)

    # Some light preprocessing on data before running through models
    train_df = train_data.dropna()
    test_df = test_data.dropna()

    ytrain = train_df['connectivity']
    ytest = test_df['connectivity']
    Xtrain, Xtest = preprocess_samples(train_df, test_df, model, features)

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

    if model == 'gb':
        gb.run_gb(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            test_latitudes,
            test_longitudes,
            experiment,
            results)
