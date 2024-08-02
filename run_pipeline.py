"""
Pipeline script to train and test ML classifier
for connectivity prediction
"""

import argparse
from classifiers import lr, rf, svm, mlp, gb, xgb
import wandb
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os


def get_args():
    parser = argparse.ArgumentParser(description='Running ML Pipeline for Connectivity or School Prediction')
    parser.add_argument('--model', help='ML Model. Must be one of rf, svm, lr, gb, mlp, xgb')
    parser.add_argument('--aoi', help='Country of interest')
    parser.add_argument('--buffer', help='Buffer extent for data')
    parser.add_argument('--root_dir', help='Root directory of project')
    parser.add_argument('--experiment_type', help='Experiment type')
    parser.add_argument('--features', help='Type of feature space. Must be one of engineer (hand crafted features),'
                                           'satclip-resnet18-l10, satclip-resnet18-l40, satclip-resnet50-l10,'
                                           'satclip-resnet50-l40, satclip-vit16-l10, satclip-vit16-l40, GeoCLIP, CSP,'
                                           'precursor-geofoundation_e011, combined, mobility_buffer, mobility_nearest')
    parser.add_argument('--parameter_tuning', help='Specify if parameter tuning. True or False.')
    parser.add_argument('--target', help='Specify model target. Must be one of connectivity or school')
    parser.add_argument('--data_split', help='Data split for the model. Either percentage (standard 70/30) or'
                                             'defined geography by user. Currently supports split of BWA admin'
                                             'zone 2 division')
    parser.add_argument('--wandb_dir', help='Wandb directory to save run information to')
    parser.add_argument('--results_dir', help='Results directory')

    return parser.parse_args()


def load_connectivity_data(country, buffer_extent, feature_space):
    """
    Connectivity data processing for ML classifier
    :return:
    """
    if feature_space == 'engineer':
        training_data = pd.read_csv(
            '{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        testing_data = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country,
                                                                                          buffer_extent))
        val_data = pd.read_csv('{}/{}/{}m_buffer/ValData_uncorrelated.csv'.format(root_dir, country, buffer_extent))

    if feature_space == 'engineer_with_aux':
        training_data = pd.read_csv(
            '{}/{}/{}m_buffer/TrainingData_uncorrelated_with_aux.csv'.format(root_dir, country, buffer_extent))
        testing_data = pd.read_csv(
            '{}/{}/{}m_buffer/TestingData_uncorrelated_with_aux.csv'.format(root_dir, country, buffer_extent))
        val_data = pd.read_csv(
            '{}/{}/{}m_buffer/ValData_uncorrelated_with_aux.csv'.format(root_dir, country, buffer_extent))

    if feature_space == 'engineer_with_aux_and_pop':
        training_data = pd.read_csv(
            '{}/{}/{}m_buffer/TrainingData_uncorrelated_with_aux_and_schoolage_pop.csv'
                .format(root_dir, country, buffer_extent))
        testing_data = pd.read_csv(
            '{}/{}/{}m_buffer/TestingData_uncorrelated_with_aux_and_schoolage_pop.csv'
                .format(root_dir, country, buffer_extent))
        val_data = pd.read_csv(
            '{}/{}/{}m_buffer/ValData_uncorrelated_with_aux_and_schoolage_pop.csv'
                .format(root_dir, country, buffer_extent))

    if feature_space in ['esa_combined', 'geoclip_combined', 'esa_combined_v04_e008_z18', 'esa_combined_v04_e008_z17',
                         'esa_combined_z17_v2-embeddings', 'csp_combined']:
        eng_train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir,
                                                                                           country, buffer_extent))
        eng_test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir,
                                                                                         country, buffer_extent))
        eng_val_df = pd.read_csv('{}/{}/{}m_buffer/ValData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        embed_name = ''
        if feature_space == 'esa_combined_v04_e008_z18':
            embed_name = 'embeddings_precursor-geofoundation_v04_e008_z18_embeddings'
        if feature_space == 'esa_combined_v04_e008_z17':
            embed_name = 'embeddings_precursor-geofoundation_v04_e008_z17_embeddings'
        if feature_space == 'esa_combined_z17_v2-embeddings':
            embed_name = 'esa_z17_v2-embeddings'
        if feature_space == 'geoclip_combined':
            embed_name = 'GeoCLIP'
        if feature_space == 'csp_combined':
            embed_name == 'CSP'

        emb_train_df = pd.read_csv('{}/{}/embeddings/{}_{}_TrainingData.csv'.
                                   format(root_dir, country, country, embed_name))
        emb_test_df = pd.read_csv('{}/{}/embeddings/{}_{}_TestingData.csv'.
                                  format(root_dir, country, country, embed_name))
        emb_val_df = pd.read_csv('{}/{}/embeddings/{}_{}_ValData.csv'.
                                 format(root_dir, country, country, embed_name))

        combined_train = eng_train_df.merge(emb_train_df, on=['giga_id_school', 'lat', 'lon', 'connectivity'],
                                            how='outer')
        combined_train.drop(columns=['Unnamed: 0_x', 'Unnamed: 0.1', 'Unnamed: 0_y'], errors='ignore')
        combined_test = eng_test_df.merge(emb_test_df, on=['giga_id_school', 'lat', 'lon', 'connectivity'],
                                          how='outer')
        combined_test.drop(columns=['Unnamed: 0_x', 'Unnamed: 0.1', 'Unnamed: 0_y'], errors='ignore')
        combined_val = eng_val_df.merge(emb_val_df, on=['giga_id_school', 'lat', 'lon', 'connectivity'], how='outer')
        combined_val.drop(columns=['Unnamed: 0_x', 'Unnamed: 0.1', 'Unnamed: 0_y'], errors='ignore')

        training_data = combined_train
        testing_data = combined_test
        val_data = combined_val

    if feature_space in ['GeoCLIP', 'CSP', 'satclip-resnet18-l10', 'satclip-resnet18-l40', 'satclip-resnet50-l10',
                         'satclip-resnet50-l40', 'satclip-vit16-l10', 'satclip-vit16-l40']:
        training_data = pd.read_csv('{}/{}/embeddings/{}_{}_embeddings_TrainingData.csv'.format(root_dir, country,
                                                                                                country, feature_space))
        testing_data = pd.read_csv('{}/{}/embeddings/{}_{}_embeddings_TestingData.csv'.format(root_dir, country,
                                                                                              country, feature_space))
        val_data = pd.read_csv('{}/{}/embeddings/{}_{}_embeddings_ValData.csv'.format(root_dir, country, country,
                                                                                      feature_space))

    if feature_space in ['embeddings_precursor-geofoundation_v04_e008_z17_embeddings',
                         'embeddings_precursor-geofoundation_v04_e008_z18_embeddings',
                         'embeddings_school-predictor_v01_e025_z17_embeddings',
                         'embeddings_school-predictor_v01_e025_z18_embeddings',
                         'esa_z17_v2-embeddings']:
        training_data = pd.read_csv('{}/{}/embeddings/{}_{}_TrainingData.csv'.format(root_dir, country, country,
                                                                                     feature_space))
        testing_data = pd.read_csv('{}/{}/embeddings/{}_{}_TestingData.csv'.format(root_dir, country, country,
                                                                                   feature_space))
        val_data = pd.read_csv('{}/{}/embeddings/{}_{}_ValData.csv'.format(root_dir, country, country, feature_space))

    training_dataset = training_data.rename(columns={'connectivity': 'label'})
    testing_dataset = testing_data.rename(columns={'connectivity': 'label'})
    val_dataset = val_data.rename(columns={'connectivity': 'label'})

    return training_dataset, testing_dataset, val_dataset


def load_schoolmapping_data(country, buffer_extent, feature_space, data_split_type):
    """
    School Mapping data processing for ML classifier
    :return:
    """
    if feature_space == 'mobility_buffer':
        mobility_df = pd.read_csv('{}/{}/Mobility/sample_df_mobility_300m_buffer_timeseries_combined.csv'.format(
            root_dir, country))
        mobility_df['class'] = mobility_df['class'].map({'school': 1, 'non_school': 0})
        mobility_df = mobility_df.loc[:, ~mobility_df.columns.str.contains('var')]
        train_df = mobility_df[mobility_df['dataset'] == 'train']
        test_df = mobility_df[mobility_df['dataset'] == 'test']
        val_df = mobility_df[mobility_df['dataset'] == 'val']

        train_df = pd.concat([train_df, val_df])

        training_dataset = shuffle(train_df)
        testing_dataset = shuffle(test_df)

        training_dataset = training_dataset.rename(columns={'class': 'label'})
        testing_dataset = testing_dataset.rename(columns={'class': 'label'})

        geo_training_df = pd.read_csv(
            '{}/{}/{}m_buffer_new/TrainingData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                                    buffer_extent))
        geo_test_df = pd.read_csv('{}/{}/{}m_buffer_new/TestingData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                                                     buffer_extent))
        geo_validation_df = pd.read_csv(
            '{}/{}/{}m_buffer_new/ValData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                               buffer_extent))
        # Combine train and val
        geo_train_df = pd.concat([geo_training_df, geo_validation_df])

        # combine feature spaces
        training_dataset = training_dataset.drop(columns='Unnamed: 0')
        geo_train_df = geo_train_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

        testing_dataset = testing_dataset.drop(columns='Unnamed: 0')
        geo_test_df = geo_test_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

        training_dataset_combined = training_dataset.merge(geo_train_df)
        testing_dataset_combined = testing_dataset.merge(geo_test_df)

        training_dataset_combined = training_dataset_combined.dropna()
        testing_dataset_combined = testing_dataset_combined.dropna()

        return training_dataset_combined, testing_dataset_combined

    if feature_space == 'mobility_nearest':
        mobility_df = pd.read_csv('{}/{}/Mobility/sample_df_mobility_nearest_timeseries_combined.csv'.format(
            root_dir, country))
        mobility_df = mobility_df.loc[:, ~mobility_df.columns.str.contains('var')]
        mobility_df['class'] = mobility_df['class'].map({'school': 1, 'non_school': 0})
        train_df = mobility_df[mobility_df['dataset'] == 'train']
        test_df = mobility_df[mobility_df['dataset'] == 'test']
        val_df = mobility_df[mobility_df['dataset'] == 'val']

        train_df = pd.concat([train_df, val_df])

        training_dataset = shuffle(train_df)
        testing_dataset = shuffle(test_df)

        training_dataset = training_dataset.rename(columns={'class': 'label'})
        testing_dataset = testing_dataset.rename(columns={'class': 'label'})

        geo_training_df = pd.read_csv(
            '{}/{}/{}m_buffer_new/TrainingData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                                    buffer_extent))
        geo_test_df = pd.read_csv(
            '{}/{}/{}m_buffer_new/TestingData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                                   buffer_extent))
        geo_validation_df = pd.read_csv(
            '{}/{}/{}m_buffer_new/ValData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                               buffer_extent))
        # Combine train and val
        geo_train_df = pd.concat([geo_training_df, geo_validation_df])

        # combine feature spaces
        training_dataset = training_dataset.drop(columns='Unnamed: 0')
        geo_train_df = geo_train_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

        testing_dataset = testing_dataset.drop(columns='Unnamed: 0')
        geo_test_df = geo_test_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

        training_dataset_combined = training_dataset.merge(geo_train_df)
        testing_dataset_combined = testing_dataset.merge(geo_test_df)

        training_dataset_combined = training_dataset_combined.dropna()
        testing_dataset_combined = testing_dataset_combined.dropna()

        return training_dataset_combined, testing_dataset_combined

    if feature_space == 'mobility_inter':
        mobility_df = pd.read_csv('{}/{}/Mobility/sample_df_mobility_inter_timeseries_combined.csv'.format(
            root_dir, country))
        mobility_df = mobility_df.loc[:, ~mobility_df.columns.str.contains('var')]
        mobility_df['class'] = mobility_df['class'].map({'school': 1, 'non_school': 0})
        train_df = mobility_df[mobility_df['dataset'] == 'train']
        test_df = mobility_df[mobility_df['dataset'] == 'test']
        val_df = mobility_df[mobility_df['dataset'] == 'val']

        train_df = pd.concat([train_df, val_df])

        training_dataset = shuffle(train_df)
        testing_dataset = shuffle(test_df)

        training_dataset = training_dataset.rename(columns={'class': 'label'})
        testing_dataset = testing_dataset.rename(columns={'class': 'label'})

        geo_training_df = pd.read_csv(
            '{}/{}/{}m_buffer_new/TrainingData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                                    buffer_extent))
        geo_test_df = pd.read_csv(
            '{}/{}/{}m_buffer_new/TestingData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                                   buffer_extent))
        geo_validation_df = pd.read_csv(
            '{}/{}/{}m_buffer_new/ValData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                               buffer_extent))
        # Combine train and val
        geo_train_df = pd.concat([geo_training_df, geo_validation_df])

        # combine feature spaces
        training_dataset = training_dataset.drop(columns='Unnamed: 0')
        geo_train_df = geo_train_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

        testing_dataset = testing_dataset.drop(columns='Unnamed: 0')
        geo_test_df = geo_test_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

        training_dataset_combined = training_dataset.merge(geo_train_df)
        testing_dataset_combined = testing_dataset.merge(geo_test_df)

        training_dataset_combined = training_dataset_combined.dropna()
        testing_dataset_combined = testing_dataset_combined.dropna()

        return training_dataset_combined, testing_dataset_combined

    if feature_space == 'engineer':
        if data_split_type == 'unicef':
            training_df = pd.read_csv('{}/{}/{}m_buffer_new/TrainingData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                                                       buffer_extent))
            test_df = pd.read_csv('{}/{}/{}m_buffer_new/TestingData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                                                       buffer_extent))
            validation_df = pd.read_csv('{}/{}/{}m_buffer_new/ValData_uncorrelated_uniceflabel.csv'.format(root_dir, country,
                                                                                                       buffer_extent))
            # Combine train and val and split later when hyp tuning
            train_df = pd.concat([training_df, validation_df])
        if data_split_type == 'percentage':
            train_df = pd.read_csv(
                '{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country,
                                                                                    buffer_extent))
            test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country,
                                                                                                     buffer_extent))
        if data_split_type == 'geo':
            # Geo split
            train_df = pd.read_csv(
                '{}/{}/{}m_buffer_combined/TrainingData_uncorrelated_{}_split.csv'.format(root_dir, country, buffer_extent,
                                                                                          data_split_type))
            test_df = pd.read_csv('{}/{}/{}m_buffer_combined/TestingData_uncorrelated_{}_split.csv'.format(root_dir, country,
                                                                                                           buffer_extent,
                                                                                                           data_split_type))
            # Removing boundary features since we are geospatial cross-validating?
            train_df.loc[:, ~train_df.columns.str.startswith('boundary')]
            test_df.loc[:, ~test_df.columns.str.startswith('boundary')]

    if country == 'BWA':
        train_df = train_df.drop(columns=['UID'])
        test_df = test_df.drop(columns=['UID'])

    # Drop rows if lat or lon are NaN
    training_data = train_df[train_df['lat'].notna()]
    training_data = training_data[training_data['lon'].notna()]
    testing_data = test_df[test_df['lat'].notna()]
    testing_data = testing_data[testing_data['lon'].notna()]

    training_dataset = shuffle(training_data)
    testing_dataset = shuffle(testing_data)

    training_dataset = training_dataset.rename(columns={'class': 'label'})
    testing_dataset = testing_dataset.rename(columns={'class': 'label'})

    return training_dataset, testing_dataset


def get_class_balance(train_df, test_df, val_df, savedir):
    """
    Calculate class balance for total, train and test set to save
    :param: train_df: training data
    :param: test_df: testing data
    :param: savedir: save directory
    :return:
    """

    positive_samples_total = sum(train_df['label']) + sum(test_df['label'] + sum(val_df['label']))
    negative_sampels_total = len(train_df) + len(test_df) + len(val_df) - positive_samples_total

    with open('{}/classbalance.txt'.format(savedir), 'w') as f:
        f.write('Number of positive samples total is: {}'.format(positive_samples_total))
        f.write('Number of negative samples total is: {}'.format(negative_sampels_total))
        f.write('Number of positive training samples is: {}'.format(sum(train_df['label'])))
        f.write('Number of negative training samples is: {}'.format(len(train_df) - sum(train_df['label'])))
        f.write('Number of positive testing samples is: {}'.format(sum(test_df['label'])))
        f.write('Number of negative testing samples is: {}'.format(len(test_df) - sum(test_df['label'])))
        f.write('Number of positive validation samples is: {}'.format(sum(val_df['label'])))
        f.write('Number of negative validation samples is: {}'.format(len(val_df) - sum(val_df['label'])))


def preprocess_samples(train_df, test_df, val_df):
    """
    Light preprocessing to data before running through models
    """
    cols_to_drop = ['connectivity', 'lat', 'lon', 'giga_id_school', 'location', 'data_split', 'label', 'Unnamed: 0',
                    'school_locations', 'class.1', 'fid', 'UID', 'dataset', 'iso', 'rurban', 'class',
                    'UID', 'class_y', 'Unnamed: 0.1', 'connectivity.1', 'Unnamed: 0.2', 'connectivity.1',
                    'Unnamed: 0_x',  'Unnamed: 0_y', 'split']

    X_test = test_df.drop(columns=cols_to_drop, errors='ignore')
    X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
    X_val = val_df.drop(columns=cols_to_drop, errors='ignore')


    cols_list = X_train.columns.tolist()
    print('Scaling data...')
    # scale the data
    train_scaler = MinMaxScaler()
    train_scaler.fit(X_train)
    X_train_scaled = train_scaler.transform(X_train)

    test_scaler = MinMaxScaler()
    test_scaler.fit(X_test)
    X_test_scaled = test_scaler.transform(X_test)

    val_scaler = MinMaxScaler()
    val_scaler.fit(X_val)
    X_val_scaled = val_scaler.transform(X_val)

    # Return as dfs
    X_train_scaled_df = pd.DataFrame(data=X_train_scaled, columns=cols_list)
    X_test_scaled_df = pd.DataFrame(data=X_test_scaled, columns=cols_list)
    X_val_scaled_df = pd.DataFrame(data=X_val_scaled, columns=cols_list)

    return X_train_scaled_df, X_test_scaled_df, X_val_scaled_df


if __name__ == '__main__':
    print('Starting run at: {}'.format(datetime.now()))
    args = get_args()
    model = args.model
    aoi = args.aoi
    buffer = args.buffer
    root_dir = args.root_dir
    experiment_type = args.experiment_type
    features = args.features
    hyper_tuning = args.parameter_tuning
    model_target = args.target
    data_split = args.data_split
    wandb_dir = args.wandb_dir
    results_dir = args.results_dir

    # Set up experiment
    experiment = wandb.init(project='giga-research',
                            tags=['aaai'],
                            mode=str(experiment_type),
                            resume='allow',
                            anonymous='must',
                            dir='wandb_env')
    experiment.config.update(
        dict(aoi=aoi, buffer=buffer, model=model, target=model_target,
             data_split=data_split, features=features)
    )

    if model_target == 'connectivity':
        label = 'connectivity'
        train_data, test_data, val_data = load_connectivity_data(aoi, buffer, features)
    if model_target == 'school':
        label = 'class'
        train_data, test_data = load_schoolmapping_data(aoi, buffer, features, data_split)

    # Make results directory
    if experiment_type == 'offline':
        results = '{}/{}/results_{}m/{}_{}'.format(root_dir, aoi, buffer, model, wandb.run.id)
        os.mkdir(results)
    else:
        if not os.path.exists('{}/{}/results_{}m/{}_{}'.format(results_dir, aoi, buffer, model, experiment.name)):
            results = '{}/{}/results_{}m/{}_{}'.format(results_dir, aoi, buffer, model, experiment.name)
            os.mkdir(results)

    cols_to_drop = ['Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 0.2', 'Unnamed: 0.1_x', 'Unnamed: 0.1_y',
                    'Unnamed: 0_x', 'Unnamed: 0_y']
    test_data = test_data.drop(columns=cols_to_drop, errors='ignore')
    train_data = train_data.drop(columns=cols_to_drop, errors='ignore')
    val_data = val_data.drop(columns=cols_to_drop, errors='ignore')

    test_data = test_data.dropna()
    train_data = train_data.dropna()
    val_data = val_data.dropna()

    # Save lat, lon for X test and then drop label and location data
    test_latitudes = test_data['lat']
    test_longitudes = test_data['lon']

    # Save data class balance breakdown
    get_class_balance(train_data, test_data, val_data, results)

    ytrain = train_data['label']
    ytest = test_data['label']
    yval = val_data['label']
    Xtrain, Xtest, Xval = preprocess_samples(train_data, test_data, val_data)


    # Run model
    if model == 'rf':
        rf.run_rf(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            Xval,
            yval,
            test_latitudes,
            test_longitudes,
            experiment,
            results,
            hyper_tuning)

    if model == 'lr':
        lr.run_lr(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            Xval,
            yval,
            test_latitudes,
            test_longitudes,
            experiment,
            results,
            hyper_tuning)

    if model == 'svm':
        svm.run_svm(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            Xval,
            yval,
            test_latitudes,
            test_longitudes,
            experiment,
            results,
            hyper_tuning)

    if model == 'mlp':
        mlp.run_mlp(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            Xval,
            yval,
            test_latitudes,
            test_longitudes,
            experiment,
            results,
            hyper_tuning)

    if model == 'gb':
        gb.run_gb(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            Xval,
            yval,
            test_latitudes,
            test_longitudes,
            experiment,
            results,
            hyper_tuning)

    if model =='xgb':
        xgb.run_xgb(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            Xval,
            yval,
            test_latitudes,
            test_longitudes,
            experiment,
            results,
            hyper_tuning)
