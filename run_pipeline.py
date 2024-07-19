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
                                           'satclip-resnet50-l40, satclip-vit16-l10, satclip-vit16-l40, '
                                           'precursor-geofoundation_e011, combined, dino-vitb14, dino-vitl14,'
                                           'dino-vits14, mobility_buffer, mobility_nearest')
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
        train_df = pd.read_csv(
            '{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        val_df = pd.read_csv('{}/{}/{}m_buffer/ValData_uncorrelated.csv'.format(root_dir, country, buffer_extent))

    if feature_space == 'engineer_with_aux':
        train_df = pd.read_csv(
            '{}/{}/{}m_buffer/TrainingData_uncorrelated_with_aux.csv'.format(root_dir, country, buffer_extent))
        test_df = pd.read_csv(
            '{}/{}/{}m_buffer/TestingData_uncorrelated_with_aux.csv'.format(root_dir, country, buffer_extent))

    if feature_space in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
        # Read in dino embeddings and append
        eng_train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country,
                                                                                           buffer_extent))
        eng_test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country,
                                                                                         buffer_extent))
        dino_emb = pd.read_csv('{}/{}/Embeddings/BWA_{}_embeds.csv'.format(root_dir, country, feature_space))

        merge_train = pd.merge(eng_train_df, dino_emb, on='UID')
        merge_test = pd.merge(eng_test_df, dino_emb, on='UID')

        train_df = train_df.rename(columns={'class_x': 'label'})
        test_df = test_df.rename(columns={'class_x': 'label'})

    if feature_space in ['dinov2_vits14_o', 'dinov2_vitb14_o', 'dinov2_vitl14_o']:
        # Read in dino embeddings and append
        eng_train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country,
                                                                                           buffer_extent))
        eng_test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country,
                                                                                         buffer_extent))
        dino_emb = pd.read_csv('{}/{}/Embeddings/BWA_{}_embeds.csv'.format(root_dir, country, feature_space[0:13]))
        dino_emb['label'] = dino_emb['class'].map({'school': 1, 'non_school': 0})

        train_uid = eng_train_df['UID']
        test_uid = eng_test_df['UID']

        # Get lat, lons so we can plot later in figs
        train_lats = eng_train_df.sort_values(by='UID')['lat']
        train_lons = eng_train_df.sort_values(by='UID')['lon']

        test_lats = eng_test_df.sort_values(by='UID')['lat']
        test_lons = eng_test_df.sort_values(by='UID')['lon']

        # Subset dinov2 to ensure we have matching UIDs to concat
        train_df = dino_emb[dino_emb.UID.isin(train_uid)]
        test_df = dino_emb[dino_emb.UID.isin(test_uid)]

        train_df = train_df.sort_values(by='UID')
        test_df = test_df.sort_values(by='UID')

        # Add lat, lon to df so we can plot later
        train_df['lat'] = train_lats.values
        train_df['lon'] = train_lons.values
        test_df['lat'] = test_lats.values
        test_df['lon'] = test_lons.values

    if feature_space == 'combined':
        eng_train_df = pd.read_csv(
            '{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        eng_test_df = pd.read_csv(
            '{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        emb_train_df = pd.read_csv(
            '{}/{}/embeddings/TrainingData_embeddings_precursor-geofoundation_v04_e008_z18_embeddings.csv'.
            format(root_dir, country))
        emb_test_df = pd.read_csv(
            '{}/{}/embeddings/TestingData_embeddings_precursor-geofoundation_v04_e008_z18_embeddings.csv'.
            format(root_dir, country))

        eng_train_df = eng_train_df.sort_values(by='giga_id_school')
        eng_test_df = eng_test_df.sort_values(by='giga_id_school')
        emb_train_df = emb_train_df.sort_values(by='giga_id_school')
        emb_test_df = emb_test_df.sort_values(by='giga_id_school')

        combined_train = pd.concat([eng_train_df, emb_train_df], axis=1)
        combined_test = pd.concat([eng_test_df, emb_test_df], axis=1)

        training_data = combined_train
        testing_data = combined_test

    if feature_space in ['GeoCLIP',
                         'CSPfMoW', 'satclip-resnet18-l10', 'satclip-resnet18-l40', 'satclip-resnet50-l10',
                         'satclip-resnet50-l40', 'satclip-vit16-l10', 'satclip-vit16-l40']:
        training_data = pd.read_csv('{}/{}/embeddings/{}_{}_embeddings_Training.csv'.format(root_dir, country, country,
                                                                                            feature_space))
        testing_data = pd.read_csv('{}/{}/embeddings/{}_{}_embeddings_Testing.csv'.format(root_dir, country, country,
                                                                                          feature_space))

    if feature_space in ['embeddings_precursor-geofoundation_v04_e025_z17',
                         'embeddings_precursor-geofoundation_v04_e025_z18',
                         'embeddings_school-predictor_v01_e025_z17',
                         'embeddings_school-predictor_v01_e025_z18',
                         'embeddings_precursor-geofoundation_v04_e008_z18',
                         'embeddings_precursor-geofoundation_v04_e008_z17']:
        training_data = pd.read_csv('{}/{}/embeddings/TrainingData_{}_embeddings.csv'.format(root_dir, country,
                                                                                             feature_space))
        testing_data = pd.read_csv('{}/{}/embeddings/TestingData_{}_embeddings.csv'.format(root_dir, country,
                                                                                           feature_space))
    if feature_space in ['esa_z18_v2-embeddings', "esa_z17_v2-embeddings"]:
        training_data = pd.read_csv('{}/{}/embeddings/{}_Train_{}.csv'.format(root_dir, country, country,
                                                                              feature_space))
        testing_data = pd.read_csv('{}/{}/embeddings/{}_Test_{}.csv'.format(root_dir, country, country,
                                                                            feature_space))

    training_dataset = training_data.rename(columns={'connectivity': 'label'})
    testing_dataset = testing_data.rename(columns={'connectivity': 'label'})

    return training_dataset, testing_dataset


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
                    'UID', 'class_y', 'Unnamed: 0.1', 'connectivity.1', 'Unnamed: 0.2', ]

    X_test = test_df.drop(columns=cols_to_drop, errors='ignore')
    X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
    X_val = val_df.drop(columns=cols_to_drop, errors='ignore')


    cols_list = X_train.columns.tolist()
    print('Scaling data...')
    # scale the data if we are not using trees/GB model
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
                            resume='allow',
                            anonymous='must',
                            dir='wandb_env')
    experiment.config.update(
        dict(aoi=aoi, buffer=buffer, model=model, target=model_target,
             data_split=data_split)
    )

    if model_target == 'connectivity':
        label = 'connectivity'
        train_data, test_data, val_data = load_connectivity_data(aoi, buffer, features)
    if model_target == 'school':
        label = 'class'
        train_data, test_data = load_schoolmapping_data(aoi, buffer, features, data_split)

    # Make results directory
    if experiment_type == 'offline':
        results = '{}/{}/results_{}m/{}'.format(root_dir, aoi, buffer, wandb.run.id)
    else:
        if not os.path.exists('{}/{}/results_{}m/{}'.format(results_dir, aoi, buffer, experiment.name)):
            results = '{}/{}/results_{}m/{}'.format(results_dir, aoi, buffer, experiment.name)

    os.mkdir(results)

    # Save data class balance breakdown
    get_class_balance(train_data, test_data, val_data, results)

    # Save lat, lon for X test and then drop label and location data
    test_latitudes = test_data['lat']
    test_longitudes = test_data['lon']

    # Some light preprocessing on data before running through models
    train_df = train_data.dropna()
    test_df = test_data.dropna()
    val_df = val_data.dropna()

    ytrain = train_df['label']
    ytest = test_df['label']
    yval = val_df['label']
    Xtrain, Xtest, Xval = preprocess_samples(train_df, test_df, val_df)

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
