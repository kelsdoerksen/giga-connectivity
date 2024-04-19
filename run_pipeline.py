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
    parser = argparse.ArgumentParser(description='Running ML Pipeline for Connectivity or School Prediction')
    parser.add_argument('--model', help='ML Model. Must be one of rf, svm, lr, gb, mlp.')
    parser.add_argument('--aoi', help='Country of interest')
    parser.add_argument('--buffer', help='Buffer extent for data')
    parser.add_argument('--root_dir', help='Root directory of project')
    parser.add_argument('--experiment_type', help='Experiment type')
    parser.add_argument('--features', help='Type of feature space. Must be one of engineer (hand crafted features),'
                                           'satclip-resnet18-l10, satclip-resnet18-l40, satclip-resnet50-l10,'
                                           'satclip-resnet50-l40, satclip-vit16-l10, satclip-vit16-l40,'
                                           'precursor-geofoundation_e011, combined, dino-vitb14, dino-vitl14,'
                                           'dino-vits14')
    parser.add_argument('--parameter_tuning', help='Specify if parameter tuning. True or False.')
    parser.add_argument('--target', help='Specify model target. Must be one of connectivity or school')

    return parser.parse_args()


def load_data(country, buffer_extent, feature_space, model_label):
    """
    Loading data
    :param aoi:
    :param buffer_extent:
    :param feature_space: refers to the features to be loaded, engineered or embeddings
    :param model_label: target of ML classifier, connectivity or school-non-school
    :return:
    """

    if feature_space == 'engineer':
        train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))

        if country == 'BWA':
            train_df = train_df.drop(columns=['UID'])
            test_df = test_df.drop(columns=['UID'])

        # Drop rows if lat or lon are NaN
        training_data = train_df[train_df['lat'].notna()]
        training_data = training_data[training_data['lon'].notna()]
        testing_data = test_df[test_df['lat'].notna()]
        testing_data = testing_data[testing_data['lon'].notna()]

    if feature_space == 'engineer_with_aux':
        train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated_with_aux.csv'.format(root_dir, country, buffer_extent))
        test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated_with_aux.csv'.format(root_dir, country, buffer_extent))

        training_data = train_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'connectivity.1', 'Unnamed: 0.2'])
        testing_data = test_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'connectivity.1', 'Unnamed: 0.2'])

        # Drop rows if lat or lon are NaN
        training_data = training_data[training_data['lat'].notna()]
        training_data = training_data[training_data['lon'].notna()]
        testing_data = testing_data[testing_data['lat'].notna()]
        testing_data = testing_data[testing_data['lon'].notna()]

    if feature_space in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
        # Read in dino embeddings and append
        eng_train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country,
                                                                                           buffer_extent))
        eng_test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country,
                                                                                           buffer_extent))
        dino_emb = pd.read_csv('{}/{}/Embeddings/BWA_{}_embeds.csv'.format(root_dir, country, feature_space))

        merge_train = pd.merge(eng_train_df, dino_emb, on='UID')
        merge_test = pd.merge(eng_test_df, dino_emb, on='UID')

        train_df = merge_train.drop(columns=['UID', 'class_y', 'dataset', 'iso', 'rurban'])
        test_df = merge_test.drop(columns=['UID', 'class_y', 'dataset', 'iso', 'rurban'])

        train_df = train_df.rename(columns={'class_x': 'label'})
        test_df = test_df.rename(columns={'class_x': 'label'})

        # Drop rows if lat or lon are NaN
        training_data = train_df[train_df['lat'].notna()]
        training_data = training_data[training_data['lon'].notna()]
        testing_data = test_df[test_df['lat'].notna()]
        testing_data = testing_data[testing_data['lon'].notna()]

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

        training_data = train_df.drop(columns=['UID', 'dataset', 'iso', 'rurban', 'class'])
        testing_data = test_df.drop(columns=['UID', 'dataset', 'iso', 'rurban', 'class'])


    if feature_space == 'combined':
        eng_train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        eng_test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(root_dir, country, buffer_extent))
        emb_train_df = pd.read_csv('{}/{}/embeddings/TrainingData_embeddings_precursor-geofoundation_v04_e008_z18_embeddings.csv'.
                                   format(root_dir, country))
        emb_test_df = pd.read_csv('{}/{}/embeddings/TestingData_embeddings_precursor-geofoundation_v04_e008_z18_embeddings.csv'.
                                   format(root_dir, country))

        emb_test_df = emb_test_df.drop(columns=['Unnamed: 0', 'fid', 'location', 'connectivity', 'lat', 'lon'])
        emb_train_df = emb_train_df.drop(columns=['Unnamed: 0', 'fid', 'location', 'connectivity', 'lat', 'lon'])
        eng_train_df = eng_train_df.drop(columns=['Unnamed: 0', 'Unnamed: 0', 'connectivity.1'])
        eng_test_df = eng_test_df.drop(columns=['Unnamed: 0', 'Unnamed: 0', 'connectivity.1'])
        eng_train_df = eng_train_df.sort_values(by='giga_id_school')
        eng_test_df = eng_test_df.sort_values(by='giga_id_school')
        emb_train_df = emb_train_df.sort_values(by='giga_id_school')
        emb_test_df = emb_test_df.sort_values(by='giga_id_school')

        combined_train = pd.concat([eng_train_df, emb_train_df],axis=1)
        combined_test = pd.concat([eng_test_df, emb_test_df],axis=1)

        combined_train = combined_train.drop(columns=['Unnamed: 0.1', 'giga_id_school', 'school_locations'])
        combined_test = combined_test.drop(columns=['Unnamed: 0.1', 'giga_id_school', 'school_locations'])

        training_data = combined_train
        testing_data = combined_test

    if feature_space in ['GeoCLIP',
                         'CSPfMoW', 'satclip-resnet18-l10', 'satclip-resnet18-l40', 'satclip-resnet50-l10',
                         'satclip-resnet50-l40', 'satclip-vit16-l10', 'satclip-vit16-l40']:

        training_data = pd.read_csv('{}/{}/embeddings/{}_{}_embeddings_Training.csv'.format(root_dir, country, country,
                                                                                             feature_space))
        testing_data = pd.read_csv('{}/{}/embeddings/{}_{}_embeddings_Testing.csv'.format(root_dir, country, country,
                                                                                             feature_space))

        training_data = training_data.drop(columns=['Unnamed: 0', 'data_split'])
        testing_data = testing_data.drop(columns=['Unnamed: 0', 'data_split'])

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

        training_data = training_data.drop(columns=['Unnamed: 0', 'giga_id_school', 'fid'])
        testing_data = testing_data.drop(columns=['Unnamed: 0', 'giga_id_school', 'fid'])

    if feature_space in ['esa_z18_v2-embeddings', "esa_z17_v2-embeddings"]:
        training_data = pd.read_csv('{}/{}/embeddings/{}_Train_{}.csv'.format(root_dir, country, country,
                                                                                             feature_space))
        testing_data = pd.read_csv('{}/{}/embeddings/{}_Test_{}.csv'.format(root_dir, country, country,
                                                                              feature_space))
        training_data = training_data.drop(columns=['Unnamed: 0'])
        testing_data = testing_data.drop(columns=['Unnamed: 0'])

    training_dataset = shuffle(training_data)
    testing_dataset = shuffle(testing_data)

    # Rename column to label for ease
    if model_label == 'connectivity':
        training_dataset = training_dataset.rename(columns={'connectivity': 'label'})
        testing_dataset = testing_dataset.rename(columns={'connectivity': 'label'})
    if model_label == 'class':
        if 'school_locations' in training_dataset.columns:
            training_dataset = training_dataset.rename(columns={'class': 'label'})
            training_dataset = training_dataset.drop(columns=['Unnamed: 0', 'class.1', 'school_locations'])
            testing_dataset = testing_dataset.rename(columns={'class': 'label'})
            testing_dataset = testing_dataset.drop(columns=['Unnamed: 0', 'class.1', 'school_locations'])

    return training_dataset, testing_dataset


def get_class_balance(train_df, test_df, savedir):
    """
    Calculate class balance for total, train and test set to save
    :param: train_df: training data
    :param: test_df: testing data
    :param: savedir: save directory
    :return:
    """

    positive_samples_total = sum(train_df['label']) + sum(test_df['label'])
    negative_sampels_total = len(train_df) + len(test_df) - positive_samples_total

    with open('{}/classbalance.txt'.format(savedir), 'w') as f:
        f.write('Number of positive samples total is: {}'.format(positive_samples_total))
        f.write('Number of negative samples total is: {}'.format(negative_sampels_total))
        f.write('Number of positive training samples is: {}'.format(sum(train_df['label'])))
        f.write('Number of negative training samples is: {}'.format(len(train_df) - sum(train_df['label'])))
        f.write('Number of positive testing samples is: {}'.format(sum(test_df['label'])))
        f.write('Number of negative testing samples is: {}'.format(len(test_df) - sum(test_df['label'])))


def preprocess_samples(train_df, test_df, model_type, feature_set, target_type):
    """
    Light preprocessing to data before running through models
    """
    cols_to_drop = []
    if target_type == 'school':
        cols_to_drop = ['lat', 'lon', 'label']

    if target_type == 'connectivity':
        if feature_set == 'engineer':
            cols_to_drop = ['connectivity', 'lat', 'lon', 'school_locations', 'giga_id_school']
        elif feature_set == 'engineer_with_aux':
            cols_to_drop = ['connectivity', 'lat', 'lon', 'school_locations', 'giga_id_school']
        elif feature_set == 'combined':
            cols_to_drop = ['connectivity', 'lat', 'lon']
        elif feature_set in ['esa_z18_v2-embeddings', "esa_z17_v2-embeddings"]:
            cols_to_drop = ['connectivity', 'lat', 'lon', 'data_split', 'giga_id_school']
        elif feature_set in ['embeddings_precursor-geofoundation_v04_e025_z17',
                             'embeddings_precursor-geofoundation_v04_e025_z18',
                             'embeddings_school-predictor_v01_e025_z17',
                             'embeddings_school-predictor_v01_e025_z18',
                             'embeddings_precursor-geofoundation_v04_e008_z18',
                             'embeddings_precursor-geofoundation_v04_e008_z17']:

            cols_to_drop = ['location', 'connectivity', 'lat', 'lon']
        elif feature_set == 'combined':
            cols_to_drop = ['lat', 'lon', 'connectivity']
        else:
            cols_to_drop = ['connectivity', 'lat', 'lon', 'giga_id_school']
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
    hyper_tuning = args.parameter_tuning
    model_target = args.target

    # Set up experiment
    experiment = wandb.init(project='giga-research',
                            resume='allow', anonymous='must')
    experiment.config.update(
        dict(aoi=aoi, buffer=buffer, model=model, target=model_target)
    )

    if model_target == 'connectivity':
        label = 'connectivity'
    if model_target == 'school':
        label = 'class'

    # Make results directory
    if experiment_type == 'offline':
        results = '{}/{}/results_{}m/{}'.format(root_dir, aoi, buffer, wandb.run.id)
    else:
        if not os.path.exists('{}/{}/results_{}m/{}'.format(root_dir, aoi, buffer, experiment.name)):
            results = '{}/{}/results_{}m/{}'.format(root_dir, aoi, buffer, experiment.name)
            os.mkdir(results)

    # Load data
    train_data, test_data = load_data(aoi, buffer, features, label)

    # Save data class balance breakdown
    get_class_balance(train_data, test_data, results)

    # Save lat, lon for X test and then drop label and location data
    test_latitudes = test_data['lat']
    test_longitudes = test_data['lon']

    # Some light preprocessing on data before running through models
    train_df = train_data.dropna()
    test_df = test_data.dropna()

    ytrain = train_df['label']
    ytest = test_df['label']
    Xtrain, Xtest = preprocess_samples(train_df, test_df, model, features, model_target)

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
            results,
            hyper_tuning)

    if model == 'lr':
        lr.run_lr(
            Xtrain,
            ytrain,
            Xtest,
            ytest,
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
            test_latitudes,
            test_longitudes,
            experiment,
            results,
            hyper_tuning)
