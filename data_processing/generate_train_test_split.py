"""
Preprocessing scripts to split feature space into 70% train, 30% test for
deterministic results
"""

import random
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import geopandas as gpd
import numpy as np

parser = argparse.ArgumentParser(description='Splitting data into training and testing sets',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--aoi", help="Country of interest")
parser.add_argument("--buffer", help="Buffer extent around school to extract data from")
parser.add_argument("--target", help="Model target, must be one of connectivity or schools")
parser.add_argument("--split_type", help="How to split the train and test set, either split it percentage or geography")
args = parser.parse_args()

aoi = args.aoi
buffer = args.buffer
target = args.target
split_type = args.split_type

if target == 'connectivity':
    base_filepath = '/Users/kelseydoerksen/Desktop/Giga/Connectivity'
    label = 'connectivity'
if target == 'schools':
    base_filepath = '/Users/kelseydoerksen/Desktop/Giga/SchoolMapping'
    label = 'class'

seed = random.seed(46)

# Admin 1 zones; admin 2 zones to spatially cross-validate
region_dict = {
    'South-East': ['Gaborone', 'Lobatse', 'South East'],
    'Kgatleng': ['Kgatleng'],
    'North-East': ['Francistown', 'North East'],
    'Central': ['Bobonong', 'Boteti', 'Mahalapye', 'Orapa', 'Selibe Phikwe', 'Serowe Palapye', 'Sowa Town', 'Tutume'],
    'Ghanzi': ['Central Kgalagadi Game Reserve', 'Ghanzi'],
    'Kweneng': ['Kweneng East', 'Kweneng West'],
    'Southern': ['Barolong', 'Jwaneng', 'Ngwaketse West', 'Southern'],
    'Kgalagadi': ['Kgalagadi North', 'Kgalagadi South'],
    'North-West': ['Ngamiland Delta', 'Ngamiland East', 'Ngamiland West'],
    'Chobe': ['Chobe']
}


def eliminate_correlated_features(df, threshold, save_dir):
    """
    Modified function from giga-ml-utils
    :return: df of uncorrelated features
    """
    # calculate correlation matrix
    corr_matrix = df.corr().abs()

    # identify pairs of highly correlated features
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = (
        upper_triangle.stack().reset_index().rename(columns={0: 'correlation'}))
    highly_correlated_pairs = highly_correlated_pairs[
        highly_correlated_pairs['correlation'] > threshold]

    # drop features from level_0
    features_to_drop = set()
    for idx, row in highly_correlated_pairs.iterrows():
        features_to_drop.add(row['level_0'])  # Drop the first feature
    print('Features removed were: {}'.format(features_to_drop))
    with open('{}/correlated_features.txt'.format(save_dir), 'w') as f:
        f.write('Features removed were: {}'.format(features_to_drop))

    return list(features_to_drop)


def subset_by_region(poly_gdf, region_subset, samples_df):
    """
    Divide samples based on geographic location
    :param: region_subset: region to subset data by to be test set
    :param: poly_gdf: polygon geodataframe of country
    :param: samples_df: df of sample features to label as train, test
    :return:
    """
    # Subset polygon into test region of interest
    region_gdf_subset = poly_gdf[poly_gdf['shapeName'].isin(region_dict['{}'.format(region_subset)])]
    region_gdf_union = region_gdf_subset.unary_union

    # Get samples df to geo dataframe
    samples_gdf = gpd.GeoDataFrame(samples_df, geometry=gpd.points_from_xy(samples_df.lon, samples_df.lat),
                                   crs='EPSG:4326')

    # Get intersection of region subset and samples_gdf so we can label as train or test
    samples_region_label = samples_gdf.within(region_gdf_union)*1
    samples_region_label = samples_region_label.map({0: 'Train', 1: 'Test'})

    # add split back to samples df and return
    samples_df['split'] = samples_region_label
    return samples_df


# Read in dataset
print('Processing features for {}'.format(aoi))
if target == 'schools':
    '''
    dataset_schools = pd.read_csv('{}/{}/{}m_buffer_school/full_feature_space.csv'.format(base_filepath, aoi, buffer))
    dataset_nonschools = pd.read_csv('{}/{}/{}m_buffer_nonschool/full_feature_space.csv'
                                     .format(base_filepath, aoi, buffer))
    dataset = pd.concat([dataset_schools, dataset_nonschools])
    dataset = dataset.drop(columns=['Unnamed: 0'])
    '''
    dataset = pd.read_csv('{}/{}/{}m_buffer_new/full_feature_space.csv'.format(base_filepath, aoi, buffer))
    # Set save directory
    save_directory = '{}/{}/{}m_buffer_new'.format(base_filepath, aoi, buffer)

    # Remove correlated features
    dataset_no_info = dataset.drop(columns=['UID', 'lat', 'lon', 'class', 'school_locations'])
    features_to_remove = eliminate_correlated_features(dataset_no_info, 0.9, save_directory)
    dataset_uncorr = dataset.drop(columns=features_to_remove)

    if split_type == 'unicef':
        print('Splitting data according to unicef-defined train/val/test')
        unicef_poly = gpd.read_file('{}/{}/{}_train.geojson'.format(base_filepath, aoi, aoi))
        unicef_poly_id = unicef_poly[['dataset', 'UID']]
        combined_df = pd.merge(dataset_uncorr, unicef_poly_id, on='UID')

        train = combined_df[combined_df['dataset'] == 'train']
        test = combined_df[combined_df['dataset'] == 'test']
        val = combined_df[combined_df['dataset'] == 'val']

        train.to_csv('{}/{}/{}m_buffer_new/TrainingData_uncorrelated_uniceflabel.csv'.format(base_filepath, aoi, buffer))
        test.to_csv('{}/{}/{}m_buffer_new/TestingData_uncorrelated_uniceflabel.csv'.format(base_filepath, aoi, buffer))
        val.to_csv('{}/{}/{}m_buffer_new/ValData_uncorrelated_uniceflabel.csv'.format(base_filepath, aoi, buffer))

    if split_type == 'percentage':
        print('Running for data split: {}'.format(split_type))
        X_train, X_test, y_train, y_test = train_test_split(dataset_uncorr, dataset_uncorr[label],
                                                            test_size=0.3, random_state = seed)
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)
        train.to_csv('{}/{}/{}m_buffer/TrainingData_uncorrelated.csv'.format(base_filepath, aoi, buffer))
        test.to_csv('{}/{}/{}m_buffer/TestingData_uncorrelated.csv'.format(base_filepath, aoi, buffer))

    if split_type == 'geography':
        for region in list(region_dict.keys()):
            print('Running for data split: {}, region: {}'.format(split_type, region))
            # Split by the geographic region of interest
            aoi_polygon = gpd.read_file('{}/{}/geoBoundaries-{}-ADM2.geojson'.format(base_filepath, aoi, aoi))
            df_of_features = subset_by_region(aoi_polygon, region, dataset_uncorr)

            train = df_of_features[df_of_features['split'] == 'Train']
            test = df_of_features[df_of_features['split'] == 'Test']

            # Drop the column denoting split, and drop columns with admin boundaries because we don't want to include this
            train = train.drop(columns='split')
            test = test.drop(columns='split')
            train.to_csv('{}/TrainingData_uncorrelated_{}_split.csv'.format(save_directory, region))
            test.to_csv('{}/TestingData_uncorrelated_{}_split.csv'.format(save_directory, region))
