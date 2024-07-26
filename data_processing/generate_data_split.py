"""
Preprocessing scripts to split feature space into 70% train, 15% test,
15% val for deterministic results
"""

import random
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import geopandas as gpd


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

seed = random.randint(0, 1000)


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


if __name__ == '__main__':
    args = get_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    target = args.target
    split_type = args.split_type
    root_dir = args.root_dir
    aoi = args.aoi

    # Read in dataset
    dataset = pd.read_csv('{}/uncorrelated_feature_space.csv'.format(data_dir))

    if target == 'connectivity':
        label = dataset['connectivity']
        dataset = dataset.drop(columns=['connectivity'])
    if target == 'schools':
        label = dataset['label']
        dataset = dataset.drop(columns=['label'])

    if split_type == 'percentage':
        print('Running for data split: {}'.format(split_type))
        X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3, random_state = seed)
        train = pd.concat([X_train, y_train], axis=1)

        # Split test in half for validation
        X_val, X_test_half, y_val, y_test_half = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)
        test = pd.concat([X_test_half, y_test_half], axis=1)
        val = pd.concat([X_val, y_val], axis=1)

        train.to_csv('{}/TrainingData.csv'.format(save_dir))
        test.to_csv('{}/TestingData.csv'.format(save_dir))
        val.to_csv('{}/ValData.csv'.format(save_dir))

    if split_type == 'geography':
        for region in list(region_dict.keys()):
            print('Running for data split: {}, region: {}'.format(split_type, region))
            # Split by the geographic region of interest
            aoi_polygon = gpd.read_file('{}/geoBoundaries-{}-ADM2.geojson'.format(root_dir, aoi))
            df_of_features = subset_by_region(aoi_polygon, aoi, dataset)

            train = df_of_features[df_of_features['split'] == 'Train']
            test = df_of_features[df_of_features['split'] == 'Test']

            # Subset a percentage of the training set as validation for hyperparameter tuning
            train = train.sample(n=len(train))
            val = train[0:int(0.15*len(train))]
            train = train.iloc[int(0.15*len(train)):]


            # Drop the column denoting split, and drop columns with admin boundaries because we don't want to include this
            train = train.drop(columns='split')
            test = test.drop(columns='split')
            val = val.drop(columns='split')
            train.to_csv('{}/TrainingData_{}_split.csv'.format(save_dir, region))
            test.to_csv('{}/TestingData_{}_split.csv'.format(save_dir, region))
            val.to_csv('{}/ValData_{}_split.csv'.format(save_dir, region))