import pandas as pd
import json
import argparse
import numpy as np
import geopandas as gpd

"""
Some scrap scripts/functions
"""

parser = argparse.ArgumentParser(description='Generating features for Random Forest',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--aoi", help="Country of interest")
parser.add_argument("--target", help="ML target, must be one of connectivity or schools")
args = parser.parse_args()
aoi = args.aoi
target = args.target


def get_lat_lon_list(df, country_name, save_dir):
    """
    Get list of latitude, longitude
    points of schools, removing duplicates
    :param: df: dataframe of school information
    :param: save_dir: save directory for lats, lons
    :return: json of lat, lon points of school locations
    """

    # remove any repeating schools based on giga id
    df_unique = df.drop_duplicates(subset=['giga_id_school'], keep='first')

    coords_dict = {
        'lats': df_unique['lat'].tolist(),
        'lons': df_unique['lon'].tolist()
    }

    with open("{}/{}_coordinates.json".format(save_dir, country_name), "w") as outfile:
        json.dump(coords_dict, outfile)


def get_lat_lon_list_from_gdp(df, country_name, save_dir):
    """
    Get lat lon list from geopandas dataframe
    of clean school/non-schools
    :return:
    """

    # change crs to EPSG:4326 for later data extraction
    df_epsg = df.to_crs(crs='EPSG:4326')

    # remove any repeating schools based on giga id
    df_unique = df_epsg.drop_duplicates(subset=['UID'], keep='first')

    coords_dict = {
        'lats': df_unique['geometry'].y.tolist(),
        'lons': df_unique['geometry'].x.tolist()
    }

    with open("{}/{}_coordinates.json".format(save_dir, country_name), "w") as outfile:
        json.dump(coords_dict, outfile)


def eliminate_correlated_features(df, threshold):
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
    with open('{}/correlated_features.txt'.format(root_dir), 'w') as f:
        f.write('Features removed were: {}'.format(features_to_drop))

    # drop the identified features
    df_filtered = df.drop(features_to_drop, axis=1)

    return df_filtered



'''
# Running uncorrelated feature selection -> uncomment if you would like to run
buffer = 1000
aois = ['BWA']
target = 'schools'
for aoi in aois:
    print('Running for buffer: {}, aoi: {}'.format(buffer, aoi))
    if target == 'connectivity':
        root_dir = '/Users/kelseydoerksen/Desktop/Giga/Connectivity/{}/{}m_buffer'.format(aoi, buffer)
        identity_cols = ['giga_id_school', 'lat', 'lon', 'connectivity', 'school_locations']
        cols_to_drop = ['Unnamed: 0', 'giga_id_school', 'lat', 'lon', 'connectivity', 'school_locations']
    if target == 'schools':
        root_dir = '/Users/kelseydoerksen/Desktop/Giga/SchoolMapping/{}/{}m_buffer_nonschool'.format(aoi, buffer)
        identity_cols = ['UID', 'lat', 'lon', 'class']
        cols_to_drop = ['Unnamed: 0', 'UID', 'lat', 'lon', 'class']

    df = pd.read_csv('{}/full_feature_space.csv'.format(root_dir))

    df_identity = df[identity_cols]
    df = df.drop(columns=cols_to_drop)
    df_filt = eliminate_correlated_features(df, 0.9)

    combined_df = pd.concat([df_identity, df_filt], axis=1)
    combined_df.to_csv('{}/uncorrelated_feature_space.csv'.format(root_dir))
'''

# Running lat, lon coordinate generation for airPy processing script
aois = ['BWA']
for aoi in aois:
    gpd_df = gpd.read_file('/Users/kelseydoerksen/Desktop/Giga/SchoolMapping/{}/{}_train.geojson'.format(aoi, aoi))
    save_dir = '/Users/kelseydoerksen/Desktop/Giga/SchoolMapping/{}'.format(aoi)
    get_lat_lon_list_from_gdp(gpd_df, aoi, save_dir)








