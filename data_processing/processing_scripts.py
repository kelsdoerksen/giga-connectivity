import json
import numpy as np

"""
Some data preprocessing functions
"""


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

    # drop the identified features
    df_filtered = df.drop(features_to_drop, axis=1)

    return df_filtered








