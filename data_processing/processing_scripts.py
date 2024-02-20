import pandas as pd
import json
import argparse
import numpy as np

"""
Some scrap scripts/functions
"""

parser = argparse.ArgumentParser(description='Generating features for Random Forest',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--aoi", help="Country of interest")
args = parser.parse_args()

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


# Running for schools
#save_dir = '/Users/kelseydoerksen/Desktop/Giga/{}'.format(args.aoi)
#df = pd.read_csv('{}/{}_school_geolocation_coverage_master.csv'.format(args.aoi, save_dir))
#get_lat_lon_list(df, args.aoi, save_dir)


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


aois = ['GIN']
for aoi in aois:
    root_dir = '/Users/kelseydoerksen/Desktop/Giga/{}/1000m_buffer'.format(aoi)
    df = pd.read_csv('{}/full_feature_space_fixed.csv'.format(root_dir))
    identity_cols = ['giga_id_school', 'lat', 'lon', 'connectivity', 'school_locations']
    df_identity = df[identity_cols]
    df = df.drop(columns=['Unnamed: 0', 'giga_id_school', 'lat', 'lon', 'connectivity', 'school_locations'])
    df_filt = eliminate_correlated_features(df, 0.9)

    combined_df = pd.concat([df_identity, df_filt], axis=1)
    combined_df.to_csv('{}/uncorrelated_feature_space_fixed.csv'.format(root_dir))


'''
# Scrap scripts for fixing the lon = lat value issue
# Aka, all I need to do is load in the coordinates for the country of interest, get the list of lon values from this
# and for each of the data csvs I have, replace the lon column data with the correct lon values
aois = ['BWA']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec']
for aoi in aois:
    coords = open('/Users/kelseydoerksen/Desktop/Giga/{}/{}_coordinates.json'.format(aoi, aoi), 'r')
    coords_data = json.loads(coords.read())
    correct_lons = coords_data['lons']

    # Fix modis
    modis_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/modis_2020_custom_buffersize_5000_with_time.csv'.
                           format(aoi))
    modis_df['lon'] = correct_lons
    modis_df.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/modis_2020_custom_buffersize_5000_with_time.csv'.
                           format(aoi))

    # Fix population
    pop_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/pop_2020_custom_buffersize_5000_with_time.csv'.
                           format(aoi))
    pop_df['lon'] = correct_lons
    pop_df.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/pop_2020_custom_buffersize_5000_with_time.csv'.
                    format(aoi))

    for month in months:
        # Fix cf_cvg
        cf_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/nightlight_cf_cvg_{}_2022_'
                            'custom_buffersize_5000_with_time.csv'.format(aoi, month))
        cf_df['lon'] = correct_lons
        cf_df.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/nightlight_avg_rad_{}_2022_'
                            'custom_buffersize_5000_with_time.csv'.format(aoi, month))
        # Fix avg_rad
        avg_rad_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/nightlight_avg_rad_{}_2022_'
                            'custom_buffersize_5000_with_time.csv'.format(aoi, month))
        avg_rad_df['lon'] = correct_lons
        avg_rad_df.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/nightlight_avg_rad_{}_2022_'
                     'custom_buffersize_5000_with_time.csv'.format(aoi, month))
'''






