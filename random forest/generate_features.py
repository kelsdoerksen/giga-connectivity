"""
Script to generate features from gee-generated csvs per country
"""

import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Generating features for Random Forest',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--aoi", help="Country of interest")
parser.add_argument("--buffer", help="Buffer extent around school to extract data from")
args = parser.parse_args()

months = ['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec']
base_filepath = '/Users/kelseydoerksen/Desktop/Giga'


def get_avg_nightlight(location, rad_band, buffer):
    '''
    Get average of monthly radiance
    :return: df of average rad
    '''

    root_dir = '{}/{}'.format(base_filepath, location)
    df_list = []
    for month in months:
        df_list.append(pd.read_csv('{}/{}m_buffer/nightlight_{}_{}_2022_custom_buffersize_{}_with_time.csv'.
                                   format(root_dir, buffer, rad_band, month, buffer)))

    df_total = pd.concat(df_list)
    by_row_index = df_total.groupby(df_total.index)
    mean_vals = by_row_index.mean()

    # rename the variables so we don't get confused
    mean_vals = mean_vals.rename(columns={"nightlight.var": "nightlight.{}.var".format(rad_band),
                              "nightlight.mean": "nightlight.{}.mean".format(rad_band),
                              "nightlight.max": "nightlight.{}.max".format(rad_band),
                              "nightlight.min": "nightlight.{}.min".format(rad_band)})

    output_df = mean_vals.drop(['Unnamed: 0'], axis=1)

    return output_df


def add_label(df):
    '''
    Add connectivity label information
    :return: df of labels
    '''
    print('Starting with {} schools'.format(len(df)))
    # Replace Yes/No with 1/0 for label
    df['connectivity'] = df['connectivity'].map({'Yes': 1, 'No': 0})

    return df['connectivity']


def unique_lat_lon(df):
    '''
    Remove lat, lon points that are repetitive
    param: df: input dataframe to query lat, lon points
    :return:
    '''
    lon_vals = df['lon'].values.tolist()
    lat_vals = df['lat'].values.tolist()

    school_locations = []
    for i in range(len(lon_vals)):
        school_locations.append((lon_vals[i][0], lat_vals[i][0]))

    df['school_locations'] = school_locations

    # Drop duplicate school locations
    df_new = df.drop_duplicates(subset=['school_locations'])
    # Drop nans
    df_new = df_new.dropna()

    print('The number of retained schools after dropping duplicates and NaNs is: {}'.format(len(df_new)))
    return df_new


def get_elec_data(aoi):
    """
    Grab the data for the tranmission line distance
    Note I am using a modified csv file from golden truth on
    ML Azure
    :return: df_dist: distance of school point to transmission line
    """

    df_trans_lines = pd.read_csv('{}/PowerGrid/{}_school_points_with_transmission_line_distance.csv'.format(
        base_filepath, aoi))

    df_dist = df_trans_lines['distance_to_transmission_line_network']

    return df_dist

def get_education_level(df):
    """
    Get school type from dataframe encoded as category
    :return:
    """
    education_level_dict = {
        'Pre-Primary': 0,
        'Primary': 1,
        'Secondary': 2,
        'Pre-Primary and Secondary': 3,
        'Primary and Secondary': 4,
        'Pre-Primary and Primary': 5,
        'Pre-Primary and Primary and Secondary': 6,
        'Pre-Primary, Primary and Secondary': 6,
        'Primary, Secondary and Post-Secondary': 7,
        'Post-Secondary': 8,
        'Other': 9
    }

    # Map education level to categories
    df['education_level'] = df['education_level'].map(education_level_dict)

    return df['education_level']


def get_feature_space(aoi, buffer):
    '''
    Grab relevant data and aggregate together
    to return final df of feature space for location
    Grabs modis, population and nightlight features
    :return: df of features for aoi specified
    '''

    df_giga = pd.read_csv('{}/{}/{}_school_geolocation_coverage_master.csv'.format(base_filepath, aoi, aoi))

    df_school_level = get_education_level(df_giga)

    df_modis = pd.read_csv('{}/{}/{}m_buffer/modis_LC_Type1_2020_custom_buffersize_{}_with_time.csv'.format(base_filepath, aoi,
                                                                                                    buffer, buffer))

    df_pop = pd.read_csv('{}/{}/{}m_buffer/pop_population_density_2020_custom_buffersize_{}_with_time.csv'.format
                         (base_filepath, aoi, buffer, buffer))

    df_ghsl = pd.read_csv('{}/{}/{}m_buffer/human_settlement_layer_built_up_built_characteristics_2018'
                          '_custom_buffersize_{}_with_time.csv'.format(base_filepath, aoi, buffer, buffer))

    df_ghm = pd.read_csv('{}/{}/{}m_buffer/global_human_modification_gHM_2016'
                         '_custom_buffersize_{}_with_time.csv'.format(base_filepath, aoi, buffer, buffer))

    df_distance = get_elec_data(aoi)

    df_avg_rad = get_avg_nightlight(aoi, 'avg_rad', buffer)
    df_cf_cvg = get_avg_nightlight(aoi, 'cf_cvg', buffer)

    df_label = add_label(df_giga)

    feature_space = pd.concat([df_modis, df_pop, df_ghsl, df_ghm, df_distance, df_school_level,
                               df_avg_rad, df_cf_cvg, df_label], axis=1)
    final_feature_space = feature_space.drop(['Unnamed: 0'], axis=1)

    # Drop nans that exist in the label -> we can't validate if there is/is not connectivity
    final_feature_space = final_feature_space[final_feature_space['connectivity'].notna()]

    # Drop if there are multiple entries of the same lat, lon point
    unique_df = unique_lat_lon(final_feature_space)

    # Remove duplicated column names from concatenating
    unique_df = unique_df.loc[:, ~unique_df.columns.duplicated()]

    unique_df.to_csv('{}/{}/{}m_buffer/full_feature_space.csv'.format(base_filepath, aoi, buffer))


def calculate_feature_correlation(data):
    """
    Calculate correlation matrix for features
    :param data:
    :return:
    """
    corrMatrix = data.corr()
    #sn.heatmap(corrMatrix, annot=True)
    #plt.show()

    # Positive correlation matrix
    corr_df = data.corr().abs()
    # Create and apply mask
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    tri_df = corr_df.mask(mask)
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.9)]



# Calling function to generate features
get_feature_space(args.aoi, args.buffer)

# Loading features and looking at correlation
'''
aoi = 'BWA'
buffer='5000'
features = pd.read_csv('{}/{}/{}m_buffer/full_feature_space.csv'.format(base_filepath, aoi, buffer))
# Drop label, lat, lon from features
only_feats = features.drop(columns=['connectivity', 'lat', 'lon', 'Unnamed: 0'])
calculate_feature_correlation(only_feats)
'''












