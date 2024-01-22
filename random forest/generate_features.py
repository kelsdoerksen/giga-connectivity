"""
Script to generate features from gee-generated csvs per country
"""

import pandas as pd
import argparse
import numpy as np
import geopandas as gpd

parser = argparse.ArgumentParser(description='Generating features for Random Forest',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--aoi", help="Country of interest")
parser.add_argument("--buffer", help="Buffer extent around school to extract data from")
args = parser.parse_args()

months = ['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec']
base_filepath = '/Users/kelseydoerksen/Desktop/Giga'

def filter_schools(aoi, giga_df):
    """
    Using Isa's cleaned dataset, filter
    the schools we don't want
    :return: clean_df
    """

    print('The number of schools before cleaning the dataset is: {}'.format(len(giga_df)))
    cleaned_df = gpd.read_file('{}/isa_clean/{}_clean.geojson'.format(base_filepath, aoi))
    cleaned_school_ids = cleaned_df[cleaned_df['clean'] <=1]
    correct_schools_list = cleaned_school_ids['giga_id_school'].tolist()

    cleaned_giga_df = giga_df[giga_df['giga_id_school'].isin(correct_schools_list)]
    print('The number of schools after cleaning the dataset is: {}'.format(len(cleaned_giga_df)))

    return cleaned_giga_df


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
    print('The number of retained schools after dropping duplicates is: {}'.format(len(df_new)))

    return df_new


def calc_nearest_point(target_df, school_df, target_name):
    """
    Calculate nearest school location to point of interest
    :param target_df: geopandas df of target data
    :param school_df: geopandas df of school data
    :return:
    """

    dist_df = gpd.sjoin_nearest(school_df, target_df, how='left', distance_col='{}_distance'.format(target_name))
    dist_df = dist_df.drop_duplicates(subset=['giga_id_school'], keep='first')

    return dist_df


def get_ookla(aoi, ookla_type):
    """
    Add ookla data to feature space
    :return:
    """
    # Load data
    school_df = gpd.read_file('{}/{}/{}_school_geolocation.geojson'.format(base_filepath, aoi, aoi))
    country_mask = gpd.read_file('{}/{}/{}_extent.geojson'.format(base_filepath, aoi, aoi))
    ookla_df = gpd.read_file('{}/Ookla/2023-10-01_performance_{}_tiles/gps_{}_tiles.shp'.
                             format(base_filepath, ookla_type, ookla_type), mask=country_mask)

    # Transform crs to 3857 for distance calculation
    school_df = school_df.to_crs(crs=3857)
    ookla_df = ookla_df.to_crs(crs=3857)

    # Calculate nearest ookla point
    combined_df = calc_nearest_point(ookla_df, school_df, 'ookla')

    combined_df = combined_df[['avg_d_kbps', 'avg_u_kbps', 'avg_lat_ms', 'tests', 'devices', 'ookla_distance']]

    combined_df = combined_df.add_suffix('_{}'.format(ookla_type))

    return combined_df


def get_elec(aoi):
    """
    Add distance to electrical grid to feature space
    :param aoi:
    :return:
    """

    school_df = gpd.read_file('{}/{}/{}_school_geolocation.geojson'.format(base_filepath, aoi, aoi))
    country_mask = gpd.read_file('{}/{}/{}_extent.geojson'.format(base_filepath, aoi, aoi))
    elec_df = gpd.read_file('{}/PowerGrid/grid.gpkg'.format(base_filepath), mask=country_mask)

    # Transform crs to 3857 for distance calculation
    school_df = school_df.to_crs(crs=3857)
    elec_df = elec_df.to_crs(crs=3857)

    combined_df = calc_nearest_point(elec_df, school_df, 'transmission_line')
    distance_df = combined_df[['transmission_line_distance']]

    return distance_df


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

    df_distance = get_elec(aoi)

    df_avg_rad = get_avg_nightlight(aoi, 'avg_rad', buffer)
    df_cf_cvg = get_avg_nightlight(aoi, 'cf_cvg', buffer)

    df_ookla_mobile = get_ookla(aoi, 'mobile')
    df_ookla_fixed = get_ookla(aoi, 'fixed')

    df_label = add_label(df_giga)

    feature_space = pd.concat([df_giga['giga_id_school'], df_modis, df_pop, df_ghsl, df_ghm, df_distance,
                               df_school_level, df_avg_rad, df_cf_cvg, df_ookla_mobile, df_ookla_fixed,
                               df_label], axis=1)

    # Filter out schools from Isabelle's methodology
    filtered = filter_schools(aoi, feature_space)

    final_feature_space = filtered.drop(['Unnamed: 0'], axis=1)

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










