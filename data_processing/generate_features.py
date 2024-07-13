"""
Script to generate features from gee-generated csvs per country
"""

import pandas as pd
import argparse
import geopandas as gpd
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser(description='Generating features for ML Classifiers',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--aoi", help="Country of interest")
parser.add_argument("--buffer", help="Buffer extent around school to extract data from")
parser.add_argument("--data_source", help='Source of school location data. If unicef, can proceed as normal'
                                          ' if new_schools we are generating from the new school samples')
parser.add_argument("--target", help='Specify is target is connectivity prediction or school prediction')
args = parser.parse_args()

months = ['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec']


def filter_schools(aoi, giga_df, base_filepath):
    """
    Using Isa's cleaned dataset, filter
    the schools we don't want
    :return: clean_df
    """

    print('The number of schools before cleaning the dataset is: {}'.format(len(giga_df)))
    cleaned_df = gpd.read_file('{}/isa_clean/{}_clean.geojson'.format(base_filepath, aoi))
    cleaned_school_ids = cleaned_df[cleaned_df['clean'] <=1]    # 1 means I am removing kindergarden
    correct_schools_list = cleaned_school_ids['giga_id_school'].tolist()

    cleaned_giga_df = giga_df[giga_df['giga_id_school'].isin(correct_schools_list)]
    print('The number of schools after cleaning the dataset is: {}'.format(len(cleaned_giga_df)))

    return cleaned_giga_df


def get_avg_nightlight(location, rad_band, buffer, b_fp):
    '''
    Get average of monthly radiance
    :return: df of average rad
    '''
    root_dir = '{}/{}'.format(b_fp, location)
    df_list = []
    for month in months:
        df_list.append(pd.read_csv('{}/{}m_buffer_new/nightlight_{}_{}_2022_custom_buffersize_{}_with_time.csv'.
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


def calc_nearest_point(target_df, school_df, target_name, data_source):
    """
    Calculate nearest school location to point of interest
    :param target_df: geopandas df of target data
    :param school_df: geopandas df of school data
    :return:
    """

    dist_df = gpd.sjoin_nearest(school_df, target_df, how='left', distance_col='{}_distance'.format(target_name))

    if 'connectivity' in school_df.columns:
        if data_source == 'unicef':
            dist_df = dist_df.drop_duplicates(subset=['giga_id_school'], keep='first')
        if data_source == 'new_schools':
            dist_df = dist_df.drop_duplicates(subset=['fid'], keep='first')
        if data_source == 'combined':
            dist_df = dist_df.drop_duplicates(subset=['UID'], keep='first')

    if 'UID' in school_df.columns:
        dist_df = dist_df.drop_duplicates(subset=['UID'], keep='first')

    return dist_df


def get_ookla(aoi, ookla_type, base_filepath, school_df, data_source):
    """
    Add ookla data to feature space
    :return:
    """
    # Load data
    if aoi == 'BWA':
        country_mask = gpd.read_file('{}/{}/{}_extent.geojson'.format(base_filepath, aoi, aoi))
    else:
        country_mask = gpd.read_file('{}/{}/geoBoundaries-{}-ADM2.geojson'.format(base_filepath, aoi, aoi))
    ookla_df = gpd.read_file('/Users/kelseydoerksen/Desktop/Giga/Ookla/2023-10-01_performance_{}_tiles/'
                             'gps_{}_tiles.shp'.format(ookla_type, ookla_type), mask=country_mask)

    # Transform crs to 3857 for distance calculation
    school_df = school_df.to_crs(crs=3857)
    ookla_df = ookla_df.to_crs(crs=3857)

    # Calculate nearest ookla point
    combined_df = calc_nearest_point(ookla_df, school_df, 'ookla', data_source)

    combined_df = combined_df[['avg_d_kbps', 'avg_u_kbps', 'avg_lat_ms', 'tests', 'devices', 'ookla_distance']]

    combined_df = combined_df.add_suffix('_{}'.format(ookla_type))

    return combined_df


def get_elec(aoi, school_df, base_filepath, data_source):
    """
    Add distance to electrical grid to feature space
    :param aoi:
    :return:
    """
    if aoi == 'BWA':
        country_mask = gpd.read_file('{}/{}/{}_extent.geojson'.format(base_filepath, aoi, aoi))
    else:
        country_mask = gpd.read_file('{}/{}/geoBoundaries-{}-ADM2.geojson'.format(base_filepath, aoi, aoi))

    elec_df = gpd.read_file('/Users/kelseydoerksen/Desktop/Giga/PowerGrid/grid.gpkg', mask=country_mask)

    # Transform crs to 3857 for distance calculation
    school_df = school_df.to_crs(crs=3857)
    elec_df = elec_df.to_crs(crs=3857)

    combined_df = calc_nearest_point(elec_df, school_df, 'transmission_line', data_source)
    distance_df = combined_df[['transmission_line_distance']]

    return distance_df


def get_education_level(df):
    """
    Get school type from dataframe encoded as category
    TO UPDATE - Want to one-hot encode this
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


def get_boundary(aoi, school_df, base_filepath, data_source):
    """
    Get the administrative adm2 boundary that the school is
    within given the country
    :return: one-hot-encoded school location df
    """

    aoi_boundaries = gpd.read_file('{}/{}/geoBoundaries-{}-ADM2.geojson'.format(base_filepath, aoi, aoi))
    if data_source == 'unicef':
        school_gdf = gpd.GeoDataFrame(school_df, geometry=gpd.points_from_xy(school_df.lon, school_df.lat),
                               crs='EPSG:4326')
    else:
        school_gdf = school_df
    district_list = []
    district_names = aoi_boundaries['shapeName'].tolist()

    country_polygon = aoi_boundaries.unary_union

    for i in range(len(school_gdf)):
        # First check if the point is within the country boundaries
        if school_gdf.loc[i]['geometry'].within(country_polygon):
            # If yes, find which district and append
            for dist in range(len(district_names)):
                polygon = aoi_boundaries.loc[aoi_boundaries['shapeName'] == district_names[dist]]['geometry']
                if school_gdf.loc[i]['geometry'].within(polygon)[dist]:
                    district_list.append(district_names[dist])
                    break
        else:
            # If no, find the nearest district to the school (sometimes the location information is slightly off)
            # transform to crs 3857 so we can calc dist
            aoi_boundaries_for_dist = aoi_boundaries.to_crs(crs='3857')
            polygon_index = aoi_boundaries_for_dist.distance(school_gdf.loc[i]['geometry']).sort_values().index[0]
            district_list.append(aoi_boundaries.loc[polygon_index]['shapeName'])

    school_gdf['boundary'] = district_list

    # Now let's one-hot encode the categorical
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(school_gdf[['boundary']]).toarray(),
                              columns=encoder.get_feature_names_out())

    encoder_df['class'] = school_gdf['class']
    encoder_df['UID'] = school_gdf['UID']
    encoder_df = encoder_df.reset_index().drop(columns='index')

    return encoder_df


def get_feature_space(aoi, buffer, data_source, target):
    '''
    Grab relevant data and aggregate together
    to return final df of feature space for location
    Grabs modis, population and nightlight features
    :return: df of features for aoi specified
    '''
    if target == 'connectivity':
        if data_source == 'unicef':
            base_filepath = '/Users/kelseydoerksen/Desktop/Giga'
            df_schools = pd.read_csv('{}/{}/{}_school_geolocation_coverage_master.csv'.format(base_filepath, aoi, aoi))
        if data_source == 'new_schools':
            base_filepath = '/Users/kelseydoerksen/Desktop/Giga/isa_new_schools'
            df_schools = gpd.read_file('{}/{}/{}_schools.geojson'.format(base_filepath, aoi, aoi))

    if target == 'school':
        base_filepath = '/Users/kelseydoerksen/Desktop/Giga/SchoolMapping'
        df_samples = gpd.read_file('{}/{}/{}_train.geojson'.format(base_filepath, aoi, aoi))
        df_samples = df_samples.to_crs(crs='EPSG:4326')

        df_samples['lat'] = df_samples['geometry'].y
        df_samples['lon'] = df_samples['geometry'].x

    print('Getting modis features...')
    df_modis = pd.read_csv('{}/{}/{}m_buffer_new/modis_LC_Type1_2020_custom_buffersize_{}_with_time.csv'.
                           format(base_filepath, aoi, buffer, buffer))

    print('Getting pop features...')
    df_pop = pd.read_csv('{}/{}/{}m_buffer_new/pop_population_density_2020_custom_buffersize_{}_with_time.csv'.format
                         (base_filepath, aoi, buffer, buffer))

    print('Getting ghsl features...')
    df_ghsl = pd.read_csv('{}/{}/{}m_buffer_new/human_settlement_layer_built_up_built_characteristics_2018'
                          '_custom_buffersize_{}_with_time.csv'.
                          format(base_filepath, aoi, buffer, buffer))

    print('Getting ghm features...')
    df_ghm = pd.read_csv('{}/{}/{}m_buffer_new/global_human_modification_gHM_2016'
                         '_custom_buffersize_{}_with_time.csv'.
                         format(base_filepath, aoi, buffer, buffer))

    print('Getting elec distance features...')
    df_distance = get_elec(aoi, df_samples, base_filepath, data_source)

    print('Getting nightlight features...')
    df_avg_rad = get_avg_nightlight(aoi, 'avg_rad', buffer, base_filepath)
    df_cf_cvg = get_avg_nightlight(aoi, 'cf_cvg', buffer, base_filepath)

    print('Getting ookla features...')
    df_ookla_mobile = get_ookla(aoi, 'mobile', base_filepath, df_samples, data_source)
    df_ookla_fixed = get_ookla(aoi, 'fixed', base_filepath, df_samples, data_source)

    if target == 'connectivity':
        print('Getting label...')
        if data_source == 'unicef':
            df_schools['connectivity'] = df_schools['connectivity'].map({'Yes': 1, 'No': 0})
            df_school_id = df_schools[['giga_id_school', 'lat','lon', 'connectivity']]
        else:
            df_school_id = df_schools['fid']
    if target == 'school':
        df_samples['class'] = df_samples['class'].map({'school': 1, 'non_school': 0})
        df_school_id = df_samples[['UID', 'lat', 'lon', 'class']]

    feature_space = pd.concat([df_school_id, df_modis, df_pop, df_ghsl, df_ghm, df_distance,
                               df_avg_rad, df_cf_cvg, df_ookla_mobile, df_ookla_fixed], axis=1)

    if target == 'connectivity':
        if data_source == 'unicef':
            # Filter out schools from Isabelle's methodology
            filtered = filter_schools(aoi, feature_space, base_filepath)
            final_feature_space = filtered.drop(['Unnamed: 0'], axis=1)
            # Drop nans that exist in the label -> we can't validate if there is/is not connectivity
            final_feature_space = final_feature_space[final_feature_space['connectivity'].notna()]
        else:
            final_feature_space = feature_space.drop(['Unnamed: 0'], axis=1)
    else:
        final_feature_space = feature_space.drop(['Unnamed: 0'], axis=1)

    # Drop if there are multiple entries of the same lat, lon point
    unique_df = unique_lat_lon(final_feature_space)

    # Remove duplicated column names from concatenating
    unique_df = unique_df.loc[:, ~unique_df.columns.duplicated()]

    if target == 'school':
        unique_df.to_csv('{}/{}/{}m_buffer_new/full_feature_space.csv'.format(base_filepath, aoi, buffer,))
    else:
        unique_df.to_csv('{}/{}/{}m_buffer/full_feature_space.csv'.format(base_filepath, aoi, buffer))

# Calling function to generate features
get_feature_space(args.aoi, args.buffer, args.data_source, args.target)










