"""
Script to generate features from gee-generated csvs per country
"""

import pandas as pd
import argparse
import geopandas as gpd
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser(description='Generating features for ML Classifiers',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--root_dir", help='Directory where data is stored')
parser.add_argument("--save_dir", help='Save directory')
parser.add_argument("--aoi", help="Country of interest")
parser.add_argument("--buffer", help="Buffer extent around school to extract data from")
parser.add_argument("--target", help='Specify is target is connectivity prediction or school prediction.'
                                     'Must be one of school or connectivity')


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


def filter_schools(region, giga_df, data_dir):
    """
    Using Isa's cleaned dataset, filter
    the schools we don't want
    :return: clean_df
    """

    print('The number of schools before cleaning the dataset is: {}'.format(len(giga_df)))
    cleaned_df = gpd.read_file('{}/isa_clean/{}_clean.geojson'.format(data_dir, region))
    cleaned_school_ids = cleaned_df[cleaned_df['clean'] <=1]    # 1 means I am removing kindergarden
    correct_schools_list = cleaned_school_ids['giga_id_school'].tolist()

    cleaned_giga_df = giga_df[giga_df['giga_id_school'].isin(correct_schools_list)]
    print('The number of schools after cleaning the dataset is: {}'.format(len(cleaned_giga_df)))

    return cleaned_giga_df


def get_avg_nightlight(region, rad_band, buffer_ext, data_dir):
    '''
    Get average of monthly radiance
    :return: df of average rad
    '''
    df_list = []
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec']
    for month in months:
        df_list.append(pd.read_csv('{}/{}/{}m_buffer/nightlight_{}_{}_2022_custom_buffersize_{}_with_time.csv'.
                                   format(data_dir, region, buffer_ext, rad_band, month, buffer)))

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


def calc_nearest_point(target_df, school_df, target_name):
    """
    Calculate nearest school location to point of interest
    :param target_df: geopandas df of target data
    :param school_df: geopandas df of school data
    :return:
    """

    dist_df = gpd.sjoin_nearest(school_df, target_df, how='left', distance_col='{}_distance'.format(target_name))

    if 'connectivity' in school_df.columns:
        dist_df = dist_df.drop_duplicates(subset=['giga_id_school'], keep='first')

    if 'UID' in school_df.columns:
        dist_df = dist_df.drop_duplicates(subset=['UID'], keep='first')

    return dist_df


def get_ookla(region, ookla_type, data_dir, sample_df):
    """
    Add ookla data to feature space
    :return:
    """
    school_gdf = gpd.GeoDataFrame(sample_df, geometry=gpd.points_from_xy(sample_df.lon, sample_df.lat),
                                  crs='EPSG:4326')
    # Load data
    country_mask = gpd.read_file('{}/{}/geoBoundaries-{}-ADM2.geojson'.format(data_dir, region, region))
    ookla_df = gpd.read_file('{}/gps_{}_tiles.shp'.format(data_dir, ookla_type), mask=country_mask)

    # Transform crs to 3857 for distance calculation
    school_df = school_gdf.to_crs(crs=3857)
    ookla_df = ookla_df.to_crs(crs=3857)

    # Calculate nearest ookla point
    combined_df = calc_nearest_point(ookla_df, school_df, 'ookla')
    combined_df = combined_df[['avg_d_kbps', 'avg_u_kbps', 'avg_lat_ms', 'tests', 'devices', 'ookla_distance']]
    combined_df = combined_df.add_suffix('_{}'.format(ookla_type))

    return combined_df


def get_elec(region, sample_df, data_dir):
    """
    Add distance to electrical grid to feature space
    """
    school_gdf = gpd.GeoDataFrame(sample_df, geometry=gpd.points_from_xy(sample_df.lon, sample_df.lat),
                                  crs='EPSG:4326')
    country_mask = gpd.read_file('{}/{}/geoBoundaries-{}-ADM2.geojson'.format(data_dir, region, region))

    elec_df = gpd.read_file('{}/grid.gpkg'.format(data_dir), mask=country_mask)

    # Transform crs to 3857 for distance calculation
    school_df = school_gdf.to_crs(crs=3857)
    elec_df = elec_df.to_crs(crs=3857)

    combined_df = calc_nearest_point(elec_df, school_df, 'transmission_line')
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


def get_boundary(region, school_df, data_dir):
    """
    Get the administrative adm2 boundary that the school is
    within given the country
    :return: one-hot-encoded school location df
    """

    aoi_boundaries = gpd.read_file('{}/{}/geoBoundaries-{}-ADM2.geojson'.format(data_dir, region, region))
    school_gdf = gpd.GeoDataFrame(school_df, geometry=gpd.points_from_xy(school_df.lon, school_df.lat),
                                  crs='EPSG:4326')
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


def get_feature_space(data_dir, region, buffer_ext, target_type, save_path):
    '''
    Grab relevant data and aggregate together
    to return final df of feature space for location
    :param: region: Country/Region of interest
    :param: buffer_ext: buffer extent to generate feature space for
    :param: target_type: ML target
    :return: csv of features for aoi specified
    '''
    if target_type == 'connectivity':
        df_samples = pd.read_csv('{}/{}/{}_school_geolocation_coverage_master.csv'.format(data_dir, region, region))

    if target == 'school':
        df_samples = gpd.read_file('{}/{}/{}_train.geojson'.format(data_dir, region, region))
        df_samples = df_samples.to_crs(crs='EPSG:4326')

        df_samples['lat'] = df_samples['geometry'].y
        df_samples['lon'] = df_samples['geometry'].x

    print('Getting modis features...')
    df_modis = pd.read_csv('{}/{}/{}m_buffer/modis_LC_Type1_2020_custom_buffersize_{}_with_time.csv'.
                           format(data_dir, region, buffer_ext, buffer_ext))

    print('Getting pop features...')
    df_pop = pd.read_csv('{}/{}/{}m_buffer/pop_population_density_2020_custom_buffersize_{}_with_time.csv'.format
                         (data_dir, region, buffer_ext, buffer_ext))

    print('Getting ghsl features...')
    df_ghsl = pd.read_csv('{}/{}/{}m_buffer/human_settlement_layer_built_up_built_characteristics_2018'
                          '_custom_buffersize_{}_with_time.csv'.format(data_dir, region, buffer_ext, buffer_ext))

    print('Getting ghm features...')
    df_ghm = pd.read_csv('{}/{}/{}m_buffer/global_human_modification_gHM_2016'
                         '_custom_buffersize_{}_with_time.csv'.format(data_dir, region, buffer_ext, buffer_ext))

    print('Getting elec distance features...')
    df_distance = get_elec(region, df_samples, data_dir)

    print('Getting nightlight features...')
    df_avg_rad = get_avg_nightlight(region, 'avg_rad', buffer_ext, data_dir)
    df_cf_cvg = get_avg_nightlight(region, 'cf_cvg', buffer_ext, data_dir)

    print('Getting ookla features...')
    df_ookla_mobile = get_ookla(region, 'mobile', data_dir, df_samples)
    df_ookla_fixed = get_ookla(region, 'fixed', data_dir, df_samples)

    if target == 'connectivity':
        print('Getting label...')
        df_samples['connectivity'] = df_samples['connectivity'].map({'Yes': 1, 'No': 0})
        df_school_id = df_samples[['giga_id_school', 'lat','lon', 'connectivity']]

    if target == 'school':
        df_samples['class'] = df_samples['class'].map({'school': 1, 'non_school': 0})
        df_school_id = df_samples[['UID', 'lat', 'lon', 'class']]

    feature_space = pd.concat([df_school_id, df_modis, df_pop, df_ghsl, df_ghm, df_distance,
                               df_avg_rad, df_cf_cvg, df_ookla_mobile, df_ookla_fixed], axis=1)

    if target == 'connectivity':
        # Filter out schools from Isabelle's methodology
        filtered = filter_schools(region, feature_space, data_dir)
        final_feature_space = filtered.drop(['Unnamed: 0'], axis=1)
        # Drop nans that exist in the label -> we can't validate if there is/is not connectivity
        final_feature_space = final_feature_space[final_feature_space['connectivity'].notna()]
    else:
        final_feature_space = feature_space.drop(['Unnamed: 0'], axis=1)

    # Drop if there are multiple entries of the same lat, lon point
    unique_df = unique_lat_lon(final_feature_space)

    # Remove duplicated column names from concatenating
    unique_df = unique_df.loc[:, ~unique_df.columns.duplicated()]
    unique_df.to_csv('{}/full_feature_space.csv'.format(save_path))

    # Remove Pearson correlated feature and save
    df_uncorr = eliminate_correlated_features(unique_df, 0.2, save_path)
    df_uncorr.to_csv('{}/uncorrelated_feature_space.csv'.format(save_path))


if __name__ == '__main__':
    args = parser.parse_args()
    root = args.root_dir
    save = args.save_dir
    aoi = args.aoi
    buffer = args.buffer
    target = args.target

    # Calling function to generate features
    get_feature_space(root, aoi, buffer, target, save)










