"""
Add auxiliary information about schools to feature space
"""

import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import OneHotEncoder
import argparse
from functools import reduce


parser = argparse.ArgumentParser(description='Adding Auxiliary Features to Model',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--root_dir", help='Directory where data is stored')
parser.add_argument("--save_dir", help='Save directory')
parser.add_argument("--aoi", help="Country of interest")
parser.add_argument("--buffer", help="Buffer extent around school to extract data from")


def get_boundary(aoi_boundaries, giga_school_df):
    """
    Get the administrative adm2 boundary that the school is
    within given the country
    :return: one-hot-encoded school location df
    """
    giga_school_gdf = gpd.GeoDataFrame(giga_school_df, geometry=gpd.points_from_xy(giga_school_df.lon,
                                                                                   giga_school_df.lat), crs='EPSG:4326')
    district_list = []
    district_names = aoi_boundaries['shapeName'].tolist()

    country_polygon = aoi_boundaries.unary_union

    for i in range(len(giga_school_gdf)):
        # First check if the point is within the country boundaries
        if giga_school_gdf.loc[i]['geometry'].within(country_polygon):
            # If yes, find which district and append
            for dist in range(len(district_names)):
                polygon = aoi_boundaries.loc[aoi_boundaries['shapeName'] == district_names[dist]]['geometry']
                if giga_school_gdf.loc[i]['geometry'].within(polygon)[dist]:
                    district_list.append(district_names[dist])
                    break
        else:
            # If no, find the nearest district to the school (sometimes the location information is slightly off)
            # transform to crs 3857 so we can calc dist
            aoi_boundaries_for_dist = aoi_boundaries.to_crs(crs='3857')
            polygon_index = aoi_boundaries_for_dist.distance(giga_school_gdf.loc[i]['geometry']).sort_values().index[0]
            district_list.append(aoi_boundaries.loc[polygon_index]['shapeName'])

    giga_school_gdf['boundary'] = district_list

    # Now let's one-hot encode the categorical
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(giga_school_gdf[['boundary']]).toarray(),
                              columns=encoder.get_feature_names_out())


    encoder_df['giga_id_school'] = giga_school_gdf['giga_id_school']
    encoder_df = encoder_df.reset_index().drop(columns='index')

    return encoder_df


def add_auxiliary_information(giga_school_df, feature_df):
    """
    Adding auxiliary information to the feature space to try
    to improve model performance
    :param giga_school_df: UNICEF-provided school information
    :param feature_df: country feature space
    :return:
    """
    # Load list of schools we have samples for
    school_list = feature_df['giga_id_school'].tolist()
    # filter the giga_school_df for this
    giga_school_df = giga_school_df[giga_school_df['giga_id_school'].isin(school_list)]

    # filter to just have the columns we are interested in
    giga_school_df_filt = giga_school_df[['giga_id_school', 'education_level',
                                          'nearest_LTE_distance', 'nearest_UMTS_distance', 'nearest_GSM_distance']]

    # encode the education level
    # 4 denotes multi-education level
    education_level_dict = {
        'Pre-Primary': 0,
        'Primary': 1,
        'Secondary': 2,
        'Post-Secondary': 3,
        'Pre-Primary and Secondary': 4,
        'Primary and Secondary': 4,
        'Pre-Primary and Primary': 4,
        'Pre-Primary and Primary and Secondary': 4,
        'Pre-Primary, Primary and Secondary': 4,
        'Primary, Secondary and Post-Secondary': 4,
        'Other': 5
    }

    giga_school_df_filt['education_level'] = giga_school_df_filt['education_level'].map(education_level_dict)

    return giga_school_df_filt


if __name__ == '__main__':
    args = parser.parse_args()
    aoi = args.aoi
    root_dir = args.root_dir
    save_dir = args.save_dir
    buffer = args.buffer

    giga_school_df = pd.read_csv('{}/{}/{}_school_geolocation_coverage_master.csv'.format(root_dir, aoi, aoi))
    aoi_boundaries_file = gpd.read_file('{}/{}/geoBoundaries-{}-ADM2.geojson'.format(root_dir, aoi, aoi))

    splits = ['Training', 'Testing', 'Val']
    for split in splits:
        feature_df = pd.read_csv('{}/{}/{}m_buffer/{}Data_uncorrelated.csv'.format(root_dir, aoi, buffer, split))
        giga_aux = add_auxiliary_information(giga_school_df, feature_df)
        encoder_df = get_boundary(aoi_boundaries_file, giga_school_df)

        combine = reduce(lambda x, y: pd.merge(x, y, on='giga_id_school', how='inner'),
                         [feature_df, encoder_df, giga_aux]).drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

        combine.to_csv('{}/{}Data_uncorrelated_with_aux.csv'.format(save_dir, split))

