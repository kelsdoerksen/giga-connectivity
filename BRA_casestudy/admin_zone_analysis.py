'''
Analysis in Admin zone in Brazil
for Connectivity Prediction using Camailot
'''

import pandas as pd
import geopandas as gpd

root_dir = '/Users/kelseydoerksen/Desktop/Giga/BRA/LocalAdminCaseStudy'


def subset_schools(zone, school_df, camailot_df, giga_df):
    """
    Subset schools based on if they overlap with the
    admin zone with camailot data to run through airpy data processing
    :return:
    """
    # Subset camailot geojson to where it overlaps with admin zone we defined
    overlapped_camailot = gpd.overlay(camailot_df, zone)

    # Subset schools to where it overlaps with the camailot in the zone we defined
    overlapped_schools = gpd.overlay(school_df, overlapped_camailot)

    # Only keep schools we have connectivity information for (UNICEF source)
    unicef_schools = overlapped_schools[overlapped_schools['source'] == 'UNICEF']
    valid_ids = unicef_schools['giga_id_school'].tolist()
    giga_keep = giga_df.loc[giga_df['giga_id_school'].isin(valid_ids)]

    print(giga_keep['connectivity'].value_counts())
    return giga_keep

# Loading data
giga_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/BRA/BRA_school_geolocation_coverage_master.csv',
                          low_memory=False)
bra_zone = gpd.read_file('{}/Manaus_extent.geojson'.format(root_dir))
bra_clean_schools = gpd.read_file('{}/BRA_clean.geojson'.format(root_dir))
camailot_df = gpd.read_file('/Users/kelseydoerksen/Desktop/Giga/Camailot/camaliot.geojson')

# Running subset function
giga_subset_zone = subset_schools(bra_zone, bra_clean_schools, camailot_df, giga_df)

# Save school subset df
giga_subset_zone.to_csv('{}/Manaus_Camailot_Subset_Schools.csv'.format(root_dir))
