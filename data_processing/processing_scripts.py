import pandas as pd
import json
import argparse

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
save_dir = '/Users/kelseydoerksen/Desktop/Giga/{}'.format(args.aoi)
df = pd.read_csv('{}/{}_school_geolocation_coverage_master.csv'.format(args.aoi, save_dir))
get_lat_lon_list(df, args.aoi, save_dir)


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






