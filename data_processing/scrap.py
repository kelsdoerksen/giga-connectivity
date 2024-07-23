import pandas as pd
import geopandas as gpd


def checking_matching_coords(new_rwa, school_rwa):
	for i in range(len(new_rwa)):
		lat = new_rwa.loc[i]['lat']
		lon = new_rwa.loc[i]['lon']
		id = new_rwa.loc[i]['giga_id_school']
		if new_rwa.loc[i]['connectivity'] == 1:
			conn = 'Yes'
		if new_rwa.loc[i]['connectivity'] == 0:
			conn = 'No'
		school_lat = school_rwa[school_rwa['giga_id_school'] == id]['lat'].values[0]
		if school_lat != lat:
			#count+=1
			print('latitude doesnt match')
			print(school_rwa[school_rwa['giga_id_school'] == id])
			print(new_rwa.loc[i])
		school_conn = school_rwa[school_rwa['giga_id_school'] == id]['connectivity'].values[0]
		if school_conn != conn:
			print('connectivity doesnt match')


def adding_coverage_availability(giga_school_df, aoi_df):
	"""
	Adding coverage availability to dataframe
	:param giga_school_df:
	:param aoi_df:
	:return:
	"""
	# Load list of schools we have samples for
	school_list = aoi_df['giga_id_school'].tolist()
	# filter the giga_school_df for this
	giga_school_df = giga_school_df[giga_school_df['giga_id_school'].isin(school_list)]

	# filter to just have the columns we are interested in
	giga_school_df_filt = giga_school_df[['giga_id_school', 'coverage_availability']]

	# merge the two dataframes on the matching giga id
	combined = pd.merge(aoi_df, giga_school_df_filt, on='giga_id_school')
	return combined


def clean_dataframes(df):
	"""
	Scrap function to remove extra columns from dataframes that were made when concatenating
	:return:
	"""
	df = df.drop(columns=['connectivity.1', 'Unnamed: 0', 'Unnamed: 0.1'])
	return df

'''
aois = ['BIH', 'BLZ', 'BWA', 'GIN', 'MNG', 'RWA', 'BRA']
data_type = ['Training', 'Testing']
for aoi in aois:
	print('Cleaning dataframe for aoi: {}'.format(aoi))
	for data in data_type:
		df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/Connectivity/{}/1000m_buffer/{}Data_uncorrelated.csv'.
						 format(aoi, data))
		df_clean = clean_dataframes(df)
		df_clean.to_csv('/Users/kelseydoerksen/Desktop/Giga/Connectivity/{}/1000m_buffer/{}Data_uncorrelated.csv'.
						format(aoi, data))
'''


def add_label(gdf_schools, feature_df):
	"""
	Adding label (school/non-school) to features
	:param: school_location_gdf: geodataframe to get locations from
	:param:
	:return:
	"""
	df_schools = pd.DataFrame()
	df_schools['lat'] = gdf_schools['geometry'].y
	df_schools['lon'] = gdf_schools['geometry'].x

	lon_vals = df_schools['lon'].values.tolist()
	lat_vals = df_schools['lat'].values.tolist()

	school_locations = []
	for i in range(len(lon_vals)):
		school_locations.append((lon_vals[i], lat_vals[i]))

	df_schools['school_locations'] = school_locations

	f_lon_vals = feature_df['lon'].values.tolist()
	f_lat_vals = feature_df['lat'].values.tolist()
	feature_locations = []
	for i in range(len(f_lon_vals)):
		feature_locations.append((f_lon_vals[i], f_lat_vals[i]))

	import ipdb
	ipdb.set_trace()
	feature_df['location'] = feature_locations

	for loc in feature_locations:
		if loc not in school_locations:
			feature_df = feature_df[feature_df.location != loc]




# Load in data and subset for only school locations
sample_location_gdf = gpd.read_file('/Users/kelseydoerksen/Desktop/Giga/SchoolMapping/BWA/BWA_train.geojson')
sample_location_gdf_epsg = sample_location_gdf.to_crs(crs='EPSG:4326')
schools = sample_location_gdf_epsg[sample_location_gdf_epsg['class'] == 'school']

feat_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/SchoolMapping/BWA/300m_buffer_toremove_nonschools/'
					  'global_human_modification_gHM_2016_custom_buffersize_300_with_time.csv')

add_label(schools, feat_df)

