import pandas as pd
from sklearn.metrics import confusion_matrix
import ast
import geopandas as gpd


#new_rwa_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/RWA/1000m_buffer/full_feature_space.csv')
#school_rwa_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/RWA/RWA_school_geolocation_coverage_master.csv')


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


def add_auxillary_information(giga_school_df, aoi_df):
	"""
	Adding auxillary information to the feature space to try
	to improve model performance
	:param giga_school_df: UNICEF-provided school information
	:param aoi_df: country feature space
	:return:
	"""
	# Load list of schools we have samples for
	school_list = aoi_df['giga_id_school'].tolist()
	# filter the giga_school_df for this
	giga_school_df = giga_school_df[giga_school_df['giga_id_school'].isin(school_list)]

	# filter to just have the columns we are interested in
	giga_school_df_filt = giga_school_df[['giga_id_school', 'education_level', 'nearest_LTE_distance',
										  'nearest_UMTS_distance', 'nearest_GSM_distance' ]]

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

	# merge the two dataframes on the matching giga id
	combined = pd.merge(aoi_df, giga_school_df_filt, on='giga_id_school')
	return combined




#aoi = 'RWA'
#giga_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/{}_school_geolocation_coverage_master.csv'.format(aoi, aoi))


'''
# Adding coverage information
feat_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/embeddings/{}_coordinates_v2-embeddings.csv'.format(
	aoi, aoi))
df_with_coverage = adding_coverage_availability(giga_df, feat_df)
df_with_coverage.to_csv('/Users/kelseydoerksen/Desktop/Giga'
						'/{}/embeddings/{}_coordinates_v2-embeddings_added_coverage.csv'.format(aoi, aoi))
'''
'''
# Adding auxillary information
train_feat_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/1000m_buffer/TrainingData_uncorrelated_fixed.csv'.format(aoi))
test_feat_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/1000m_buffer/TestingData_uncorrelated_fixed.csv'.format(aoi))
train_df_with_aux = add_auxillary_information(giga_df, train_feat_df)
test_df_with_aux = add_auxillary_information(giga_df, test_feat_df)

train_df_with_aux.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/1000m_buffer/TrainingData_uncorrelated_fixed_with_aux.csv'.format(aoi))
test_df_with_aux.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/1000m_buffer/TestingData_uncorrelated_fixed_with_aux.csv'.format(aoi))
'''

def calc_confusion_matrix(y_test, y_pred, savedir):
	"""
	Calculates confusion matrix
	"""
	predictions = (y_pred >= 0.5)
	CM = confusion_matrix(y_test, predictions)
	TN = CM[0][0]
	FN = CM[1][0]
	TP = CM[1][1]
	FP = CM[0][1]

	print('True Positive is {}'.format(TP))
	print('True Negative is {}'.format(TN))
	print('False Positive is {}'.format(FP))
	print('False Negative is {}'.format(FN))

	FP_Rate = FP / (FP + TN)
	TP_Rate = TP / (TP + FN)
	FN_Rate = FN / (FN + TP)
	TN_Rate = TN / (TN + FP)

	print('False positive rate is {}'.format(FP_Rate))
	print('True positive rate is {}'.format(TP_Rate))
	print('False negative rate is {}'.format(FN_Rate))
	print('True negative rate is {}'.format(TN_Rate))

	with open('{}/confusionmatrix.txt'.format(savedir), 'w') as f:
		f.write('False positive rate is {}'.format(FP_Rate))
		f.write('True positive rate is {}'.format(TP_Rate))
		f.write('False negative rate is {}'.format(FN_Rate))
		f.write('True negative rate is {}'.format(TN_Rate))


BWA_folders = ['dazzling-dragon-229/rf_results_for_plotting',
			   'vibrant-dragon-236/mlp_results_for_plotting',
			   'lambent-fireworks-237/svm_results_for_plotting',
			   'dazzling-ox-238/lr_results_for_plotting',
			   'glittering-fuse-239/gb_results_for_plotting']

RWA_folders = ['incandescent-bao-227/rf_results_for_plotting',
			   'glittering-horse-240/gb_results_for_plotting',
			   'lambent-fireworks-241/lr_results_for_plotting',
			   'red-mandu-242/svm_results_for_plotting',
			   'prosperous-pig-243/mlp_results_for_plotting']

'''
# Running confusion matrix plotting
aoi = 'BWA'
root_dir = '/Users/kelseydoerksen/Desktop/Giga/{}/results_1000m'.format(aoi)
for i in BWA_folders:
	print('running for: {}'.format(i))
	df = pd.read_csv('{}/{}.csv'.format(root_dir, i))
	df["prediction"] = df["prediction"].astype(int)
	folder = i.split('/')[0]
	save_directory = root_dir + '/' + folder
	calc_confusion_matrix(df['prediction'], df['label'], save_directory)
'''
'''
def emb_df_formatting(df, zoom):
	"""
	Quick script for embedding df formatting
	:return:
	"""
	emb_list = []
	for i in range(len(df)):
		literal_list = ast.literal_eval(df.loc[i]['embed_{}'.format(zoom)])
		emb_list.append(literal_list)

	emb_df_only = pd.DataFrame(emb_list)
	emb_df_only['giga_id_school'] = df['giga_id_school']
	emb_df_only['lat'] = df['lat']
	emb_df_only['lon'] = df['lon']
	emb_df_only['connectivity'] = df['connectivity']
	emb_df_only['data_split'] = df['data_split']

	return emb_df_only

aoi = 'RWA'
z = 'z17'
full_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/embeddings/{}_coordinates_v2-embeddings.csv'.
					  format(aoi, aoi))
emb_df = emb_df_formatting(full_df, z)
train_emb_df = emb_df[emb_df['data_split'] == 'Train']
test_emb_df = emb_df[emb_df['data_split'] == 'Test']

train_emb_df.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/embeddings/{}_Train_esa_{}_v2-embeddings.csv'.
					format(aoi, aoi, z))
test_emb_df.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/embeddings/{}_Test_esa_{}_v2-embeddings.csv'.
					format(aoi, aoi, z))
'''

'''
# Processing matching fid from Casper embeddings to giga_id correctly
aoi = 'RWA'
rwa_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/RWA/embeddings/RWA_coordinates_v2-embeddings.csv')
fid_rwa_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/RWA/embeddings/old/rwa_coordinates_for_embeddings.csv')

rwa_gdf = gpd.GeoDataFrame(rwa_df, geometry=gpd.points_from_xy(rwa_df.lon, rwa_df.lat))
fid_rwa_gdf = gpd.GeoDataFrame(fid_rwa_df, geometry=gpd.points_from_xy(fid_rwa_df.lon, fid_rwa_df.lat))
dist_df = gpd.sjoin_nearest(rwa_gdf, fid_rwa_gdf, how='left', distance_col='nearest_fid')

fid_corrected_subset = dist_df[['giga_id_school', 'fid_right']]
fid_corrected_subset = fid_corrected_subset.rename(columns={'fid_right': 'fid'})
result_df = fid_corrected_subset.drop_duplicates(subset=['giga_id_school'])
rwa_subset = rwa_df[['giga_id_school', 'lat', 'lon', 'connectivity','school_locations','data_split']]
combined = rwa_subset.merge(result_df, on='giga_id_school')
combined.to_csv('/Users/kelseydoerksen/Desktop/Giga/RWA/embeddings/RWA_coordinates_for_embeddings_with_fid.csv')
'''


def clean_dataframes(df):
	"""
	Scrap function to remove extra columns from dataframes that were made when concatenating
	:return:
	"""
	df = df.drop(columns=['connectivity.1', 'Unnamed: 0', 'Unnamed: 0.1'])
	return df


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




