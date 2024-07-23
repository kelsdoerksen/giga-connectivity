"""
Add auxiliary information about schools to feature space
"""

import pandas as pd


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

	# merge the two dataframes on the matching giga id
	combined = pd.merge(aoi_df, giga_school_df_filt, on='giga_id_school')
	return combined