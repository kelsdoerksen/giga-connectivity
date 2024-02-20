import pandas as pd

new_rwa = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/RWA/1000m_buffer/full_feature_space.csv')
school_rwa = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/RWA/RWA_school_geolocation_coverage_master.csv')

count=0

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
		count+=1
		print('connectivity doesnt match')