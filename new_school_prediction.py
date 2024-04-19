"""
Script for running our trained models on new schools identified
by school mapping model
"""

import pandas as pd
import pickle

# For now hard-coding to the BWA schools
old_model_dict = {
    'gb': 'jolly-voice-91/gb_model.pkl',
    'mlp': 'expert-plasma-95/mlp_model.pkl',
    'svm': 'tough-fire-93/svm_model.pkl',
    'rf': 'light-fog-92/rf_model.pkl',
    'lr': 'glorious-resonance-94/lr_model.pkl'
}

model_dict = {
    'gb': 'legendary-galaxy-417/gb_model.pkl',
    'mlp': 'lyric-river-440/mlp_model.pkl',
    'svm': 'jolly-morning-416/svm_model.pkl',
    'rf': 'colorful-river-413/rf_model.pkl',
    'lr': 'youthful-sun-415/lr_model.pkl'
}


for model_type in model_dict.keys():
    # load model
    print('processing for model {}'.format(model_type))
    filename = '/Users/kelseydoerksen/Desktop/Giga/BWA/results_1000m/{}'.format(model_dict[model_type])
    loaded_model = pickle.load(open(filename, 'rb'))
    # load dataset to predict from
    new_schools = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/isa_new_schools/BWA/1000m_buffer/'
                              'full_feature_space.csv')

    # Drop features we didn't use that were highly correlated
    corr_feats = ['human_settlement_layer_built_up.built_characteristics.built_res_3-6', 'global_human_modification.gHM.max', 'global_human_modification.gHM.mode', 'human_settlement_layer_built_up.built_characteristics.built_res_15-30', 'nightlight.cf_cvg.mean', 'human_settlement_layer_built_up.built_characteristics.built_res_6-15', 'global_human_modification.gHM.mean', 'pop.population_density.mean', 'human_settlement_layer_built_up.built_characteristics.built_res_3', 'nightlight.avg_rad.mean']
    new_schools = new_schools.drop(columns=corr_feats)

    # If there are missing school location overlaps, let's add these features as 0s in the feature space
    new_schools['boundary_Bobonong'] = 0
    new_schools['boundary_Boteti'] = 0
    new_schools['boundary_Chobe'] = 0
    new_schools['boundary_Francistown'] = 0
    new_schools['boundary_Ghanzi'] = 0
    new_schools['boundary_Kgalagadi North'] = 0
    new_schools['boundary_Kgalagadi South'] = 0
    new_schools['boundary_Lobatse'] = 0
    new_schools['boundary_Ngamiland East'] = 0
    new_schools['boundary_Ngamiland West'] = 0
    new_schools['boundary_North East'] = 0
    new_schools['boundary_Orapa'] = 0
    new_schools['boundary_Selibe Phikwe'] = 0
    new_schools['boundary_Tutume'] = 0
    new_schools['boundary_Sowa Town'] = 0

    location_info = new_schools[['lat', 'lon']]

    new_schools = new_schools.drop(columns=['Unnamed: 0', 'lat', 'lon', 'fid', 'school_locations'])

    # correcting so the features are in the same order
    new_schools =  new_schools[['modis.LC_Type1.mode','modis.LC_Type1.var', 'modis.LC_Type1.evg_conif',
        'modis.LC_Type1.evg_broad', 'modis.LC_Type1.dcd_needle',
        'modis.LC_Type1.dcd_broad', 'modis.LC_Type1.mix_forest',
        'modis.LC_Type1.cls_shrub', 'modis.LC_Type1.open_shrub',
        'modis.LC_Type1.woody_savanna', 'modis.LC_Type1.savanna',
        'modis.LC_Type1.grassland', 'modis.LC_Type1.perm_wetland',
        'modis.LC_Type1.cropland', 'modis.LC_Type1.urban',
        'modis.LC_Type1.crop_nat_veg', 'modis.LC_Type1.perm_snow',
        'modis.LC_Type1.barren', 'modis.LC_Type1.water_bds',
        'pop.population_density.var', 'pop.population_density.max',
        'pop.population_density.min',
        'human_settlement_layer_built_up.built_characteristics.mode',
        'human_settlement_layer_built_up.built_characteristics.var',
        'human_settlement_layer_built_up.built_characteristics.open_low_veg',
        'human_settlement_layer_built_up.built_characteristics.open_med_veg',
        'human_settlement_layer_built_up.built_characteristics.open_high_veg',
        'human_settlement_layer_built_up.built_characteristics.open_water',
        'human_settlement_layer_built_up.built_characteristics.open_road',
        'human_settlement_layer_built_up.built_characteristics.built_res_30',
        'human_settlement_layer_built_up.built_characteristics.built_non_res_3',
        'human_settlement_layer_built_up.built_characteristics.built_non_res_3-6',
        'human_settlement_layer_built_up.built_characteristics.built_non_res_6-15',
        'human_settlement_layer_built_up.built_characteristics.built_non_res_15-30',
        'human_settlement_layer_built_up.built_characteristics.build_non_res_30',
        'human_settlement_layer_built_up.built_characteristics.built',
        'global_human_modification.gHM.var',
        'global_human_modification.gHM.min', 'transmission_line_distance',
        'nightlight.avg_rad.var', 'nightlight.avg_rad.max',
        'nightlight.avg_rad.min', 'nightlight.cf_cvg.var',
        'nightlight.cf_cvg.max', 'nightlight.cf_cvg.min', 'avg_d_kbps_mobile',
        'avg_u_kbps_mobile', 'avg_lat_ms_mobile', 'tests_mobile',
        'devices_mobile', 'ookla_distance_mobile', 'avg_d_kbps_fixed',
        'avg_u_kbps_fixed', 'avg_lat_ms_fixed', 'tests_fixed', 'devices_fixed',
        'ookla_distance_fixed', 'boundary_Barolong', 'boundary_Bobonong',
        'boundary_Boteti', 'boundary_Chobe', 'boundary_Francistown',
        'boundary_Gaborone', 'boundary_Ghanzi', 'boundary_Kgalagadi North',
        'boundary_Kgalagadi South', 'boundary_Kgatleng',
        'boundary_Kweneng East', 'boundary_Kweneng West', 'boundary_Lobatse',
        'boundary_Mahalapye', 'boundary_Ngamiland East',
        'boundary_Ngamiland West', 'boundary_Ngwaketse West',
        'boundary_North East', 'boundary_Orapa', 'boundary_Selibe Phikwe',
        'boundary_Serowe Palapye', 'boundary_South East', 'boundary_Southern',
        'boundary_Sowa Town', 'boundary_Tutume']]

    probs = loaded_model.predict_proba(new_schools)
    print(probs)

    results_df = pd.DataFrame(data=probs, columns=['unconnected', 'connected'])
    results_df['lat'] = location_info['lat']
    results_df['lon'] = location_info['lon']

    # Save for plotting and looking at results
    results_df.to_csv('/Users/kelseydoerksen/Desktop/Giga/isa_new_schools/BWA/BWA_results_new_schools_corrected_{}_model.csv'.
                      format(model_type))
