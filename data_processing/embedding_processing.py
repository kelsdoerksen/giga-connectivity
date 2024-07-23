"""
Processing embedding features from various sources
Hard-coding this for now for Botswana
"""

import json
import pandas as pd
import regex as re
import numpy as np
import ast


def process_pretrained_encoders(embedding_df, verified_school_list):
    """
    Catch-all to process embeddings from pre-trained models SatCLIP,
    GeoCLIP and CSP (tbd GPS2Vec)
    :return:
    """
    # drop duplicates (left over from original UNICEF school location)
    e_df = embedding_df.drop_duplicates(subset=['location']).reset_index()
    e_df = e_df.drop(columns=['Unnamed: 0', 'index'])

    emb_list = []
    matching_schools = []
    for i in range(len(e_df)):
        # embedding location
        loc = e_df['location'].loc[i]
        # check if it is in the train or test set we want samples for
        for school in verified_school_list:
            if loc == str(school):
                emb_list.append(e_df[e_df['location'] == loc].values[0].tolist())
                matching_schools.append(loc)

    # ok now let's get this in a dataframe format to save
    arr = np.array(emb_list)
    emb_df = pd.DataFrame(arr)
    emb_df = emb_df.rename(columns={int(emb_df.shape[1] - 1): 'location'})
    return emb_df


def process_esa_emb_old(esa_json, s, z,  verified_school_list):
    """
    Processing esa foundational model embeddings to match
    verified schools list
    :param esa_json: json file of esa embeddings
    :param s: chosen source (s0 or s1)
    :param z: chosen z source (z17 or z18)
    :param: verified_df: df of verified schools for either train or test set
    :return:
    """

    lon_re = '_x(.*)_y'
    lat_re = '_y(.*)_z'

    # Get list of keys we are interested in from json
    emb_keys = []
    for k in esa_json.keys():
        if all(x in k for x in [s,z]):
            emb_keys.append(k)

    locs = []
    for emb in emb_keys:
        locs.append(str((float(re.search(lon_re, emb).group(1)), float(re.search(lat_re, emb).group(1)))))

    df = pd.DataFrame(emb_keys)
    df['location'] = locs

    updated_list = []
    for i in range(len(verified_school_list)):
        tup = ast.literal_eval(verified_school_list[i])
        tup = tuple([round(x, 6) if isinstance(x, float) else x for x in tup])
        updated_list.append(tup)

    emb_list = []
    for school in updated_list:
        lon = school[0]
        lat = school[1]
        embedding = dict(filter(lambda item: '_x{}_y{}_z17_s0_256x25'.format(lon, lat) in item[0], esa_json.items()))
        emb_list.append([*embedding.values()][0])


    # ok now let's get this in a dataframe format to save
    arr = np.array(emb_list)
    emb_df = pd.DataFrame(arr)
    emb_df['location'] = updated_list
    return emb_df


def process_esa_emb_old2(esa_json, z,  verified_school_list):
    """
    Processing esa foundational model embeddings to match
    verified schools list
    :param esa_json: json file of esa embeddings
    :param z: chosen z source (z17 or z18)
    :param: verified_df: df of verified schools for either train or test set
    :return:
    """

    # Get list of keys we are interested in from json
    emb_keys = []
    for k in esa_json.keys():
        if all(x in k for x in [z]):
            emb_keys.append(k)

    updated_list = []
    for i in range(len(verified_school_list)):
        if verified_school_list[i] == '(nan, nan)':
            continue
        tup = ast.literal_eval(verified_school_list[i])
        tup = tuple([round(x, 6) if isinstance(x, float) else x for x in tup])
        updated_list.append(tup)

    emb_list = []
    count = 0
    for school in updated_list:
        lon = school[0]
        lat = school[1]
        embedding = dict(filter(lambda item: '_x{}_y{}_{}'.format(lon, lat, z) in item[0], esa_json.items()))
        vals = [*embedding.values()]
        #print(len(vals))
        if len(vals) > 0:
            emb_list.append(vals[0])
        elif len(vals) == 0:
            print('Missing data for location {}, {}'.format(lon, lat))
            count+=1

    # ok now let's get this in a dataframe format to save
    arr = np.array(emb_list)
    emb_df = pd.DataFrame(arr)
    emb_df['location'] = updated_list
    return emb_df


def process_esa_emb(aoi_df, esa_json, z, datasplit_type):
    '''
    Fixed processing esa emb given json file of embeddings
    fid columns of aoi_df will match esa json key start
    :return:
    '''

    emb_new_dict = {
        'fid': [],
        'emb': [],
        'location': [],
        'giga_id_school': [],
        'connectivity': [],
        'lat': [],
        'lon': []
    }
    emb_keys = [key for key, val in esa_json.items()]
    filt_df = aoi_df[aoi_df['data_split'] == datasplit_type]
    filt_df = filt_df.reset_index()
    for i in range(len(filt_df)):
        fid = filt_df.loc[i]['fid']
        fid_str = 'i{}_x'.format(fid)
        for key in emb_keys:
            if fid_str in key:
                if z in key:
                    emb = esa_json['{}'.format(key)]
                    emb_new_dict['fid'].append(fid)
                    emb_new_dict['emb'].append(emb)
                    emb_new_dict['location'].append(filt_df.loc[i]['school_locations'])
                    emb_new_dict['giga_id_school'].append(filt_df.loc[i]['giga_id_school'])
                    emb_new_dict['connectivity'].append(filt_df.loc[i]['connectivity'])
                    emb_new_dict['lat'].append(filt_df.loc[i]['lat'])
                    emb_new_dict['lon'].append(filt_df.loc[i]['lon'])


    emb_df = pd.DataFrame(emb_new_dict['emb'])
    emb_df['fid'] = emb_new_dict['fid']
    emb_df['location'] = emb_new_dict['location']
    emb_df['giga_id_school'] = emb_new_dict['giga_id_school']
    emb_df['connectivity'] = emb_new_dict['connectivity']
    emb_df['lat'] = emb_new_dict['lat']
    emb_df['lon'] = emb_new_dict['lon']

    return emb_df


# ------ Running sat embedding feature generation script for Botswana, ESA model

data_type = ['Train', 'Test']
aoi = 'BWA'
aoi_full = 'botswana'
for datasplit in data_type:
    aoi_giga_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/embeddings/{}_coordinates_for_embeddings_with_fid.csv'.
                            format(aoi, aoi))

    esa_models = ['embeddings_precursor-geofoundation_e011',
                  'embeddings_precursor-geofoundation_{}_v04_e008'.format(aoi_full),
                  'embeddings_precursor-geofoundation_{}_v04_e025'.format(aoi_full),
                  'embeddings_school-predictor_{}_v01_e025'.format(aoi_full)]
    esa_model = esa_models[1]
    print('Running {} for aoi {} for esa model {}'.format(datasplit, aoi, esa_model))

    with open('/Users/kelseydoerksen/Desktop/Giga/{}/embeddings/{}.json'.format(aoi, esa_model)) as file:
        esa_data = json.load(file)

    z_val = 'z18'
    esa_emb = process_esa_emb(aoi_giga_df, esa_data, z_val, datasplit)
    esa_emb.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/embeddings/{}ingData_{}_{}_embeddings.csv'.
                   format(aoi, datasplit, esa_model, z_val))

'''
esa_emb = process_esa_emb_old(esa_data, 's1', 'z17', test_locs)
esa_emb = esa_emb.sort_values(by=['location'])
test_df = test_df.sort_values(by=['school_locations'])
esa_emb['connectivity'] = test_df['connectivity']
esa_emb.to_csv('/Users/kelseydoerksen/Desktop/Giga/BWA/embeddings/TrainingData_precursor-geofoundation_e011_s1_z17.csv')
'''



# ------ Running sat embedding feature generation script pre-trained embedding models
'''
aoi = 'RWA'
# Testing information
test_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/1000m_buffer/TestingData_uncorrelated_fixed.csv'.format(aoi))
test_locs = test_df['school_locations'].tolist()

# Training information
train_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/1000m_buffer/TrainingData_uncorrelated_fixed.csv'.format(aoi))
train_locs = train_df['school_locations'].tolist()

# Let's run for the foundational models we have
embedding_models = ['satclip-resnet18-l10', 'satclip-resnet18-l40', 'satclip-resnet50-l10',
                 'satclip-resnet50-l40', 'satclip-vit16-l10', 'satclip-vit16-l40', 'GeoCLIP', 'CSPfMoW']

for model in embedding_models:
    print('Running train test sample split for country {} location encoder: {}'.format(aoi, model))
    emb_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/embeddings/{}_{}_embeddings.csv'.format(aoi, aoi, model))

    # Test set
    model_test_df = process_pretrained_encoders(emb_df, test_locs)
    test_df = test_df.sort_values(by=['school_locations'])
    model_test_df = model_test_df.sort_values(by=['location'])
    model_test_df['connectivity'] = test_df['connectivity']
    model_test_df.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/embeddings/TestingData_{}_embeddings_fixed.csv'.
                         format(aoi, model))

    # Training set
    model_train_df = process_pretrained_encoders(emb_df, train_locs)
    train_df = train_df.sort_values(by=['school_locations'])
    model_train_df = model_train_df.sort_values(by=['location'])
    model_train_df['connectivity'] = train_df['connectivity']
    model_train_df.to_csv('/Users/kelseydoerksen/Desktop/Giga/{}/embeddings/TrainingData_{}_embeddings_fixed.csv'.
                          format(aoi, model))
'''
