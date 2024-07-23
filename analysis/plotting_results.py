"""
Script for plotting different results for
visualization of connectivity
"""

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import argparse

parser = argparse.ArgumentParser(description='Plotting Random Forest Results',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--aoi", help="Country of interest")
parser.add_argument("--buffer", help="Buffer extent around school to extract data from")
parser.add_argument("--model", help="ML Classifier used. Must be one of mlp, gb, rf, svm, lr")
parser.add_argument("--wandb_exp", help="Wandb experiment to plot results for")
args = parser.parse_args()

# lons then lats for countries of interest (up and down goes second)
bbox_dict = {
    'BWA': [18, 30, -17, -27],
    'RWA': [28.5, 31, -1, -3],
    'BIH': [15, 19.75, 45.5, 42.5],
    'BLZ': [-89.5, -87.75, 18.75, 15.75],
    'GIN': [-15.5, -7.25, 13, 7],
    'SLV': [-90.25, -87.5, 14.5, 13],
    'PAN': [-83.5, -76.5, 10, 6.75],
    'BRA': [-76.0, -34.0, 8, -35]
}

country_dict = {
    'BWA': 'Botswana',
    'RWA': 'Rwanda',
    'BIH': 'Bosnia',
    'BLZ': 'Belize',
    'GIN': 'Guinea',
    'SLV': 'El Salvador',
    'PAN': 'Panama',
    'BRA': 'Brazil'
}

def plot_2d_array(array, country, buffer_extent, title, savename, connectivity_status):
    """
    Plotting 2d array of school connectivity
    green points: schools w/ connectivity
    red points: schools w/o connectivity
    :param: array: 2d array of data to plot
    :param: country: country to plot to query for the boundary lines
    :param: buffer_extent: buffer region around central point
    :param: title: title of plot
    :param: savename: save name for the file
    :param: connectivity_status: denotes the plot type of connectivity to update the cmap colour list (full, no, yes)
    :return:
    """

    cmap_colour = {'full': ['red', 'green'],
                   'no': ['red', 'green'],
                   'yes': ['green', 'red']}

    array_copy = np.copy(array)
    # Filter array accordingly for the y/n connectivity plot
    if connectivity_status == 'no':
        array_copy[array_copy == 1] = np.nan
    if connectivity_status == 'yes':
        array_copy[array_copy == 0] = np.nan

    array_to_plot = {'full': array,
                     'no': array_copy,
                     'yes': array_copy}

    x, y = np.meshgrid(lon_vals, lat_vals, indexing='xy')

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", cmap_colour['{}'.format(connectivity_status)])
    plt.scatter(x, y, c=array_to_plot['{}'.format(connectivity_status)], cmap=cmap)
    country_borders = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')
    ax.add_feature(country_borders, edgecolor='gray')
    ax.coastlines()
    ax.stock_img()
    ax.set_extent(bbox_dict['{}'.format(country)], crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    plt.title('{}'.format(title))
    #plt.show()
    '''
    plt.savefig('/Users/kelseydoerksen/Desktop/Giga/{}/results_{}m/{}/{}.png'
                .format(country, buffer_extent, wandb_exp, savename))
    '''
    plt.savefig('/Users/kelseydoerksen/Desktop/Giga/isa_new_schools/{}/{}.png'.format(country, savename))

def plot_feature_importance(aoi, buffer):
    """
    Plotting features by importance
    :return:
    """
    fi_df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/results_{}m/{}/FI.csv'.
                        format(aoi, buffer, wandb_exp))
    importances = fi_df.loc[0][1:].tolist()
    features = fi_df.columns[1:].tolist()

    # Normalize importances to 0-1
    norm_importances = (importances-np.min(importances))/(np.max(importances)-np.min(importances))

    # Remove the prefix from the human settlement layer/human modification because super long
    strings_to_remove = ['human_settlement_layer_built_up', 'global_human_modification']
    new_names = []
    for name in features:
        prefix = name.split('.',1)[0]
        if prefix in strings_to_remove:
            if 'built_characteristics' in name:
                updated_name = name.replace("built_characteristics", "built_c")
                new_names.append(updated_name.split('.', 1)[1])
            else:
                new_names.append(name.split('.', 1)[1])
        else:
            new_names.append(name)

    plt.barh(new_names, norm_importances, height=0.3)
    plt.rc('ytick', labelsize=6)
    plt.subplots_adjust(top=0.963, bottom=0.054)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.title('Feature Importance for {}'.format(country_dict[aoi]))
    plt.savefig('/Users/kelseydoerksen/Desktop/Giga/{}/results_{}m/{}/FeatureImportance_{}.png'
                .format(aoi, buffer, wandb_exp, aoi))

aoi = args.aoi
buffer = args.buffer
model = args.model

wandb_exp = args.wandb_exp
'''
df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/results_{}m/{}/{}_results_for_plotting.csv'
                 .format(aoi, buffer, wandb_exp, model))
'''
df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/isa_new_schools/BWA/'
                 'BWA_results_new_schools_{}_model.csv'.format(model))

lon_vals = df['lon']
lat_vals = df['lat']

true_locations = []
for i in range(len(lon_vals)):
    true_locations.append((lon_vals[i], lat_vals[i]))

df['tuple loc'] = true_locations

# Get the 2D array for plotting
total_pred_data = []
total_groundtruth = []
for lat in lat_vals:
    for lon in lon_vals:
        # Check if there is school at the lat, lon I am specifying, if yes, grab data, if no, set to NaN
        if (lon, lat) in true_locations:
            prediction = df[df['tuple loc'] == (lon, lat)]['final_pred']*1
            prediction = float(prediction)
            #groundtruth = float(df[df['tuple loc'] == (lon, lat)]['label'])
            total_pred_data.append(prediction)
            #total_groundtruth.append(groundtruth)
        else:
            total_pred_data.append(np.nan)
            #total_groundtruth.append(np.nan)

total_data_arr = np.array(total_pred_data)
total_data_2d = np.reshape(total_data_arr, (len(lat_vals), len(lon_vals)))

#total_gt_arr = np.array(total_groundtruth)
#total_gt_2d = np.reshape(total_gt_arr, (len(lat_vals), len(lon_vals)))

# Get the array of correctly identified (y/n) school connectivity status
#correct_preds = (total_gt_2d==total_data_2d)*1
#correct_preds = correct_preds.astype('float')
#correct_preds[correct_preds == 0] = np.nan

# Full Y/N connectivity schools
plot_2d_array(total_data_2d, aoi, buffer, '{} Predictions'.format(model), '{}_pred_map'.format(model), 'full')
#plot_2d_array(total_gt_2d, aoi, buffer, 'Ground Truth', 'groundtruth_map', 'full')

# No connectivity schools
plot_2d_array(total_data_2d, aoi, buffer, '{} Predictions No Connectivity'.format(model),
              '{}_pred_map_no_connectivity'.format(model), 'no')
#plot_2d_array(total_gt_2d, aoi, buffer, 'Ground Truth No Connectivity', 'groundtruth_map_no_connectivity', 'no')


# Yes connectivity schools
plot_2d_array(total_data_2d, aoi, buffer, '{} Predictions Yes Connectivity'.format(model),
              '{}_pred_map_yes_connectivity'.format(model), 'yes')
#plot_2d_array(total_gt_2d, aoi, buffer, 'Ground Truth Yes Connectivity', 'groundtruth_map_yes_connectivity', 'yes')

# Run feature importance plot
#plot_feature_importance(aoi, buffer)