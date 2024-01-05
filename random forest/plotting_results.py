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
parser.add_argument("--experiment", help="Experiment type for feature space. Must be one of full or limited.")
args = parser.parse_args()

# lons then lats for countries of interest (up and down goes second)
bbox_dict = {
    'BWA': [15, 35, -15, -30],
    'RWA': [28.5, 31, -1, -3],
    'BIH': [15, 19.75, 45.5, 42.5],
    'BLZ': [-89.25, -88, 18.75, 15.75],
    'GIN': [-15.5, -7.25, 13, 7],
    'SLV': [-90.25, -87.5, 14.5, 13]
}

def plot_2d_array(array, country, buffer_extent, title, savename, connectivity_status, experiment_name):
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
    plt.savefig('/Users/kelseydoerksen/Desktop/Giga/{}/{}_features_results_{}m/{}.png'
                .format(country, experiment_name, buffer_extent, savename))

aoi = args.aoi
buffer = args.buffer
exp = args.experiment

df = pd.read_csv('/Users/kelseydoerksen/Desktop/Giga/{}/{}_features_results_{}m/results_for_plotting.csv'
                 .format(aoi, exp, buffer))
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
            prediction = df[df['tuple loc'] == (lon, lat)]['prediction']*1
            prediction = float(prediction)
            groundtruth = float(df[df['tuple loc'] == (lon, lat)]['label'])
            total_pred_data.append(prediction)
            total_groundtruth.append(groundtruth)
        else:
            total_pred_data.append(np.nan)
            total_groundtruth.append(np.nan)

total_data_arr = np.array(total_pred_data)
total_data_2d = np.reshape(total_data_arr, (len(lat_vals), len(lon_vals)))

total_gt_arr = np.array(total_groundtruth)
total_gt_2d = np.reshape(total_gt_arr, (len(lat_vals), len(lon_vals)))

# Get the array of correctly identified (y/n) school connectivity status
correct_preds = (total_gt_2d==total_data_2d)*1
correct_preds = correct_preds.astype('float')
correct_preds[correct_preds == 0] = np.nan

# Full Y/N connectivity schools

plot_2d_array(total_data_2d, aoi, buffer, 'Random Forest Predictions', 'rf_pred_map', 'full', exp)
plot_2d_array(total_gt_2d, aoi, buffer, 'Ground Truth', 'groundtruth_map', 'full', exp)

# No connectivity schools
plot_2d_array(total_data_2d, aoi, buffer, 'Random Forest Predictions No Connectivity', 'rf_pred_map_no_connectivity',
              'no', exp)
plot_2d_array(total_gt_2d, aoi, buffer, 'Ground Truth No Connectivity', 'groundtruth_map_no_connectivity', 'no', exp)


# Yes connectivity schools
plot_2d_array(total_data_2d, aoi, buffer, 'Random Forest Predictions Yes Connectivity', 'rf_pred_map_yes_connectivity',
              'yes', exp)
plot_2d_array(total_gt_2d, aoi, buffer, 'Ground Truth Yes Connectivity', 'groundtruth_map_yes_connectivity', 'yes', exp)