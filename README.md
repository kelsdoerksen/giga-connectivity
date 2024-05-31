<div align="center">

# UNICEF-Giga: School Mapping & Connectivity Prediction with Geospatial Data

<p>
<b><a href="#-description">Description</a></b>
|
<b><a href="#-dataset">Dataset</a></b>
|
<b><a href="#-data-processing">Data Processing</a></b>
|
<b><a href="#-code-organization">Code Organization</a></b>
</p>

</div>

## 📄 Description
This work presents the data processing, model training, testing, and analysis for the purposes of school mapping and school connectivity prediction. This work is developed under Giga, a global initiative by UNICEF-ITU to connect every school to the internet by 2030.

Obtaining complete and accurate information on schools locations is a critical first step to accelerating digital connectivity and driving progress towards SDG4: Quality Education. However, precise GPS coordinate of schools are often inaccurate, incomplete, or even completely non-existent in many developing countries.  In support of the Giga initiative, we leverage machine learning and a combination of remote sensing and auxillary data to accelerate school mapping. We also investigate the ability of geospatial information to be used for predicting connectivity status of schools.

This work aims to support government agencies and connectivity providers in improving school location data to better estimate the costs of digitally connecting schools and plan the strategic allocation of their financial resources.

<p>

## 🌍 Dataset
[MODIS Landcover](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1)

[Global Human Settlement Layer](https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_C)

[Global Human Modification](https://developers.google.com/earth-engine/datasets/catalog/CSP_HM_GlobalHumanModification)

[VIIRS Nighttime Lights](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMCFG)

[Ookla Speedtest](https://www.ookla.com/ookla-for-good/open-data)

[World Bank Electrical Power Grid](https://energydata.info/dataset/derived-map-global-electricity-transmission-and-distribution-lines)

## ⚙️ Data Processing
Prior to generating features, the coordinates of the school and non-school samples were extracted from the `AOI_train.geojson` file provided by UNICEF with the `get_lat_lon_list_from_gdp` function in the `processing_scripts.py` script.

To generate the tabular features extracted from Google Earth Engine data, the [airPy](https://github.com/kelsdoerksen/airPy) package was used with the following command: 
```
python run_airpy.py --gee_data <QUERIED DATA> --band <QUERIED DATA BAND> --region <COORDINATES OF SCHOOLS/NON-SCHOOLS> --date <DATE> --analysis_type <COLLECTION> --buffer_size <BUFFER_SIZE> --configs_dir <DIRECTORY TO SAVE CONFIGS> --save_dir <DIRECTORY TO SAVE TABULAR FEATURES> --add_time no --save_type <CSV>
```

Distance to electrical transmission line and ookla speedtest data features were calculated in the `get_elec` and `get_ookla` functions in the `generate_features.py` script.


## 📚 Code Organization
To run the pipeline, the following command is used:
```
python run_pipeline.py --model <MODEL> --aoi <COUNTRY> --buffer <BUFFER_EXTENT> --root_dir <DIRECTORY OF DATA> --experiment_type <ONLINE/OFFLINE> --features <FEATURES_SPACE> --parameter_tuning <TRUE/FALSE> --target <SCHOOL/CONNECTIVITY> --data_split <PERCENTAGE OR SPATIAL CV>
```
The available configurable parameters are:
* `--model`: Model
    *   `rf`: random forest
    *   `gb`: gradient boosted
    *   `mlp`: multi-layer perceptron
    *   `svm`: support vector machine
    *   `lr`: logistic regression
* `--aoi`: Country
* `--buffer`: Buffer extent surrounding school/non-school. Currently supports 300, 500, 1000m extents
* `--root_dir`: Directory of data
* `--experiment_type`: Wandb experiment type. Online or Offline to save and push run directly to Wandb project.
* `--features`: Feature space to use to train/test model
* `--parameter_tuning`: Specify if you would like to hyperparamter tune model
* `--target`: Model target. School or Connectivity.
* `--data_split`: Specify if percentage split of data (ie 70/30 train/test) or spatial cross validation.

The below folders host the following code:

`data_processing`: all pre-processing scripts to generate tabular feature space.

`classifiers`: each ML classifier used.

`analysis`: scripts for post-processing results into figures and maps.

