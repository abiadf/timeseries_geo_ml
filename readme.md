
This folder is empty, please download the datasets from the provided links

# # Dataset 1: ECG
- Downloaded the entire dataset from https://physionet.org/content/ptb-xl/1.0.3/
- y = `ptbxl_database.csv`
- X = records<100>
Combined all patient files to form a large X

X:
- pages: 1 for each patient record
- cols: ECG measurements
- rows: col values in time
y:
- rows: 1 for each patient
- cols: 23 features, each representing the chance one of 23 health conditions



# # Dataset 2: Argoverse
- Downloaded the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html#download-link) (`validation` set)
- Scenario: multiple agents (cars, pedestrians...) are moving in a city. Argoverse car observes their past positions and velocities. We want to predict the future trajectory of the focal agent (`focal_track_id`), using other agents (`track_id`) as context.

- Task: predict the future (x,y) positions of focal agent `focal_track_id` using past trajectories of all agents (including its own past trajectory) in this scenario.
- Each JSON file (out of 24988 files) is a 'scenario' focused on predicting exactly 1 `focal_track_id` (found in the `focal_track_id` field)
- The ground truth y is the future timesteps (`observed`=false) of the focal agent, provided in each scenario

Time:
- Measurements are uniformly sampled at 10 Hz.
- Each agent (`target_id`) has a local `timestep` starting from 0. Note that `timestep` is local to the agent and cannot be used to align agents
- Instead, use `start_timestamp/end_timestamp` (ns) for global alignment
- For prediction, best to align all agents to focal agent's `start_timestamp/end_timestamp`

Data:
X (3D array):
- pages = agents, 1 page per `track_id`
- cols = position_x, position_y, heading (radian), velocity_x, velocity_y
- rows = past timesteps where `observed` = true.
Note that when agents disappear/appear during measurement, the # of rows varies across pages

y (2D array):
- rows: future timesteps of focal agent (where `observed` = false)
- cols: x-pos y-pos of focal agent
- we predict y_pred from X, then compare it to the ground truth y

Data usage
Note that `observed` = true implies measurement occurred (we use past measurements to predict the future)
`track_id` != `focal_track_id` AND `observed` = true -> use in X
`track_id` = `focal_track_id` AND `observed` = true -> use in X (it is a strong predictor of focal agent's future position)
`track_id` = `focal_track_id` AND `observed` = false -> ground truth y (don't use in X/y arrays), compare y_pred with it
`track_id` != `focal_track_id` AND `observed` = false -> discard from everywhere

All pages in X are predicting focal agent's y for a few timesteps. Best predictor is possibly when `track_id` = `focal_track_id` AND `observed` = true, as it is the focal agent's past position (which we use in X).

Problem definition: flexible depending on data need
- seq2seq prediction of form 3D-2-2D, where y is a timeseries
- seq2seq 2D-2-2D if we only consider the focal agent's history in X, where y is a timeseries
- seq2point if we consider a single position point (ie, last or average) to predict, where y is NOT a timeseries



# # Dataset 3: Weather Bench 2
Weather Bench 2 [intro page](https://weatherbench2.readthedocs.io/en/latest/data-guide.html), ERA5 dataset found [here](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))

Combined many datasets into one:
- `2m_temperature` [link](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/2m_temperature;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
- `10m_u_component_of_wind` [link](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/10m_u_component_of_wind;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
- `10m_v_component_of_wind` [link](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/10m_v_component_of_wind;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
- `mean_sea_level_pressure` [link](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/mean_sea_level_pressure;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
- [optional] `total_precipitation_6hr` [link](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/total_precipitation_6hr?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))


X (3D array):
- 2m_temperature + 10m_u_component_of_wind + 10m_v_component_of_wind
y (2D array):
- mean sea level pressure


The Zarr folder contains ~92k timesteps (6-hourly 2m temperature from 1959–2022), but the data is chunked across ~921 files for efficient access. Each file stores a piece of the array, not a single timestep.

weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/2m_temperature [link here](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/2m_temperature?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))

Download the folder like so:
`$ gsutil -m cp -r "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/<dataset>"`
replace `<dataset>` with `10m_v_component_of_wind` or `10m_u_component_of_wind` or `2m_temperature` or 


# # Dataset 4: India catchments (CAMELS-IND)
From [this page](https://zenodo.org/records/14999580), download folder `CAMELS_IND_All_Catchments.zip`

X: combine all 472 files from folder `catchment_mean_forcings`, where each file represents a timeseries of the forcing of 1 catchment location

y: `attributes_csv/camels_ind_clim.csv` or `camels_ind_hydro.csv` (missing values), but not other files in that folder as they are static attributes, so are not correlated with X.

If results are nonsensical, implying X is a weak predictor of y (due to data being collected separately, see pdf in data folder), then another option is using `streamflow_timeseries/lstm_pred_streamflow.csv`. However this file is a timeseries, so need to turn each col into a vector of statistical attributes first to have a non-timebased y.


# # Dataset 5: Germany catchments (CAMELS-DE)
From [this page](https://camels-de.org/), very similar to India catchments dataset

X files are in folder `timeseries`
y file is called `CAMELS_DE_hydrogeology_attributes.csv`, about hydrogeology attributes, in main dir


Both X and y contain NaNs, so set these NaN values to 0

X shape: (1582, 25568, 21)
y shape: (1582, 100)


# # Dataset 6: China weather
From [this page](https://huggingface.co/datasets/BUPT-PRIS-727/Weather2K/blob/main/README.md)

The data came in shape X: (1866, 13, 13632), 1866 stations, 13632 timestamps, for 13 properties (first 3 are position), with notably no y.
We reshape X to (0, 2, 1) then extract properties [T, mnt, mxt, rh, ws] (of indices [4, 5, 6, 7, 10]) and make a y from it, taking only the last value per sample, then deleting these cols from X. This produces:

X: (1866, 13632, 8)
y: (1866, 5)



# # Dataset 3: NASA engine
From [this page](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and [this page](https://data.nasa.gov/dataset/phm-2008-challenge)
@article{Saxena2008DamagePM,
  title={Damage propagation modeling for aircraft engine run-to-failure simulation},
  author={Abhinav Saxena and Kai Goebel and Donald C. Simon and Neil H. W. Eklund},
  journal={2008 International Conference on Prognostics and Health Management},
  year={2008},
  pages={1-9},
  url={https://api.semanticscholar.org/CorpusID:206508104}
}

The data is in shape 3D X and 2d y
4 different file groups, each consisting of a 1 train + 1 test + 1 RUL file, total of (4 groupd x (3 files/group) =) 12 files
This dataset is different from the rest in that it has train and test sets at 50% of the dataset (instead of the usual 80%-20%).
We apply our algorithm on scenario 1


# # General data preprocessing
For all datasets, we take the first 50 `pages` (which are equivalent to files) in the dir, then apply a random train-test split (except for the NASA dataset), maintaining the timeseries component intact.

After train-test splitting we scale, after which we run a feature selection based on mutual information between X and y, selecting the 20-30% most correlated features; this helps reduce the effect of poorly-correlated or noisy features.


# # Turkey Gas turbine
https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set


# # Milling
https://www.kaggle.com/datasets/tonylschmitz/digital-machining-database?select=Dataset+6+mat


