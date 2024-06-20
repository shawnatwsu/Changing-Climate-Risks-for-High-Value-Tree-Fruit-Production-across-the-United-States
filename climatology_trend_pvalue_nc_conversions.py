import xarray as xr
import numpy as np
from scipy.stats import linregress

# Function for linear regression
# Calculates the slope and p-value of the linear trend between x and y
def linear_trend(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, p_value

# Function for calculating growing degree days (GDD)
# GDD is calculated as the accumulated heat units above a baseline temperature but below an upper threshold
def calc_gdd(data, baseline, upper_threshold):
    gdd = data.where(data > baseline, baseline).where(data < upper_threshold, upper_threshold) - baseline
    return gdd.where(gdd > 0, 0)

# Function for calculating cooling degree days (CDD)
# CDD is calculated as the accumulated cool units below a baseline temperature
def calc_cdd(data, baseline):
    cdd = data.where(data < baseline) - baseline
    return cdd.where(cdd < 0, 0)

# Function for calculating the last frost day
# The last day of the year with a minimum temperature below 0°C before day 212 (end of July)
def calc_last_frost_day(data):
    last_frost_day = data['day'].dt.dayofyear.where((data['tmin'] <= 0) & (data['day'].dt.dayofyear <= 212)).groupby('day.year').max(dim='day')
    return last_frost_day.assign_coords(year=last_frost_day.year.values)

# Load temperature datasets and convert from Kelvin to Celsius
ds_tmax = xr.open_dataset('/home/shawn_preston/tmax_gridmet.nc')
ds_tmin = xr.open_dataset('/home/shawn_preston/SHAWN_Data/tmin_gridmet.nc')
ds_tmax['tmax'] = ds_tmax['tmax'] - 273.15
ds_tmin['tmin'] = ds_tmin['tmin'] - 273.15

# Calculate mean daily temperature for GDD/CDD
ds = xr.Dataset({'tmean': (ds_tmax['tmax'] + ds_tmin['tmin']) / 2}, coords={'day': ds_tmax['day'], 'lat': ds_tmax['lat'], 'lon': ds_tmax['lon']})

# Lists to hold climatology and trend DataArrays
climatology_dataarrays = []
trend_dataarrays = []
p_value_dataarrays = []

# List of analysis configurations
analyses = [
    {'name': 'gdd1', 'months': range(1, 5), 'baseline': 6, 'upper_threshold': 28, 'function': calc_gdd},
    {'name': 'gdd2', 'months': range(1, 10), 'baseline': 6, 'upper_threshold': 28, 'function': calc_gdd},
    {'name': 'cdd', 'months': [11, 12, 1, 2, 3], 'baseline': 0, 'function': calc_cdd},
    {'name': 'tmin', 'months': [8, 9], 'threshold': 15, 'dataset': ds_tmin, 'variable': 'tmin'},
    {'name': 'tmax', 'months': [6, 7, 8], 'threshold': 34, 'dataset': ds_tmax, 'variable': 'tmax'},
    {'name': 'last_frost_day', 'months': range(1, 8), 'function': calc_last_frost_day, 'dataset': ds_tmin}
]

for analysis in analyses:
    # Common parameters
    climatology_years = range(1991, 2021)
    trend_years = range(1979, 2023)
    months = analysis['months']

    if 'function' in analysis:
        # GDD, CDD, or last frost day calculations
        function = analysis['function']
        dataset = analysis.get('dataset', ds)
        mask_climatology = (dataset['day'].dt.month.isin(months)) & (dataset['day'].dt.year.isin(climatology_years))
        mask_trend = (dataset['day'].dt.month.isin(months)) & (dataset['day'].dt.year.isin(trend_years))
        data_climatology = dataset.sel(day=mask_climatology)
        data_trend = dataset.sel(day=mask_trend)

        if analysis['name'] == 'last_frost_day':
            climatology = function(data_climatology).mean(dim='year')
            trend_data = function(data_trend)
        elif analysis['name'] in ['gdd1', 'gdd2']:
            baseline = analysis['baseline']
            upper_threshold = analysis['upper_threshold']
            climatology_data = function(data_climatology['tmean'], baseline, upper_threshold).groupby('day.year').sum(dim='day')
            climatology = climatology_data.mean(dim='year')
            trend_data = function(data_trend['tmean'], baseline, upper_threshold).groupby('day.year').sum(dim='day')
        elif analysis['name'] == 'cdd':
            baseline = analysis['baseline']
            climatology_data = function(data_climatology['tmean'], baseline).groupby('day.year').sum(dim='day')
            climatology = climatology_data.mean(dim='year')
            trend_data = function(data_trend['tmean'], baseline).groupby('day.year').sum(dim='day')
        else:
            baseline = analysis['baseline']
            climatology_data = function(data_climatology['tmean'], baseline).groupby('day.year').sum(dim='day')
            climatology = climatology_data.mean(dim='year')
            trend_data = function(data_trend['tmean'], baseline).groupby('day.year').sum(dim='day')
    else:
        # Tmin > 15°C or Tmax > 34°C calculations
        threshold = analysis['threshold']
        dataset = analysis['dataset']
        variable = analysis['variable']
        mask_climatology = (dataset['day'].dt.month.isin(months)) & (dataset['day'].dt.year.isin(climatology_years))
        mask_trend = (dataset['day'].dt.month.isin(months)) & (dataset['day'].dt.year.isin(trend_years))
        data_climatology = dataset[variable].sel(day=mask_climatology)
        data_trend = dataset[variable].sel(day=mask_trend)
        climatology_data = (data_climatology >= threshold).groupby('day.year').sum(dim='day')
        climatology = climatology_data.mean(dim='year')
        trend_data = (data_trend >= threshold).groupby('day.year').sum(dim='day')

    # Calculate trend and p-value
    x = np.arange(len(trend_data.year))  # x-values are just the index of years
    trend, p_value = xr.apply_ufunc(linear_trend, x, trend_data, vectorize=True, input_core_dims=[['year'], ['year']], output_core_dims=[[], []])

    # Add the climatology, trend, and p_value DataArrays to their lists
    climatology_dataarrays.append(climatology.rename(f'climatology_{analysis["name"]}'))
    trend_dataarrays.append(trend.rename(f'trend_{analysis["name"]}'))
    p_value_dataarrays.append(p_value.rename(f'p_value_{analysis["name"]}'))

# Create the climatology, trend, and p_value Datasets
climatology_dataset = xr.merge(climatology_dataarrays)
trend_dataset = xr.merge(trend_dataarrays)
p_value_dataset = xr.merge(p_value_dataarrays)

# Save the Datasets to NC files
climatology_dataset.to_netcdf('climatology.nc')
trend_dataset.to_netcdf('trend.nc')
p_value_dataset.to_netcdf('p_value.nc')
