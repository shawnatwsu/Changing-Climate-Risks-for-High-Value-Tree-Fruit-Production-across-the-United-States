import os  # Import os for interacting with the operating system
import pandas as pd  # Import pandas for data manipulation and analysis
import numpy as np  # Import numpy for numerical operations
import re  # Import regular expressions module
import xarray as xr  # Import xarray for working with labeled multi-dimensional arrays
from scipy.stats import linregress  # Import linregress for linear regression analysis

# Helper function to round lat/lon values to match the gridMET resolution
def round_to_nearest(value, decimal_places=2):
    multiplier = 10 ** decimal_places
    return round(value * multiplier) / multiplier

# Open gridMET file and extract lat/lon values
gridmet_file = '/home/shawn_preston/tmax_gridmet.nc'  # Path to the gridMET file
ds = xr.open_dataset(gridmet_file)
lat_values = ds['lat'].values  # Extract latitude values
lon_values = ds['lon'].values  # Extract longitude values

# Define the date of interest for climatology and trend (March 31st)
date_of_interest = (3, 31)

# Directory containing the data files
data_dir = '/weka/data/lab/adam/matthew.yourek/chill_portion_4km/dynamic/sept/observed/'  # Data directory path

# Create a mapping from rounded lat/lon to filename
filename_mapping = {}
for filename in os.listdir(data_dir):  # Iterate over files in the data directory
    match = re.search(r'(-?\d+\.\d+)_(-?\d+\.\d+).txt$', filename)  # Use regex to extract lat/lon from filename
    if match:
        lat, lon = map(float, match.groups())  # Convert extracted strings to floats
        rounded_lat = round_to_nearest(lat)  # Round latitude
        rounded_lon = round_to_nearest(lon)  # Round longitude
        filename_mapping[(rounded_lat, rounded_lon)] = filename  # Map the rounded lat/lon to the filename

# Initialize arrays to store climatology and trend data
climatology_values = np.full((len(lat_values), len(lon_values)), np.nan)  # Array for climatology values
trend_values = np.full((len(lat_values), len(lon_values)), np.nan)  # Array for trend values

# Process the files using the mapping
for lat_idx, lat in enumerate(lat_values):  # Iterate over latitude values
    for lon_idx, lon in enumerate(lon_values):  # Iterate over longitude values
        rounded_lat = round_to_nearest(lat)  # Round latitude
        rounded_lon = round_to_nearest(lon)  # Round longitude
        filename = filename_mapping.get((rounded_lat, rounded_lon))  # Get the corresponding filename
        if filename:  # If a corresponding file exists
            filepath = os.path.join(data_dir, filename)  # Construct the full file path
            df = pd.read_csv(filepath, delim_whitespace=True)  # Read the file into a dataframe
            df_date = df[(df['month'] == date_of_interest[0]) & (df['day'] == date_of_interest[1])]  # Filter data for the date of interest
            
            # Calculate climatology for 1991-2020
            climatology = df_date[(df_date['year'] >= 1991) & (df_date['year'] <= 2020)]['cume_portions'].mean()  # Calculate mean value for 1991-2020
            climatology_values[lat_idx, lon_idx] = climatology  # Store climatology value
            
            # Calculate trend for 1979-2022
            years = df_date['year'].values  # Extract years
            values = df_date['cume_portions'].values  # Extract values
            slope, intercept, r_value, p_value, std_err = linregress(years, values)  # Perform linear regression
            trend_values[lat_idx, lon_idx] = slope  # Store trend value (slope)

# Save the climatology and trend data as .npy files with distinct filenames
np.save('/home/shawn_preston/Chill_Portion_Climatology.npy', climatology_values)  # Save climatology values
np.save('/home/shawn_preston/Chill_Portion_Trend.npy', trend_values)  # Save trend values
