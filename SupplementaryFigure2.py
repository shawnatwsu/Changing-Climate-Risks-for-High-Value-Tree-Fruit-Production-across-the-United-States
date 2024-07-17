import os  # Import os module for interacting with the operating system
import pandas as pd  # Import pandas for data manipulation and analysis
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import cartopy.crs as ccrs  # Import cartopy for map projections
import cartopy.feature as cfeature  # Import cartopy features (like borders and coastlines)
import re  # Import regular expressions module
import xarray as xr  # Import xarray for working with labeled multi-dimensional arrays
import matplotlib.colors as mcolors  # Import color mapping utilities from matplotlib
import numpy as np  # Import numpy for numerical operations
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Import for adjusting axis positions

# Function to format a date as 'Month Day'
def format_date(month, day):
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    return f"{month_names[month]} {day}"

# Define a list of colors for the colormap
colors = ['#ffffff', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe', '#0868ac', '#084081', '#063057', '#041c2c']
cmap = mcolors.ListedColormap(colors)  # Create a colormap from the list of colors
bounds = [0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140]  # Define boundaries for data values
norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Create a normalization instance for the colormap

# Function to round a value to the nearest specified decimal places
def round_to_nearest(value, decimal_places=2):
    multiplier = 10 ** decimal_places
    return round(value * multiplier) / multiplier

# Define dates of interest for analysis
dates_of_interest = [(9, 30), (10, 31), (11, 30), (12, 31), (1, 31), (2, 28), (3, 31)]

# Directory containing the data files
data_dir = '/weka/data/lab/adam/matthew.yourek/chill_portion_4km/dynamic/sept/observed/'

# Open the gridMET file and extract latitude and longitude values
gridmet_file = '/home/shawn_preston/tmax/tmax_gridmet.nc'
ds = xr.open_dataset(gridmet_file)
lat_values = ds['lat'].values  # Extract latitude values
lon_values = ds['lon'].values  # Extract longitude values

# Create a mapping from rounded lat/lon to filename
filename_mapping = {}
for filename in os.listdir(data_dir):  # Iterate over files in the data directory
    match = re.search(r'(-?\d+\.\d+)_(-?\d+\.\d+).txt$', filename)  # Use regex to extract lat/lon from filename
    if match:
        lat, lon = map(float, match.groups())  # Convert extracted strings to floats
        rounded_lat = round_to_nearest(lat)  # Round latitude
        rounded_lon = round_to_nearest(lon)  # Round longitude
        filename_mapping[(rounded_lat, rounded_lon)] = filename  # Map the rounded lat/lon to the filename

# Create an empty dictionary to store climatology data
data_dict = {date: [] for date in dates_of_interest}

# Process the files and store climatology data in the dictionary
for lat in lat_values:  # Iterate over latitude values
    for lon in lon_values:  # Iterate over longitude values
        rounded_lat = round_to_nearest(lat)  # Round latitude
        rounded_lon = round_to_nearest(lon)  # Round longitude
        filename = filename_mapping.get((rounded_lat, rounded_lon))  # Get the corresponding filename
        if filename:  # If a corresponding file exists
            filepath = os.path.join(data_dir, filename)  # Construct the full file path
            df = pd.read_csv(filepath, delim_whitespace=True)  # Read the file into a dataframe
            df = df[df['year'].between(1979, 2022)]  # Filter rows to include years between 1979 and 2022
            for date in dates_of_interest:  # Iterate over dates of interest
                month, day = date  # Unpack the month and day
                avg_value = df[(df['month'] == month) & (df['day'] == day)]['cume_portions'].mean()  # Calculate the mean value for the specified date
                data_dict[date].append((lat, lon, avg_value))  # Append the data to the dictionary

# Create an empty dictionary to store trend data
trend_data_dict = {date: [] for date in dates_of_interest}

# Process the files and store trend data in the dictionary
for lat in lat_values:  # Iterate over latitude values
    for lon in lon_values:  # Iterate over longitude values
        rounded_lat = round_to_nearest(lat)  # Round latitude
        rounded_lon = round_to_nearest(lon)  # Round longitude
        filename = filename_mapping.get((rounded_lat, rounded_lon))  # Get the corresponding filename
        if filename:  # If a corresponding file exists
            filepath = os.path.join(data_dir, filename)  # Construct the full file path
            df = pd.read_csv(filepath, delim_whitespace=True)  # Read the file into a dataframe
            df = df[df['year'].between(1979, 2022)]  # Filter rows to include years between 1979 and 2022
            for date in dates_of_interest:  # Iterate over dates of interest
                month, day = date  # Unpack the month and day
                avg_value = df[(df['month'] == month) & (df['day'] == day)]['cume_portions'].mean()  # Calculate the mean value for the specified date
                data_dict[date].append((lat, lon, avg_value))  # Append the data to the dictionary
                
                # Calculate the trend for each (lat, lon) point
                years = df['year'].unique()  # Get the unique years
                values = [df[(df['year'] == year) & (df['month'] == month) & (df['day'] == day)]['cume_portions'].mean() for year in years]  # Calculate mean value for each year
                trend = np.polyfit(years, values, 1)[0]  # Get the slope of the linear fit (trend)
                trend_data_dict[date].append((lat, lon, trend))  # Append the trend data to the dictionary

# Define colormap and normalization for trend data
trend_bounds = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Define boundaries for trend data values
trend_cmap = plt.cm.bwr  # Define colormap for trend data
trend_norm = mcolors.BoundaryNorm(trend_bounds, trend_cmap.N)  # Create a normalization instance for the trend colormap

# Create figure and axes for the plot
fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(35, 10), subplot_kw={'projection': ccrs.PlateCarree()})  # Create a 2x7 grid of subplots
fig.suptitle('Chill Portion Per Month Climatology & Trend (1979-2022)', fontsize=20, fontweight='bold')  # Set the main title for the figure

# Plotting the climatology data in the first row
for ax, (date, data) in zip(axs[0], data_dict.items()):  # Iterate over axes and climatology data
    ax.set_extent([-125, -66.5, 24.4, 49.384358], crs=ccrs.PlateCarree())  # Set the map extent
    ax.coastlines()  # Add coastlines to the plot
    ax.add_feature(cfeature.STATES, linestyle=':')  # Add state borders
    ax.add_feature(cfeature.BORDERS, linestyle=':')  # Add country borders
    lats, lons, values = zip(*data)  # Unpack latitude, longitude, and values
    scatter = ax.scatter(lons, lats, c=values, s=1, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())  # Create a scatter plot
    ax.set_title(format_date(*date), fontsize=12, fontweight='bold')  # Set the title for the subplot

# Plotting the trend data in the second row
for ax, (date, trend_data) in zip(axs[1], trend_data_dict.items()):  # Iterate over axes and trend data
    ax.set_extent([-125, -66.5, 24.4, 49.384358], crs=ccrs.PlateCarree())  # Set the map extent
    ax.coastlines()  # Add coastlines to the plot
    ax.add_feature(cfeature.STATES, linestyle=':')  # Add state borders
    ax.add_feature(cfeature.BORDERS, linestyle=':')  # Add country borders
    lats, lons, trends = zip(*trend_data)  # Unpack latitude, longitude, and trends
    scatter_trend = ax.scatter(lons, lats, c=trends, s=1, cmap=trend_cmap, norm=trend_norm, transform=ccrs.PlateCarree())  # Create a scatter plot
    ax.set_title(format_date(*date), fontsize=12, fontweight='bold')  # Set the title for the subplot

# Adjusting figure layout for colorbars
plt.tight_layout()  # Adjust layout to fit elements
plt.subplots_adjust(bottom=0.25, top=0.85)  # Adjust the spacing at the top and bottom of the figure

# Adding colorbar for climatology
cbar_ax_clim = fig.add_axes([0.08, 0.49, 0.8, 0.05])  # Create an axis for the colorbar
cbar_clim = fig.colorbar(scatter, cax=cbar_ax_clim, orientation='horizontal', label='Chill Portion')  # Create the colorbar
cbar_clim.set_ticks([(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)])  # Set the ticks
cbar_clim.set_ticklabels(['0-14', '15-28', '29-42', '43-56', '57-70', '71-84', '85-98', '99-112', '113-126', '127-140'])  # Set the tick labels
cbar_clim.ax.tick_params(labelsize=12)  # Set the size of the tick labels

# Adding colorbar for trend
cbar_ax_trend = fig.add_axes([0.08, 0.09, 0.8, 0.05])  # Create an axis for the colorbar
cbar_trend = fig.colorbar(scatter_trend, cax=cbar_ax_trend, orientation='horizontal', label='Trend (CP/year)')  # Create the colorbar
cbar_trend.set_ticks(trend_bounds)  # Set the ticks
cbar_trend.set_ticklabels(['-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0', '0.1', '0.2', '0.3', '0.4', '0.5'])  # Set the tick labels
cbar_trend.ax.tick_params(labelsize=12)  # Set the size of the tick labels

# Resize trend plots
for ax in axs[1, :]:  # Iterate over the axes in the second row
    ax_pos = ax.get_position()  # Get the current position of the axis
    ax.set_position([ax_pos.x0, ax_pos.y0 - 0.05, ax_pos.width, ax_pos.height * 1.1])  # Adjust the position and height

plt.tight_layout()  # Adjust layout to fit elements
plt.subplots_adjust(bottom=0.15, top=0.85, hspace=0.1, wspace=0.2)  # Adjust the spacing between subplots
plt.savefig('chill_accumulation_climatology_gridmet_panel_trend_plot_eachmonth.png', dpi=300)  # Save the figure
plt.show()  # Display the figure
