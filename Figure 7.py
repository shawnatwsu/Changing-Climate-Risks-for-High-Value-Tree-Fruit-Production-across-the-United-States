#Plotting the Potential Climate Damage Index per Decade
import xarray as xr  
import numpy as np 
import matplotlib.pyplot as plt  
import cartopy.crs as ccrs  
import cartopy.feature as cfeature  
import matplotlib.colors as mcolors  
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  

# Load climatology and trend datasets
climatology_ds = xr.open_dataset('/home/shawn_preston/climatology.nc')
trend_ds = xr.open_dataset('/home/shawn_preston/trend.nc')

# Load the interpolated Chill Portion datasets
chill_trend_ds = xr.open_dataset('/home/shawn_preston/Prestonetal20241stPaper/Chill_Portion_Trend_interpolated.nc')
chill_climatology_ds = xr.open_dataset('/home/shawn_preston/Prestonetal20241stPaper/Chill_Portion_Climatology_interpolated.nc')

# Ensure that the variables exist
if 'trend_chill' not in chill_trend_ds.variables:
    raise KeyError("Variable 'trend_chill' not found in Chill Portion trend dataset.")

if 'climatology_chill' not in chill_climatology_ds.variables:
    raise KeyError("Variable 'climatology_chill' not found in Chill Portion climatology dataset.")

# Extract 'trend_chill' from chill_trend_ds
trend_chill_array = chill_trend_ds['trend_chill'].values

# Extract 'climatology_chill' from chill_climatology_ds
climatology_chill_array = chill_climatology_ds['climatology_chill'].values

# Scale trend data to represent changes per decade
trend_variables = ['trend_cdd', 'trend_tmax', 'trend_gdd1', 'trend_gdd2', 'trend_tmin']
for var in trend_variables:
    if var in trend_ds.variables:
        trend_ds[var] = trend_ds[var] * 10  # Multiply by 10 for per-decade trends
    else:
        print(f"Warning: Variable '{var}' not found in trend dataset.")

# Scale 'trend_chill' data
trend_chill_array = trend_chill_array * 10  # Multiply by 10 for per-decade trends

# Extract trend data arrays for each metric from trend_ds
trend_cdd_array = trend_ds['trend_cdd'].values        # Cooling Degree Days
trend_tmax_array = trend_ds['trend_tmax'].values      # High Heat Days (Extreme Tmax > 34°C)
trend_gdd1_array = trend_ds['trend_gdd1'].values      # GDD Bud Break
trend_gdd2_array = trend_ds['trend_gdd2'].values      # GDD General Growth
trend_tmin_array = trend_ds['trend_tmin'].values      # Fall Minimum Temperature (Min Temp > 15°C)

# Print statements to verify the number of valid (non-NaN) values
def print_valid_data_count(array, label):
    non_nan_count = np.count_nonzero(~np.isnan(array))
    print(f"{label}: {non_nan_count} valid data points")

print_valid_data_count(trend_cdd_array, 'Cooling Degree Days (trend_cdd)')
print_valid_data_count(trend_tmax_array, 'High Heat Days (trend_tmax)')
print_valid_data_count(trend_gdd1_array, 'GDD Bud Break (trend_gdd1)')
print_valid_data_count(trend_gdd2_array, 'GDD General Growth (trend_gdd2)')
print_valid_data_count(trend_tmin_array, 'Fall Minimum Temperature (trend_tmin)')
print_valid_data_count(trend_chill_array, 'Chill Portion (trend_chill) after multiplying by -1')

# Ensure that the Chill Portion data arrays have the same shape as the other datasets
if trend_chill_array.shape != trend_cdd_array.shape:
    print("Interpolating Chill Portion data to match the grid of other datasets.")
    # Create DataArrays for trend_chill and climatology_chill
    trend_chill_da = xr.DataArray(trend_chill_array, coords=[chill_trend_ds['lat'], chill_trend_ds['lon']], dims=['lat', 'lon'])
    climatology_chill_da = xr.DataArray(climatology_chill_array, coords=[chill_climatology_ds['lat'], chill_climatology_ds['lon']], dims=['lat', 'lon'])
    # Interpolate to the grid of trend_ds
    lats = trend_ds['lat'].values
    lons = trend_ds['lon'].values
    trend_chill_da_interp = trend_chill_da.interp(lat=lats, lon=lons, method='linear')
    climatology_chill_da_interp = climatology_chill_da.interp(lat=lats, lon=lons, method='linear')
    # Update arrays
    trend_chill_array = trend_chill_da_interp.values 
    climatology_chill_array = climatology_chill_da_interp.values 
else:
    lats = trend_ds['lat'].values
    lons = trend_ds['lon'].values

# Create 2D grids of latitude and longitude coordinates
lons_grid, lats_grid = np.meshgrid(lons, lats)

# Check for NaN values in trend_chill_array after interpolation
nan_count = np.count_nonzero(np.isnan(trend_chill_array))
total_cells = trend_chill_array.size
print(f"Total cells in trend_chill_array: {total_cells}")
print(f"Number of NaN cells in trend_chill_array: {nan_count}")

# **Adjust the detrimental trend condition for Chill Portions**
detrimental_trend_cdd = (trend_cdd_array > 0)
detrimental_trend_tmax = (trend_tmax_array > 0)
detrimental_trend_gdd1 = (trend_gdd1_array > 0)
detrimental_trend_gdd2 = (trend_gdd2_array > 0)
detrimental_trend_tmin = (trend_tmin_array > 0)
detrimental_trend_chill = (trend_chill_array < 0) 

# Calculate number of grid cells with detrimental Chill Portion trend
num_detrimental_chill = np.count_nonzero(detrimental_trend_chill)
total_chill_cells = np.count_nonzero(~np.isnan(trend_chill_array))
print(f"Total grid cells with Chill Portion data: {total_chill_cells}")
print(f"Number of grid cells with detrimental Chill Portion trend: {num_detrimental_chill}")

# Sum the number of detrimental trends at each grid cell
detrimental_trend_count = (
    detrimental_trend_cdd.astype(int) +
    detrimental_trend_tmax.astype(int) +
    detrimental_trend_gdd1.astype(int) +
    detrimental_trend_gdd2.astype(int) +
    detrimental_trend_tmin.astype(int) +
    detrimental_trend_chill.astype(int)  # Include Chill Portion
)

# Set grid cells with zero detrimental trends to NaN
detrimental_trend_count = np.where(detrimental_trend_count == 0, np.nan, detrimental_trend_count)

# Extract climatology data arrays for each metric
climatology_cdd_array = climatology_ds['climatology_cdd'].values
climatology_tmax_array = climatology_ds['climatology_tmax'].values
climatology_gdd1_array = climatology_ds['climatology_gdd1'].values
climatology_gdd2_array = climatology_ds['climatology_gdd2'].values
climatology_tmin_array = climatology_ds['climatology_tmin'].values

# Handle missing values in climatology arrays
climatology_cdd_array = np.where(climatology_cdd_array == 0, np.nan, climatology_cdd_array)
climatology_tmax_array = np.where(climatology_tmax_array == 0, np.nan, climatology_tmax_array)
climatology_gdd1_array = np.where(climatology_gdd1_array == 0, np.nan, climatology_gdd1_array)
climatology_gdd2_array = np.where(climatology_gdd2_array == 0, np.nan, climatology_gdd2_array)
climatology_tmin_array = np.where(climatology_tmin_array == 0, np.nan, climatology_tmin_array)

# Compute relative trends as percentage change from climatology (Method 2)
with np.errstate(divide='ignore', invalid='ignore'):
    relative_trend_cdd = (trend_cdd_array * 100) / climatology_cdd_array
    relative_trend_tmax = (trend_tmax_array * 100) / climatology_tmax_array
    relative_trend_gdd1 = (trend_gdd1_array * 100) / climatology_gdd1_array
    relative_trend_gdd2 = (trend_gdd2_array * 100) / climatology_gdd2_array
    relative_trend_tmin = (trend_tmin_array * 100) / climatology_tmin_array
    relative_trend_chill = (trend_chill_array * 100) / climatology_chill_array  # Chill Portion

# **Adjust the detrimental trend condition for Chill Portions with threshold**
detrimental_trend_cdd_threshold = (relative_trend_cdd > 10)
detrimental_trend_tmax_threshold = (relative_trend_tmax > 10)
detrimental_trend_gdd1_threshold = (relative_trend_gdd1 > 10)
detrimental_trend_gdd2_threshold = (relative_trend_gdd2 > 10)
detrimental_trend_tmin_threshold = (relative_trend_tmin > 10)
detrimental_trend_chill_threshold = (relative_trend_chill > -10)  

# Calculate number of grid cells with detrimental Chill Portion trend
num_detrimental_chill_threshold = np.count_nonzero(detrimental_trend_chill_threshold)
print(f"Number of grid cells with detrimental Chill Portion trend (>10% increase): {num_detrimental_chill_threshold}")

# Sum the number of detrimental trends at each grid cell based on threshold
detrimental_trend_count_threshold = (
    detrimental_trend_cdd_threshold.astype(int) +
    detrimental_trend_tmax_threshold.astype(int) +
    detrimental_trend_gdd1_threshold.astype(int) +
    detrimental_trend_gdd2_threshold.astype(int) +
    detrimental_trend_tmin_threshold.astype(int) +
    detrimental_trend_chill_threshold.astype(int)  # Include Chill Portion
)

# Set grid cells with zero detrimental trends to NaN
detrimental_trend_count_threshold = np.where(detrimental_trend_count_threshold == 0, np.nan, detrimental_trend_count_threshold)

# Verify shapes to ensure consistency
print(f"Shape of lats_grid: {lats_grid.shape}")
print(f"Shape of lons_grid: {lons_grid.shape}")
print(f"Shape of detrimental_trend_count: {detrimental_trend_count.shape}")
print(f"Shape of detrimental_trend_count_threshold: {detrimental_trend_count_threshold.shape}")

# Custom labels for the colorbars
custom_labels = {
    1: '1 Trend',
    2: '2 Trends',
    3: '3 Trends',
    4: '4 Trends',
    5: '5 Trends',
    6: '6 Trends'
}

# Define the colormap and normalization for six categories
levels = np.arange(0.5, 7, 1)  # 0.5 to 6.5 to center the colors on integers 1-6
cmap = plt.get_cmap('OrRd', 6)
norm = mcolors.BoundaryNorm(levels, cmap.N)

# Create a figure with two subplots side by side
fig = plt.figure(figsize=(22, 8))  # Increased width to accommodate

# First subplot
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax1.set_extent([-125, -67, 25, 50], crs=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle='-')
ax1.add_feature(cfeature.STATES, linestyle='-')

masked_data = np.ma.array(detrimental_trend_count, mask=np.isnan(detrimental_trend_count))
plot1 = ax1.pcolormesh(
    lons_grid, lats_grid, masked_data,
    cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
    shading='auto'
)
ax1.set_title('Count of Detrimental Climate Trends', fontsize=14)

# Add colorbar under ax1
cax1 = inset_axes(ax1,
                  width="100%",  # width = 100% of parent_bbox width
                  height="5%",   # height = 5% of parent_bbox height
                  loc='lower center',
                  bbox_to_anchor=(0., -0.05, 1, 1),
                  bbox_transform=ax1.transAxes,
                  borderpad=0)
cbar1 = plt.colorbar(plot1, cax=cax1, orientation='horizontal', ticks=np.arange(1, 7, 1))
cbar_labels = [custom_labels.get(int(level), '') for level in np.arange(1, 7, 1)]
cbar1.ax.set_xticklabels(cbar_labels)
cbar1.set_label('Number of Trends', size=12)

# Second subplot 
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
ax2.set_extent([-125, -67, 25, 50], crs=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle='-')
ax2.add_feature(cfeature.STATES, linestyle='-')

masked_data_threshold = np.ma.array(detrimental_trend_count_threshold, mask=np.isnan(detrimental_trend_count_threshold))
plot2 = ax2.pcolormesh(
    lons_grid, lats_grid, masked_data_threshold,
    cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
    shading='auto'
)
ax2.set_title('Count of Detrimental Climate Trends (>10% Threshold Method)', fontsize=14)

# Add colorbar under ax2
cax2 = inset_axes(ax2,
                  width="100%",  # width = 100% of parent_bbox width
                  height="5%",   # height = 5% of parent_bbox height
                  loc='lower center',
                  bbox_to_anchor=(0., -0.05, 1, 1),
                  bbox_transform=ax2.transAxes,
                  borderpad=0)
cbar2 = plt.colorbar(plot2, cax=cax2, orientation='horizontal', ticks=np.arange(1, 7, 1))
cbar_labels = [custom_labels.get(int(level), '') for level in np.arange(1, 7, 1)]
cbar2.ax.set_xticklabels(cbar_labels)
cbar2.set_label('Number of Trends', size=12)

# Adjust layout 
plt.tight_layout(rect=[0, 0.05, 1, 1])  

# Save the plot
plt.savefig('Detrimental_Trends_With_Chill_Portion.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
