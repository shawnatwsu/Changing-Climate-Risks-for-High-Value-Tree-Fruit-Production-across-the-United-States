import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from mpl_toolkits.basemap import Basemap


def calculate_trends(da, time_dim, freq):
    if freq == 'monthly':
        resampled_da = da.resample({time_dim: '1MS'}).mean(dim=time_dim)
    elif freq == 'seasonal':
        resampled_da = da.resample({time_dim: 'QS-DEC'}).mean(dim=time_dim)
        #print(resampled_da.shape)
    elif freq == 'annual':
        resampled_da = da.resample({time_dim: '1YS'}).mean(dim=time_dim)
    else:
        raise ValueError("Invalid frequency. Choose from 'monthly', 'seasonal', or 'annual'.")

    resampled_da = resampled_da.assign_coords(season=resampled_da[time_dim].dt.season)

    if freq == 'seasonal':
        grouped_data = resampled_da.groupby('season')
        trend_data = np.zeros((4, *da.shape[1:3],))
        season_data_list = [resampled_da.where(resampled_da.season == season, drop=True) for season in ['DJF', 'MAM', 'JJA', 'SON']]
        for i, season_data in enumerate(season_data_list):
            for lat_idx in range(da.shape[1]):
                for lon_idx in range(da.shape[2]):
                    slope, _, _, _, _ = linregress(np.arange(len(season_data[time_dim])), season_data[:, lat_idx, lon_idx])
                    trend_data[i, lat_idx, lon_idx] = slope * 10# Multiply by 10 to get the trend per decade
    else:
        trend_data = np.zeros((*da.shape[1:3],))
        for lat_idx in range(da.shape[1]):
            for lon_idx in range(da.shape[2]):
                slope, _, _, _, _ = linregress(np.arange(len(resampled_da[time_dim])), resampled_da[:, lat_idx, lon_idx])
                trend_data[lat_idx, lon_idx] = slope * 10# Multiply by 10 to get the trend per decade

    return trend_data, resampled_da
#calculates teh climatology per month
def calculate_monthly_climatology(da, time_dim):
    monthly_climatology = da.groupby(da[time_dim].dt.month).mean(dim=time_dim)
    return monthly_climatology




def calculate_monthly_trends(da, time_dim):
    resampled_da = da.resample({time_dim: '1MS'}).mean(dim=time_dim)
    trend_data = np.zeros((12, *da.shape[1:]))

    for month in range(12):
        month_data = resampled_da.sel(day=resampled_da[time_dim].dt.month == (month + 1))
        #print(month_data.shape)
        for lat_idx in range(da.shape[1]):
            for lon_idx in range(da.shape[2]):
                slope, _, _, _, _ = linregress(np.arange(len(month_data[time_dim])), month_data[:, lat_idx, lon_idx])
                trend_data[month, lat_idx, lon_idx] = slope * 10  # Multiply by 10 to get the trend per decade

    return trend_data, resampled_da



# Define the bounding box for CONUS
lon_min, lon_max = -124.736342, -66.945392
lat_min, lat_max = 24.25, 49.75

# Read the NetCDF file using xarray
ds = xr.open_dataset('/home/shawn_preston/SHAWN_Data/tmin_gridmet.nc')

# Convert the longitudes from 0-360 to -180 to 180 format
#ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby('lon')

# Set the time range for the analysis
start_date = '1979-01-01'
end_date = '2022-12-31'

# Slice the dataset to the desired region and time period
ds_subset = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min), day=slice(start_date, end_date)).sortby('lat')

# Calculate trends for each specified frequency
freqs = ['monthly', 'seasonal', 'annual']
trends = {}
resampled_data = {}
temp_data = ds_subset['tmin']-273.15

for freq in freqs:
    trends[freq], resampled_data[freq] = calculate_trends(temp_data, 'day', freq)

# Calculate monthly trends separately
monthly_trends, monthly_resampled_data = calculate_monthly_trends(temp_data, 'day')

# Calculate monthly climatology
monthly_climatology = calculate_monthly_climatology(ds_subset['tmin'] - 273.15, 'day')


# Plot the trends
cmap = plt.get_cmap('bwr')

# Plot the monthly trends
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 12))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for idx, month in enumerate(months):
    ax = axs[idx // 4, idx % 4]
    m = Basemap(ax=ax, projection='cyl',resolution = 'i', llcrnrlon=lon_min, urcrnrlon=lon_max, llcrnrlat=lat_min, urcrnrlat=lat_max)
    m.drawstates()
    m.drawcoastlines()
    m.drawcountries()
    lon, lat = np.meshgrid(ds_subset.lon, ds_subset.lat)
    im = m.pcolormesh(lon, lat, monthly_trends[idx, :, :], cmap=cmap,vmin = -1, vmax = 1, latlon=True)
    cbar = m.colorbar(im, location='bottom')
    cbar.set_label('Minimum Temperature Trend (°C/decade)')
    plt.annotate('Data Source: Gridmet', xy=(0.01, 0.03), xycoords='axes fraction', fontsize=6,fontweight = 'bold',  color='k')
    ax.set_title(f'{month} Minimum Temperature Trends (1979-2022)')

plt.savefig('/home/shawn_preston/FIGURES/Gridmet_monthly_tmin_trends_CONUS_1979_2022.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot the seasonal trends
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
seasons = ['DJF', 'MAM', 'JJA', 'SON']

for idx, season in enumerate(seasons):
    ax = axs[idx]
    m = Basemap(ax=ax, projection='cyl',resolution = 'i', llcrnrlon=lon_min, urcrnrlon=lon_max, llcrnrlat=lat_min, urcrnrlat=lat_max)
    m.drawstates()
    m.drawcoastlines()
    m.drawcountries()
    lon, lat = np.meshgrid(ds_subset.lon, ds_subset.lat)
    im = m.pcolormesh(lon, lat, trends['seasonal'][idx, :, :], cmap=cmap,vmin = -1, vmax =1, latlon=True)
    cbar = m.colorbar(im, location='bottom')
    cbar.set_label('Minimum Temperature Trend (°C/decade)')
    plt.annotate('Data Source: Gridmet', xy=(0.01, 0.03), xycoords='axes fraction', fontsize=6,fontweight = 'bold',  color='k')
    ax.set_title(f'{season} MinimumTemperature Trends (1979-2022)')

    
    
    
    
plt.savefig('/home/shawn_preston/FIGURES/Gridmet_seasonal_tmin_trends_CONUS_1979_2022.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot the monthly climatology
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 12))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for idx, month in enumerate(months):
    
    vmin = np.min(monthly_climatology[idx,:,:])
    vmax = np.max(monthly_climatology[idx,:,:])
    row = idx // 4
    col = idx % 4
    ax = axs[row, col]
    m = Basemap(ax=ax, projection='cyl', resolution='i', llcrnrlon=lon_min, urcrnrlon=lon_max, llcrnrlat=lat_min, urcrnrlat=lat_max)
    m.drawstates()
    m.drawcoastlines()
    m.drawcountries()
    lon, lat = np.meshgrid(ds_subset.lon, ds_subset.lat)
    im = m.pcolormesh(lon, lat, monthly_climatology[idx, :, :], cmap='viridis', latlon=True, vmin = vmin, vmax = vmax)
    cbar = m.colorbar(im, location='bottom')
    cbar.set_label('Minimum Temperature (°C)')
    plt.annotate('Data Source: Gridmet', xy=(0.01, 0.03), xycoords='axes fraction', fontsize=6, fontweight='bold', color='k')
    ax.set_title(f'{month} Min. Temp Climatology (79-22)')

plt.savefig('/home/shawn_preston/FIGURES/Gridmet_monthly_tmin_climatology_CONUS_1979_20221.png', dpi=300, bbox_inches='tight')
plt.show()


# Plot the annual trends
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
m = Basemap(ax=ax, projection='cyl',resolution = 'i', llcrnrlon=lon_min, urcrnrlon=lon_max, llcrnrlat=lat_min, urcrnrlat=lat_max)
m.drawstates()
m.drawcoastlines()
m.drawcountries()
lon, lat = np.meshgrid(ds_subset.lon, ds_subset.lat)
im = m.pcolormesh(lon, lat, trends['annual'], cmap=cmap,latlon=True, vmin = -1, vmax =1)
cbar = m.colorbar(im, location='bottom')
cbar.set_label('Minimum Temperature Trend (°C/decade)')
plt.annotate('Data Source: Gridmet', xy=(0.01, 0.03), xycoords='axes fraction', fontsize=6,fontweight = 'bold',  color='k')
ax.set_title(f'Annual Minimum Temperature Trends (1979-2022)')

plt.savefig('/home/shawn_preston/FIGURES/Gridmet_annual_tmin_trends_CONUS_1979_2022.png', dpi=300, bbox_inches='tight')
plt.show()

