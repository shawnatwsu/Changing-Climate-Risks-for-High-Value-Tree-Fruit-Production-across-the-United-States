import xarray as xr  # For handling multi-dimensional data arrays
import matplotlib.pyplot as plt  # For plotting
import cartopy.crs as ccrs  # For map projections
import cartopy.feature as cfeature  # For adding geographic features to plots
import numpy as np  # For numerical operations
from matplotlib.lines import Line2D  # For creating custom lines in plots
import matplotlib.colors as mcolors  # For color mapping in plots
import matplotlib.cm as cm  # For colormap handling
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For adjusting axes positions
import geopandas as gpd  # For geographic data operations
from shapely.geometry import Point  # For handling geometric objects
import regionmask  # For masking regions in geographic data
import matplotlib.ticker as mticker  # For custom tick formatting
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # For formatting longitude and latitude

# Function to create a custom colormap from a list of colors and their positions
def make_colormap(colors, position):
    return mcolors.LinearSegmentedColormap.from_list('', list(zip(position, colors)))

# Define the discrete levels and the colors for the color maps for various climatology and trend variables
level_bounds = {
    'climatology_cdd': np.linspace(0, 1500, 11),
    'climatology_gdd1': np.linspace(0, 1500, 11),
    'climatology_gdd2': np.linspace(0, 6000, 11),
    'climatology_tmax': np.linspace(0, 90, 11),
    'climatology_tmin': np.linspace(0, 60, 11),
    'climatology_chill_portion': np.linspace(0, 140, 11),
    'trend_cdd': np.linspace(-40, 40, 10),
    'trend_chill_portion': np.linspace(-5, 5, 10),
    'trend_gdd1': np.linspace(-40, 40, 10),
    'trend_gdd2': np.linspace(-70, 70, 10),
    'trend_tmin': np.linspace(-3, 3, 10),
    'trend_tmax': np.linspace(-4, 4, 10)
}

# Load the climatology and trend datasets using xarray
climatology_ds = xr.open_dataset('/home/shawn_preston/climatology.nc')
trend_ds = xr.open_dataset('/home/shawn_preston/trend.nc')

# Apply masks to the climatology dataset to filter out invalid data points
climatology_ds['climatology_cdd'] = abs(climatology_ds['climatology_cdd'].where(climatology_ds['climatology_cdd'] < 0))
climatology_ds['climatology_gdd1'] = climatology_ds['climatology_gdd1'].where(climatology_ds['climatology_gdd1'] < 2500)
climatology_ds['climatology_gdd2'] = climatology_ds['climatology_gdd2'].where(climatology_ds['climatology_gdd2'] < 6000)
climatology_ds['climatology_tmin'] = climatology_ds['climatology_tmin'].where(climatology_ds['climatology_tmin'] > 0)
climatology_ds['climatology_tmax'] = climatology_ds['climatology_tmax'].where(climatology_ds['climatology_tmax'] > 0)

# Load .npy data for chill portion climatology and trends
climatology_chill_portion_data = np.load('/home/shawn_preston/Chill_Portion_Climatology.npy')
trend_chill_portion_data = np.load('/home/shawn_preston/Chill_Portion_Trend.npy')
trend_chill_portion_data *= 10  # Scale the trend data for visualization

# Generate lat/lon ranges based on the shape of the .npy data
lat_range = np.linspace(49.4, 25.07, climatology_chill_portion_data.shape[0])  # Latitude range (note reversed order)
lon_range = np.linspace(-124.8, -67.06, climatology_chill_portion_data.shape[1])  # Longitude range
lons, lats = np.meshgrid(lon_range, lat_range)  # Create 2D grids for longitude and latitude

# Multiply trend values by 10 for better visualization, except for chill portion
for var in ['trend_cdd', 'trend_gdd1', 'trend_gdd2', 'trend_tmax', 'trend_tmin']:
    trend_ds[var] = trend_ds[var] * 10

# Define color schemes for different variables
color_lists = {
    'climatology_cdd': plt.cm.Blues,
    'climatology_gdd1': plt.cm.YlGn,
    'climatology_gdd2': plt.cm.YlGn,
    'climatology_tmax': plt.cm.RdPu,
    'climatology_tmin': plt.cm.RdPu,
    'climatology_chill_portion': plt.cm.Blues,
    'trend_cdd': plt.cm.RdBu_r,
    'trend_gdd1': plt.cm.RdBu_r,
    'trend_gdd2': plt.cm.RdBu_r,
    'trend_tmax': plt.cm.RdBu_r,
    'trend_tmin': plt.cm.RdBu_r,
    'trend_chill_portion': plt.cm.RdBu_r
}

# Create colormaps and norms for climatology and trend variables based on the predefined level bounds
colormaps = {}
norms = {}
for var, bounds in level_bounds.items():
    colormaps[var] = color_lists[var]
    norms[var] = mcolors.BoundaryNorm(bounds, colormaps[var].N)

# Utility function to add latitude and longitude ticks to plots
def add_ticks(ax):
    lon_formatter = LONGITUDE_FORMATTER
    lat_formatter = LATITUDE_FORMATTER
    ax.tick_params(axis='both', which='major', direction='out', labelsize=10)
    lon_ticks = np.arange(-180, 181, 5)
    lat_ticks = np.arange(-90, 91, 5)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    ax.tick_params(labeltop=False, labelright=False)

# Function to format trend colorbar ticks
def format_trend_ticks(cbar, var_name, colorbar_ticks_cp):
    if 'trend_' in var_name:
        ticks = colorbar_ticks_cp[var_name]
        cbar.set_ticks(ticks)
        integer_ticks = [f"{int(tick)}" for tick in ticks]
        cbar.set_ticklabels(integer_ticks)
        cbar.ax.tick_params(labelsize=12)

# Define labels for the colorbars used in the plots
colorbar_labels = {
    'climatology_cdd': 'Accumulation Days',
    'trend_cdd': 'Degree Days/Decade',
    'climatology_chill_portion': 'Chill Portion',
    'trend_chill_portion': 'CP/Decade',
    'climatology_gdd1': 'Accumulation Days',
    'trend_gdd1': 'Degree Days/Decade',
    'climatology_gdd2': 'Accumulation Days',
    'trend_gdd2': 'Degree Days/Decade',
    'climatology_tmax': 'Days Above',
    'trend_tmax': 'Days/Decade',
    'climatology_tmin': 'Days Above',
    'trend_tmin': 'Days/Decade'
}

# --- Figure 2: CDD & Chill Portion ---
def plot_cdd_chill_portion():
    # Create a figure with a 2x2 grid of subplots
    fig_cp, axes_cp = plt.subplots(nrows=2, ncols=2, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    fig_cp.suptitle('Cold Degree Days/Chill Portion Climatology and Trends', fontsize=28, fontweight='bold')

    # Map of subplot positions to labels
    label_map = {
        (0, 0): 'A',
        (0, 1): 'B',
        (1, 0): 'i',
        (1, 1): 'ii'
    }

    # Titles for the subplots
    titles = [
        'Cold Degree Days (November-March)',
        'Chill Portion March 31st'
    ]

    # Define colorbar ticks for different variables
    colorbar_ticks_cp = {
        'climatology_cdd': np.arange(0, 1501, 300),
        'trend_cdd': np.arange(-50, 50, 5),
        'climatology_chill_portion': np.arange(0, 141, 28),
        'trend_chill_portion': np.linspace(-5, 5, 10)
    }

    for row in range(2):  # Loop through the rows
        for col in range(2):  # Loop through the columns
            # Determine if plotting climatology or trend
            trend_status = 'climatology_' if row == 0 else 'trend_'
            # Determine if plotting CDD or Chill Portion
            var = 'cdd' if col == 0 else 'chill_portion'
            var_name = f'{trend_status}{var}'
            label = label_map[(row, col)]

            ax = axes_cp[row, col]
            add_ticks(ax)  # Add latitude and longitude ticks
            ax.text(0.05, 0.05, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='left')

            if var_name == 'trend_cdd':  # Special handling for trend CDD data
                original_trend_ds = xr.open_dataset('/home/shawn_preston/trend.nc')
                data_values = -1 * original_trend_ds[var_name].values * 10
                levels = np.arange(-70, 71, 10)
                cmap = plt.get_cmap('RdBu', len(levels) - 1)
                norm = mcolors.BoundaryNorm(levels, cmap.N, clip=True)
                plot_element = ax.pcolormesh(lons, lats, data_values, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            elif var_name == 'climatology_cdd':  # Handling for climatology CDD data
                data_values = climatology_ds[var_name].values
                plot_element = ax.pcolormesh(lons, lats, data_values, cmap=colormaps[var_name], norm=norms[var_name], transform=ccrs.PlateCarree())
            elif var == 'chill_portion':  # Handling for Chill Portion data
                data_values = climatology_chill_portion_data if 'climatology' in var_name else trend_chill_portion_data
                plot_element = ax.scatter(lons.flatten(), lats.flatten(), c=data_values, s=0.9, cmap=colormaps[var_name], norm=norms[var_name], transform=ccrs.PlateCarree(), clip_on=True)

            ax.coastlines()  # Add coastlines to the plot
            ax.add_feature(cfeature.BORDERS, linestyle='-')  # Add country borders
            ax.add_feature(cfeature.STATES, linestyle='-')  # Add state borders

            if row == 0:  # Add title for the first row (climatology)
                title = titles[col]
                ax.set_title(title, fontsize=22)
            extend = 'both' if 'trend_' in var_name else 'neither'
            cbar = fig_cp.colorbar(plot_element, ax=ax, orientation='horizontal', fraction=0.09, pad=0.09, aspect=25, extend=extend)

            if var_name == 'trend_cdd':  # Special handling for trend CDD colorbar
                cbar.set_ticks(levels)
                cbar.ax.set_xticklabels([str(int(level)) for level in levels], fontsize=12)
            else:
                format_trend_ticks(cbar, var_name, colorbar_ticks_cp)
            cbar.set_label(colorbar_labels[var_name], size=12)

    # Adjust layout and add annotations for climatology and trends
    fig_cp.tight_layout(pad=0, h_pad=0, w_pad=0.1, rect=[0, 0, 0, 0])
    fig_cp.text(0.07, 0.74, 'Climatology', rotation=90, verticalalignment='center', fontsize=20, fontweight='bold')
    fig_cp.text(0.09, 0.74, '1991-2020', rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')
    fig_cp.text(0.07, 0.33, 'Trend', rotation=90, verticalalignment='center', fontsize=20, fontweight='bold')
    fig_cp.text(0.09, 0.33, '1979-2022', rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')

    # Save the figure
    #plt.savefig('CDD_and_Chill_Portion_Subsections.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- Figure 3: GDD1 & GDD2 ---
def plot_gdd1_gdd2():
    # Create a figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle('Growing Degree Day Climatology and Trends', fontsize=28, fontweight='bold')

    # Variables to plot
    variables = ['gdd1', 'gdd2']
    titles = ['GDD1 Climatology (1991-2020)', 'GDD1 Trend (1979-2022)',
              'GDD2 Climatology (1991-2020)', 'GDD2 Trend (1979-2022)']

    # Map of subplot positions to labels
    label_map = {
        (0, 0): 'A',
        (0, 1): 'B',
        (1, 0): 'i',
        (1, 1): 'ii'
    }

    # Colorbar labels for GDD variables
    colorbar_labels_gdd = {
        'climatology_gdd1': 'Accumulation Days',
        'trend_gdd1': 'Degree Days/Decade',
        'climatology_gdd2': 'Accumulation Days',
        'trend_gdd2': 'Degree Days/Decade'
    }

    for j, trend_type in enumerate(['climatology_', 'trend_']):  # Loop through climatology and trend
        for i, var in enumerate(variables):  # Loop through GDD1 and GDD2
            ax = axes[j, i]
            label = label_map[(j, i)]
            ax.text(0.05, 0.05, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='left')

            var_name = f"{trend_type}{var}"
            data_values = climatology_ds[var_name].values if 'climatology' in trend_type else trend_ds[var_name].values

            if 'trend' in trend_type:  # Special handling for trend data
                if var == 'gdd1':
                    levels = np.arange(-40, 41, 5)
                elif var == 'gdd2':
                    levels = np.arange(-70, 71, 10)
                masked_data_values = np.ma.array(data_values, mask=np.isnan(data_values))
                cmap = plt.get_cmap('RdBu_r', len(levels) - 1)
                norm = mcolors.BoundaryNorm(levels, cmap.N, clip=True)
                plot_element = ax.pcolormesh(lons, lats, masked_data_values, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            else:  # Handling for climatology data
                cmap = colormaps[var_name]
                norm = norms[var_name]
                plot_element = ax.pcolormesh(lons, lats, data_values, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

            ax.coastlines()  # Add coastlines to the plot
            ax.add_feature(cfeature.BORDERS, linestyle='-')  # Add country borders
            ax.add_feature(cfeature.STATES, linestyle='-')  # Add state borders
            ax.set_title(titles[j * 2 + i], fontsize=22)

            extend = 'both' if 'trend_' in var_name else 'neither'
            cbar = fig.colorbar(plot_element, ax=ax, orientation='horizontal', pad=0.09, fraction=0.066, aspect=30, extend=extend)
            cbar.set_label(colorbar_labels_gdd[var_name], size=12)

            if 'trend_' in var_name:  # Special handling for trend colorbar ticks
                cbar.set_ticks(levels)
                cbar.ax.set_xticklabels([str(int(level)) for level in levels], fontsize=12)

    # Adjust layout and add annotations for climatology and trends
    fig.tight_layout(pad=0, h_pad=0, w_pad=0, rect=[0, 0, 0, 0])
    fig.text(0.07, 0.74, 'Climatology', rotation=90, verticalalignment='center', fontsize=20, fontweight='bold')
    fig.text(0.09, 0.74, '1991-2020', rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')
    fig.text(0.07, 0.33, 'Trend', rotation=90, verticalalignment='center', fontsize=20, fontweight='bold')
    fig.text(0.09, 0.33, '1979-2022', rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')

    # Save the figure
    #plt.savefig('GDD1_GDD2_Subsections.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- Figure 4: Tmin & Tmax ---
def plot_tmin_tmax():
    # Create a figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle('Extreme Temperature Climatology and Trends', fontsize=28, fontweight='bold')

    # Variables to plot
    variables = ['tmax', 'tmin']
    # Colorbar labels for temperature variables
    colorbar_labels_temp = {
        'climatology_tmax': 'Days Above',
        'trend_tmax': 'Days/Decade',
        'climatology_tmin': 'Days Above',
        'trend_tmin': 'Days/Decade'
    }

    # Titles for the subplots
    titles = [
        'Extreme Heat Days (June-August)',
        'Warm Nights (August-September)', '', ''
    ]

    # Map of subplot positions to labels
    label_map = {
        (0, 0): 'A',
        (0, 1): 'B',
        (1, 0): 'i',
        (1, 1): 'ii'
    }

    for j, trend_type in enumerate(['climatology_', 'trend_']):  # Loop through climatology and trend
        for i, var in enumerate(variables):  # Loop through Tmax and Tmin
            ax = axes[j, i]
            add_ticks(ax)  # Add latitude and longitude ticks
            label = label_map[(j, i)]
            ax.text(0.05, 0.05, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='left')

            var_name = f"{trend_type}{var}"
            data_values = climatology_ds[var_name].values if 'climatology' in trend_type else trend_ds[var_name].values

            if 'trend' in trend_type:  # Special handling for trend data
                if var == 'tmax':
                    levels = np.arange(-5, 6, .5)
                else:
                    levels = np.arange(-5, 6, .5)
                cmap = plt.get_cmap('RdBu_r', len(levels) - 1)
                norm = mcolors.BoundaryNorm(levels, cmap.N, clip=True)
                data_values = np.ma.masked_invalid(data_values)
            else:  # Handling for climatology data
                cmap = colormaps[var_name]
                norm = norms[var_name]

            masked_data_values = np.ma.array(data_values, mask=np.isnan(data_values))
            mesh = ax.pcolormesh(lons, lats, masked_data_values, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

            ax.coastlines()  # Add coastlines to the plot
            ax.add_feature(cfeature.BORDERS, linestyle='-')  # Add country borders
            ax.add_feature(cfeature.STATES, linestyle='-')  # Add state borders
            ax.set_title(titles[j * 2 + i], fontsize=22)

            extend = 'both' if 'trend' in trend_type else 'neither'
            cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.09, fraction=0.066, aspect=30, extend=extend)
            cbar.set_label(colorbar_labels_temp[var_name], size=12)

            if 'trend' in trend_type:  # Special handling for trend colorbar ticks
                cbar.set_ticks(levels)
                cbar.ax.set_xticklabels([str(int(level)) if level.is_integer() else str(level) for level in levels], fontsize=12)

    # Adjust layout and add annotations for climatology and trends
    fig.tight_layout(pad=0, h_pad=0, w_pad=0, rect=[0, 0, 0, 0])
    fig.text(0.07, 0.74, 'Climatology', rotation=90, verticalalignment='center', fontsize=20, fontweight='bold')
    fig.text(0.09, 0.74, '1991-2020', rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')
    fig.text(0.07, 0.33, 'Trend', rotation=90, verticalalignment='center', fontsize=20, fontweight='bold')
    fig.text(0.09, 0.33, '1979-2022', rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')

    # Save the figure
    #plt.savefig('Tmx_Tmn_Subsections.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call the plotting functions to generate the figures
plot_cdd_chill_portion()
plot_gdd1_gdd2()
plot_tmin_tmax()
