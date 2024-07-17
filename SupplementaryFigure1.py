import xarray as xr  # Import xarray for working with labeled multi-dimensional arrays
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import cartopy.crs as ccrs  # Import cartopy for map projections
import numpy as np  # Import numpy for numerical operations
import cartopy.feature as cfeature  # Import cartopy features (like borders and coastlines)
import matplotlib.patches as mpatches  # Import patches for creating custom legends
from matplotlib.colors import ListedColormap  # Import colormap utilities from matplotlib
import matplotlib.lines as mlines  # Import lines for custom legend entries

# Load datasets
ds = xr.open_dataset('/home/shawn_preston/p_value.nc')  # Load p-value dataset
ds_original = xr.open_dataset('/home/shawn_preston/tmax/tmax_gridmet.nc')  # Load original dataset for land mask

# Create subplots with 2 rows and 3 columns
fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# List of metrics to plot
metrics = [
    'p_value_gdd1',
    'p_value_gdd2',
    'p_value_cdd',
    'p_value_tmin',
    'p_value_tmax',
    'p_value_last_frost_day'
]

# Define custom levels and colors for p-values
pvalue_discrete_levels = [0, 0.05, 0.1, 0.2, 1]  # Define discrete levels for p-values
pvalue_colors = ['royalblue', 'yellow', 'salmon', 'lightgray']  # Define colors for each level

# Create a colormap from the list of colors
cmap = ListedColormap(pvalue_colors)

# Create a land mask based on the original dataset (non-NaN values indicate land)
mask_land = ~np.isnan(ds_original['tmax'].isel(day=0))

# Loop through metrics and create subplots
for i, ax in enumerate(axs.flat):  # Iterate over axes and metrics
    metric = metrics[i]  # Get the current metric
    
    # Extract data for the current metric
    data = ds[metric]
    
    # Apply the land mask to the data
    data = data.where(mask_land)
        
    # Plot data
    p = data.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, cmap=cmap,
                  levels=pvalue_discrete_levels)
    
    # Add hatching for non-significant areas (p > 0.05) considering the mask
    non_significant = (data > 0.05) & mask_land
    ax.contourf(ds['lon'], ds['lat'], non_significant,
                levels=[0.5, 1.5], hatches=['...', None], colors='none',
                transform=ccrs.PlateCarree())
    
    # Set the title for each subplot
    if metric == 'p_value_cdd':
        ax.set_title('p_value_days_below_0C')
    else:
        ax.set_title(metric)
    
    # Add coastlines, borders, and state boundaries to the plot
    ax.coastlines()
    ax.set_extent([-125, -66, 24, 50])  # CONUS bounding box
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':')

    # Add a colorbar to each subplot
    cbar = plt.colorbar(p, ax=ax, orientation='horizontal', fraction=0.06, pad=0.1)
    cbar.set_label('p-value', rotation=0)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.2)

# Create custom legend for p-values
legend_labels = ['0-0.05', '0.05 - 0.1', '0.1 - 0.2', '0.2-1', 'No Data']
legend_colors = pvalue_colors + ['white']
legend_patches = [mpatches.Patch(facecolor=color, edgecolor='black' if color == 'white' else 'none', label=label) 
                  for color, label in zip(legend_colors, legend_labels)]

# Add custom legend for p-values to the plot
legend1 = plt.legend(handles=legend_patches, bbox_to_anchor=(0.5, 0.44), loc='center', title='P-value',
                     bbox_transform=fig.transFigure, ncol=5)
plt.gca().add_artist(legend1)

# Create custom legend for hatches indicating non-significant areas
hatch_patch = mpatches.Patch(facecolor='none', hatch='...', edgecolor='black', label='Not \nSignificant')
legend2 = plt.legend(handles=[hatch_patch], bbox_to_anchor=(0.75, 0.44), loc='center', title='',
                     bbox_transform=fig.transFigure, ncol=1, frameon=True, framealpha=1)

# Add legends to the plot
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

# Set the main title for the figure
plt.suptitle('P-Values for Temperature-Based Metrics', y=0.83, fontweight='bold', fontsize=20)

# Save the plot to a file
plt.savefig('pvalue_signficance_hashes.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
