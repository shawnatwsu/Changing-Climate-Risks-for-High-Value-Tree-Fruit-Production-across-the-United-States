import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import sem, t
from matplotlib.lines import Line2D

# Function to plot with confidence intervals
def plot_with_confidence_interval(ax, x, y, color, label):
    # Calculate the trend line
    slope, intercept, _, _, _ = stats.linregress(x, y)
    trend = intercept + slope * x
    
    # Calculate the confidence interval
    stderr = sem(y)  # Standard error of the mean
    interval = stderr * t.ppf((1 + 0.95) / 2., len(y) - 1)  # Confidence interval
    
    # Plot the trend line
    ax.plot(x, trend, color=color, linestyle='dashed', label=label)
    
    # Shade the confidence interval
    ax.fill_between(x, trend - interval, trend + interval, color=color, alpha=0.2)

# Function to print results to a file
def print_results(data, label, county, file):
    years = data.year.values
    values = data.values
    slope, intercept, _, p_value, _ = stats.linregress(years, values)
    max_value = max(values)
    min_value = min(values)
    std_dev = np.std(values)

    file.write(f"{county} - {label}\n")
    file.write(f"Years: {years}\n")
    file.write(f"Values: {values}\n")
    file.write(f"Max Value: {max_value:.4f}\n")
    file.write(f"Min Value: {min_value:.4f}\n")
    file.write(f"Standard Deviation: {std_dev:.4f}\n")
    file.write(f"Trend: {slope:.4f}\n")
    file.write(f"P-Value: {p_value:.9f}\n")
    file.write("------\n")

# Load the shapefile
shapefile = '/home/shawn_preston/shapefile_counties/tl_2019_us_county.shp'
watershed = gpd.read_file(shapefile)

# Filter to regions of interest
regions_of_interest_yakima = watershed[watershed['GEOID'] == '53077']  # Yakima County, WA
regions_of_interest_third = watershed[watershed['GEOID'] == '26081']  # Kent County, MI
regions_of_interest_wayne = watershed[watershed['GEOID'] == '36117']  # Wayne County, NY

# Unify the geometries of the regions
unified_geometry_yakima = regions_of_interest_yakima.unary_union
unified_geometry_third = regions_of_interest_third.unary_union
unified_geometry_wayne = regions_of_interest_wayne.unary_union

# Load the datasets
ds_tmax = xr.open_dataset('/home/shawn_preston/tmax/tmax_gridmet.nc')
ds_tmin = xr.open_dataset('/home/shawn_preston/SHAWN_Data/tmin_gridmet.nc')

# Convert temperatures to Celsius
ds_tmax['tmax'] = ds_tmax['tmax'] - 273.15
ds_tmin['tmin'] = ds_tmin['tmin'] - 273.15

# Generate mask for coordinates inside the unified regions
mask_unified_yakima = np.zeros((len(ds_tmax['lat']), len(ds_tmax['lon'])), dtype=bool)
mask_unified_third = np.zeros((len(ds_tmax['lat']), len(ds_tmax['lon'])), dtype=bool)
mask_unified_wayne = np.zeros((len(ds_tmax['lat']), len(ds_tmax['lon'])), dtype=bool)

# Creating mask for Yakima
for i, lat in enumerate(ds_tmax['lat'].values):
    for j, lon in enumerate(ds_tmax['lon'].values):
        mask_unified_yakima[i, j] = unified_geometry_yakima.contains(Point(lon, lat))

# Creating mask for Kent
for i, lat in enumerate(ds_tmax['lat'].values):
    for j, lon in enumerate(ds_tmax['lon'].values):
        mask_unified_third[i, j] = unified_geometry_third.contains(Point(lon, lat))

# Creating mask for Wayne
for i, lat in enumerate(ds_tmax['lat'].values):
    for j, lon in enumerate(ds_tmax['lon'].values):
        mask_unified_wayne[i, j] = unified_geometry_wayne.contains(Point(lon, lat))

# Create xarray DataArray from the masks
mask_unified_da_yakima = xr.DataArray(mask_unified_yakima, coords=[ds_tmax['lat'], ds_tmax['lon']])
mask_unified_da_third = xr.DataArray(mask_unified_third, coords=[ds_tmax['lat'], ds_tmax['lon']])
mask_unified_da_wayne = xr.DataArray(mask_unified_wayne, coords=[ds_tmax['lat'], ds_tmax['lon']])

# Apply the mask to the datasets
ds_tmax_masked_yakima = ds_tmax.where(mask_unified_da_yakima)
ds_tmin_masked_yakima = ds_tmin.where(mask_unified_da_yakima)

ds_tmax_masked_third = ds_tmax.where(mask_unified_da_third)
ds_tmin_masked_third = ds_tmin.where(mask_unified_da_third)

ds_tmax_masked_wayne = ds_tmax.where(mask_unified_da_wayne)
ds_tmin_masked_wayne = ds_tmin.where(mask_unified_da_wayne)

# Compute the number of grid points in the unified regions
n_grid_points_yakima = mask_unified_da_yakima.sum()
n_grid_points_third = mask_unified_da_third.sum()
n_grid_points_wayne = mask_unified_da_wayne.sum()

# Restrict data to specific year range
ds_tmin_masked_yakima = ds_tmin_masked_yakima.where(ds_tmin_masked_yakima['day'].dt.year.isin(range(1979, 2023)), drop=True)
ds_tmin_masked_third = ds_tmin_masked_third.where(ds_tmin_masked_third['day'].dt.year.isin(range(1979, 2023)), drop=True)
ds_tmin_masked_wayne = ds_tmin_masked_wayne.where(ds_tmin_masked_wayne['day'].dt.year.isin(range(1979, 2023)), drop=True)

# Define function to compute the last frost day of a year from January to July
def compute_last_frost_day(year_group):
    frost_days = (year_group['tmin'] <= 0) & (year_group['day'].dt.month.isin(range(1, 8)))
    dayofyear = year_group['day'].dt.dayofyear
    return xr.where(frost_days, dayofyear, np.nan).max('day', skipna=True)

# Apply the function to each year group
last_frost_days_yakima = ds_tmin_masked_yakima.groupby('day.year').map(compute_last_frost_day)
last_frost_days_third = ds_tmin_masked_third.groupby('day.year').map(compute_last_frost_day)
last_frost_days_wayne = ds_tmin_masked_wayne.groupby('day.year').map(compute_last_frost_day)

# Average over lat, lon
last_frost_days_avg_yakima = last_frost_days_yakima.sum(dim=['lat', 'lon']) / n_grid_points_yakima
last_frost_days_avg_third = last_frost_days_third.sum(dim=['lat', 'lon']) / n_grid_points_third
last_frost_days_avg_wayne = last_frost_days_wayne.sum(dim=['lat', 'lon']) / n_grid_points_wayne

# Clip to ensure days count starts from 1
last_frost_days_avg_yakima = last_frost_days_avg_yakima.where(last_frost_days_avg_yakima >= 1)
last_frost_days_avg_third = last_frost_days_avg_third.where(last_frost_days_avg_third >= 1)
last_frost_days_avg_wayne = last_frost_days_avg_wayne.where(last_frost_days_avg_wayne >= 1)

# Compute metrics
fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # Adjusted layout to 2 rows and 3 columns
ax_A, ax_B, ax_C, ax_D, ax_E, ax_F = axs.ravel()

# Count of days with Tmax > 34°C for June, July, August for Yakima County
ds_tmax_masked_summer_yakima = ds_tmax_masked_yakima.where(ds_tmax_masked_yakima['day'].dt.month.isin([6, 7, 8]), drop=True)
count_tmax_above_35_yakima = (ds_tmax_masked_summer_yakima['tmax'] > 34).groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_yakima

# Count of days with Tmax > 34°C for June, July, August for Kent County
ds_tmax_masked_summer_third = ds_tmax_masked_third.where(ds_tmax_masked_third['day'].dt.month.isin([6, 7, 8]), drop=True)
count_tmax_above_35_third = (ds_tmax_masked_summer_third['tmax'] > 34).groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_third

# Count of days with Tmax > 34°C for June, July, August for Wayne County
ds_tmax_masked_summer_wayne = ds_tmax_masked_wayne.where(ds_tmax_masked_wayne['day'].dt.month.isin([6, 7, 8]), drop=True)
count_tmax_above_35_wayne = (ds_tmax_masked_summer_wayne['tmax'] > 34).groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_wayne

# Count of days with Tmin > 15°C for August, September for Yakima County
ds_tmin_masked_late_summer_yakima = ds_tmin_masked_yakima.where(ds_tmin_masked_yakima['day'].dt.month.isin([8, 9]), drop=True)
count_tmin_above_15_yakima = (ds_tmin_masked_late_summer_yakima['tmin'] > 15).groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_yakima

# Count of days with Tmin > 15°C for August, September for Kent County
ds_tmin_masked_late_summer_third = ds_tmin_masked_third.where(ds_tmin_masked_third['day'].dt.month.isin([8, 9]), drop=True)
count_tmin_above_15_third = (ds_tmin_masked_late_summer_third['tmin'] > 15).groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_third

# Count of days with Tmin > 15°C for August, September for Wayne County
ds_tmin_masked_late_summer_wayne = ds_tmin_masked_wayne.where(ds_tmin_masked_wayne['day'].dt.month.isin([8, 9]), drop=True)
count_tmin_above_15_wayne = (ds_tmin_masked_late_summer_wayne['tmin'] > 15).groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_wayne

# GDD (Jan-Apr) for Yakima County with 6°C as baseline and 28°C as max threshold using average temperature
gdd_jan_apr_yakima = ((ds_tmax_masked_yakima['tmax'].where(ds_tmax_masked_yakima['day'].dt.month <= 4) + 
                      ds_tmin_masked_yakima['tmin'].where(ds_tmin_masked_yakima['day'].dt.month <= 4)) / 2).clip(min=6, max=28) - 6
gdd_jan_apr_yakima = gdd_jan_apr_yakima.groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_yakima

# GDD (Jan-Apr) for Kent County with 6°C as baseline and 28°C as max threshold using average temperature
gdd_jan_apr_third = ((ds_tmax_masked_third['tmax'].where(ds_tmax_masked_third['day'].dt.month <= 4) + 
                     ds_tmin_masked_third['tmin'].where(ds_tmin_masked_third['day'].dt.month <= 4)) / 2).clip(min=6, max=28) - 6
gdd_jan_apr_third = gdd_jan_apr_third.groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_third

# GDD (Jan-Apr) for Wayne County with 6°C as baseline and 28°C as max threshold using average temperature
gdd_jan_apr_wayne = ((ds_tmax_masked_wayne['tmax'].where(ds_tmax_masked_wayne['day'].dt.month <= 4) + 
                     ds_tmin_masked_wayne['tmin'].where(ds_tmin_masked_wayne['day'].dt.month <= 4)) / 2).clip(min=6, max=28) - 6
gdd_jan_apr_wayne = gdd_jan_apr_wayne.groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_wayne

# GDD (Jan-Sep) for Yakima County with 6°C as baseline and 28°C as max threshold using average temperature
gdd_jan_sep_yakima = ((ds_tmax_masked_yakima['tmax'].where(ds_tmax_masked_yakima['day'].dt.month <= 9) + 
                      ds_tmin_masked_yakima['tmin'].where(ds_tmin_masked_yakima['day'].dt.month <= 9)) / 2).clip(min=6, max=28) - 6
gdd_jan_sep_yakima = gdd_jan_sep_yakima.groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_yakima

# GDD (Jan-Sep) for Kent County with 6°C as baseline and 28°C as max threshold using average temperature
gdd_jan_sep_third = ((ds_tmax_masked_third['tmax'].where(ds_tmax_masked_third['day'].dt.month <= 9) + 
                     ds_tmin_masked_third['tmin'].where(ds_tmin_masked_third['day'].dt.month <= 9)) / 2).clip(min=6, max=28) - 6
gdd_jan_sep_third = gdd_jan_sep_third.groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_third

# GDD (Jan-Sep) for Wayne County with 6°C as baseline and 28°C as max threshold using average temperature
gdd_jan_sep_wayne = ((ds_tmax_masked_wayne['tmax'].where(ds_tmax_masked_wayne['day'].dt.month <= 9) + 
                     ds_tmin_masked_wayne['tmin'].where(ds_tmin_masked_wayne['day'].dt.month <= 9)) / 2).clip(min=6, max=28) - 6
gdd_jan_sep_wayne = gdd_jan_sep_wayne.groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_wayne

# Days Below 0°C (Nov-Mar) for Yakima County using average tempreatures below 0°C
cdd_nov_mar_yakima = ((ds_tmax_masked_yakima['tmax'].where(ds_tmax_masked_yakima['day'].dt.month.isin([11, 12, 1, 2, 3])) + 
                      ds_tmin_masked_yakima['tmin'].where(ds_tmin_masked_yakima['day'].dt.month.isin([11, 12, 1, 2, 3]))) / 2).where(lambda x: x < 0)
cdd_nov_mar_yakima = (-1) * cdd_nov_mar_yakima.groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_yakima

# Days Below 0°C (Nov-Mar) for Kent County using average tempreatures below 0°C
cdd_nov_mar_third = ((ds_tmax_masked_third['tmax'].where(ds_tmax_masked_third['day'].dt.month.isin([11, 12, 1, 2, 3])) + 
                     ds_tmin_masked_third['tmin'].where(ds_tmin_masked_third['day'].dt.month.isin([11, 12, 1, 2, 3]))) / 2).where(lambda x: x < 0)
cdd_nov_mar_third = (-1) * cdd_nov_mar_third.groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_third

# Days Below 0°C (Nov-Mar) for Wayne County using average tempreatures below 0°C
cdd_nov_mar_wayne = ((ds_tmax_masked_wayne['tmax'].where(ds_tmax_masked_wayne['day'].dt.month.isin([11, 12, 1, 2, 3])) + 
                     ds_tmin_masked_wayne['tmin'].where(ds_tmin_masked_wayne['day'].dt.month.isin([11, 12, 1, 2, 3]))) / 2).where(lambda x: x < 0)
cdd_nov_mar_wayne = (-1) * cdd_nov_mar_wayne.groupby('day.year').sum(dim='day').sum(dim=['lat', 'lon']) / n_grid_points_wayne

# Labels for the plots
labels = [
    'Degree Days',           # For CDD (Nov-Mar)
    'Days Since Jan 1',      # For Average Last Day of Frost
    'Degree Days',           # For GDD (Jan-Apr)
    'Degree Days',           # For GDD (Jan-Sep)
    'Days Above',            # For Days with Tmax > 34°C
    'Days Above'             # For Days with Tmin > 15°C
]

# Reordered plots for Yakima County
plots_yakima = [
    (cdd_nov_mar_yakima, 'Cold Degree Days (November-March)'),
    (last_frost_days_avg_yakima, 'Last Day of Frost (January-July)'),
    (gdd_jan_apr_yakima, 'GDD (January-April)'),
    (gdd_jan_sep_yakima, 'GDD (January-September)'),
    (count_tmax_above_35_yakima, 'Extreme Heat Days (June-August)'),
    (count_tmin_above_15_yakima, 'Warm Nights (August-September)')
]

# Reordered plots for Kent County (Third County)
plots_third = [
    (cdd_nov_mar_third, 'Cold Degree Days (November-March)'),
    (last_frost_days_avg_third, 'Last Day of Frost (January-July)'),
    (gdd_jan_apr_third, 'GDD (January-April)'),
    (gdd_jan_sep_third, 'GDD (January-September)'),
    (count_tmax_above_35_third, 'Extreme Heat Days (June-August)'),
    (count_tmin_above_15_third, 'Warm Nights (August-September)')
]

# Reordered plots for Wayne County
plots_wayne = [
    (cdd_nov_mar_wayne, 'Cold Degree Days (November-March)'),
    (last_frost_days_avg_wayne, 'Last Day of Frost (January-July)'),
    (gdd_jan_apr_wayne, 'GDD (January-April)'),
    (gdd_jan_sep_wayne, 'GDD (January-September)'),
    (count_tmax_above_35_wayne, 'Extreme Heat Days (June-August)'),
    (count_tmin_above_15_wayne, 'Warm Nights (August-September)')
]

# Helper function to format the p-value string
def p_value_format(slope, p_val):
    if p_val < 0.01:
        return f'{slope:.2f} (p<<0.01)'
    elif p_val < 0.05:
        return f'{slope:.2f} (p<0.05)'
    else:
        return f'{slope:.2f} (p={p_val:.2f})'

# Function to annotate trend on the plot
def annotate_trend(ax, slope, p_val, color, county, title=None):
    trend_text = f"{county}: {p_value_format(slope, p_val)}"
    
    if title == "Last Day of Frost (January-July)":
        if color == 'black':
            ax.text(0.98, 0.50, trend_text, color=color, fontsize=8, transform=ax.transAxes, ha='right', va='center')
        elif color == 'blue':
            ax.text(0.98, 0.45, trend_text, color=color, fontsize=8, transform=ax.transAxes, ha='right', va='center')
        elif color == 'green':  # For Wayne County or similar
            ax.text(0.98, 0.40, trend_text, color=color, fontsize=8, transform=ax.transAxes, ha='right', va='center')
    else:
        y_positions = {
            'black': 0.98,
            'blue': 0.93,
            'green': 0.88
        }
        ax.text(0.05, y_positions[color], trend_text, color=color, fontsize=8, transform=ax.transAxes, ha='left', va='top')

# Plotting data for each county and adding confidence intervals and trends
for idx, (plot_yakima, plot_third, plot_wayne) in enumerate(zip(plots_yakima, plots_third, plots_wayne)):
    row = idx // 3
    col = idx % 3
    ax = axs[row, col]
    data_yakima, title_yakima = plot_yakima
    data_third, title_third = plot_third
    data_wayne, title_wayne = plot_wayne

    # Plotting the data
    ax.plot(data_yakima.year, data_yakima, marker='o', markersize=5, linestyle='-', color='black', fillstyle='none')
    ax.plot(data_third.year, data_third, marker='o', markersize=5, linestyle='-', color='blue', fillstyle='none')
    ax.plot(data_wayne.year, data_wayne, marker='o', markersize=5, linestyle='-', color='green', fillstyle='none')
    
    # Plotting the data with confidence intervals
    plot_with_confidence_interval(ax, data_yakima.year, data_yakima, 'black', 'Yakima')
    plot_with_confidence_interval(ax, data_third.year, data_third, 'blue', 'Kent')
    plot_with_confidence_interval(ax, data_wayne.year, data_wayne, 'green', 'Wayne')

    # Annotating trends
    # Yakima County
    slope_yakima, _, _, p_value_yakima, _ = stats.linregress(data_yakima.year, data_yakima) 
    annotate_trend(ax, slope_yakima, p_value_yakima, 'black', 'Yakima', title_yakima)

    # Kent County
    slope_third, _, _, p_value_third, _ = stats.linregress(data_third.year, data_third) 
    annotate_trend(ax, slope_third, p_value_third, 'blue', 'Kent', title_third)

    # Wayne County
    slope_wayne, _, _, p_value_wayne, _ = stats.linregress(data_wayne.year, data_wayne) 
    annotate_trend(ax, slope_wayne, p_value_wayne, 'green', 'Wayne', title_wayne)
    
    # Adds A-F for each subplot
    ax.text(0.94, 0.98, chr(65 + idx), transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_xticks(range(int(data_yakima.year.min()), int(data_yakima.year.max()) + 1, 5))
    ax.set_title(title_yakima, fontsize=16, fontweight='bold')

    # Adjust tick parameters
    ax.tick_params(axis='x', which='major', labelsize=10, width=2, length=6)
    ax.tick_params(axis='y', which='major', labelsize=10, width=2, length=6)
    labels_a = [item.get_text() for item in ax.get_xticklabels()]
    new_labels = ["'" + label[-2:] for label in labels_a]
    ax.set_xticklabels(new_labels)
    ax.set_ylabel(labels[idx], fontsize=12, fontweight='bold')

# Define the legend handles and labels
legend_elements = [Line2D([0], [0], color='black', label='Yakima'),
                   Line2D([0], [0], color='blue', label='Kent'),
                   Line2D([0], [0], color='green', label='Wayne')]

# Creating a unified legend
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='medium')

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the spacing from the top for suptitle
fig.suptitle('Climate Metrics Analysis: County Trends 1979-2022', fontsize=18, fontweight='bold')
plt.savefig('timeseries_counties_with_3Counties_elong1.png', dpi=300)
plt.show()

# Open a text file in write mode
with open('resultscounties.txt', 'w') as file:
    for (data_yakima, label), (data_third, _), (data_wayne, _) in zip(plots_yakima, plots_third, plots_wayne):
        print_results(data_yakima, label, "Yakima County", file)
        print_results(data_third, label, "Kent County", file)
        print_results(data_wayne, label, "Wayne County", file)

# Function to print yearly values for each metric in each county
def print_yearly_values(data, label, county):
    print(f"\n{county} - {label}")
    print("Yearly Values:")
    for year, value in zip(data.year.values, data.values):
        print(f"Year: {year}, Value: {value:.4f}")

# Print yearly values for Yakima County
print("\nYakima County Metrics:")
for data, label in plots_yakima:
    print_yearly_values(data, label, "Yakima County")

# Print yearly values for Kent County
print("\nKent County Metrics:")
for data, label in plots_third:
    print_yearly_values(data, label, "Kent County")

# Print yearly values for Wayne County
print("\nWayne County Metrics:")
for data, label in plots_wayne:
    print_yearly_values(data, label, "Wayne County")

# Save the yearly values to a text file
with open('yearly_values.txt', 'w') as f:
    def print_yearly_values(data, label, county):
        f.write(f"\n{county} - {label}\n")
        f.write("Yearly Values:\n")
        for year, value in zip(data.year.values, data.values):
            f.write(f"Year: {year}, Value: {value:.4f}\n")

    f.write("\nYakima County Metrics:\n")
    for data, label in plots_yakima:
        print_yearly_values(data, label, "Yakima County")

    f.write("\nKent County Metrics:\n")
    for data, label in plots_third:
        print_yearly_values(data, label, "Kent County")

    f.write("\nWayne County Metrics:\n")
    for data, label in plots_wayne:
        print_yearly_values(data, label, "Wayne County")
