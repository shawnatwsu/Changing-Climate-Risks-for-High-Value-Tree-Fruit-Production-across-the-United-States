README
This repository provides the code for conducting the experiment for the 2024 Preston et al. paper.

Files and Descriptions
climatology_trend_pvalue_nc_conversions.py
This code takes in the raw daily data of maximum and minimum temperatures from the gridMET dataset. You can access the data at the following link: https://www.northwestknowledge.net/metdata/data/.

It then creates the functions for each climate metric analyzed and produces:

The 1991-2020 climatology for each metric
The 1979-2022 trends for each metric
The p-value (p=0.05) for each metric across the United States
To change values or dates, you will need to edit this code to your liking.

Figure2-4.py
This code takes in the files from climatology.nc, trend.nc, chillportionclimatology.npy, and chillportiontrend.npy and creates three distinct figures. These figures highlight different growing and developmental phases for apple fruits:

Figure 2: CDD/Chill Portions
Figure 3: GDD Budbreak/GDD General Growth
Figure 4: Extreme Temperatures > 34°C/Extreme Temperatures > 15°C
Figure5.py
This code takes in the data for all three counties analyzed: Yakima County, WA; Kent County, MI; and Wayne County, NY. The code processes six climate metrics:

Cold Degree Days
Last Day of Spring Frost
GDD Bud Break
GDD General Growth
Extreme Heat Days
Warm Nights
The code also implements trend line annotations with a 95% CI for each county and prints out a table of all metrics with p-values included.
