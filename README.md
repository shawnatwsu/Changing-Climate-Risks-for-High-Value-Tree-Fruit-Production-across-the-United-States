This repository provides the code for conducting the experiement for the 2024 Preston etal paper. 


######-----climatology_trend_pvalue_nc_conversions.py-----######
This code takes in the raw daily data of maximum and minimum temperatures from gridMET dataset metdata https://www.northwestknowledge.net/metdata/data/

It then creates the functions for each climate metric analyzed and produces the 1991-2020 climatology for each metric, the 1979-2022 trends for each metric, and the p-value (p=0.05) for each metric across the United States. 
To change values or dates, you will need to edit this code to your liking.

######-----Figure2-4.py-----######

This code takes in the files from the climatology.nc, trend.nc and chillportionclimatology.npy and chillportiontrend.npy and creates 3 distinct figures. These figures highlight different growing and developemental phases for apple fruits
Figure 2: CDD/Chill Portions
Figure 3: GDD Budbreak/GDD General Growth
Figure 4: Extreme Temperatures > 34C/Extreme Temperatures > 15C

