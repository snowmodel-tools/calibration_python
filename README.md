# calibration_python
This is for python scripts associated with the process of automating calibration of SnowModel

This is separate from the acquisition of SNOTEL data (or other) for the purposes of calibration. Those scripts reside in the preprocess repositories

## Calibrate_SnowModel_full.ipynb
This script runs allows a user to select calibration parameters. SnowModel will be run in line mode for all parameter combinations at each SNOTEL location. Objective functions (RMSE, R^2, NSE, KGE, MBE) are computed between modeled and observed SWE. 

Requires:
**par_base.json** in directory with notebook 

Files saved out:
- .csv of all parameter combinations 
- .nc of calibration metrics for each station for each calibration run

## Calibration_postprocess.ipynb
This script evaluates the output of the calibration runs based on:

1. OFs averages over the domain 
- OFs are averaged across the stations for each calibration run
- The highest score for each of is used to identify the top model parameter combinations

2. OF rankings 
- each OF is ranked independently
- the rank scores are summed across each model run 
- The summed scores are then ranked and the top scores are used to identify the top model parameter combinations

Using both of the above method, SnowModel is run again in line mode and a top parameter combination is identified. 

A final full-domain calibration run is initiated using the top parameter combination.