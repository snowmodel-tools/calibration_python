# calibration_python
This is for python scripts associated with the process of automating calibration of SnowModel

This is separate from the acquisition of SNOTEL data (or other) for the purposes of calibration. Those scripts reside in the preprocess repositories

## Calibrate_SnowModel_full.ipynb
This script runs allows a user to select calibration parameters. SnowModel will be run in line mode for all parameter combinations at each SNOTEL location. Calibration metrics (RMSE, R^2, NSE, KGE, MBE) are computed between modeled and observed SWE. 
Files saved out:
- .csv of all parameter combinations 
- .nc of calibration metrics for each station for each clibration run

## Calibration_postprocess.ipynb

## Calibration_plots.ipynb
