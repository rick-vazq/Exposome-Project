# Exposome-Project
ML in Healthcare Exposome Project



- Public Repository for the Exposome Project


# Documentation

This Project contains multiple files and folders:

1. AirQuality Folder:
   - preprocessing_pipeline.py 
   - calc_airquality.py
   - informacion_estaciones_red_calidad_aire(1).xls

#### preprocessing_pipeline.py

This includes the following functions that are used to preprocess the Google Takeout Data and also the Air Quality Data.

1. Requires a folder path for the Google Takeout data that contains Yearly folders with `.json` files inside. Iterates through the files and aggregates all the timeline data into one Pandas DataFrame.
2. Requires a folder path for the Air Quality Data that has `.zip` files for the years in the format `AnioYEAR....zip` Gets only the years relevant in the google takeout data.

#### calc_airquality.py

This includes functions that are used to calculate the final air quality value for the minute to minute Google Takeout data.

- Air Quality data is only in hours and so we interpolate for the minutes fidelity to match the Google Takeout Data.
- Only the necessary timestamps that are included in the google takeout are interpolated to reduce computational complexity (and storage space). NOTE: For one month of Air quality data, one type reading type ($NO_{2}$) and for all stations, the interpolated data (from hours to minutes) was an almost 500MB .csv file.
- 
   


