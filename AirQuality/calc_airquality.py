# 40.329657	-3.765773 lat long example

import math
import time
import pandas as pd
import numpy as np




# Function to convert degrees to radians
def deg_to_rad(deg):
    """
    Converts degrees to radians.

    Parameters:
        deg (float): Angle in degrees.

    Returns:
        float: Angle in radians.
    """
    return deg * (math.pi / 180)


# Haversine function to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two geographic points.

    Parameters:
        lat1 (float): Latitude of the first point (in decimal degrees).
        lon1 (float): Longitude of the first point (in decimal degrees).
        lat2 (float): Latitude of the second point (in decimal degrees).
        lon2 (float): Longitude of the second point (in decimal degrees).

    Returns:
        float: Distance between the two points in kilometers.
    """
    R = 6371  # Radius of the Earth in kilometers
    phi1 = deg_to_rad(lat1)
    phi2 = deg_to_rad(lat2)
    delta_phi = deg_to_rad(lat2 - lat1)
    delta_lambda = deg_to_rad(lon2 - lon1)

    # Compute the Haversine formula
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


# Function to calculate weights based on distances to the nearest stations
def obtain_weights(my_lat, my_long, info_estaciones):
    """
    Computes the top 3 closest stations to a given point and returns their normalized inverse-distance weights.

    Parameters:
        my_lat (float): Latitude of the reference point.
        my_long (float): Longitude of the reference point.
        info_estaciones (pd.DataFrame): DataFrame containing station information with columns [station_code, longitude, latitude].

    Returns:
        dict: Dictionary with station codes as keys and their normalized weights as values.
    """
    distances = {}

    for i in range(info_estaciones.shape[0]):
        station = info_estaciones.iloc[i, 0]
        station_longitude = info_estaciones.iloc[i, 1]
        station_latitude = info_estaciones.iloc[i, 2]

        # Calculate Haversine distance
        distance = haversine(my_lat, my_long, station_latitude, station_longitude)
        distances[station] = distance

    # Get the top 3 closest stations
    sorted_keys = sorted(distances, key=distances.get)
    best_3 = sorted_keys[:3]

    # Apply inverse weighting (1 / distance)
    weights_inverse = {key: 1 / distances[key] for key in best_3}

    # Normalize the weights so they sum to 1
    total_weight = sum(weights_inverse.values())
    normalized_weights = {key: weight / total_weight for key, weight in weights_inverse.items()}

    return normalized_weights


# Function to interpolate data for a specific station
def interpolate_stations(station_code, aq_2021):
    """
    Interpolates air quality data for a given station.

    Parameters:
        station_code (int): Station code to filter data.
        aq_2021 (pd.DataFrame): Air quality data containing 'ESTACION' and 'H_Value'.

    Returns:
        pd.DataFrame: Interpolated and forward-filled data for the station.
    """
    my_data = aq_2021[aq_2021['ESTACION'] == station_code]
    my_data = my_data[~my_data.index.duplicated(keep='first')]

    # Resample and interpolate the H_Value column
    data_resampled = my_data.resample('min').asfreq()
    data_resampled['H_Value'] = data_resampled['H_Value'].interpolate(method='linear')
    data_resampled = data_resampled.ffill()

    return data_resampled


# Function to interpolate air quality data for a given timestamp
def interpolate_for_timeline(normalized_weight, timestamp, final_aq_data):
    """
    Interpolates air quality data for a given timestamp based on weighted station values.

    Parameters:
        normalized_weight (dict): Dictionary with station codes as keys and weights as values.
        timestamp (pd.Timestamp): Timestamp for which interpolation is required.
        final_aq_data (pd.DataFrame): DataFrame containing air quality data with 'ESTACION', 'H_Value', and timestamp index.

    Returns:
        float: Weighted sum of interpolated air quality values for the given timestamp.
    """
    timestamp = timestamp.tz_localize(None) if timestamp.tzinfo else timestamp
    final_aq_data.index = final_aq_data.index.tz_localize(None) if final_aq_data.index.tzinfo else final_aq_data.index

    temp_air_quality = 0

    for station, weight in normalized_weight.items():
        station_aq_data = final_aq_data[final_aq_data['ESTACION'] == station]

        # Filter rows matching the timestamp's date and hour
        filtered_data = station_aq_data[(station_aq_data.index.year == timestamp.year) &
                                        (station_aq_data.index.month == timestamp.month) &
                                        (station_aq_data.index.day == timestamp.day) &
                                        (station_aq_data.index.hour == timestamp.hour)]

        # Drop duplicates
        filtered_data = filtered_data.loc[~filtered_data.index.duplicated(keep='first')]

        if filtered_data.empty:
            continue

        # Interpolate the value at the given timestamp
        interpolated_value = (
            filtered_data['H_Value']
            .reindex(filtered_data.index.union([timestamp]))
            .sort_index()
            .interpolate(method='linear')
            .loc[timestamp]
        )

        temp_air_quality += interpolated_value * weight

    return temp_air_quality


# Function to load station information from a file
def get_info_estaciones(info_estaciones_folder_path):
    """
    Reads station information from an Excel file.

    Parameters:
        info_estaciones_folder_path (str): Path to the Excel file containing station data.

    Returns:
        pd.DataFrame: DataFrame with station information [station_code, longitude, latitude].
    """
    info_estaciones = pd.read_excel(info_estaciones_folder_path, engine='xlrd')
    info_estaciones = info_estaciones[['CODIGO', 'CODIGO_CORTO', 'LONGITUD', 'LATITUD']]
    info_estaciones = info_estaciones.drop(columns=['CODIGO'])

    return info_estaciones


# Function to compute air quality data for a timeline
def get_aq_from_timeline(timeline, final_aq_data, info_estaciones_folder_path):
    """
    Computes interpolated air quality data for a timeline based on station distances.

    Parameters:
        timeline (pd.DataFrame): Timeline DataFrame with 'latitude', 'longitude', and 'start_timestamp' columns.
        final_aq_data (pd.DataFrame): Air quality data DataFrame with 'ESTACION', 'H_Value', and timestamp index.
        info_estaciones_folder_path (str): Path to the Excel file containing station information.

    Returns:
        pd.DataFrame: Timeline DataFrame with interpolated air quality data.
    """
    info_estaciones = get_info_estaciones(info_estaciones_folder_path)

    # Compute weights for each row in the timeline
    timeline['normalized_weights'] = timeline.apply(lambda row: obtain_weights(row['latitude'], row['longitude'], info_estaciones=info_estaciones), axis=1)
    timeline = timeline.rename(columns={'start_timestamp': 'timestamp'})

    # Compute interpolated air quality values for each timestamp
    timeline['interpolated_air_quality'] = timeline.apply(lambda row: interpolate_for_timeline(row['normalized_weights'], row['timestamp'], final_aq_data), axis=1)

    return timeline


import preprocessing_pipeline as pre
import os



if __name__ == '__main__':

    timeline_path = 'C:/Users/Ricardo/Desktop/Exposome-Project/Files/takeout-Ricardo-timeline'
    aq_zips = "C:/Users/Ricardo/Desktop/Exposome-Project/Files/Exposome_ds_UC3M/Database/Calidad del aire"
    info_estaciones_folder_path = 'C:/Users/Ricardo/Desktop/Exposome-Project/AirQuality/informacion_estaciones_red_calidad_aire(1).xls'

    os.makedirs('./stored_results', exist_ok=True)

    # Getting the timeline data from our json files
    time1 = time.time()
    timeline = pre.full_timeline_data(timeline_path)

    

    print(timeline)

    print('\n----------------------------------------------\n')
    time2 = time.time()
    print(f"\nGetting timeline data took:   {time2-time1} [s]\n\n")
    timeline.to_csv('./stored_results/timeline_final.csv')

    

    print('\n----------------------------------------------\n')
    # getting the Air Quality data from our zip files (only the relevant ones)
    time1 = time.time()
    final_aq_data = pre.obtain_final_AQ(timeline, aq_zips)
    time2 = time.time()
    print(f"\nGetting final aq data took:   {time2-time1} [s]\n\n")
    final_aq_data.to_csv('./stored_results/final_aq_data.csv')



    print('\n----------------------------------------------\n')
    # Getting the air quality for our timeline calculating distances
    time1 = time.time()
    final_timeline = get_aq_from_timeline(timeline, final_aq_data, info_estaciones_folder_path)
    time2 = time.time()
    print(f"\nGetting the air quality of timeline data took:   {time2-time1} [s]\n\n")
    final_timeline.to_csv('./stored_results/final_timeline.csv')


    print('\n----------------------------------------------\n')
    # Getting the air quality time series
    air_quality_timeseries = final_timeline[['timestamp', 'interpolated_air_quality', 'latitude', 'longitude']]

    # Storing the time series

    air_quality_timeseries.to_csv('./stored_results/air_quality_timeseries.csv')



