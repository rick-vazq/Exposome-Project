import zipfile
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import time


def extract_coordinates(location, fallback=None):
    """Extract latitude and longitude from the location object with a fallback."""
    lat = location.get('latitudeE7')
    lon = location.get('longitudeE7')
    if lat is None or lon is None:  # Use fallback if necessary
        if fallback and 'waypoints' in fallback:
            lat = fallback['waypoints'][0].get('latE7')
            lon = fallback['waypoints'][0].get('lngE7')
    return (lat / 1e7 if lat else None, lon / 1e7 if lon else None)

def filter_by_bounding_box(df, lat_min = 40.3, lat_max = 40.6, lon_min=-3.8, lon_max=-3.6):
    """Filters rows by latitude and longitude bounding box."""
    return df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
              (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]

def process_json_files(folder_path):
    # Initialize a list to hold all rows of extracted data
    rows = []

    # Walk through the folder and subfolders to read JSON files
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.json'):  # Check for JSON files
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        for item in data.get('timelineObjects', []):
                            if 'activitySegment' in item:
                                activity = item['activitySegment']

                                # Extract coordinates (with fallback to waypointPath)
                                start_location = activity.get('startLocation', {})
                                end_location = activity.get('endLocation', {})
                                waypoints = activity.get('waypointPath', {})

                                start_lat, start_lon = extract_coordinates(start_location, waypoints)
                                end_lat, end_lon = extract_coordinates(end_location, waypoints)

                                # Extract timestamps
                                start_timestamp = activity.get('duration', {}).get('startTimestamp')
                                end_timestamp = activity.get('duration', {}).get('endTimestamp')

                                # Create a row with relevant data
                                row = {
                                    'file_name': file_name,
                                    'start_timestamp': start_timestamp,
                                    'end_timestamp': end_timestamp,
                                    'start_latitude': start_lat,
                                    'start_longitude': start_lon,
                                    'end_latitude': end_lat,
                                    'end_longitude': end_lon,
                                    'address': None  # No address in activitySegment
                                }
                                rows.append(row)

                            elif 'placeVisit' in item:
                                place = item['placeVisit']

                                # Extract visit location data
                                location = place['location']
                                latitude = location.get('latitudeE7') / 1e7
                                longitude = location.get('longitudeE7') / 1e7
                                address = location.get('address')

                                # Extract timestamps
                                start_timestamp = place.get('duration', {}).get('startTimestamp')
                                end_timestamp = place.get('duration', {}).get('endTimestamp')

                                # Create a row with relevant data
                                row = {
                                    'file_name': file_name,
                                    'start_timestamp': start_timestamp,
                                    'end_timestamp': end_timestamp,
                                    'start_latitude': latitude,
                                    'start_longitude': longitude,
                                    'end_latitude': None,
                                    'end_longitude': None,
                                    'address': address
                                }
                                rows.append(row)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Create a DataFrame from the extracted rows
    df = pd.DataFrame(rows)

    # Preprocessing:
    # Dropping unnecessary columns
    df = df.drop(columns=['end_latitude', 'end_longitude', 'end_timestamp'])

    # Renaming columns for general use
    df = df.rename(columns={'start_latitude': 'latitude', 'start_longitude': 'longitude'})

    # Dropping rows with NA in latitude or longitude
    df = df.dropna(subset=['latitude', 'longitude'])

    

    return df




"""

Preprocessing pipeline:

1. Only get the rows that are in Madrid
2. Convert the timestamp to be rounded to the nearest minute (to be compatible with the air quality data)
    2.1 Remove any duplicates in the timestamp (we round so we dont care about being in two places at once)

3. Make sure it is compatible with the air quality data


"""


def preprocessing_timeline(df:pd.DataFrame):


    """
    Input: 'df' type: pd.DataFrame

    output: 'df' type: pd.Dataframe

    Preprocessing pipeline:

    1. Only get the rows that are in Madrid
    2. Convert the timestamp to be rounded to the nearest minute (to be compatible with the air quality data)
        2.1 Remove any duplicates in the timestamp (we round so we dont care about being in two places at once)

    3. Make sure it is compatible with the air quality data
    
    """

# 1. Only getting the rows that are in Madrid
    df = filter_by_bounding_box(df)

# 2. Convert the time stamp to be rounded to the nearest minute and remove duplicates

    # Convert start_timestamp to datetime
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], format='mixed')


    # Round to the nearest minute
    df['start_timestamp'] = df['start_timestamp'].dt.round('min')

    # removing the duplicates rows that result from rounding to the nearest minute.
    df = df.drop_duplicates(subset=['start_timestamp'])

    # 3. Make sure it is compatible with the Air quality data
    # CHANGE THE NAME HERE
    df = df.rename(columns={'start_timestamp':'timestamp'})


    return df


def full_timeline_data(folder_path):

    df = process_json_files(folder_path)

    timeline = preprocessing_timeline(df)

    return timeline



def get_years(df:pd.DataFrame):

    years = df['timestamp'].dt.year.unique()


    return years

def get_months(years:list, df: pd.DataFrame):

    months = []
    for i in range(len(years)):

        temp_df = df[df['timestamp'].dt.year == years[i]]

        # Getting the unique months here:

        

        month = temp_df['timestamp'].dt.month.unique()
        months.append(month)

    return months







"""
----------------------------------------------------------------------------------------------------


NOTE: THIS BELOW IS ONLY FOR AIR QUALITY

Air Quality Dataset reading and preprocessing:


----------------------------------------------------------------------------------------------------

"""

import os
import zipfile
import pandas as pd
import re

def extract_air_quality_data(directory, years):
    """
    Extract air quality data from ZIP files for specified years.
    
    Args:
        directory (str): Path to the directory containing ZIP files.
        years (list): List of years (as integers or strings) to filter files.

    Returns:
        pd.DataFrame: Combined data from the specified years.
    """
    # Convert years to strings for comparison
    years = set(map(str, years))
    
    # Initialize an empty DataFrame
    combined_data = pd.DataFrame()

    # Loop through files in the directory
    for file in os.listdir(directory):
        # Check if the file is a ZIP file
        if file.endswith(".zip"):
            # Extract the year from the filename using regex
            match = re.search(r'Anio(\d{4})', file)
            if match:
                file_year = match.group(1)
                # Process only if the year is in the provided list
                if file_year in years:
                    zip_path = os.path.join(directory, file)
                    print(f"Processing file: {file}")
                    # Open the ZIP file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        # List all files in the ZIP archive
                        for inner_file in zip_ref.namelist():
                            # Check if the file is a CSV
                            if inner_file.endswith(".csv"):
                                # Read the CSV file directly from the ZIP
                                with zip_ref.open(inner_file) as csv_file:
                                    df = pd.read_csv(csv_file, sep=';')
                                    
                                    # Optionally, ensure timestamps are in datetime format
                                    if 'timestamp' in df.columns:
                                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                                    
                                    # Append to the combined DataFrame
                                    combined_data = pd.concat([combined_data, df], ignore_index=True)
    
    return combined_data

# Usage example
# directory_path = "C:/Users/Ricardo/Desktop/ML in health/ML-Healthcare-Project/Files/Exposome_ds_UC3M/Database/Calidad del aire"
# years_to_process = get_years(timeline)  # Specify the years to process
# result_df = extract_air_quality_data(directory_path, years_to_process)

def get_relevant_AQ(timeline:pd.DataFrame, result_df:pd.DataFrame):

    # timeline == the data from the time line google takeout
    # result_df == the data from the extract_air_quality_data



    years_to_process = get_years(timeline) # Specify the years to process
    months_to_process = get_months(years_to_process, timeline)

    df_collection = []

    for year in range(0,len(years_to_process)):
        temp_df = result_df[result_df['ANO'] == years_to_process[year]]
        # Filter rows for the months in the current year's list of months
        temp_df = temp_df[temp_df['MES'].isin(months_to_process[year])]

        df_collection.append(temp_df)

    # Combine all DataFrames in the list into one DataFrame
    combined_df = pd.concat(df_collection, ignore_index=True)

    return combined_df




def preprocess_aq_dat(df):



    """
    Given the read dataframe for the zip files and years we wanted to capture. We are preprocessing the dataframe

    - Only getting the rows with the codigo_tecnica for NO2 == 8

    returns preprocessed df
    
    """
    # location = 'Files/Exposome_ds_UC3M/Database/Calidad del aire/READ_Anio202112/'
    # location += filename

    # april_2021_air = pd.read_csv(location, sep=';')


    april_2021_air = df.copy()
    april_2021_air["codigo_tecnica"] = april_2021_air["PUNTO_MUESTREO"].str.split("_").str[2].astype(int)

    april_2021_air = april_2021_air[april_2021_air['codigo_tecnica'] == 8]

    columns_to_drop = ['V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07', 'V08', 'V09',
                'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24']
    
    april_2021_air.drop(columns=columns_to_drop, inplace=True)
    
    def generate_times(row):
        base_time = pd.Timestamp(f"{row['ANO']}-{row['MES']}-{row['DIA']} 00:00")
        # Create a list of times for H01 to H24
        hours = [base_time + pd.Timedelta(hours=i) for i in range(24)]
        return hours

    # Apply to the DataFrame
    april_2021_air['Times'] = april_2021_air.apply(generate_times, axis=1)
    

    # Assuming your DataFrame is named 'df' and you have a list of H column names:
    h_columns = ['H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24']


    # Assuming you want to include 'PROVINCIA', 'MUNICIPIO', 'ESTACION', 'MAGNITUD', 'PUNTO_MUESTREO', 'ANO', 'MES', 'DIA', and 'codigo_tecnica'
    columns_to_include = ['PROVINCIA', 'MUNICIPIO', 'ESTACION', 'MAGNITUD', 'PUNTO_MUESTREO', 'ANO', 'MES', 'DIA', 'codigo_tecnica'] + h_columns

    def extract_data(row):
        other_data = row[columns_to_include[:-len(h_columns)]]
        h_values = row[h_columns]
        timestamps = row['Times']
        return [(dict(zip(columns_to_include[:-len(h_columns)], other_data)), h, t) for h, t in zip(h_values, timestamps)]

    df_long = april_2021_air.apply(extract_data, axis=1).explode()
    df_long = pd.DataFrame(df_long.tolist(), columns=['Other_Data', 'H_Value', 'Timestamp'])

    other_data_df = pd.json_normalize(df_long['Other_Data'])

    # Concatenate with the rest of the columns from df_long (except 'Other_Data')
    df_long_expanded = pd.concat([df_long.drop(columns=['Other_Data']), other_data_df], axis=1)

    df_long_expanded.set_index(df_long_expanded['Timestamp'], inplace=True)

    df_long_expanded = df_long_expanded.drop(columns=['Timestamp', 'ANO', 'MES', 'DIA', 'PUNTO_MUESTREO'])

    
    # df_long_expanded

    return df_long_expanded






def obtain_final_AQ(timeline, directory_path):


    years = get_years(timeline)
    months =get_months(years,timeline)

    print('Extracting the Air quality data for the specific number of years in timeline')
    t0 = time.time()
    years_to_process = get_years(timeline)  # Specify the years to process
    result_df = extract_air_quality_data(directory_path, years_to_process)
    t1 = time.time()
    print(f"Took {t1-t0}")


    print("Getting only the relevant Air Quality data for the months and features")
    t0 = time.time()

    combined_df = get_relevant_AQ(timeline, result_df)
    t1 = time.time()
    print(f"Took {t1-t0}")


    print("Preprocessing the Air Quality Data to get the final aq dataset")
    t0 = time.time()
    final_aq_data = preprocess_aq_dat(combined_df)
    t1 = time.time()
    print(f"Took {t1-t0}")

    return final_aq_data



if __name__ == '__main__':
    # print("main")
    timeline_path = 'C:/Users/Ricardo/Desktop/Exposome-Project/Files/takeout-Ricardo-timeline'


    timeline = full_timeline_data(timeline_path)


    aq_zips = "C:/Users/Ricardo/Desktop/ML in health/ML-Healthcare-Project/Files/Exposome_ds_UC3M/Database/Calidad del aire"

    final_aq_data = obtain_final_AQ(timeline, aq_zips)

    print(timeline.head())

    print(final_aq_data)



    
