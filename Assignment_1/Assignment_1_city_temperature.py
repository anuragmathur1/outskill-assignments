#!/usr/bin/env python
import csv
import numpy as np
from datetime import datetime
import pandas as pd

def get_city_list(csv_file_name):
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        column_data = next(reader)
        colmap = dict(zip(column_data, range(len(column_data))))
        # print(colmap)
        colmap_list = [None] * (max(colmap.values()) + 1)
        ## code to change colmap to a list with city_names as values in the list indexes by the column number
        ## e.g. colmap[0] = 'Date', colmap[1] = 'London', colmap[2] = 'Tokyo', colmap[3] = 'New York'.
        for k, v in colmap.items():
            # print(k)
            # print(v)
            colmap_list[v] = k        
        # print(colmap_list)
    return colmap, colmap_list[1:]

def max_temp(city,csv_file_name):
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data_array = np.array(data)
        # print(data_array)
        max_temp = float(max(data_array[:, colmap[city]][1:]))
    return {city: max_temp}

## Function to identify the monthly average temperature for each of the city.
## Return dict.

## Note : Instead of returning a dict, 
##   The function is returning city, month and avg temp to look complete

def monthly_avg_temp(df, city):
    df['Month'] = df['Date'].dt.month
    # city_cols = [col for col in df.columns if col not in ['Date', 'Month']]
    # print(city_cols)
    max_per_month = df.groupby('Month')[city].max()
    for month, value in max_per_month.items():
        # print(f"{month}\t{value}")
        return city, month, value
    
def sliding_window(df, city, colmap_list):
# For each city column (exclude 'Date' and 'Month')
    results = {}
    # city_columns = [col for col in df.columns if col not in ['Date', 'Month']]
    # print(colmap_list)
    results[city] = {}
    all_months = sorted(df['Month'].unique())
    # for city in city_columns:
    #     results[city] = {}
    for month in all_months:
        city_month_data = df[df['Month'] == month][city].reset_index(drop=True)
        if city_month_data.empty or len(city_month_data) < 5:
            results[city][month] = 0
            continue
        monthly_avg = city_month_data.mean()
        windows = np.lib.stride_tricks.sliding_window_view(city_month_data.values, 5)
        count = np.sum(np.all(windows > monthly_avg, axis=1))
        results[city][month] = int(count)

        # Print results
        # print(results)
        # print(type(results))
    # Print only once per city, after all months are processed
    for city, months in results.items():
        print(f"{city}:")
        for month in sorted(months):
            print(f"  Month {month}: {months[month]} stretch(es)")



if __name__ == "__main__":
    ##Function to identify the day when max temperature recorded for each city. 
    ## Return as a dict (city - temp pair)
    # filename = '../Assignments/city_temperature.csv'
    csv_file_name = 'city_temperature.csv'
    df = pd.read_csv(csv_file_name, parse_dates=['Date'], dayfirst=True)
    colmap, colmap_list = get_city_list(csv_file_name)
    # print(colmap_list)
    max_temp_list = [max_temp(city, csv_file_name) for city in colmap_list]

    ## Assignment 1.1
    print("=================== Assignemnt 1.1 ===================")
    for item in max_temp_list:
        for k, v in item.items():
            matching_dates = df.loc[df[k] == v, 'Date']
            for date in matching_dates:
                print(item, date.strftime('%Y-%m-%d'))

    print("=================== Assignemnt 1.2 ===================")
    ## Assignment 1.2
    for city in colmap_list:
        print(monthly_avg_temp(df, city))
        
    print("=================== Assignemnt 1.3 ===================")
    ## Assignment 1.3
    for city in colmap_list:
        sliding_window(df, city, colmap_list)



