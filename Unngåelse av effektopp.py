#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:54:47 2023

@author: njolsen
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from itertools import combinations

raw_df = pd.read_csv("VVB data.csv", skiprows=3, sep=';') # VVB-data
raw_tot_df = pd.read_csv("Total data.csv", skiprows=3, sep=';') # Totalt forbruk på Gardermoen
# New row data
v = str(0)
# Identify the indices where you want to add the new row
index_before = 2401  # Replace with the actual index value
index_after = 2402   # Replace with the actual index value
new_row_data = {'Tid (Time)': '26.03.2023 02:00 - 03:00', 'Energi': v, 'CO₂': v, 'Kostnad': v, 'Peak High': v,'Snittlast': v, 'Utetemperatur': v, 'Oslo NO1 (NOK) øre/kWh': v}
new_tot_data = {'Tid (Time)': '26.03.2023 02:00 - 03:00', 'El Energi': raw_tot_df.at[index_before, 'El Energi'], 'El Kostnad': raw_tot_df.at[index_before, 'El Kostnad'], 'El CO₂': v}

# Add the new row between the specified indices
raw_df = pd.concat([raw_df.loc[:index_before], pd.DataFrame([new_row_data]), raw_df.loc[index_before + 1:]]).reset_index(drop=True)
raw_tot_df = pd.concat([raw_tot_df.loc[:index_before], pd.DataFrame([new_tot_data]), raw_tot_df.loc[index_before + 1:]]).reset_index(drop=True)


# Fjerning av unødvendige kolonner
df = raw_df.drop('CO₂', axis=1)   # Måling av CO₂
df = df.drop('Peak High', axis=1) # Verdi for "Peak High" = "Energi" i alle rader på timesbasis
df = df.drop('Snittlast', axis=1) # Verdi for "Snittlast" = "Energi" i alle rader på timesbasis

# Fjerning av NaN-verdier
df = df.dropna()

# Formatkonvertering av tidskolonne
df['Tid (Time)'] = df['Tid (Time)'].str.replace(' - \d+:\d+', '', regex=True)

# Fjerning av unødvendig kolonne
tot_df = raw_tot_df.drop("El CO₂", axis=1)

# Fjerning av NaN-verdier
tot_df = tot_df.dropna()

# Formatkonvertering av tidskolonne
tot_df['Tid (Time)'] = tot_df['Tid (Time)'].str.replace(' - \d+:\d+', '', regex=True)

#%%
def assign_time_columns(df, date_column_name = "Tid (Time)"):
    # Convert the date column to a datetime object
    df[date_column_name] = pd.to_datetime(df[date_column_name], format='%d.%m.%Y %H:%M')

    # Assign the day of the week to each date and create a new column
    df['DayOfWeek'] = df[date_column_name].dt.strftime('%A')
    df['WeekNumber'] = df[date_column_name].dt.strftime('%U')
    df['Month'] = df[date_column_name].dt.month
    
    # Extract the hour part and create a new column for it
    df['Hour'] = df['Tid (Time)'].dt.hour

    return df

def merge_missing_columns(df1, df2, date_time_column_name):
    # Iterate through columns in df2 that are not in df1
    for col_name in df2.columns.difference(df1.columns):
        # Merge the matching columns based on the date and time column
        df1 = df1.merge(df2[['Tid (Time)', col_name]], on=date_time_column_name, how='left')

        # Rename the merged column to match the column name from df2
        df1.rename(columns={col_name: col_name}, inplace=True)

    return df1

def fix_numbers(df):

    df['El Energi'] = pd.to_numeric(df['El Energi'].str.replace(',', '.'), errors='coerce')
    df['El Kostnad'] = pd.to_numeric(df['El Kostnad'].str.replace(',', '.'), errors='coerce')
    df['Energi'] = pd.to_numeric(df['Energi'].str.replace(',', '.'), errors='coerce')
    df['Oslo NO1 (NOK) øre/kWh'] = pd.to_numeric(
        df['Oslo NO1 (NOK) øre/kWh'].str.replace(',', '.'), errors='coerce')
    df['Kostnad'] = pd.to_numeric(df['Kostnad'].str.replace(',', '.'), errors='coerce')
    df['Utetemperatur'] = pd.to_numeric(df['Utetemperatur'].str.replace(',', '.'), errors='coerce')

    return df

def add_day_count(df, date_column_name):
    # Ensure the date column is in datetime format
    df[date_column_name] = pd.to_datetime(df[date_column_name])

    # Initialize day count and previous date variables
    day_count = 1
    prev_date = None

    # Iterate through rows and update day count when a new date is encountered
    for index, row in df.iterrows():
        current_date = row[date_column_name]

        if prev_date is None:
            prev_date = current_date
        elif current_date.date() != prev_date.date():
            day_count += 1
            prev_date = current_date

        df.at[index, 'Day_count'] = day_count

    return df


def distribute_energy(df):
    # Initialize a new column "Fordelt bereder" with 0 values
    df["Fordelt bereder"] = 0

    data = df.copy()

    # Get the unique values in the "Day_count" column
    day_counts = data["Day_count"].unique()
    
    df["Peak_count"] = 0
    peak_count = 1

    # Removing some rows for visualization
    data = data.drop('El Kostnad', axis=1)
    data = data.drop('DayOfWeek', axis=1)
    data = data.drop('Oslo NO1 (NOK) øre/kWh', axis=1)
    data = data.drop('WeekNumber', axis=1)
    data = data.drop('Tid (Time)', axis=1)
    data = data.drop('Kostnad', axis=1)
    data = data.drop('Utetemperatur', axis=1)

    for current_day in day_counts[:-1]:  # Exclude the last day
        next_day = current_day + 1

        # Filter rows for the current_day and next_day
        selected_rows = data[(data["Day_count"] == current_day) | (
            data["Day_count"] == next_day)]

        # Find the rows with the highest "El Energi" values for both days
        max_el_energi_current_day = selected_rows[selected_rows["Day_count"]
                                                  == current_day]["El Energi"].max()
        max_el_energi_next_day = selected_rows[selected_rows["Day_count"]
                                               == next_day]["El Energi"].max()

        # Find the index of the row with max_el_energi_current_day
        max_el_energi_current_day_index = selected_rows[selected_rows["El Energi"]
                                                        == max_el_energi_current_day].index[0]
        #print(max_el_energi_current_day_index)

        # Find the index of the row with max_el_energi_next_day
        max_el_energi_next_day_index = selected_rows[selected_rows["El Energi"]
                                                     == max_el_energi_next_day].index[0]
        #print(max_el_energi_next_day_index)

        # Calculate the total sum of "Energi" for the next_day
        total_energy_next_day = data[(
            data["Day_count"] == next_day)]["Energi"].sum()

        # Slice the DataFrame to include rows between max_el_energi_current_day and max_el_energi_next_day
        slice_data = selected_rows.loc[max_el_energi_current_day_index: max_el_energi_next_day_index-1]

        # Sort the sliced DataFrame by "El Energi" in ascending order
        slice_data = slice_data.sort_values(by="El Energi")
        
        for index, row in slice_data.iterrows():
            df.at[index, "Peak_count"] = peak_count
            
        peak_count += 1

        for index, row in slice_data.iterrows():
            if total_energy_next_day <= 0:
                break

            available_capacity = 60 - row["Fordelt bereder"]
            distribution = min(total_energy_next_day, available_capacity)
            df.at[index, "Fordelt bereder"] += distribution
            
            total_energy_next_day -= distribution
        
    return df

def new_curve(data, column_to_subtract="Energi", column_to_subtract_from="El Energi", plus="Fordelt bereder"):
    # Check if the specified columns exist in the DataFrame
    if column_to_subtract not in data.columns or column_to_subtract_from not in data.columns:
        raise ValueError(
            "One or both specified columns do not exist in the DataFrame.")

    # Perform the subtraction and store the result in a new column
    data["Nytt totalt forbruk"] = data[column_to_subtract_from] - data[column_to_subtract] + data[plus]

    return data

def on_off_columns(df):
    df['Nr on'] = np.select([df['Energi'] < 15, (df['Energi'] >= 15) & (df['Energi'] < 30), (df['Energi'] >= 30) 
                             & (df['Energi'] < 45), df['Energi'] >= 45],
                            [1, 2, 3, 4],
                            default=0)

    df['Time ON'] = df.apply(lambda row: row['Energi'] / 15 if row['Nr on'] == 1 else
                                         (row['Energi'] - 15) / 15 if row['Nr on'] == 2 else
                                         (row['Energi'] - 30) / 15 if row['Nr on'] == 3 else
                                         (row['Energi'] - 45) / 15 if row['Nr on'] == 4 else 0, axis=1)

    df['Time OFF'] = df.apply(lambda row: 1 - (row['Energi'] / 15) if row['Nr on'] == 1 else
                                           1 - ((row['Energi'] - 15) / 15) if row['Nr on'] == 2 else
                                           1 - ((row['Energi'] - 30) / 15) if row['Nr on'] == 3 else
                                           1 - ((row['Energi'] - 45) / 15) if row['Nr on'] == 4 else 0, axis=1)
    return df

def mass_calculation(df):
    m_dot = (15/(4.186*45))*3600
    
    df["Mass"] = df.apply(lambda row: row['Time ON'] * m_dot if row['Nr on'] == 1 else
                                         (1 + row['Time ON']) * m_dot if row['Nr on'] == 2 else
                                         (2 + row['Time ON']) * m_dot if row['Nr on'] == 3 else
                                         (3 + row['Time ON']) * m_dot if row['Nr on'] == 4 else 0, axis=1)
    return df


#%% 
def create_timescale_df(df, input1, input2):
    # Create a new DataFrame with unique input1 values as columns
    df_timescale = pd.DataFrame()

    # Initialize the "Average" column with zeros
    df_average = pd.DataFrame()

    # Iterate through unique input1 values
    for value in df[input1].unique():
        # Calculate the average value of input2 for each distinct "Hour"-value with the same input1-value
        avg_values = df[df[input1] == value].groupby('Hour')[input2].mean()

        # Add each individual average value to the "Average" column
        df_average[value] = avg_values

        # Rank the average values in descending order
        ranked_hours = avg_values.sort_values(ascending=True).index

        # Add corresponding "Hour" values to df_timescale as columns
        df_timescale[value] = ranked_hours

    # Add a "Ranking" column
    df_timescale['Ranking'] = range(1, len(df_timescale) + 1)

    return df_timescale, df_average
def add_priority_column(df, df_timescale, interval_column):
    """
    Add a "priority" column to df_result based on values in df_timescale.

    Parameters:
    - df: DataFrame, input DataFrame containing columns "Hour", interval_column, and other columns.
    - df_timescale: DataFrame, DataFrame with "Hour" values in the column names.
    - interval_column: str, the column in df used to identify which column in df_timescale is being processed.

    Returns:
    - df_result: DataFrame, a new DataFrame based on df with added "priority" column.
    """
    # Create a copy of df, dropping specified columns
    df_result = df.drop(['Utetemperatur', 'Fordelt bereder', 'Peak_count', 'Nytt totalt forbruk'], axis=1).copy()

    #unique_values = df[interval_column].unique()
    #print(unique_values)
    
    for col in df_timescale:
        # Filter df_result for rows where interval_column is equal to col
        sliced_df = df_result[df_result[interval_column] == col].copy()
    
        # Merge sliced_df with df_timescale based on 'Hour'
        merged_df = pd.merge(sliced_df, df_timescale, left_on='Hour', right_on=col, how='left')
    
        # Assign "Ranking" values to the "Prior" column
        df_result.loc[df_result[interval_column] == col, 'Priority'] = merged_df['Ranking'].values

    return df_result  

def find_top60_energy_values(df):
    # Assuming df is your DataFrame with columns 'Month', 'Day_count', 'Hour', and 'El Energi'
    
    # Group by 'Month'
    grouped_df = df.groupby('Month')
    sum_list = []
    # Define a function to get the top 60 values for each group
    def top60(group):
        top_values = group.nlargest(120, 'El Energi')
        lowest_value = top_values['El Energi'].min()
        highest_value = top_values['El Energi'].max()
        sum_list.append(highest_value)
        # Find the row with the highest 'Nytt totalt forbruk' value and get the corresponding 'Tid (Time)' value
        highest_time = top_values[top_values['El Energi'] == highest_value]['Tid (Time)'].iloc[0]

        print(f"Month: {group['Month'].iloc[0]}, Lowest of Top 60: {lowest_value}, Highest of Top 60: {highest_value}, Time of Highest Value: {highest_time}")
        return top_values
    # Apply the function to each group and concatenate the results
    result_df = grouped_df.apply(top60).reset_index(drop=True)
    print(sum(sum_list))
    return result_df

def new_top60(df):
    # Assuming df is your DataFrame with columns 'Month', 'Day_count', 'Hour', and 'El Energi'
    
    # Group by 'Month'
    grouped_df = df.groupby('Month')
    sum_list = []
    
    # Define a function to get the top 60 values for each group
    def top60(group):
        top_values = group.nlargest(120, 'Nytt totalt forbruk')
        lowest_value = top_values['Nytt totalt forbruk'].min()
        highest_value = top_values['Nytt totalt forbruk'].max()
        sum_list.append(highest_value)
        
        # Find the row with the highest 'Nytt totalt forbruk' value and get the corresponding 'Tid (Time)' value
        highest_time = top_values[top_values['Nytt totalt forbruk'] == highest_value]['Tid (Time)'].iloc[0]

        print(f"Month: {group['Month'].iloc[0]}, Lowest of Top 60: {lowest_value}, Highest of Top 60: {highest_value}, Time of Highest Value: {highest_time}")
        return top_values
    # Apply the function to each group and concatenate the results
    result_df = grouped_df.apply(top60).reset_index(drop=True)
    print(sum(sum_list))
    return result_df

def find_highest_values(df):
    # Assuming df is your DataFrame with columns 'Month', 'El Energi', and 'Nytt totalt forbruk'
    
    # Group by 'Month'
    grouped_df = df.groupby('Month')

    # Define a function to get the highest value for each column in each group and apply calculations
    def highest_values_with_calc(group):
        highest_totalt_forbruk = group['Totalt forbruk'].max()
        highest_nytt_totalt_forbruk = group['Nytt totalt forbruk'].max()

        # Determine the multiplier based on the month
        multiplier = 86 if group.name in [1, 2, 3, 10, 11, 12] else 36

        # Apply the multiplier
        calc_totalt_forbruk = highest_totalt_forbruk * multiplier
        calc_nytt_totalt_forbruk = highest_nytt_totalt_forbruk * multiplier

        return pd.Series({
            'Highest Totalt forbruk': highest_totalt_forbruk, 
            'Highest Nytt totalt forbruk': highest_nytt_totalt_forbruk,
            'KG': calc_totalt_forbruk,
            'KN': calc_nytt_totalt_forbruk
        })

    # Apply the function to each group
    result_df = grouped_df.apply(highest_values_with_calc)

    return result_df

def find_matching_rows(df1, df2):
    # Ensure both dataframes have the same columns
    if list(df1.columns) != list(df2.columns):
        raise ValueError("Dataframes must have the same columns")

    # Find matching rows and get their index values from df1
    matching_indices = df1[df1.isin(df2.to_dict(orient='list')).all(axis=1)].index.tolist()

    return matching_indices

def add_running_column(df, index_list):
    # Add a new column named 'No running' with default value 'Yes'
    df['No running'] = 'Yes'
    
    # Set 'No' for rows with indices in the provided list
    df.loc[index_list, 'No running'] = 'No'
    
    return df

def find_consecutive_lists(input_list):
    consecutive_lists = []
    current_list = []

    for i, value in enumerate(input_list):
        if i > 0 and value != input_list[i - 1] + 1:
            consecutive_lists.append(current_list)
            current_list = []

        current_list.append(value)

    if current_list:
        consecutive_lists.append(current_list)

    return consecutive_lists

def calculate_sum_for_lists(index_lists, df):
    results = []

    for indices in index_lists:
        result = 0
        for i in indices:
            result += df.at[i, "Mass"]
        results.append(result)

    # Create a new DataFrame from the results
    result_df = pd.DataFrame({'index_list':index_lists, 'Sum': results})

    return result_df

def calculate_and_print(df):
    # Sort the DataFrame by 'Month' for consecutive months
    #df_sorted = df.sort_values(by='Month')

    # Initialize a dictionary to store the 'rest' value for each month
    rest_dict = {}
    
    # Initialize a list to store the result rows
    result_rows = []

    for index, row in df.iterrows():
        # Check if the current 'Month' value is the first distinct one
        if row['Month'] not in rest_dict:
            # Calculate 'p' for the first distinct 'Month'
            p = row['Energi'] * 5

            # Calculate 'rest' for the first distinct 'Month'
            rest = row['El Energi'] - p
            rest_dict[row['Month']] = rest
            
        # Check the condition and add the row to the result list
        if row['El Energi'] > rest_dict[row['Month']]:
            result_rows.append(row.to_dict())

    # Convert the result list to a DataFrame
    result_df = pd.DataFrame(result_rows)

    return result_df

def add_running_column(df, index_lists):
    # Add a new column 'Running' with default value 1
    df['Running'] = 1

    # Set 'Running' to 0 for rows with indices in the provided list of lists
    for indices in index_lists:
        df.loc[indices, 'Running'] = 0

    return df

def process_data(df, input_column):
    # Ensure the DataFrame is sorted by 'Day_count' and 'input_column' in descending order
    df_sorted = df.sort_values(by=['Day_count', input_column], ascending=[True, False])

    # Initialize a counter for the number of rows looped through for each 'Day_count'
    # Loop through every row of every distinct 'Day_count' value
    for day_count, group_df in df_sorted.groupby('Day_count'):
        # Initialize a counter for the number of rows looped through for each 'Day_count'
        rows_looped = 0
        mass_sum = 0 
        #print(group_df["El Energi"])
        # Start with the highest values of 'input_column' for each 'Day_count'
        for index, row in group_df.iterrows():
            # Sum together the 'Mass' value of the rows in descending 'input_column' order
            mass_sum += row["Mass"]
            #print(rows_looped)
            #print(mass_sum)
            if mass_sum < 3000 and rows_looped < 20: 
                df.at[index, 'Running'] = 0
                rows_looped += 1
            else: 
                break
                    
    return df

def analyze_zero_running_df(df):
    # Check if required columns exist
    if 'Day_count' not in df.columns or 'Running' not in df.columns:
        raise ValueError("DataFrame must contain 'Day_count' and 'Running' columns")

    # Group by 'Day_count' and count the zeros in 'Running' for each group
    zero_counts = df[df['Running'] == 0].groupby('Day_count').size()

    # Calculate the average and most common counts
    average_zero_count = zero_counts.mean()
    most_common_count = zero_counts.mode()[0]

    # Calculate the lowest count that occurs at least 3 times
    counts_occurrence = zero_counts.value_counts()
    lowest_recurring_count = counts_occurrence[counts_occurrence >= 3].index.min()

    # Creating a DataFrame to return the results
    result_df = pd.DataFrame({
        'Average Count': [average_zero_count],
        'Most Common Count': [most_common_count],
        'Lowest Recurring Count (min 3 occurrences)': [lowest_recurring_count]
    })

    return result_df

def analyze_zero_running_expanded(df):
    # Check if required columns exist
    if 'Day_count' not in df.columns or 'Running' not in df.columns or 'VVB tradisjonelt forbruk' not in df.columns:
        raise ValueError("DataFrame must contain 'Day_count', 'Running', and 'VVB tradisjonelt forbruk' columns")

    # Group by 'Day_count' and count the zeros in 'Running' for each group
    zero_counts = df[df['Running'] == 0].groupby('Day_count').size()

    # Get the value counts of each unique count
    count_frequencies = zero_counts.value_counts()

    # Create a DataFrame with columns for each distinct count
    max_count = zero_counts.max()
    result_df = pd.DataFrame(index=['Count', 'Percentage (%)'], columns=range(max_count + 1)).fillna(0)

    # Populate the DataFrame with counts and calculate percentages
    total_days = zero_counts.count()
    for count, freq in count_frequencies.items():
        result_df.at['Count', count] = freq
        result_df.at['Percentage (%)', count] = (freq / total_days * 100).round(2)

    # Calculate the average 'VVB tradisjonelt forbruk' for days with zeros in 'Running'
    daily_avg_vvb = df[df['Running'] == 0].groupby('Day_count')['VVB tradisjonelt forbruk'].mean()

    # Get the average of averages for each distinct count value
    avg_vvb_for_each_count = daily_avg_vvb.groupby(zero_counts).mean()
    
    # Add a row for the average 'VVB tradisjonelt forbruk' for each count
    result_df.loc['Avg VVB for Zero Count'] = avg_vvb_for_each_count.reindex(range(max_count + 1))
    
    # Add a new row with the average values multiplied by 3.325
    result_df.loc['Avg VVB for Zero Count (Adjusted)'] = result_df.loc['Avg VVB for Zero Count'] * 3.325

    return result_df






#%%
def calculate_heated_mass(df):
    # Step 1: Add a new columns "Heated mass", "New on" and heat and time columns with default value 0
    df['Heated mass'] = 0
    df["New on"] = 0
    
    df["Heat 1"] = 0
    df["Heat 2"] = 0
    df["Heat 3"] = 0
    #df["Heat 3.4"] = 0
    df["Heat 4"] = 0
    
    df["Time 1"] = 0
    df["Time 2"] = 0
    df["Time 3"] = 0
    df["Time 4"] = 0
    
    #df.at[2893, "Heated mass"] = df.at[2893, "Mass"]
    #df.at[2725, "Heated mass"] = df.at[2725, "Mass"]


    # Step 2: Define constant max_heat and total mass caluclation for the first row
    max_heat = 286
    
    df.at[6213, 'Mass'] = 0
    
    df.at[0, 'Total mass'] = 4000 - df.at[0, 'Mass'] + df.at[0, 'Heated mass']
    
    # Step 3: Iterate through the DataFrame
    for i in range(1, len(df)):
        
        if df.at[i, 'Running'] == 0:
            df.at[i, 'Total mass'] = df.at[i-1, 'Total mass'] - df.at[i, 'Mass']
            
        elif df.at[i, 'Running'] == 1:
            missing_heat = 4000 - df.at[i-1, 'Total mass']
            #if missing_heat < 0:
                #missing_heat = df.at[i, 'Mass']
                
            if missing_heat <= 1000:
                if missing_heat > max_heat:
                    df.at[i, 'Heat 1'] = max_heat
                elif missing_heat < max_heat:
                    df.at[i, 'Heat 1'] = missing_heat
            elif missing_heat <= 2000:
                df.at[i, 'Heat 1'] = max_heat
                if (missing_heat - 1000) > max_heat:
                    df.at[i, 'Heat 2'] = max_heat 
                else: 
                    df.at[i, 'Heat 2'] = missing_heat - 1000
            elif missing_heat <= 3000:
                df.at[i, 'Heat 1'] = max_heat
                df.at[i, 'Heat 2'] = max_heat
                if (missing_heat - 2000) > max_heat:
                    df.at[i, 'Heat 3'] = max_heat
                else: 
                    df.at[i, 'Heat 3'] = missing_heat - 2000
            elif missing_heat <= 4000:
                df.at[i, 'Heat 1'] = max_heat
                df.at[i, 'Heat 2'] = max_heat
                df.at[i, 'Heat 3'] = max_heat
                if (missing_heat - 2000) > max_heat:
                    df.at[i, 'Heat 4'] = max_heat
                else: 
                    df.at[i, 'Heat 4'] = missing_heat - 3000
                    
            df.at[i, 'Heated mass'] = df.at[i, 'Heat 1'] + df.at[i, 'Heat 2'] + df.at[i, 'Heat 3'] + df.at[i, 'Heat 4']
            if df.at[i, 'Mass'] > df.at[i, 'Heated mass'] and df.at[i, 'Heated mass'] != 0:
                diff = df.at[i, 'Mass'] - df.at[i, 'Heated mass']
                if diff < (286 - df.at[i, 'Heated mass']):
                    df.at[i, 'Heated mass'] += diff
                    df.at[i, 'Heat 1'] += diff
                elif diff > (286 - df.at[i, 'Heated mass']):
                    if df.at[i, 'Heat 1'] == 286:
                        df.at[i, 'Heat 2'] += diff
                        df.at[i, 'Heated mass'] += diff
                    elif df.at[i, 'Heat 1'] < 286:
                        diff_1_remain = 286 - df.at[i, 'Heat 1']
                        df.at[i, 'Heat 1'] += diff_1_remain
                        df.at[i, 'Heated mass'] += diff_1_remain
                        diff -= diff_1_remain
                        df.at[i, 'Heat 2'] += diff
                        df.at[i, 'Heated mass'] += diff
                    
                            
                    else: 
                        print(diff)
                        
                        #print(df.at[i, 'Heat 2'])
                #elif diff > (286 - df.at[i, 'Heat 1']) and (286 - df.at[i, 'Heat 1']) == 0:
                 #   df.at[i, 'Heated mass'] += diff
                  #  df.at[i, 'Heat 2'] += diff
                #else: 
                    #print(diff)
            
            #df.at[i, 'Heated mass'] = df.at[i, 'Heat 1'] + df.at[i, 'Heat 2'] + df.at[i, 'Heat 3'] + df.at[i, 'Heat 4']
            df.at[i, 'Total mass'] = max(0, min(4000, df.at[i-1, 'Total mass'] + df.at[i, 'Heated mass'] - df.at[i, 'Mass']))
                
                
                    
        #df.at[i, 'Total mass'] = max(0, min(4000, df.at[i-1, 'Total mass'] - df.at[i, 'Mass'] + df.at[i, 'Heated mass']))
        # Step 6: Adding heated mass if total mass gets below 1000
        
    return df

def time_calc(df):
    df['Time Total'] = 0
    for i in range(len(df)):
        df.at[i, 'Time 1'] = df.at[i, 'Heat 1'] / 286
        df.at[i, 'Time 2'] = df.at[i, 'Heat 2'] / 286
        df.at[i, 'Time 3'] = df.at[i, 'Heat 3'] / 286
        df.at[i, 'Time 4'] = df.at[i, 'Heat 4'] / 286
        df.at[i, 'Time Total'] = df.at[i, 'Time 1'] + df.at[i, 'Time 2'] + df.at[i, 'Time 3'] + df.at[i, 'Time 4'] 
    return df

def new_energy(df):
    df['New energy'] = 0
    df['New cost'] = 0
    for i in range(len(df)):
        df.at[i, 'New energy'] = df.at[i, 'Time Total']*15
        df.at[i, 'New cost'] = df.at[i, 'New energy'] * (df.at[i, 'Oslo NO1 (NOK) øre/kWh']/100)
    return df
    
def calculate_vvb(df, input1, input2='VVB tradisjonelt forbruk', factor=3.325):
    # Ensure the input columns exist in the DataFrame
    if input1 not in df or input2 not in df:
        raise ValueError("Input columns not found in the DataFrame")

    # Calculating 'Total VVB tradisjonell' and 'Total VVB flyttet'
    df['Total VVB tradisjonell'] = df[input2] * factor
    df['Total VVB flyttet'] = df[input1] * factor

    # Calculating 'Nytt totalt forbruk'
    df['Nytt totalt forbruk'] = df['Totalt forbruk'] - df['Total VVB tradisjonell'] + df['Total VVB flyttet']
    
    # Calculating 'Differanse'
    df['Differanse'] = df['Nytt totalt forbruk'] - df['Totalt forbruk']


    return df

def analyze_column(df, column_name):
    # Check if the column exists
    if column_name not in df.columns:
        raise ValueError(f"The column '{column_name}' does not exist in the DataFrame")

    # Check if 'WeekNumber' column exists
    if 'WeekNumber' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'WeekNumber' column")

    # Convert 'WeekNumber' to integer to ensure correct comparison
    df['WeekNumber'] = pd.to_numeric(df['WeekNumber'], errors='coerce').fillna(0).astype(int)

    # Function to calculate sums and zero count
    def calculate_sums_and_zero_count(data, col_name):
        negative_sum = data[data[col_name] < 0][col_name].sum()
        positive_sum = data[data[col_name] > 0][col_name].sum()

        # Count as zero if value is between -0.5 and 0.5
        zero_count = data[(data[col_name] >= -0.5) & (data[col_name] <= 0.5)][col_name].count()

        return negative_sum, positive_sum, zero_count

    # General calculations for the entire DataFrame
    neg_sum, pos_sum, zero_cnt = calculate_sums_and_zero_count(df, column_name)

    # Filter the DataFrame for rows where 'WeekNumber' is 26, then perform calculations
    week_26_data = df[df['WeekNumber'] == 26]
    neg_sum_26, pos_sum_26, zero_cnt_26 = calculate_sums_and_zero_count(week_26_data, column_name)

    # Creating a DataFrame to return the results
    result_df = pd.DataFrame({
        'Negative Sum': [neg_sum, neg_sum_26],
        'Positive Sum': [pos_sum, pos_sum_26],
        'Zero Count': [zero_cnt, zero_cnt_26]
    }, index=['Overall', 'Week 26'])

    return result_df

def count_running_by_hour(df):
    # Check if 'WeekNumber' column exists
    if 'WeekNumber' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'WeekNumber' column")

    # Convert 'WeekNumber' to integer to ensure correct comparison
    df['WeekNumber'] = pd.to_numeric(df['WeekNumber'], errors='coerce').fillna(0).astype(int)

    # General count for all data
    general_grouped_counts = df.groupby(['Hour', 'Running']).size().unstack(fill_value=0)

    # Calculate percentage of 0s for the entire dataset
    general_grouped_counts['Percent_0'] = (general_grouped_counts[0] / general_grouped_counts.sum(axis=1)) * 100

    # Filter the DataFrame for 'WeekNumber' 26 and count
    week_26_data = df[df['WeekNumber'] == 26]
    week_26_grouped_counts = week_26_data.groupby(['Hour', 'Running']).size().unstack(fill_value=0)

    # Calculate percentage of 0s for 'WeekNumber' 26
    week_26_grouped_counts['Percent_0_Week26'] = (week_26_grouped_counts[0] / week_26_grouped_counts.sum(axis=1)) * 100

    # Reset the index to make 'Hour' a column
    general_grouped_counts.reset_index(inplace=True)
    week_26_grouped_counts.reset_index(inplace=True)

    # Merge the general counts and week 26 counts
    result_df = pd.merge(general_grouped_counts, week_26_grouped_counts, on='Hour', suffixes=('', '_Week26'))

    # Rename the columns for clarity
    result_df.columns = ['Hour', 'Count_0', 'Count_1', 'Percent_0', 'Count_0_Week26', 'Count_1_Week26', 'Percent_0_Week26']

    return result_df

def group_counts_and_calculate_percentages(df):
    # Create a new column for 4-hour group intervals
    df['Hour_Group'] = df['Hour'] // 4

    # Group by the new 'Hour_Group' column
    grouped_df = df.groupby('Hour_Group').sum()

    # Calculate new percentages based on grouped counts
    grouped_df['New_Percent_0'] = (grouped_df['Count_0'] / (grouped_df['Count_0'] + grouped_df['Count_1'])) * 100
    grouped_df['New_Percent_0_Week26'] = (grouped_df['Count_0_Week26'] / (grouped_df['Count_0_Week26'] + grouped_df['Count_1_Week26'])) * 100

    return grouped_df.reset_index()



#%%
def uke_dobbel_plot(df, column1, column2, unit, equal=True, date_column_name = 'Tid (Time)'):
    # Ensure the date column is in datetime format
    df[date_column_name] = pd.to_datetime(
        df[date_column_name], format='%d.%m.%Y %H:%M')

    # Group the DataFrame by WeekNumber and DayOfWeek
    df['WeekNumber'] = df[date_column_name].dt.strftime('%U')
    df['DayOfWeek'] = df[date_column_name].dt.strftime('%A')

    # Check if there's a full week of data for each week number
    full_weeks = df.groupby('WeekNumber')['DayOfWeek'].nunique() == 7

    # Iterate through full weeks and create tub plots
    for week_number in full_weeks[full_weeks].index:
        week_data = df[df['WeekNumber'] == week_number]

        # Create an array for the x-axis representing hours from Monday 00:00 to Sunday 23:00
        x = np.arange(7 * 24)

        # Rearrange x-axis to start on Monday (Shift the array 24 positions to the left)
        x = np.roll(x, 0)

        # Check if the lengths of x and y are the same
        if len(x) != len(week_data[column1]):
            # If the lengths don't match, skip to the next week
            continue

        # Create subplots
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot "column1" on the left y-axis
        ax1.plot(x, week_data[column1], color='tab:blue', label=column1)
        ax1.set_xlabel('Ukedag', fontsize=18)  # Increase x-axis label font size
        ax1.set_ylabel(f"{column1} [{unit}]", color='tab:blue', fontsize=20)  # Increase y-axis label font size
        ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)  # Increase y-axis tick label font size

        # Plot "column2" on the right y-axis
        ax2 = ax1.twinx()
        ax2.plot(x, week_data[column2], color='tab:orange', label=column2)
        ax2.set_ylabel(f"{column2} [{unit}]", color='tab:orange', fontsize=20)  # Increase y-axis label font size
        ax2.tick_params(axis='y', labelcolor='tab:orange', labelsize=16)  # Increase y-axis tick label font size


        # Set y-axis limits to be the same for both axes
        if equal == True:
            min_y = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
            max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
            ax1.set_ylim(min_y, max_y)
            ax2.set_ylim(min_y, max_y)

        # Set x-axis ticks and labels to show only the days of the week
        day_labels = ['Mandag', 'Tirsdag', 'Onsdag', 
                      'Torsdag', 'Fredag', 'Lørdag', 'Søndag']
        day_ticks = [24 * i for i in range(len(day_labels))]
        ax1.set_xticks(day_ticks)
        # Show only days of the week
        ax1.set_xticklabels(day_labels, rotation=0, fontsize=16, ha='left')

        # Set the title and legend
        plt.title(f'{column1} og {column2} for uke {week_number}', fontsize=22)
        plt.grid()
        #plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
def uke_enkel_plot(df, column1, unit, date_column_name='Tid (Time)'):
    # Ensure the date column is in datetime format
    df[date_column_name] = pd.to_datetime(
        df[date_column_name], format='%d.%m.%Y %H:%M')

    # Group the DataFrame by WeekNumber and DayOfWeek
    df['WeekNumber'] = df[date_column_name].dt.strftime('%U')
    df['DayOfWeek'] = df[date_column_name].dt.strftime('%A')

    # Check if there's a full week of data for each week number
    full_weeks = df.groupby('WeekNumber')['DayOfWeek'].nunique() == 7

    # Iterate through full weeks and create plots
    for week_number in full_weeks[full_weeks].index:
        week_data = df[df['WeekNumber'] == week_number]

        # Create an array for the x-axis representing hours from Monday 00:00 to Sunday 23:00
        x = np.arange(7 * 24)

        # Check if the lengths of x and y are the same
        if len(x) != len(week_data[column1]):
            continue

        # Create subplots
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot "column1" on the y-axis
        ax1.plot(x, week_data[column1], color='tab:blue', label=column1)
        ax1.set_xlabel('Ukedag', fontsize=18)  # Increase x-axis label font size
        ax1.set_ylabel(f"{column1} [{unit}]", color='tab:blue', fontsize=20)  # Increase y-axis label font size
        ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)  # Increase y-axis tick label font size

        # Set x-axis ticks and labels to show only the days of the week
        day_labels = ['Mandag', 'Tirsdag', 'Onsdag', 'Torsdag', 'Fredag', 'Lørdag', 'Søndag']
        day_ticks = [24 * i for i in range(len(day_labels))]
        ax1.set_xticks(day_ticks)
        ax1.set_xticklabels(day_labels, rotation=0, fontsize=16, ha='left')

        # Set the title
        plt.title(f'{column1} for uke {week_number}', fontsize=22)
        plt.grid()
        plt.tight_layout()
        plt.show()
        
def plot_percent_0(df):
    # Ensure 'Hour' and 'Percent_0' columns are in the DataFrame
    if 'Hour' not in df.columns or 'Percent_0' not in df.columns:
        raise ValueError("DataFrame must contain 'Hour' and 'Percent_0' columns")

    # Plotting as a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(df['Hour'], df['Percent_0'], color='blue')

    # Setting plot labels and title with specified font size
    plt.xlabel('Time på døgnet', fontsize=16)
    plt.ylabel('Andel nektet forbruk [%]', fontsize=16)
    plt.title('Prosentandel av perioden med forbruksnekt per time', fontsize=16)

    # Set x-ticks to show every hour with specified font size
    plt.xticks(df['Hour'], fontsize=16)

    # Setting y-ticks font size
    plt.yticks(fontsize=16)

    # Show grid
    plt.grid(axis='y')

    # Display the plot
    plt.show()


#%%

df = assign_time_columns(df)
tot_df = assign_time_columns(tot_df)

df = merge_missing_columns(df, tot_df, "Tid (Time)")

df = fix_numbers(df)

df = add_day_count(df, "Tid (Time)")


df = distribute_energy(df)

df = new_curve(df)

df = on_off_columns(df)

df = mass_calculation(df)

#%%

df_month_ElEnergi, df_avg = create_timescale_df(df, "Month", "El Energi")
df_month_price, df_avg_price = create_timescale_df(df, "Month", 'Oslo NO1 (NOK) øre/kWh')


df = add_priority_column(df, df_month_ElEnergi, "Month")

chack_df = find_top60_energy_values(df)

#df_test = add_viz_column(df, chack_df)
index_list = find_matching_rows(df, chack_df)
index_lists = find_consecutive_lists(index_list)
sliced_df = df.loc[index_list]
#df = add_running_column(df, index_list)

go_df = calculate_sum_for_lists(index_lists, df)

#%% 
rest = calculate_and_print(chack_df)

index_list2 = find_matching_rows(df, rest)
index_lists2 = find_consecutive_lists(index_list2)
sliced_df2 = df.loc[index_list2]
go_df2 = calculate_sum_for_lists(index_lists2, df)

#%%
df = add_running_column(df, index_lists2)
df_sjekk = process_data(df.copy(), 'El Energi')
df_sjekk2 = calculate_heated_mass(df_sjekk)
df_sjekk2 = time_calc(df_sjekk2)
df_sjekk2 = new_energy(df_sjekk2)
df_sjekk_price = process_data(df.copy(), 'Oslo NO1 (NOK) øre/kWh')

df_done = df_sjekk2.copy()
df_done["Tradisjonelt varmtvannsnivå"] = 4000 - df_done["Mass"]

# Dictionary to map old column names to new column names
column_name_mapping = {'El Energi': 'Totalt forbruk', 'Energi': 'VVB tradisjonelt forbruk', 'New energy': 'VVB flyttet forbruk', 'Total mass': 'Nytt varmtvannsnivå'}

# Rename columns using the mapping
df_done.rename(columns=column_name_mapping, inplace=True)

df_done = calculate_vvb(df_done, 'VVB flyttet forbruk')

done_sum = analyze_column(df_done.copy(), 'Differanse')

new_top60(df_done)

df_run = count_running_by_hour(df_done.copy())
df_run4 = group_counts_and_calculate_percentages(df_run)

df_topp = find_highest_values(df_done.copy())
df_cero_count = analyze_zero_running_df(df_done.copy())
df_c = analyze_zero_running_expanded(df_done.copy())

#%%
#uke_dobbel_plot(df_done, "Totalt forbruk", "VVB flyttet forbruk")
#%%
#uke_dobbel_plot(df_done, 'Tradisjonelt varmtvannsnivå', 'Nytt varmtvannsnivå', True)
#%%
#uke_dobbel_plot(df_done, 'Totalt forbruk', 'Nytt totalt forbruk', True)
#%%
#uke_enkel_plot(df_done, 'Differanse')


#%% 
#uke_dobbel_plot(df_done, 'Totalt forbruk', 'VVB tradisjonelt forbruk', 'kWh/h')


#%%
#uke_dobbel_plot(df_done, 'VVB tradisjonelt forbruk', 'VVB flyttet forbruk', 'kWh/h')
#uke_dobbel_plot(df_done, 'Tradisjonelt varmtvannsnivå', 'Nytt varmtvannsnivå', 'kg')
#uke_dobbel_plot(df_done, 'Nytt totalt forbruk', 'Total VVB flyttet', 'kWh/h', False)
#uke_dobbel_plot(df_done, 'Nytt totalt forbruk', 'Totalt forbruk', 'kWh/h')
#uke_enkel_plot(df_done, 'Differanse', 'kWh/h')

plot_percent_0(df_run)
