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

    # Iterate through unique input1 values
    for value in df[input1].unique():
        # Calculate the average value of input2 for each distinct "Hour"-value with the same input1-value
        avg_values = df[df[input1] == value].groupby('Hour')[input2].mean()

        # Rank the average values in descending order
        ranked_hours = avg_values.sort_values(ascending=True).index

        # Add corresponding "Hour" values to df_timescale as columns
        df_timescale[value] = ranked_hours
    df_timescale['Ranking'] = range(1, len(df_timescale) + 1)

    return df_timescale

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


def subtract_and_add_values(df, column1, column2):
    check_result = (df[column1] != 0) & (df[column2] != 0)
    rows_meeting_condition = df[check_result]

    for index in rows_meeting_condition.index:
        # Subtract the value from column2 in the current row
        subtracted_value = df.at[index, column2]
        df.at[index, column2] = 0

        # Add the subtracted value to column2 in the previous row
        if index > 0:
            df.at[index - 1, column2] += subtracted_value

    return df


#%% 
def calculate_heated_mass(df):
    # Step 1: Add a new columns "Heated mass", "New on" and heat and time columns with default value 0
    df['Heated mass'] = 0
    df["New on"] = 0
    
    df["Heat 1"] = 0
    df["Heat 2"] = 0
    df["Heat 3.0"] = 0
    df["Heat 3.4"] = 0
    df["Heat 4"] = 0
    
    df["Time 1"] = 0
    df["Time 2"] = 0
    df["Time 3"] = 0
    df["Time 4"] = 0
    
    #df.at[2893, "Heated mass"] = df.at[2893, "Mass"]
    #df.at[2725, "Heated mass"] = df.at[2725, "Mass"]


    # Step 2: Define constant max_heat and total mass caluclation for the first row
    max_heat = 286
    
    df.at[0, 'Total mass'] = 4000 - df.at[0, 'Mass'] + df.at[0, 'Heated mass']
    
    # Step 3: Iterate through the DataFrame
    for i in range(1, len(df)):
        # Step 5: Total mass caluculation for the given row
        df.at[i, 'Total mass'] = max(0, min(4000, df.at[i-1, 'Total mass'] - df.at[i, 'Mass'] + df.at[i, 'Heated mass']))
        # Step 6: Adding heated mass if total mass gets below 1000
        if df.at[i, 'Total mass'] < 1000:
            if df.at[i-1, 'Total mass'] > 1000:
                df.at[i, 'Heated mass'] += df.at[i, 'Mass'] - df.at[i-1, 'Total mass'] + 1000 
            else: 
                df.at[i, 'Heated mass'] += df.at[i, 'Mass']
            if df.at[i, 'Heated mass'] < max_heat:
                df.at[i, 'New on'] = 1
                df.at[i, 'Nr off'] = 3
                df.at[i, "Heat 4"] = df.at[i, 'Heated mass']
                df.at[i, "Time 4"] = df.at[i, "Heated mass"]/max_heat
            else:
                df.at[i, 'New on'] = 2
                df.at[i, 'Nr off'] = 2
                df.at[i, "Heat 4"] = max_heat
                df.at[i, "Time 4"] = 1
                df.at[i, "Heat 3.4"] = df.at[i, 'Heated mass'] - max_heat
                df.at[i, "Time 4"] = (df.at[i, "Heated mass"] - max_heat)/(2*max_heat)
            
            df.at[i, 'Total mass'] = max(0, min(4000, df.at[i-1, 'Total mass'] - df.at[i, 'Mass'] + df.at[i, 'Heated mass']))
    

        # Step 7: Add "Nr off" column
        if df.at[i, 'Heated mass'] == 0:
            if 3000 <= df.at[i, 'Total mass'] <= 4000:
                df.at[i, 'Nr off'] = 1
            elif 2000 <= df.at[i, 'Total mass'] < 3000:
                df.at[i, 'Nr off'] = 2
            elif 1000 <= df.at[i, 'Total mass'] < 2000:
                df.at[i, 'Nr off'] = 3
            elif df.at[i, 'Total mass'] < 1000:
                df.at[i, 'Nr off'] = 4

            
        # Step 8: Check next 5 values in "Priority"
        if i + 6 < len(df):
            next_5_priorities = df.loc[i + 1:i + 6, ['Priority']]
            #print(next_5_priorities['Priority'])
            if all(priority <= 6 for priority in next_5_priorities['Priority']):
                # Original implementation for mass_sum calculation
                #print(next_5_priorities.index)
                sorted_df = next_5_priorities.sort_values(by='Priority')
                #print(sorted_list)
                
                total_heat = 4000 - df.at[i, 'Total mass']

                # Check "Nr off" value and calculate last_tank
                nr_off = df.at[i, 'Nr off']
                last_tank = min(total_heat - (nr_off - 1) * 1000, 1000) 

                # Assign values based on total_heat and "Nr off"
                if 2000 <= total_heat <= 3000:
                    heat_1, heat_2, heat_3 = 1000, 1000, last_tank
                elif 1000 <= total_heat < 2000:
                    heat_1, heat_2, heat_3 = 1000, last_tank, 0
                elif total_heat < 1000:
                    heat_1, heat_2, heat_3 = last_tank, 0, 0
                else:
                    heat_1, heat_2, heat_3 = 0, 0, 0  # Default values

                # Iterate through the prioritized rows and distribute heat
                for row in sorted_df.index:  # Select the three lowest priorities
                    #row = df[(df['Priority'] == heat_priority)].index[0]
                    #print(row)
                    
                  #  if df.at[row, "Heat 1"] is not 0:
                   #     df.at[row, "Nr off"] = 0
                        
                    if heat_1 >= max_heat:
                        df.at[row, "Heat 1"] = max_heat
                        heat_1 -= max_heat
                    elif heat_1 < max_heat:
                        df.at[row, "Heat 1"] = heat_1
                        heat_1 = 0
                    
                    if heat_2 >= max_heat:
                        df.at[row, "Heat 2"] = max_heat
                        heat_2 -= max_heat
                    elif heat_2 < max_heat:
                        df.at[row, "Heat 2"] = heat_2
                        heat_2 = 0

                    if heat_3 >= max_heat:
                        df.at[row, "Heat 3.0"] = max_heat
                        heat_3 -= max_heat
                    elif heat_3 < max_heat:
                        df.at[row, "Heat 3.0"] = heat_3
                        heat_3 = 0
                
                df.at[i + 7, 'Heated mass'] += total_heat 
                #print(df.at[i + 5, 'Total mass'])
                
        if df.at[i, "Heated mass"] > 1000: 
            previous_rows = df.loc[max(0, i-5):i-1, ['Heat 1', 'Heat 2', 'Heat 3.0', 'Priority']]
            remaining_value = 4000 - df.at[i, "Total mass"] - df.at[i, "Mass"]
            previous_rows = previous_rows.sort_values(by='Priority')
            df.at[i, "Heated mass"] += remaining_value
            df.at[i, 'Total mass'] = max(0, min(4000, df.at[i-1, 'Total mass'] - df.at[i, 'Mass'] + df.at[i, 'Heated mass']))
            for r in previous_rows.index:
                if remaining_value > 5 and df.at[r, "Heat 1"] < 286:
                 # Calculate the amount to distribute to the current column
                    distribute_amount = min(286 - df.at[r, "Heat 1"], remaining_value)
                    
                    # Update the column value and remaining_value
                    df.at[r, "Heat 1"] += distribute_amount
                    remaining_value -= distribute_amount
                elif remaining_value > 5 and df.at[r, "Heat 2"] < 286:
                 # Calculate the amount to distribute to the current column
                    distribute_amount = min(286 - df.at[r, "Heat 2"], remaining_value)
                    
                    # Update the column value and remaining_value
                    df.at[r, "Heat 2"] += distribute_amount
                    remaining_value -= distribute_amount
                elif remaining_value > 5 and df.at[r, "Heat 3.0"] < 286:
                 # Calculate the amount to distribute to the current column
                    distribute_amount = min(286 - df.at[r, "Heat 3.0"], remaining_value)
                    
                    # Update the column value and remaining_value
                    df.at[r, "Heat 3.0"] += distribute_amount
                    remaining_value -= distribute_amount
                    
            if df.at[i, "Priority"] < 12: 
                df.at[i, "Heated mass"] += df.at[i, "Mass"] 
                df.at[i, "Heat 1"] += df.at[i, "Mass"] 
                df.at[i, 'Total mass'] = max(0, min(4000, df.at[i-1, 'Total mass'] - df.at[i, 'Mass'] + df.at[i, 'Heated mass']))
        
        #for 
        #if df.at[i, "Heat 4"] != 0 and df.at[i]
    
    #df.at[2893, "Heat 1"] = df.at[2893, "Mass"]
    #df.at[2725, "Heat 1"] = df.at[2725, "Mass"]
    return df


def sum_heat_columns(df):
    # Filter rows where "Priority" is twelve and above
    filtered_df = df[df['Priority'] >= 12]
    
    # Calculate the sum of "Heat 3.4" and "Heat 4" for each "Priority"
    result_df = filtered_df.groupby('Priority')[['Heat 3.4', 'Heat 4']].sum().reset_index()
    
    return result_df

def process_heat4_checks(df):
    j = 1
    sliced_dfs = []

    for i in range(1, len(df)):
        #print(j)
        if j == i:
            if df.at[j, 'Heat 4'] > 0 and df.at[j, 'Priority'] >= 12:
                # Identify the row where "Heated mass" is greater than 1000
                #heated_mass_row = df.index[df['Heated mass'] > 1000].max()

                # Slice out the new df with the 5 next rows and the 10 previous rows
                sliced_df = df.loc[max(0, j - 14):j + 5].copy()
                
                # CHECKS!!!!
                # Check every row until the row where "Heated mass" > 1000
                for idx in sliced_df.index:
                    if sliced_df.at[idx, "Heated mass"] > 1000:
                        break
                    if sliced_df.at[idx, 'Heated mass'] == 0 and sliced_df.at[idx, 'Heat 1'] == 0:
                        print(idx)
                        
                # Check if there are more than two values of "Heated mass" that are above 1000
                if sliced_df[sliced_df['Heated mass'] > 1000]['Heated mass'].count() > 1:
                    print("Warning 2")
                #CHECKS DONE!!!!!!!

                # Append the sliced df to the list
                sliced_dfs.append(sliced_df)
                
                

                # Skip the next 5 rows
                j += 6  # Skip the next 5 rows plus the current row
        elif i > j:
            j = i+1

    return sliced_dfs

def process_heat4_rows(df):
    j = 1
    sliced_dfs = []
    df["Avail"] = 0

    for i in range(1, len(df)):
        if j == i:
            if df.at[j, 'Heat 4'] > 0 and df.at[j, 'Priority'] >= 12:
                # Identify the row where "Heated mass" is greater than 1000
                heated_mass_row = df.index[df['Heated mass'] > 1000].max()

                # Slice out the new df with the 5 next rows and the 10 previous rows
                sliced_df = df.loc[max(0, j - 14):j + 5].copy()

                # Identify rows where "Heated mass" and "Heat 1" are 0, and "Priority" is <= 12
                condition = (sliced_df['Heated mass'] == 0) & (sliced_df['Heat 1'] == 0) & (sliced_df['Priority'] <= 12)

                # Extract rows that meet the condition
                filtered_rows = sliced_df[condition]
                
                for r in filtered_rows.index:
                    df.at[r, "Avail"] = 4000 - df.at[r, "Total mass"]
                

                # Skip the next 5 rows
                j += 6  # Skip the next 5 rows plus the current row
        elif i > j:
            j = i + 1

    return df

def calculate_TM2(df):
    # Assuming "Heat 1", "Heat 2", and "Heat 3.0" columns exist in the DataFrame
    heat_columns = ["Heat 1", "Heat 2", "Heat 3.0", "Heat 3.4", "Heat 4"]

    # Create a new column "HM2" as the sum of specified heat columns
    df['HM2'] = df[heat_columns].sum(axis=1)
    
    df.at[0, 'TM2'] = 4000 - df.at[0, 'Mass'] + df.at[0, 'HM2']
    df.at[6213, 'TM2'] = 4000
    # Step 3: Iterate through the DataFrame
    for i in range(1, len(df)):
        # Step 5: Total mass caluculation for the given row
        df.at[i, 'TM2'] = max(0, min(4000, df.at[i-1, 'TM2'] - df.at[i, 'Mass'] + df.at[i, 'HM2']))
    


    return df

def sum_HM2_by_priority(df):
    # Group by "Priority" and calculate the sum of "HM2" for each group
    result_df = df.groupby('Priority')['HM2'].agg(['sum', 'count']).reset_index()
    
    # Calculate the percentage column
    result_df['Percentage'] = result_df['sum'] / result_df['sum'].sum() * 100

    return result_df

def sum_NE_by_priority(df):
    # Group by "Priority" and calculate the sum of "HM2" for each group
    result_df = df.groupby('Priority')['New energy'].agg(['sum', 'count']).reset_index()
    
    # Calculate the percentage column
    result_df['Percentage'] = result_df['sum'] / result_df['sum'].sum() * 100

    return result_df
def sum_E_by_priority(df):
    # Group by "Priority" and calculate the sum of "HM2" for each group
    result_df = df.groupby('Priority')['Energi'].agg(['sum', 'count']).reset_index()
    
    # Calculate the percentage column
    result_df['Percentage'] = result_df['sum'] / result_df['sum'].sum() * 100

    return result_df


def create_new_df(df):
    # Create a new DataFrame to store the calculated means
    new_df = pd.DataFrame()

    # Specify the columns to keep in the new DataFrame
    columns_to_keep = ['Month', 'Hour', 'Priority', 'Heat 1', 'Heat 2', 'Heat 3.0', 'Heat 3.4', 'Heat 4', 'Mass', 'Heated mass', 'Total mass']

    # Iterate over each distinct month in the original DataFrame
    for month in df['Month'].unique():
        # Slice out rows for each month
        month_slice = df[df['Month'] == month]

        # Create a new row for the means
        mean_row = month_slice.groupby('Hour')[columns_to_keep].mean().reset_index()

        # Update the 'Month' and 'Priority' columns in the mean row
        mean_row['Month'] = month
        mean_row['Priority'] = month_slice['Priority'].iloc[0]  # Assuming 'Priority' is the same for all hours in a month

        # Append the mean row to the new DataFrame
        new_df = new_df.append(mean_row, ignore_index=True)

    return new_df


def time_calc(df):
    df['Time Total'] = 0
    for i in range(len(df)):
        df.at[i, 'Time 1'] = df.at[i, 'Heat 1'] / 286
        df.at[i, 'Time 2'] = df.at[i, 'Heat 2'] / 286
        df.at[i, 'Time 3'] = (df.at[i, 'Heat 3.0'] + df.at[i, 'Heat 3.4']) / 286
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


def new_top60(df):
    # Assuming df is your DataFrame with columns 'Month', 'Day_count', 'Hour', and 'El Energi'
    
    # Group by 'Month'
    grouped_df = df.groupby('Month')
    sum_list = []
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
df_month_price = create_timescale_df(df, "Month", 'Oslo NO1 (NOK) øre/kWh')
df_price = add_priority_column(df.copy(), df_month_price, "Month")
df_price = calculate_heated_mass(df_price)
#%%

df_month_ElEnergi = create_timescale_df(df, "Month", "El Energi")

df = add_priority_column(df, df_month_ElEnergi, "Month")

df = calculate_heated_mass(df)

df_new_2 = subtract_and_add_values(df.copy(), "Heat 3.0", "Heat 3.4")

test_df = sum_heat_columns(df_new_2)
df_test = process_heat4_rows(df.copy())

df_try = calculate_TM2(df_new_2)
#process_heat4_rows(df)
result = sum_HM2_by_priority(df_try)


#%%
df_try["Tradisjonelt varmtvannsnivå"] = 4000 - df_try["Mass"]
df_try = new_energy(time_calc(df_new_2))
result_power = sum_NE_by_priority(df_try)
result_energi = sum_E_by_priority(df_try)

df_done = df_try.copy()

# Dictionary to map old column names to new column names
column_name_mapping = {'El Energi': 'Totalt forbruk', 'Energi': 'VVB tradisjonelt forbruk', 'New energy': 'VVB flyttet forbruk',
                       'TM2': 'Nytt varmtvannsnivå'}

# Rename columns using the mapping
df_done.rename(columns=column_name_mapping, inplace=True)

df_done = calculate_vvb(df_done, 'VVB flyttet forbruk')

done_sum = analyze_column(df_done.copy(), 'Differanse')

df_topp_max = find_highest_values(df_done.copy())

new_top60(df_done.copy())


#%% Volum i tanken
#uke_dobbel_plot(df_done, 'Tradisjonelt varmtvannsnivå', 'Nytt varmtvannsnivå', True)

#%% Lastflytting
#uke_dobbel_plot(df_done, 'Totalt forbruk', 'VVB flyttet forbruk')

#%% Totalt forbruk
#uke_dobbel_plot(df_done, 'Totalt forbruk', 'Nytt totalt forbruk', True)
#%%
#uke_enkel_plot(df_done, 'Differanse')


#%% 
#uke_dobbel_plot(df_done, 'VVB tradisjonelt forbruk', 'VVB flyttet forbruk', 'kWh/h')
#uke_dobbel_plot(df_done, 'Tradisjonelt varmtvannsnivå', 'Nytt varmtvannsnivå', 'kg')
#uke_dobbel_plot(df_done, 'Nytt totalt forbruk', 'Total VVB flyttet', 'kWh/h', False)
#uke_dobbel_plot(df_done, 'Nytt totalt forbruk', 'Totalt forbruk', 'kWh/h')
#uke_enkel_plot(df_done, 'Differanse', 'kWh/h')

















