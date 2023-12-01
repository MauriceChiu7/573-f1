#!/usr/bin/env python
# coding: utf-8

# In[177]:


import pandas as pd
import numpy as np

input = 'C:/Users/Daniel/Downloads/final_df.csv'
df = pd.read_csv(input, nrows=3720)

df['season'] = df['season'].astype('category')

df['round'] = df['round'].astype('category')

df['date'] = pd.to_datetime(df['date'])

df['start_time'] = df['start_time'].astype('category')

df['driver_name'] = df['driver_name'].astype('category')

df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
df['age2'] = (df['date'] - df['date_of_birth']).astype('<m8[Y]')
df['date_of_birth'] = df['age2']
df.rename(columns={'date_of_birth': 'age'}, inplace=True)
df = df.drop('age2', axis=1)

df['driver_nationality'] = df['driver_nationality'].astype('category')

category_order0 = [0.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 
                  12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
df['driver_standings_pos'] = pd.Categorical(df['driver_standings_pos'], categories=category_order0, ordered=True)

df['initial_tyre'] = df['initial_tyre'].fillna(df.groupby('circuit_id')['initial_tyre'].transform(lambda x: x.mode().iloc[0]))
df['initial_tyre'] = df['initial_tyre'].astype('category')

max_values0 = df.groupby(['season', 'round'])['qualifying_position'].transform('max')
df['qualifying_position'] = df['qualifying_position'].fillna(max_values0 + 1)
category_order1 = [22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 
                   11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

max_time_values0 = df.groupby(['season', 'round'])['qualifying_time'].transform('max')
df['qualifying_time'] = df['qualifying_time'].fillna(max_time_values0)

df['circuit_id'] = df['circuit_id'].astype('category')

df['circuit_name'] = df['circuit_name'].astype('category')

df['circuit_city'] = df['circuit_city'].astype('category')

df['circuit_country'] = df['circuit_country'].astype('category')

df['constructor_name'] = df['constructor_name'].astype('category')

df['constructor_country'] = df['constructor_country'].astype('category')

category_order2 = [0.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
df['constructor_standings_pos'] = pd.Categorical(df['constructor_standings_pos'], categories=category_order2, ordered=True)

mask1 = (df['season'] == 2020) & (df['round'] == 11)
df.loc[mask1, ['fp_pos_1', 'fp_time_1']] = df.loc[mask1, ['fp_pos_3', 'fp_time_3']].values
max_values1 = df.groupby(['season', 'round'])['fp_pos_1'].transform('max')
max_time_values1 = df.groupby(['season', 'round'])['fp_time_1'].transform('max')
df['fp_pos_1'] = df['fp_pos_1'].fillna(max_values1 + 1)
df['fp_time_1'] = df['fp_time_1'].fillna(max_time_values1)

mask2 = (df['season'] == 2015) & (df['round'] == 16)
avg_fp_time = round(df.loc[mask2, ['fp_time_1', 'fp_time_3']].mean(axis=1))
df.loc[mask2, 'fp_time_2'] = avg_fp_time
df.loc[mask2, 'fp_pos_2'] = df.loc[mask2, 'fp_time_2'].rank(method='min')
mask3 = (df['season'] == 2017) & (df['round'] == 2)
avg_fp_time2 = round(df.loc[mask3, ['fp_time_1', 'fp_time_3']].mean(axis=1))
df.loc[mask3, 'fp_time_2'] = avg_fp_time2
df.loc[mask3, 'fp_pos_2'] = df.loc[mask3, 'fp_time_2'].rank(method='min')
mask4 = (df['season'] == 2020) & (df['round'] == 11)
df.loc[mask4, ['fp_pos_2', 'fp_time_2']] = df.loc[mask4, ['fp_pos_3', 'fp_time_3']].values
mask5 = (df['season'] == 2020) & (df['round'] == 13)
df.loc[mask5, ['fp_pos_2', 'fp_time_2']] = df.loc[mask5, ['fp_pos_1', 'fp_time_1']].values
mask6 = (df['season'] == 2023) & (df['round'] == 4)
df.loc[mask6, ['fp_pos_2', 'fp_time_2']] = df.loc[mask6, ['fp_pos_1', 'fp_time_1']].values
mask7 = (df['season'] == 2023) & (df['round'] == 9)
df.loc[mask7, ['fp_pos_2', 'fp_time_2']] = df.loc[mask7, ['fp_pos_1', 'fp_time_1']].values
mask8 = (df['season'] == 2023) & (df['round'] == 12)
df.loc[mask8, ['fp_pos_2', 'fp_time_2']] = df.loc[mask8, ['fp_pos_1', 'fp_time_1']].values
mask9 = (df['season'] == 2023) & (df['round'] == 17)
df.loc[mask9, ['fp_pos_2', 'fp_time_2']] = df.loc[mask9, ['fp_pos_1', 'fp_time_1']].values
mask10 = (df['season'] == 2023) & (df['round'] == 18)
df.loc[mask10, ['fp_pos_2', 'fp_time_2']] = df.loc[mask10, ['fp_pos_1', 'fp_time_1']].values
mask11 = (df['season'] == 2023) & (df['round'] == 20)
df.loc[mask11, ['fp_pos_2', 'fp_time_2']] = df.loc[mask11, ['fp_pos_1', 'fp_time_1']].values
max_values2 = df.groupby(['season', 'round'])['fp_pos_2'].transform('max')
max_time_values2 = df.groupby(['season', 'round'])['fp_time_2'].transform('max')
df['fp_pos_2'] = df['fp_pos_2'].fillna(max_values2 + 1)
df['fp_time_2'] = df['fp_time_2'].fillna(max_time_values2)

mask12 = (df['season'] == 2019) & (df['round'] == 17)
avg_fp_time3 = round(df.loc[mask12, ['fp_time_1', 'fp_time_2']].mean(axis=1))
df.loc[mask12, 'fp_time_3'] = avg_fp_time3
df.loc[mask12, 'fp_pos_3'] = df.loc[mask12, 'fp_time_3'].rank(method='min')
mask13 = (df['season'] == 2020) & (df['round'] == 2)
avg_fp_time4 = round(df.loc[mask13, ['fp_time_1', 'fp_time_2']].mean(axis=1))
df.loc[mask13, 'fp_time_3'] = avg_fp_time4
df.loc[mask13, 'fp_pos_3'] = df.loc[mask13, 'fp_time_3'].rank(method='min')
mask14 = (df['season'] == 2020) & (df['round'] == 13)
df.loc[mask14, ['fp_pos_3', 'fp_time_3']] = df.loc[mask14, ['fp_pos_1', 'fp_time_1']].values
mask15 = (df['season'] == 2021) & (df['round'] == 10)
avg_fp_time5 = round(df.loc[mask15, ['fp_time_1', 'fp_time_2']].mean(axis=1))
df.loc[mask15, 'fp_time_3'] = avg_fp_time5
df.loc[mask15, 'fp_pos_3'] = df.loc[mask15, 'fp_time_3'].rank(method='min')
mask16 = (df['season'] == 2021) & (df['round'] == 14)
avg_fp_time6 = round(df.loc[mask16, ['fp_time_1', 'fp_time_2']].mean(axis=1))
df.loc[mask16, 'fp_time_3'] = avg_fp_time6
df.loc[mask16, 'fp_pos_3'] = df.loc[mask16, 'fp_time_3'].rank(method='min')
mask17 = (df['season'] == 2021) & (df['round'] == 15)
avg_fp_time7 = round(df.loc[mask17, ['fp_time_1', 'fp_time_2']].mean(axis=1))
df.loc[mask17, 'fp_time_3'] = avg_fp_time7
df.loc[mask17, 'fp_pos_3'] = df.loc[mask17, 'fp_time_3'].rank(method='min')
mask18 = (df['season'] == 2021) & (df['round'] == 19)
avg_fp_time8 = round(df.loc[mask18, ['fp_time_1', 'fp_time_2']].mean(axis=1))
df.loc[mask18, 'fp_time_3'] = avg_fp_time8
df.loc[mask18, 'fp_pos_3'] = df.loc[mask18, 'fp_time_3'].rank(method='min')
mask19 = (df['season'] == 2022) & (df['round'] == 4)
avg_fp_time9 = round(df.loc[mask19, ['fp_time_1', 'fp_time_2']].mean(axis=1))
df.loc[mask19, 'fp_time_3'] = avg_fp_time9
df.loc[mask19, 'fp_pos_3'] = df.loc[mask19, 'fp_time_3'].rank(method='min')
mask20 = (df['season'] == 2022) & (df['round'] == 11)
avg_fp_time10 = round(df.loc[mask20, ['fp_time_1', 'fp_time_2']].mean(axis=1))
df.loc[mask20, 'fp_time_3'] = avg_fp_time10
df.loc[mask20, 'fp_pos_3'] = df.loc[mask20, 'fp_time_3'].rank(method='min')
mask21 = (df['season'] == 2022) & (df['round'] == 21)
avg_fp_time11 = round(df.loc[mask21, ['fp_time_1', 'fp_time_2']].mean(axis=1))
df.loc[mask21, 'fp_time_3'] = avg_fp_time11
df.loc[mask21, 'fp_pos_3'] = df.loc[mask21, 'fp_time_3'].rank(method='min')
mask22 = (df['season'] == 2023) & (df['round'] == 4)
df.loc[mask22, ['fp_pos_3', 'fp_time_3']] = df.loc[mask22, ['fp_pos_1', 'fp_time_1']].values
mask23 = (df['season'] == 2023) & (df['round'] == 9)
df.loc[mask23, ['fp_pos_3', 'fp_time_3']] = df.loc[mask23, ['fp_pos_1', 'fp_time_1']].values
mask24 = (df['season'] == 2023) & (df['round'] == 12)
df.loc[mask24, ['fp_pos_3', 'fp_time_3']] = df.loc[mask24, ['fp_pos_1', 'fp_time_1']].values
mask25 = (df['season'] == 2023) & (df['round'] == 17)
df.loc[mask25, ['fp_pos_3', 'fp_time_3']] = df.loc[mask25, ['fp_pos_1', 'fp_time_1']].values
mask26 = (df['season'] == 2023) & (df['round'] == 18)
df.loc[mask26, ['fp_pos_3', 'fp_time_3']] = df.loc[mask26, ['fp_pos_1', 'fp_time_1']].values
mask27 = (df['season'] == 2023) & (df['round'] == 20)
df.loc[mask27, ['fp_pos_3', 'fp_time_3']] = df.loc[mask27, ['fp_pos_1', 'fp_time_1']].values
max_values3 = df.groupby(['season', 'round'])['fp_pos_3'].transform('max')
max_time_values3 = df.groupby(['season', 'round'])['fp_time_3'].transform('max')
df['fp_pos_3'] = df['fp_pos_3'].fillna(max_values3 + 1)
df['fp_time_3'] = df['fp_time_3'].fillna(max_time_values3)

category_order3 = [23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 
                  12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
df['fp_pos_1'] = pd.Categorical(df['fp_pos_1'], categories=category_order3, ordered=True)
category_order4 = [22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 
                  12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
df['fp_pos_2'] = pd.Categorical(df['fp_pos_2'], categories=category_order4, ordered=True)
category_order5 = [22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 
                  12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
df['fp_pos_3'] = pd.Categorical(df['fp_pos_3'], categories=category_order5, ordered=True)

df['weather_cloudy'] = df['weather_cloudy'].astype('category')

df['weather_cold'] = df['weather_cold'].astype('category')

df['weather_dry'] = df['weather_dry'].astype('category')

df['weather_warm'] = df['weather_warm'].astype('category')

df['weather_wet'] = df['weather_wet'].astype('category')

mask28 = ~df['has_sprint']

df['has_sprint'] = df['has_sprint'].astype('category')

df.loc[mask28, 'sprint_qualifying_position'] = df.loc[mask28, 'qualifying_position'].values
max_values4 = df.groupby(['season', 'round'])['sprint_qualifying_position'].transform('max')
df['sprint_qualifying_position'] = df['sprint_qualifying_position'].fillna(max_values4 + 1)
category_order6 = [22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 
                   11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
df['sprint_qualifying_position'] = pd.Categorical(df['sprint_qualifying_position'], categories=category_order6, ordered=True)

df['qualifying_position'] = pd.Categorical(df['qualifying_position'], categories=category_order1, ordered=True)

df.loc[mask28, 'sprint_qualifying_time'] = df.loc[mask28, 'qualifying_time'].values
max_time_values4 = df.groupby(['season', 'round'])['sprint_qualifying_time'].transform('max')
df['sprint_qualifying_time'] = df['sprint_qualifying_time'].fillna(max_time_values4)

df = df.drop('sprint_laps', axis=1)

category_order7 = [22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 
                   11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

df['num_o_ps'] = df['num_o_ps'].fillna(0)

max_values5 = df.groupby(['season', 'round'])['time'].transform('max')
max_values6 = df.groupby(['season', 'round'])['fl_time'].transform('max')
df['time'] = df['time'].fillna(max_values5 + max_values6)

df['status'] = df['status'].astype('category')

mask29 = (df['season'] == 2021) & (df['round'] == 12)
df.loc[mask29, ['fl_pos', 'fl_time']] = df.loc[mask29, ['podium', 'time']].values

max_values7 = df.groupby(['season', 'round'])['fl_pos'].transform('max')
df['fl_pos'] = df['fl_pos'].fillna(max_values7 + 1)

df.loc[mask28, 'sprint_fl_pos'] = df.loc[mask28, 'fl_pos'].values

max_values8 = df.groupby(['season', 'round'])['sprint_fl_pos'].transform('max')
df['sprint_fl_pos'] = df['sprint_fl_pos'].fillna(max_values8 + 1)
mask30 = (df['season'] == 2023) & (df['round'] == 12)
df.loc[mask30, 'sprint_fl_time'] = np.maximum(df.loc[mask30, 'sprint_qualifying_time'], df.loc[mask30, 'qualifying_time'])
df.loc[mask30, 'sprint_fl_pos'] = df.loc[mask30, 'sprint_fl_time'].rank(method='min')

category_order8 = [22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 
                   11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
df['sprint_fl_pos'] = pd.Categorical(df['sprint_fl_pos'], categories=category_order8, ordered=True)
df['fl_pos'] = pd.Categorical(df['fl_pos'], categories=category_order8, ordered=True)

df['fl_time'] = df['fl_time'].fillna(max_values6)

max_values9 = df.groupby(['season', 'round'])['sprint_fl_time'].transform('max')
df['sprint_fl_time'] = df['sprint_fl_time'].fillna(max_values9)
df.loc[mask28, 'sprint_fl_time'] = df.loc[mask28, 'fl_time'].values

df['sprint_position'] = df['sprint_position'].replace(['dns', 'dnf'], np.nan)
df['sprint_position'] = df['sprint_position'].astype(float)
df.loc[mask28, 'sprint_position'] = df.loc[mask28, 'podium'].values
max_values10 = df.groupby(['season', 'round'])['sprint_position'].transform('max')
df['sprint_position'] = df['sprint_position'].fillna(max_values10+1)
df['podium'] = pd.Categorical(df['podium'], categories=category_order7, ordered=True)
df['sprint_position'] = pd.Categorical(df['sprint_position'], categories=category_order7, ordered=True)

df.loc[mask28, 'sprint_time'] = df.loc[mask28, 'time'].values
max_values11 = df.groupby(['season', 'round'])['sprint_time'].transform('max')
max_values12 = df.groupby(['season', 'round'])['sprint_fl_time'].transform('max')
df['sprint_time'] = df['sprint_time'].fillna(max_values11 + max_values12)

condition1 = (df['season'] == 2021) & (df['round'] == 12) & (df['driver_name'] == 'max verstappen')
df.loc[condition1, 'points'] = 13.0
condition2 = (df['season'] == 2021) & (df['round'] == 12) & (df['driver_name'] == 'lewis hamilton')
df.loc[condition2, 'points'] = 8.0
condition3 = (df['season'] == 2021) & (df['round'] == 12) & (df['driver_name'] == 'carlos sainz')
df.loc[condition3, 'points'] = 1.0




# In[186]:


print("Columns with NaN values:")
print(df.isna().any())


# In[185]:


print(df.dtypes)

