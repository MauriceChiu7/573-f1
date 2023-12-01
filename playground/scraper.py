import pandas as pd
import numpy as np
import requests
FROM_YEAR = 2019
THIS_YEAR = 2023
# query API

# races = {'season': [],
#         'round': [],
#         'circuit_id': [],
#         'lat': [],
#         'long': [],
#         'country': [],
#         'date': [],
#         'url': []}

# for year in list(range(FROM_YEAR,THIS_YEAR)):
#     print(f"processing races year {year}")    
#     url = 'https://ergast.com/api/f1/{}.json'
#     r = requests.get(url.format(year))
#     json = r.json()

#     for item in json['MRData']['RaceTable']['Races']:
#         try:
#             races['season'].append(int(item['season']))
#         except:
#             races['season'].append(None)

#         try:
#             races['round'].append(int(item['round']))
#         except:
#             races['round'].append(None)

#         try:
#             races['circuit_id'].append(item['Circuit']['circuitId'])
#         except:
#             races['circuit_id'].append(None)

#         try:
#             races['lat'].append(float(item['Circuit']['Location']['lat']))
#         except:
#             races['lat'].append(None)

#         try:
#             races['long'].append(float(item['Circuit']['Location']['long']))
#         except:
#             races['long'].append(None)

#         try:
#             races['country'].append(item['Circuit']['Location']['country'])
#         except:
#             races['country'].append(None)

#         try:
#             races['date'].append(item['date'])
#         except:
#             races['date'].append(None)

#         try:
#             races['url'].append(item['url'])
#         except:
#             races['url'].append(None)
        
# races = pd.DataFrame(races)
# races.to_csv("./data/races.csv", sep=",", index=False)

# # append the number of rounds to each season from the races_df

races = pd.read_csv("./data/races.csv")
results = pd.read_csv("./data/results.csv")
driver_standings = pd.read_csv("./data/driver_standings.csv")
constructor_standings = pd.read_csv("./data/constructor_standings.csv")
qualifying_results = pd.read_csv("./data/qualifying_results.csv")
weather = pd.read_csv("./data/weather.csv")
weather_info = pd.read_csv("./data/weather_info.csv")
final_df = pd.read_csv("./data/final_df.csv")

rounds = []
for year in np.array(races.season.unique()):
    rounds.append([year, list(races[races.season == year]['round'])])

# print(rounds)

# # query API
    
# results = {'season': [],
#           'round':[],
#            'circuit_id':[],
#           'driver': [],
#            'date_of_birth': [],
#            'nationality': [],
#           'constructor': [],
#           'grid': [],
#           'time': [],
#           'status': [],
#           'points': [],
#           'podium': []}

# for n in list(range(len(rounds))):
#     for i in rounds[n][1]:
#         print(f"processing results round: {n}, {i}")
#         url = 'http://ergast.com/api/f1/{}/{}/results.json'
#         r = requests.get(url.format(rounds[n][0], i))
#         json = r.json()

#         for item in json['MRData']['RaceTable']['Races'][0]['Results']:
#             try:
#                 results['season'].append(int(json['MRData']['RaceTable']['Races'][0]['season']))
#             except:
#                 results['season'].append(None)

#             try:
#                 results['round'].append(int(json['MRData']['RaceTable']['Races'][0]['round']))
#             except:
#                 results['round'].append(None)

#             try:
#                 results['circuit_id'].append(json['MRData']['RaceTable']['Races'][0]['Circuit']['circuitId'])
#             except:
#                 results['circuit_id'].append(None)

#             try:
#                 results['driver'].append(item['Driver']['driverId'])
#             except:
#                 results['driver'].append(None)
            
#             try:
#                 results['date_of_birth'].append(item['Driver']['dateOfBirth'])
#             except:
#                 results['date_of_birth'].append(None)
                
#             try:
#                 results['nationality'].append(item['Driver']['nationality'])
#             except:
#                 results['nationality'].append(None)

#             try:
#                 results['constructor'].append(item['Constructor']['constructorId'])
#             except:
#                 results['constructor'].append(None)

#             try:
#                 results['grid'].append(int(item['grid']))
#             except:
#                 results['grid'].append(None)

#             try:
#                 results['time'].append(int(item['Time']['millis']))
#             except:
#                 results['time'].append(None)

#             try:
#                 results['status'].append(item['status'])
#             except:
#                 results['status'].append(None)

#             try:
#                 results['points'].append(int(item['points']))
#             except:
#                 results['points'].append(None)

#             try:
#                 results['podium'].append(int(item['position']))
#             except:
#                 results['podium'].append(None)

           
# results = pd.DataFrame(results)
# results.to_csv("./data/results.csv", sep=",", index=False)

# driver_standings = {'season': [],
#                     'round':[],
#                     'driver': [],
#                     'driver_points': [],
#                     'driver_wins': [],
#                    'driver_standings_pos': []}

# for n in list(range(len(rounds))):     
#     for i in rounds[n][1]:    # iterate through rounds of each year
#         print(f"processing driver standings round: {n}, {i}")
#         url = 'https://ergast.com/api/f1/{}/{}/driverStandings.json'
#         r = requests.get(url.format(rounds[n][0], i))
#         json = r.json()

#         print(f"url: https://ergast.com/api/f1/{rounds[n][0]}/{i}/driverStandings.json")
#         print(f"r: {r}")
#         print(f"json: {json}")
#         print("================")

#         for item in json['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']:
#             try:
#                 driver_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
#             except:
#                 driver_standings['season'].append(None)

#             try:
#                 driver_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
#             except:
#                 driver_standings['round'].append(None)
                                         
#             try:
#                 driver_standings['driver'].append(item['Driver']['driverId'])
#             except:
#                 driver_standings['driver'].append(None)
            
#             try:
#                 driver_standings['driver_points'].append(int(item['points']))
#             except:
#                 driver_standings['driver_points'].append(None)
            
#             try:
#                 driver_standings['driver_wins'].append(int(item['wins']))
#             except:
#                 driver_standings['driver_wins'].append(None)
                
#             try:
#                 driver_standings['driver_standings_pos'].append(int(item['position']))
#             except:
#                 driver_standings['driver_standings_pos'].append(None)
            
# driver_standings = pd.DataFrame(driver_standings)
# driver_standings.to_csv("./data/driver_standings.csv", sep=",", index=False)

# define lookup function to shift points and number of wins from previous rounds

# def lookup (df, team, points):
#     df['lookup1'] = df.season.astype(str) + df[team] + df['round'].astype(str)
#     df['lookup2'] = df.season.astype(str) + df[team] + (df['round']-1).astype(str)
#     new_df = df.merge(df[['lookup1', points]], how = 'left', left_on='lookup2',right_on='lookup1')
#     new_df.drop(['lookup1_x', 'lookup2', 'lookup1_y'], axis = 1, inplace = True)
#     new_df.rename(columns = {points+'_x': points+'_after_race', points+'_y': points}, inplace = True)
#     new_df[points].fillna(0, inplace = True)
#     return new_df
  
# driver_standings = lookup(driver_standings, 'driver', 'driver_points')
# driver_standings = lookup(driver_standings, 'driver', 'driver_wins')
# driver_standings = lookup(driver_standings, 'driver', 'driver_standings_pos')

# driver_standings.drop(['driver_points_after_race', 'driver_wins_after_race', 'driver_standings_pos_after_race'], 
#                       axis = 1, inplace = True)

# constructor_rounds = rounds[:]

# constructor_standings = {'season': [],
#                     'round':[],
#                     'constructor': [],
#                     'constructor_points': [],
#                     'constructor_wins': [],
#                    'constructor_standings_pos': []}
# # query API

# print(constructor_rounds)

# for n in list(range(len(constructor_rounds))):
#     for i in constructor_rounds[n][1]:
    
#         url = 'https://ergast.com/api/f1/{}/{}/constructorStandings.json'
#         r = requests.get(url.format(constructor_rounds[n][0], i))
#         json = r.json()

#         print(f"url: https://ergast.com/api/f1/{constructor_rounds[n][0]}/{i}/constructorStandings.json")

#         for item in json['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']:
#             try:
#                 constructor_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
#             except:
#                 constructor_standings['season'].append(None)

#             try:
#                 constructor_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
#             except:
#                 constructor_standings['round'].append(None)
                                         
#             try:
#                 constructor_standings['constructor'].append(item['Constructor']['constructorId'])
#             except:
#                 constructor_standings['constructor'].append(None)
            
#             try:
#                 constructor_standings['constructor_points'].append(int(item['points']))
#             except:
#                 constructor_standings['constructor_points'].append(None)
            
#             try:
#                 constructor_standings['constructor_wins'].append(int(item['wins']))
#             except:
#                 constructor_standings['constructor_wins'].append(None)
                
#             try:
#                 constructor_standings['constructor_standings_pos'].append(int(item['position']))
#             except:
#                 constructor_standings['constructor_standings_pos'].append(None)
            
# constructor_standings = pd.DataFrame(constructor_standings)

# constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_points')
# constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_wins')
# constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_standings_pos')

# constructor_standings.drop(['constructor_points_after_race', 'constructor_wins_after_race','constructor_standings_pos_after_race' ],
#                            axis = 1, inplace = True)

# constructor_standings.to_csv("./data/constructor_standings.csv", sep=",", index=False)


# ==========
# Qualifying Results
# ==========

# import bs4
# from bs4 import BeautifulSoup

# qualifying_results = pd.DataFrame()

# # Qualifying times are only available from 1983

# for year in list(range(FROM_YEAR,THIS_YEAR)):
#     url = 'https://www.formula1.com/en/results.html/{}/races.html'
#     r = requests.get(url.format(year))
#     soup = BeautifulSoup(r.text, 'html.parser')
    
#     # find links to all circuits for a certain year
    
#     year_links = []
#     for page in soup.find_all('a', attrs = {'class':"resultsarchive-filter-item-link FilterTrigger"}):
#         link = page.get('href')
#         if f'/en/results.html/{year}/races/' in link: 
#             year_links.append(link)
    
#     # for each circuit, switch to the starting grid page and read table

#     year_df = pd.DataFrame()
#     new_url = 'https://www.formula1.com{}'
#     for n, link in list(enumerate(year_links)):
#         link = link.replace('race-result.html', 'starting-grid.html')
#         df = pd.read_html(new_url.format(link))
#         df = df[0]
#         df['season'] = year
#         df['round'] = n+1
#         for col in df:
#             if 'Unnamed' in col:
#                 df.drop(col, axis = 1, inplace = True)

#         year_df = pd.concat([year_df, df])

#     # concatenate all tables from all years  
        
#     qualifying_results = pd.concat([qualifying_results, year_df])

# # rename columns
    
# qualifying_results.rename(columns = {'Pos': 'grid', 'Driver': 'driver_name', 'Car': 'car',
#                                      'Time': 'qualifying_time'}, inplace = True)
# # drop driver number column

# qualifying_results.drop('No', axis = 1, inplace = True)

# qualifying_results.to_csv("./data/qualifying_results.csv", sep=",", index=False)


# ============
# Weather
# ============

# from selenium import webdriver

# weather = races.iloc[:,[0,1,2]]

# info = []

# # read wikipedia tables

# for link in races.url:
#     try:
#         df = pd.read_html(link)[0]
#         if 'Weather' in list(df.iloc[:,0]):
#             n = list(df.iloc[:,0]).index('Weather')
#             info.append(df.iloc[n,1])
#         else:
#             df = pd.read_html(link)[1]
#             if 'Weather' in list(df.iloc[:,0]):
#                 n = list(df.iloc[:,0]).index('Weather')
#                 info.append(df.iloc[n,1])
#             else:
#                 df = pd.read_html(link)[2]
#                 if 'Weather' in list(df.iloc[:,0]):
#                     n = list(df.iloc[:,0]).index('Weather')
#                     info.append(df.iloc[n,1])
#                 else:
#                     df = pd.read_html(link)[3]
#                     if 'Weather' in list(df.iloc[:,0]):
#                         n = list(df.iloc[:,0]).index('Weather')
#                         info.append(df.iloc[n,1])
#                     else:
#                         driver = webdriver.Chrome()
#                         driver.get(link)

#                         # click language button
#                         button = driver.find_element_by_link_text('Italiano')
#                         button.click()
                        
#                         # find weather in italian with selenium
                        
#                         clima = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div/table[1]/tbody/tr[9]/td').text
#                         info.append(clima) 
                                
#     except:
#         info.append('not found')

# # append column with weather information to dataframe  
  
# weather['weather'] = info

# # set up a dictionary to convert weather information into keywords

# weather_dict = {'weather_warm': ['soleggiato', 'clear', 'warm', 'hot', 'sunny', 'fine', 'mild', 'sereno'],
#                'weather_cold': ['cold', 'fresh', 'chilly', 'cool'],
#                'weather_dry': ['dry', 'asciutto'],
#                'weather_wet': ['showers', 'wet', 'rain', 'pioggia', 'damp', 'thunderstorms', 'rainy'],
#                'weather_cloudy': ['overcast', 'nuvoloso', 'clouds', 'cloudy', 'grey', 'coperto']}

# # map new df according to weather dictionary

# weather_df = pd.DataFrame(columns = weather_dict.keys())
# for col in weather_df:
#     weather_df[col] = weather['weather'].map(lambda x: 1 if any(i in weather_dict[col] for i in x.lower().split()) else 0)
   
# weather_info = pd.concat([weather, weather_df], axis = 1)

# weather.to_csv("./data/weather.csv", sep=',', index=False)
# weather_info.to_csv("./data/weather_info.csv", sep=',', index=False)

# weather.head(10)


# ============
# merge df
# ============

# df1 = pd.merge(races, weather, how='inner', 
#                on=['season', 'round', 'circuit_id']).drop(['lat', 'long','country','weather'], 
#                                                           axis = 1)
# # df2 = pd.merge(df1, results, how='inner', on=['season', 'round', 'circuit_id', 'url']).drop(['url','points', 'status', 'time'], axis = 1)
# df2 = pd.merge(df1, results, how='inner', on=['season', 'round', 'circuit_id']).drop(['points', 'status', 'time'], axis = 1)
# df3 = pd.merge(df2, driver_standings, how='left', 
#                on=['season', 'round', 'driver']) 
# df4 = pd.merge(df3, constructor_standings, how='left', 
#                on=['season', 'round', 'constructor']) #from 1958

# final_df = pd.merge(df4, qualifying_results, how='inner', 
#                     on=['season', 'round', 'grid']).drop(['driver_name', 'car'], 
#                                                          axis = 1) #from 1983

# final_df.to_csv("./data/final_df.csv", sep=",", index=False)


"""
# calculate age of drivers

from dateutil.relativedelta import *
final_df['date'] = pd.to_datetime(final_df.date)
final_df['date_of_birth'] = pd.to_datetime(final_df.date_of_birth)
final_df['driver_age'] = final_df.apply(lambda x: 
                                        relativedelta(x['date'], x['date_of_birth']).years, axis=1)
final_df.drop(['date', 'date_of_birth'], axis = 1, inplace = True)


# fill/drop nulls

for col in ['driver_points', 'driver_wins', 'driver_standings_pos', 'constructor_points', 
            'constructor_wins' , 'constructor_standings_pos']:
    final_df[col].fillna(0, inplace = True)
    final_df[col] = final_df[col].map(lambda x: int(x))
    
final_df.dropna(inplace = True )


# convert to boolean to save space

for col in ['weather_warm', 'weather_cold','weather_dry', 'weather_wet', 'weather_cloudy']:
    final_df[col] = final_df[col].map(lambda x: bool(x))


# calculate difference in qualifying times

final_df['qualifying_time'] = final_df.qualifying_time.map(lambda x: 0 if str(x) == '00.000' 
                             else(float(str(x).split(':')[1]) + 
                                  (60 * float(str(x).split(':')[0])) if x != 0 else 0))
final_df = final_df[final_df['qualifying_time'] != 0]
final_df.sort_values(['season', 'round', 'grid'], inplace = True)
final_df['qualifying_time_diff'] = final_df.groupby(['season', 'round']).qualifying_time.diff()
final_df['qualifying_time'] = final_df.groupby(['season', 
                                                'round']).qualifying_time_diff.cumsum().fillna(0)
final_df.drop('qualifying_time_diff', axis = 1, inplace = True)


# get dummies

df_dum = pd.get_dummies(final_df, columns = ['circuit_id', 'nationality', 'constructor'] )

for col in df_dum.columns:
    if 'nationality' in col and df_dum[col].sum() < 140:
        df_dum.drop(col, axis = 1, inplace = True)
        
    elif 'constructor' in col and df_dum[col].sum() < 140:
        df_dum.drop(col, axis = 1, inplace = True)
        
    elif 'circuit_id' in col and df_dum[col].sum() < 70:
        df_dum.drop(col, axis = 1, inplace = True)
    
    else:
        pass
      

# scoring function for regression

def score_regression(model):
    score = 0
    for circuit in df[df.season == 2019]['round'].unique():

        test = df[(df.season == 2019) & (df['round'] == circuit)]
        X_test = test.drop(['driver', 'podium'], axis = 1)
        y_test = test.podium

        #scaling
        X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

        # make predictions
        prediction_df = pd.DataFrame(model.predict(X_test), columns = ['results'])
        prediction_df['podium'] = y_test.reset_index(drop = True)
        prediction_df['actual'] = prediction_df.podium.map(lambda x: 1 if x == 1 else 0)
        prediction_df.sort_values('results', ascending = True, inplace = True)
        prediction_df.reset_index(inplace = True, drop = True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)

        score += precision_score(prediction_df.actual, prediction_df.predicted)

    model_score = score / df[df.season == 2019]['round'].unique().max()
    return model_score

# scoring function for classification

def score_classification(model):
    score = 0
    for circuit in df[df.season == 2019]['round'].unique():

        test = df[(df.season == 2019) & (df['round'] == circuit)]
        X_test = test.drop(['driver', 'podium'], axis = 1)
        y_test = test.podium

        #scaling
        X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

        # make predictions
        prediction_df = pd.DataFrame(model.predict_proba(X_test), columns = ['proba_0', 'proba_1'])
        prediction_df['actual'] = y_test.reset_index(drop = True)
        prediction_df.sort_values('proba_1', ascending = False, inplace = True)
        prediction_df.reset_index(inplace = True, drop = True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)

        score += precision_score(prediction_df.actual, prediction_df.predicted)

    model_score = score / df[df.season == 2019]['round'].unique().max()
    return model_score

df = data.copy()

#train split

train = df[df.season <2019]
X_train = train.drop(['driver', 'podium'], axis = 1)
y_train = train.podium

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)

  
# Linear Regression

params={'fit_intercept': ['True', 'False']}

for fit_intercept in params['fit_intercept']:
    model_params = (fit_intercept)
    model = LinearRegression(fit_intercept = fit_intercept)
    model.fit(X_train, y_train)
            
    model_score = score_regression(model)
            
    comparison_dict['model'].append('linear_regression')
    comparison_dict['params'].append(model_params)
    comparison_dict['score'].append(model_score)

    
# Random Forest Regressor

params={'criterion': ['mse'],
        'max_features': [0.8, 'auto', None],
        'max_depth': list(np.linspace(5, 55, 26)) + [None]}

for criterion in params['criterion']:
    for max_features in params['max_features']:
        for max_depth in params['max_depth']:
            model_params = (criterion, max_features, max_depth)
            model = RandomForestRegressor(criterion = criterion,
                                          max_features = max_features, max_depth = max_depth, random_state = 1)
            model.fit(X_train, y_train)
            
            model_score = score_regression(model)
            
            comparison_dict['model'].append('random_forest_regressor')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score)

            
# Support Vector Machines

params={'gamma': np.logspace(-4, -1, 10),
        'C': np.logspace(-2, 1, 10),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']} 

for gamma in params['gamma']:
    for c in params['C']:
        for kernel in params['kernel']:
            model_params = (gamma, c, kernel)
            model = svm.SVR(gamma = gamma, C = c, kernel = kernel)
            model.fit(X_train, y_train)
            
            model_score = score_regression(model)
            
            comparison_dict['model'].append('svm_regressor')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score)

            
# Neural network

params={'hidden_layer_sizes': [(80,20,40,5), (75,30,50,10,3)], 
        'activation': ['identity', 'relu','logistic', 'tanh',], 
        'solver': ['lbfgs','sgd', 'adam'], 
        'alpha': np.logspace(-4,1,20)} 

for hidden_layer_sizes in params['hidden_layer_sizes']:
    for activation in params['activation']:
        for solver in params['solver']:
            for alpha in params['alpha']:
                model_params = (hidden_layer_sizes, activation, solver, alpha )
                model = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes,
                                      activation = activation, solver = solver, alpha = alpha, random_state = 1)
                model.fit(X_train, y_train)

                model_score = score_regression(model)

                comparison_dict['model'].append('nn_regressor')
                comparison_dict['params'].append(model_params)
                comparison_dict['score'].append(model_score)
  
#print best models  
pd.DataFrame(comparison_dict).groupby('model')['score'].max()

df = data.copy()
df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)

#split train

train = df[df.season <2019]
X_train = train.drop(['driver', 'podium'], axis = 1)
y_train = train.podium

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)


# gridsearch dictionary

comparison_dict ={'model':[],
                  'params': [],
                  'score': []}

# Logistic Regression

params={'penalty': ['l1', 'l2'],
        'solver': ['saga', 'liblinear'],
        'C': np.logspace(-3,1,20)}

for penalty in params['penalty']:
    for solver in params['solver']:
        for c in params['C']:
            model_params = (penalty, solver, c)
            model = LogisticRegression(penalty = penalty, solver = solver, C = c, max_iter = 10000)
            model.fit(X_train, y_train)
            
            model_score = score_classification(model)
            
            comparison_dict['model'].append('logistic_regression')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score)

# Random Forest Classifier

params={'criterion': ['gini', 'entropy'],
        'max_features': [0.8, 'auto', None],
        'max_depth': list(np.linspace(5, 55, 26)) + [None]}

for criterion in params['criterion']:
    for max_features in params['max_features']:
        for max_depth in params['max_depth']:
            model_params = (criterion, max_features, max_depth)
            model = RandomForestClassifier(criterion = criterion, max_features = max_features, max_depth = max_depth)
            model.fit(X_train, y_train)
            
            model_score = score_classification(model)
            
            comparison_dict['model'].append('random_forest_classifier')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score)

# Support Vector Machines

params={'gamma': np.logspace(-4, -1, 20),
        'C': np.logspace(-2, 1, 20),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']} 

for gamma in params['gamma']:
    for c in params['C']:
        for kernel in params['kernel']:
            model_params = (gamma, c, kernel)
            model = svm.SVC(probability = True, gamma = gamma, C = c, kernel = kernel )
            model.fit(X_train, y_train)
            
            model_score = score_classification(model)
            
            comparison_dict['model'].append('svm_classifier')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score)

# Neural network

params={'hidden_layer_sizes': [(80,20,40,5), (75,25,50,10)], 
        'activation': ['identity', 'logistic', 'tanh', 'relu'], 
        'solver': ['lbfgs', 'sgd', 'adam', 'logistic'], 
        'alpha': np.logspace(-4,2,20)} 

for hidden_layer_sizes in params['hidden_layer_sizes']:
    for activation in params['activation']:
        for solver in params['solver']:
            for alpha in params['alpha']:
                model_params = (hidden_layer_sizes, activation, solver, alpha )
                model = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes,
                                      activation = activation, solver = solver, alpha = alpha, random_state = 1)
                model.fit(X_train, y_train)

                model_score = score_classification(model)

                comparison_dict['model'].append('neural_network_classifier')
                comparison_dict['params'].append(model_params)
                comparison_dict['score'].append(model_score)

"""