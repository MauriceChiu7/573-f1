import pandas as pd
import requests

# Initialize dictionary to hold race data
races = {
    'season': [], 'round': [], 'race_url': [], 'raceName': [], 'circuit_id': [],
    'circuit_url': [], 'circuitName': [], 'lat': [], 'long': [], 'locality': [],
    'country': [], 'date': [], 'time': [], 'fp1_date': [], 'fp1_time': [],
    'fp2_date': [], 'fp2_time': [], 'fp3_date': [], 'fp3_time': [],
    'qualifying_date': [], 'qualifying_time': []
}

# Iterate over years and collect race data
for year in range(1950, 2024):
    url = f'https://ergast.com/api/f1/{year}.json'
    r = requests.get(url)
    r.raise_for_status()  # Raise an error for HTTP errors
    json = r.json()

    for item in json.get('MRData', {}).get('RaceTable', {}).get('Races', []):
        races['season'].append(item.get('season'))
        races['round'].append(item.get('round'))
        races['race_url'].append(item.get('url'))
        races['raceName'].append(item.get('raceName'))
        
        circuit = item.get('Circuit', {})
        races['circuit_id'].append(circuit.get('circuitId'))
        races['circuit_url'].append(circuit.get('url'))
        races['circuitName'].append(circuit.get('circuitName'))
        
        location = circuit.get('Location', {})
        races['lat'].append(location.get('lat'))
        races['long'].append(location.get('long'))
        races['locality'].append(location.get('locality'))
        races['country'].append(location.get('country'))
        
        races['date'].append(item.get('date'))
        races['time'].append(item.get('time'))
        
        # Practice and Qualifying
        races['fp1_date'].append(item.get('FirstPractice', {}).get('date'))
        races['fp1_time'].append(item.get('FirstPractice', {}).get('time'))
        
        races['fp2_date'].append(item.get('SecondPractice', {}).get('date'))
        races['fp2_time'].append(item.get('SecondPractice', {}).get('time'))
        
        races['fp3_date'].append(item.get('ThirdPractice', {}).get('date'))
        races['fp3_time'].append(item.get('ThirdPractice', {}).get('time'))
        
        races['qualifying_date'].append(item.get('Qualifying', {}).get('date'))
        races['qualifying_time'].append(item.get('Qualifying', {}).get('time'))

# Create DataFrame and save to CSV
races_df = pd.DataFrame(races)
races_df.to_csv("./races.csv", sep=',', index=False)
