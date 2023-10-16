import cfbd
from cfbd.rest import ApiException
import datetime
import json
import csv

# Configure API key authorization: ApiKeyAuth
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = 'Bearer jV0Q0EwcJ9CPh/tsccoErh75xJVcvtRl8sqE3xxmMRimM1i7BL9LkkHyeeCy933S'

# Create an instance of the API class
api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))

# Calculate the current year
current_year = datetime.datetime.now().year

# Calculate the year 10 years ago
ten_years_ago = current_year - 10

try:
    # Retrieve games data for the last 10 years
    api_response = api_instance.get_games(year=ten_years_ago, season_type='regular')

    # Initialize an empty list to store the extracted data
    games_data = []

    # Iterate through the Game objects and extract relevant data
    for game in api_response:
        if game.season_type == 'regular':
            game_dict = {
                'id': game.id,
                'season': game.season,
                'start_date': game.start_date,
                'home_team': game.home_team,
                'away_team': game.away_team,
                'home_points': game.home_points,
                'away_points': game.away_points,
                'neutral_site': game.neutral_site,
            }
            games_data.append(game_dict)

    # Define the JSON file path to store the data
    json_file_path = 'games_data.json'

    # Define the CSV file path to store the data
    csv_file_path = 'games_data.csv'

    # Write the data to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(games_data, json_file, indent=4)

    print(f'Data has been stored in {json_file_path}')

    # Write the data to a CSV file
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = games_data[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(games_data)

    print(f'Data has been stored in {csv_file_path}')

except ApiException as e:
    print("Exception when calling GamesApi->get_games: %s\n" % e)
