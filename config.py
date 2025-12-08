# config.py

CONFIG = {
    'FL': {
        'state': 'FL',
        'outage_files': [
            "/Users/ryanchen/Downloads/data/FL/florida_fpl_sept_oct_2022_hourly_county.csv",
            "/Users/ryanchen/Downloads/data/FL/florida_duke_sept_oct_2022_hourly_county.csv"
        ],
        'census_file': "/Users/ryanchen/Downloads/data/FL/florida_census_data_2021.csv",
        'weather_folder': "/Users/ryanchen/Downloads/20220901-20221011_Daily",
        'weather_file_pattern': '*_Daily.csv',
        'start_date': "2022-09-25",
        'end_date': "2022-10-14",
        'county_total_customer': 1000,
        'county_count_threshold': 50,
        'outage_start_threshold': 0.02,
        'storm_periods': None
    },
    'MA': {
        'state': 'MA',
        'start_date': "2018-03-01",
        'end_date': "2018-03-15",
        'outage_files': [
            "/Users/ryanchen/Downloads/data/MA/ma_jan_march_2018_hourly_county.csv"
        ],
        'census_file': "/Users/ryanchen/Downloads/data/MA/Massachusets Census Data - 2018.csv",  # no census data for MA
        'weather_folder': "/Users/ryanchen/Downloads/data/MA/weather",
        'weather_file_pattern': '*_201803.csv',
        'storm_periods': {
            'first': {"start_date": "2018-03-01", "end_date": "2018-03-07"},
            'second': {"start_date": "2018-03-07", "end_date": "2018-03-14"},
            'third': {"start_date": "2018-03-13", "end_date": "2018-03-15"}
        },
        'county_total_customer': 1000,
        'county_count_threshold': 50,
        'outage_start_threshold': 0.01
    }
}
