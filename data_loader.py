# data_loader.py
import pandas as pd
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
from pandas import date_range

def load_outage_data(file_paths: list, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load and combine outage data from multiple CSV files with optional date filtering.
    """
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        # Keep only desired columns and convert county names to title case
        df = df[['date_', 'hour_', 'county', 'total_outage', 'total_customer']]
        df['county'] = df['county'].str.title()
        if start_date and end_date:
            df = df[(df['date_'] >= start_date) & (df['date_'] <= end_date)]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df.groupby(['date_', 'hour_', 'county'], as_index=False).sum()

def load_census_data(census_path: str) -> pd.DataFrame:
    """
    Load census data and process it into a DataFrame.
    """
    census_data = pd.read_csv(census_path)
    census_data['County'] = census_data['County'].str.title()
    return census_data

def load_weather_data(weather_folder: str,
                      start_date: str = None,
                      end_date: str = None,
                      state: str = "FL",
                      pattern: str = None) -> dict:
    """
    Load weather data dynamically based on available files, with date filtering.
    
    Args:
        weather_folder (str): Path to the folder containing weather data CSVs.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format.
        state (str, optional): State name ("FL" or others). Defaults to "FL".
        pattern (str, optional): Glob file pattern to match weather files.
            This must be provided via the config.
            
    Returns:
        dict: Dictionary mapping county names to weather DataFrames.
    """
    if pattern is None:
        raise ValueError("weather_file_pattern must be provided in the configuration.")
    file_pattern = f'{weather_folder}/{pattern}'
    weather_files = glob.glob(file_pattern)
    weather_data = {}
    
    for path in weather_files:
        county_name = path.split('/')[-1].split('_')[0].title()
        df = pd.read_csv(path, delimiter=None, engine='python')
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        date_col = next((col for col in df.columns 
                         if 'date' in col.lower() or 'datetime' in col.lower()), None)
        if date_col is None:
            print(f"Warning: No valid date column found in {path}. Skipping file.")
            continue
        df.rename(columns={date_col: 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        if 'unknown' in df.columns:
            df.drop(columns=['unknown'], inplace=True)
        numeric_cols = df.columns.difference(['County', 'date'])
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        weather_data[county_name] = df
    return weather_data

def filter_counties(df: pd.DataFrame, min_occurrences: int = 50) -> pd.DataFrame:
    """
    Filter counties based on minimum occurrences in the dataset.
    """
    county_counts = df['county'].value_counts()
    counties_to_keep = county_counts[county_counts >= min_occurrences].index
    return df[df['county'].isin(counties_to_keep)].reset_index(drop=True)

def calculate_total_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total customers per county.
    """
    return df.groupby('county')['total_customer'].max().reset_index()

def aggregate_outage_data(df_list: list) -> pd.DataFrame:
    """
    Aggregate outage data from multiple sources.
    """
    combined_df = pd.concat(df_list)
    return combined_df.groupby(['date_', 'hour_'], as_index=False)['total_outage'].sum()

def plot_sum_of_outages_over_time(filtered_data: pd.DataFrame, save_path: str = "") -> None:
    """
    Plots the sum of outages across all counties at each time step.
    """
    filtered_data = filtered_data.copy()
    filtered_data['time_step'] = filtered_data.groupby('county').cumcount() + 1
    outage_sum_over_time = filtered_data.groupby('time_step')['total_outage'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(outage_sum_over_time['time_step'], outage_sum_over_time['total_outage'], linestyle='-')
    plt.title("Sum of Total Outages Across All Counties at Each Time Step")
    plt.xlabel("Time Step (Starting from 1)")
    plt.ylabel("Sum of Total Outages")
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_outage_trends(df: pd.DataFrame, title: str = "Total Outages Over Time", save_path: str = "plots/") -> None:
    """
    Plots aggregated outage data over time.
    """
    df['datetime'] = pd.to_datetime(df['date_']) + pd.to_timedelta(df['hour_'], unit='h')
    plt.figure(figsize=(14, 7))
    plt.plot(df['datetime'], df['total_outage'], label='Total Outages', linestyle='-')
    plt.title(title)
    plt.xlabel('Date and Hour')
    plt.ylabel('Total Outages')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_county_level_outages(df: pd.DataFrame, title: str = "County-Level Outage Data", save_path: str = "plots/") -> None:
    """
    Plots county-level outage data in a grid of subplots.
    """
    df['datetime'] = pd.to_datetime(df['date_']) + pd.to_timedelta(df['hour_'], unit='h')
    counties = df['county'].unique()
    num_counties = len(counties)
    cols = 5
    rows = math.ceil(num_counties / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()
    for i, county in enumerate(counties):
        county_data = df[df['county'] == county]
        axes[i].plot(county_data['datetime'], county_data['total_outage'], linestyle='-')
        axes[i].set_title(county)
        axes[i].set_xlabel('Date and Hour')
        axes[i].set_ylabel('Total Outages')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

def fill_missing_hours(df: pd.DataFrame,
                       date_col: str = 'date_',
                       hour_col: str = 'hour_',
                       value_col: str = 'total_outage') -> pd.DataFrame:
    """
    Fills missing hourly data within each county using linear interpolation.
    Preserves 'total_customer' if present by forward/backward fill.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[hour_col], unit='h')

    out = []
    for county, county_data in df.groupby('county', sort=False):
        # Full hourly span for this county
        full_dt = date_range(county_data['datetime'].min(),
                             county_data['datetime'].max(),
                             freq='h')  # <- use 'h' (lowercase)

        # Reindex on datetime
        county_data = county_data.set_index('datetime').sort_index()
        county_data = county_data.reindex(full_dt)

        # Keep county column and any static columns
        county_data['county'] = county

        # Interpolate outages
        county_data[value_col] = pd.to_numeric(county_data[value_col], errors='coerce')
        county_data[value_col] = county_data[value_col].interpolate(method='linear', limit_direction='both')

        # Preserve total_customer if present (carry forward/back)
        if 'total_customer' in county_data.columns:
            county_data['total_customer'] = pd.to_numeric(county_data['total_customer'], errors='coerce')
            county_data['total_customer'] = county_data['total_customer'].ffill().bfill()

        # Restore columns
        county_data = county_data.reset_index().rename(columns={'index': 'datetime'})
        county_data[date_col] = county_data['datetime'].dt.date.astype(str)
        county_data[hour_col] = county_data['datetime'].dt.hour

        out.append(county_data[[date_col, hour_col, 'county', value_col] +
                               (['total_customer'] if 'total_customer' in county_data.columns else [])])

    return pd.concat(out, ignore_index=True)

def preprocess_outage_data(file_paths: list,
                           start_date: str = None,
                           end_date: str = None,
                           min_occurrences: int = 50) -> pd.DataFrame:
    """
    Load, filter, and interpolate outage data.
    """
    df = load_outage_data(file_paths, start_date, end_date)
    df = filter_counties(df, min_occurrences)
    df = fill_missing_hours(df)
    return df


# ============================================================
# Outage Processing Pipeline
# ============================================================
class OutageProcessor:
    def __init__(self,
                 state: str,
                 outage_files: list,
                 start_date: str,
                 end_date: str,
                 county_total_customer: int,
                 county_count_threshold: int,
                 outage_start_threshold: float) -> None:
        self.state: str = state
        self.outage_files: list = outage_files
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.county_total_customer: int = county_total_customer
        self.county_count_threshold: int = county_count_threshold
        self.outage_start_threshold: float = outage_start_threshold
        
        self.processed_outage_data: list = []
        self.combined_outage_data: pd.DataFrame = pd.DataFrame()
        self.aligned_outage_data: pd.DataFrame = pd.DataFrame()
        self.filtered_outage_data: pd.DataFrame = pd.DataFrame()
        self.total_customer_dict: dict = {}

    def _log(self, message: str) -> None:
        print(message)

    def process_outages(self) -> None:
        self._log("\nProcessing outage data for all providers...")
        self.processed_outage_data = [
            preprocess_outage_data([path], self.start_date, self.end_date)
            for path in self.outage_files
        ]
        for i, data in enumerate(self.processed_outage_data):
            plot_county_level_outages(
                data,
                title=f"County-Level Outages (Provider {i + 1}, {self.state}) | Date: {self.start_date} to {self.end_date}",
                save_path="plots/process_outages.png"
            )

    def aggregate_outages(self) -> None:
        self._log("\nAggregating outage data across all providers...")

        dfc = pd.concat(self.processed_outage_data, ignore_index=True)

        group_cols = ['date_', 'hour_', 'county']
        agg_map = {}

        if 'total_outage' in dfc.columns:
            agg_map['total_outage'] = 'sum'
        if 'total_customer' in dfc.columns:
            # total customer is static for a county; keep the max observed
            agg_map['total_customer'] = 'max'

        # Fallback: if nothing to aggregate, just keep distinct rows
        if not agg_map:
            self.combined_outage_data = dfc[group_cols].drop_duplicates().copy()
        else:
            self.combined_outage_data = (
                dfc.groupby(group_cols, as_index=False)
                .agg(agg_map)
            )

        # Create datetime AFTER aggregation (avoid summing datetimes)
        self.combined_outage_data['datetime'] = (
            pd.to_datetime(self.combined_outage_data['date_']) +
            pd.to_timedelta(self.combined_outage_data['hour_'], unit='h')
        )

        plot_county_level_outages(
            self.combined_outage_data,
            title=f"County-Level Outages (Aggregated, {self.state}) | Date: {self.start_date} to {self.end_date}",
            save_path="plots/aggregate_outages.png"
        )


    def compute_total_customers(self) -> None:
        self._log("\nComputing total customers per county...")
        total_customer_df = self.combined_outage_data.groupby('county')['total_customer'].max().reset_index()
        self.combined_outage_data = self.combined_outage_data.merge(
            total_customer_df, on='county', suffixes=('', '_max')
        )
        self.total_customer_dict = total_customer_df.set_index('county')['total_customer'].to_dict()
        self._log("Total customers per county computed.")

    def filter_outage_threshold(self) -> None:
        self._log("\nFiltering outages exceeding {:.0f}% of total customers...".format(self.outage_start_threshold * 100))
        self.aligned_outage_data = self.combined_outage_data[
            self.combined_outage_data['total_outage'] >= self.outage_start_threshold * self.combined_outage_data['total_customer_max']
        ].copy()
        self._log(f"Number of counties remained: {len(self.aligned_outage_data['county'].unique())}")

        first_threshold_exceedance = self.aligned_outage_data.groupby('county')['datetime'].min().reset_index()
        first_threshold_exceedance.columns = ['county', 'start_datetime']
        self.aligned_outage_data = self.aligned_outage_data.merge(first_threshold_exceedance, on='county')
        self.aligned_outage_data = self.aligned_outage_data[
            self.aligned_outage_data['datetime'] >= self.aligned_outage_data['start_datetime']
        ]
        self.aligned_outage_data.drop(columns=['start_datetime'], inplace=True)
        plot_county_level_outages(
            self.aligned_outage_data,
            title=f"Outages Exceeding {self.outage_start_threshold * 100}% Customers ({self.state})",
            save_path="plots/filter_outage_threshold.png"
        )

    def refine_filtered_data(self) -> None:
        self._log("\nRemoving counties with fewer than {} total customers...".format(self.county_total_customer))
        self.aligned_outage_data = self.aligned_outage_data[
            self.aligned_outage_data['county'].isin(
                self.combined_outage_data.county[self.combined_outage_data.total_customer >= self.county_total_customer]
            )
        ]
        self._log(f"Remaining counties: {len(self.aligned_outage_data['county'].unique())}")
        plot_county_level_outages(
            self.aligned_outage_data,
            title=f"Outages for Counties with More than {self.county_total_customer} Customers ({self.state})"
        )

    def final_filter(self) -> None:
        self._log("\nFiltering counties with at least {} outage occurrences...".format(self.county_count_threshold))
        county_counts = self.aligned_outage_data['county'].value_counts()
        counties_to_keep = county_counts[county_counts >= self.county_count_threshold].index
        self.filtered_outage_data = self.aligned_outage_data[
            self.aligned_outage_data['county'].isin(counties_to_keep)
        ].reset_index(drop=True)
        self._log(f"Final number of counties retained: {len(self.filtered_outage_data['county'].unique())}")
        plot_county_level_outages(
            self.filtered_outage_data,
            title=f"Outages with Frequent Reports (At Least {self.county_count_threshold} Occurrences) ({self.state})"
        )

    def run(self) -> pd.DataFrame:
        self.process_outages()
        self.aggregate_outages()
        self.compute_total_customers()
        self.filter_outage_threshold()
        self.refine_filtered_data()
        self.final_filter()
        self._log("\nProcessing complete.")
        return self.filtered_outage_data[['datetime', 'hour_', 'county', 'total_outage']]


def align_by_threshold(data: pd.DataFrame, total_customer_dict: dict, outage_start_threshold: float) -> pd.DataFrame:
    """
    For each county in the provided data, find the first datetime when the county's outage 
    exceeds its threshold (computed as outage_start_threshold * total_customer). Then, 
    re-align the county's time series so that this event is time step 1.
    Counties with no threshold event are skipped.
    """
    aligned_list = []
    for county in data['county'].unique():
        county_data = data[data['county'] == county].copy()
        threshold = outage_start_threshold * total_customer_dict[county]
        county_data_threshold = county_data[county_data['total_outage'] >= threshold]
        if county_data_threshold.empty:
            continue
        first_threshold_time = county_data_threshold['datetime'].min()
        county_data = county_data[county_data['datetime'] >= first_threshold_time].copy()
        county_data = county_data.sort_values('datetime')
        county_data['time_step'] = range(1, len(county_data) + 1)
        aligned_list.append(county_data)
    if aligned_list:
        return pd.concat(aligned_list, ignore_index=True)
    else:
        return pd.DataFrame()
