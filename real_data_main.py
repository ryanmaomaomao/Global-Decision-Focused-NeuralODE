"""
Cleaned implementation of SIR, RNN, and LSTM models with Decision-Focused Learning (DFL)
for outage prediction and SAIDI evaluation.

This script provides:
1. Clean, organized code structure
2. SIR ODE models (base and weather-enhanced) with MSE and DFL training
3. RNN and LSTM baseline models with MSE and DFL training
4. Comprehensive evaluation with MSE and SAIDI metrics
5. Model comparison and visualization
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from sklearn.model_selection import train_test_split
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB
import os
import time
from datetime import datetime

from data_loader import *
from utils import set_random_seed, calc_SAIDI
from models import BaseSIRModel, SIRWeatherModel, RNNModel, LSTMModel, MultiCityRNNModel, MultiCityLSTMModel
from config import CONFIG

# ============================================================
# Gurobi Optimization Functions for SAIDI Evaluation
# ============================================================

def select_cities_for_hardening(predictions, customer, max_selected_cities, label=""):
    """
    Select cities for hardening based on predictions using Gurobi knapsack formulation.
    
    Args:
        predictions (np.array): 2D array of predicted outages (num_cities x num_time_steps)
        customer (list or np.array): Customer counts for each city
        max_selected_cities (int): Maximum number of cities to select for hardening
        label (str): Label for logging purposes
        
    Returns:
        list: Indices of cities selected for hardening
    """
    # Calculate SAIDI for each city based on predictions
    saidi_values = calc_SAIDI(np.array(customer), predictions)
    
    num_cities = len(saidi_values)
    
    # Set up the Gurobi model
    gurobi_model = gp.Model("City_Selection_Knapsack")
    gurobi_model.setParam('OutputFlag', 0)  # Suppress Gurobi output
    
    # Create binary decision variables for each city
    x = gurobi_model.addVars(num_cities, vtype=GRB.BINARY, name="x")
    
    # Set objective: maximize the total SAIDI in the selected cities
    gurobi_model.setObjective(gp.quicksum(saidi_values[i] * x[i] for i in range(num_cities)), GRB.MAXIMIZE)
    
    # Constraint: select at most max_selected_cities cities
    gurobi_model.addConstr(gp.quicksum(x[i] for i in range(num_cities)) <= max_selected_cities)
    
    # Optimize
    gurobi_model.optimize()
    
    if gurobi_model.status == GRB.OPTIMAL:
        selected_cities = [i for i in range(num_cities) if x[i].x > 0.5]
        if label:
            print(f"--- {label} City Selection ---")
            print(f"Selected cities (indices): {selected_cities}")
            print(f"Selected SAIDI values: {[saidi_values[i] for i in selected_cities]}")
            print(f"Total selected SAIDI: {sum([saidi_values[i] for i in selected_cities]):.4f}")
        return selected_cities
    else:
        print(f"No optimal solution found for {label}")
        return []

def evaluate_saidi_hardened(selected_cities, customer, true_outages, label=""):
    """
    Evaluate SAIDI based on hardened selection using ground truth outages.
    
    Args:
        selected_cities (list): Indices of selected cities
        customer (list or np.array): Customer counts for each city
        true_outages (np.array): 2D array of true outages (num_cities x num_time_steps)
        label (str): Label for logging purposes
        
    Returns:
        dict: Dictionary containing SAIDI evaluation results
    """
    # Calculate SAIDI for all cities using ground truth
    all_saidi = calc_SAIDI(np.array(customer), true_outages)
    
    # Calculate SAIDI for selected cities only
    selected_saidi = [all_saidi[i] for i in selected_cities]
    selected_customers = [customer[i] for i in selected_cities]
    
    # Calculate remaining SAIDI (total - selected)
    total_saidi = np.sum(all_saidi)
    selected_total_saidi = np.sum(selected_saidi)
    remaining_saidi = total_saidi - selected_total_saidi
    
    results = {
        'selected_cities': selected_cities,
        'selected_saidi': selected_saidi,
        'selected_customers': selected_customers,
        'total_saidi': total_saidi,
        'selected_total_saidi': selected_total_saidi,
        'remaining_saidi': remaining_saidi
    }
    
    if label:
        print(f"--- {label} SAIDI Evaluation ---")
        print(f"Selected cities: {selected_cities}")
        print(f"Selected SAIDI values: {[f'{s:.4f}' for s in selected_saidi]}")
        print(f"Selected customers: {selected_customers}")
        print(f"Total SAIDI: {total_saidi:.4f}")
        print(f"Selected total SAIDI: {selected_total_saidi:.4f}")
        print(f"Remaining SAIDI: {remaining_saidi:.4f}")
    
    return results

# ============================================================
# Helper Functions for Evaluation
# ============================================================

def calculate_mse(predictions, targets):
    """Calculate Mean Squared Error between predictions and targets."""
    return np.mean((predictions - targets) ** 2)

def calculate_saidi_metrics(customer, outage):
    """Calculate SAIDI metrics for each city."""
    return calc_SAIDI(np.array(customer), outage)

# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser(description='Train and evaluate SIR, RNN, and LSTM models with DFL')
parser.add_argument('--model', type=str, choices=['sir-ode', 'sir-weather-ode'], default='sir-weather-ode',
                    help="Select the SIR model: 'sir-ode' (base) or 'sir-weather-ode' (with weather data)")
parser.add_argument('--state', type=str, choices=['MA', 'FL'], default='MA',
                    help="Select the state configuration: 'MA' or 'FL'")
parser.add_argument('--seed', type=int, default=42,
                    help="Set the random seed for reproducibility (default: 42)")
parser.add_argument('--num_MSE_epochs', type=int, default=200,
                    help="Number of epochs for MSE training (default: 3000)")
parser.add_argument('--num_gdf_epochs', type=int, default=100,
                    help="Number of epochs for decision-focused learning (default: 1000)")
parser.add_argument('--lambda_smoothing', type=float, default=0.1,
                    help="Smoothing parameter for CVX layer (default: 0.1)")
parser.add_argument('--lambda_mse', type=float, default=0.01,
                    help="Weight for MSE loss in DFL (default: 0.01)")
parser.add_argument('--max_selected_cities', type=int, default=6,
                    help="Maximum number of cities to select in optimization (default: 6)")
parser.add_argument('--seq_len', type=int, default=24,
                    help="Sequence length for RNN/LSTM models (default: 24)")
parser.add_argument('--hidden_dim', type=int, default=64,
                    help="Hidden dimension for RNN/LSTM models (default: 64)")
parser.add_argument('--num_layers', type=int, default=2,
                    help="Number of layers for RNN/LSTM models (default: 2)")
parser.add_argument('--random_seed', type=int, default=42,
                    help="Random seed for torch and numpy reproducibility (default: 42)")
parser.add_argument('--num_subsampled_cities', type=int, default=None,
                    help="Number of randomly subsampled cities to use (default: None, uses all cities)")
parser.add_argument('--sir_pretrained_path', type=str, default='weights/SIR_MA_sir-weather-ode_1000epochs_MSE_20250916_214002.pth',
                    help="Path to pretrained SIR model weights (if provided, skip MSE training)")
parser.add_argument('--rnn_pretrained_path', type=str, default='weights/RNN_MA_multi-city_1000epochs_MSE_20250916_232352.pth',
                    help="Path to pretrained RNN model weights (if provided, skip MSE training)")
parser.add_argument('--lstm_pretrained_path', type=str, default='weights/LSTM_MA_multi-city_1000epochs_MSE_20250916_232352.pth',
                    help="Path to pretrained LSTM model weights (if provided, skip MSE training)")

args = parser.parse_args()

# ============================================================
# Configuration and setup
# ============================================================
def setup_configuration():
    """Setup configuration based on command line arguments."""
    state_config = CONFIG[args.state]
    set_random_seed(args.seed)
    
    return {
        'state': state_config['state'],
        'outage_files': state_config['outage_files'],
        'start_date': state_config['start_date'],
        'end_date': state_config['end_date'],
        'county_total_customer': state_config['county_total_customer'],
        'county_count_threshold': state_config['county_count_threshold'],
        'outage_start_threshold': state_config['outage_start_threshold'],
        'census_file': state_config.get('census_file'),
        'weather_folder': state_config['weather_folder'],
        'weather_file_pattern': state_config.get('weather_file_pattern'),
        'storm_periods': state_config.get('storm_periods')
    }

# ============================================================
# Data loading and preprocessing
# ============================================================
def load_and_preprocess_data(config):
    """Load and preprocess outage, census, and weather data."""
    print("Loading and preprocessing data...")
    
    # Process outage data
    processor = OutageProcessor(
        config['state'], 
        config['outage_files'], 
        config['start_date'], 
        config['end_date'], 
        config['county_total_customer'], 
        config['county_count_threshold'], 
        config['outage_start_threshold']
    )
    final_outage_data = processor.run()
    total_customer_dict = processor.total_customer_dict
    
    # Load census data if available
    census_data = None
    if config['census_file']:
        census_data = load_census_data(config['census_file'])
    
    # Load weather data
    weather_data = load_weather_data(
        config['weather_folder'], 
        config['start_date'], 
        config['end_date'], 
        state=config['state'], 
        pattern=config['weather_file_pattern']
    )
    
    # Align data by county
    final_outage_data = final_outage_data.copy()
    final_outage_data['county'] = final_outage_data['county'].str.title()
    if census_data is not None:
        census_counties = census_data['County'].str.title().unique()
        final_outage_data = final_outage_data[final_outage_data['county'].isin(census_counties)]
    
    weather_counties = [county.title() for county in weather_data.keys()]
    final_outage_data = final_outage_data[final_outage_data['county'].isin(weather_counties)]
    
    # Align data by padding
    max_length = final_outage_data['county'].value_counts().max()
    aligned_data = []
    for county in final_outage_data['county'].unique():
        county_data = final_outage_data[final_outage_data['county'] == county].copy()
        first_outage_time = county_data['datetime'].min()
        full_time_range = pd.date_range(start=first_outage_time, periods=max_length, freq='h')
        county_data = county_data.set_index('datetime').reindex(full_time_range, fill_value=0)
        county_data = county_data.reset_index().rename(columns={'index': 'datetime'})
        county_data['county'] = county
        county_data['hour_'] = county_data['datetime'].dt.hour
        aligned_data.append(county_data)
    
    filtered_aligned_outage_data_padded = pd.concat(aligned_data, ignore_index=True)
    
    return {
        'outage_data': filtered_aligned_outage_data_padded,
        'total_customer_dict': total_customer_dict,
        'census_data': census_data,
        'weather_data': weather_data
    }

# ============================================================
# Train/Test split
# ============================================================
def split_train_test(data, total_customer_dict, config):
    """Split data into train and test sets based on configuration."""
    if args.state == "MA" and config['storm_periods']:
        # Use storm periods for MA
        storm_train = config['storm_periods']['first']
        storm_test = config['storm_periods']['second']
        train_start = pd.to_datetime(storm_train['start_date'])
        train_end = pd.to_datetime(storm_train['end_date'])
        test_start = pd.to_datetime(storm_test['start_date'])
        test_end = pd.to_datetime(storm_test['end_date'])
        
        train_data_raw = data[
            (data['datetime'] >= train_start) & 
            (data['datetime'] <= train_end)
        ]
        test_data_raw = data[
            (data['datetime'] >= test_start) & 
            (data['datetime'] <= test_end)
        ]
        
        # Align by threshold
        train_data = align_by_threshold(train_data_raw, total_customer_dict, config['outage_start_threshold'])
        test_data = align_by_threshold(test_data_raw, total_customer_dict, config['outage_start_threshold'])
        
        train_counties = train_data['county'].unique()
        test_counties = test_data['county'].unique()
    else:
        # Random split for other states
        unique_counties = data['county'].unique()
        train_counties, test_counties = train_test_split(unique_counties, test_size=0.3, random_state=args.seed)
        train_data = data[data['county'].isin(train_counties)]
        test_data = data[data['county'].isin(test_counties)]
    
    return train_data, test_data, train_counties, test_counties

def align_by_threshold(data, total_customer_dict, outage_start_threshold):
    """Align data by threshold crossing events."""
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
    return pd.concat(aligned_list, ignore_index=True) if aligned_list else pd.DataFrame()

# ============================================================
# Evaluation functions
# ============================================================
def calculate_mse(predictions, targets):
    """Calculate Mean Squared Error."""
    return np.mean((predictions - targets) ** 2)

def calculate_saidi_metrics(customer_counts, outage_data):
    """Calculate SAIDI metrics for given data."""
    if isinstance(customer_counts, list):
        customer_counts = np.array(customer_counts)
    if isinstance(outage_data, torch.Tensor):
        outage_data = outage_data.detach().cpu().numpy()
    
    return calc_SAIDI(customer_counts, outage_data)

def create_outage_array_from_df(data, counties, max_time_points):
    """Create outage array from DataFrame."""
    data = data.copy()
    if 'time_step' not in data.columns:
        data['time_step'] = data.groupby('county').cumcount() + 1
    
    outage_array = np.zeros((len(counties), max_time_points))
    for i, county in enumerate(counties):
        county_data = data[data['county'] == county].copy()
        county_data = county_data.sort_values(by='time_step')
        county_outages = county_data['total_outage'].values
        num = min(len(county_outages), max_time_points)
        outage_array[i, :num] = county_outages[:num]
    return outage_array

# ============================================================
# Decision-Focused Learning (DFL) functions
# ============================================================
def create_cvxpy_layer(lambda_smoothing, max_selected_cities, n_cities):
    """Create CVXPY layer for decision-focused optimization."""
    x = cp.Variable(n_cities)
    t = cp.Variable()
    bl_SAIDI_param = cp.Parameter(n_cities)
    
    saidi_cost = cp.matmul(bl_SAIDI_param, x)
    objective = cp.Maximize(saidi_cost - lambda_smoothing * t)
    
    constraints = [
        cp.sum(x) <= max_selected_cities,
        x >= 0,
        x <= 1,
        cp.sum_squares(x) <= t
    ]
    
    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[bl_SAIDI_param], variables=[x, t])
    return cvxpylayer

def calc_SAIDI_tensor(customer, outage):
    """Calculate SAIDI using tensor operations."""
    if not isinstance(customer, torch.Tensor):
        customer = torch.tensor(customer, dtype=torch.float32)
    if not isinstance(outage, torch.Tensor):
        outage = torch.tensor(outage, dtype=torch.float32)
    
    total_outages = torch.sum(outage, dim=1)
    SAIDI = total_outages / customer
    return SAIDI

def train_decision_focused_sir(model, cvxpylayer, optimizer, customer, outage, 
                              initial_conditions, counties, time_points, 
                              lambda_mse=0.01, num_epochs=1000, model_option='sir-ode',
                              census_data_dict=None, weather_data_dict=None,
                              demographic_dim=0, weather_dim=0):
    """Train SIR model using decision-focused learning."""
    decision_losses = []
    mse_losses = []
    
    print(f"Starting SIR DFL training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        y0 = torch.tensor([initial_conditions[county] for county in counties], dtype=torch.float32)
        N_values = y0.sum(dim=1)
        model.set_population(N_values)
        
        if model_option == 'sir-weather-ode' and census_data_dict and weather_data_dict:
            batch_demographic = []
            batch_weather = []
            for county in counties:
                if county in census_data_dict:
                    batch_demographic.append(census_data_dict[county])
                else:
                    batch_demographic.append(torch.zeros(demographic_dim))
                if county in weather_data_dict:
                    batch_weather.append(weather_data_dict[county])
                else:
                    batch_weather.append(torch.zeros(weather_dim))
            batch_demographic = torch.stack(batch_demographic, dim=0)
            batch_weather = torch.stack(batch_weather, dim=0)
            model.set_demographic(batch_demographic)
            model.set_weather(batch_weather)
        # elif model_option == 'sir-weather-ode':
        #     # If weather model but no data provided, create dummy data
        #     batch_demographic = torch.zeros(len(counties), demographic_dim)
        #     batch_weather = torch.zeros(len(counties), weather_dim)
        #     model.set_demographic(batch_demographic)
        #     model.set_weather(batch_weather)
        
        y_pred = odeint(model, y0, time_points, method='rk4')
        predicted_outages = y_pred[:, :, 1].T
        
        # Calculate SAIDI for decision making
        bl_SAIDI = calc_SAIDI_tensor(customer, predicted_outages)
        
        # Decision layer
        x_opt_pred, _ = cvxpylayer(bl_SAIDI, solver_args={"solve_method": "ECOS"})
        
        # Calculate decision loss using true SAIDI
        bl_SAIDI_true = torch.tensor(calc_SAIDI(customer, outage), dtype=torch.float32)
        x_opt_true, _ = cvxpylayer(bl_SAIDI_true)
        
        true_obj_opt = 0
        true_obj_pred = torch.matmul(bl_SAIDI_true, x_opt_pred)
        decision_loss = true_obj_opt - true_obj_pred
        decision_losses.append(decision_loss.item())
        
        # MSE loss
        true_outage = torch.tensor(outage, dtype=torch.float32)
        mse_loss = torch.mean((predicted_outages - true_outage) ** 2)
        mse_losses.append(mse_loss.item())
        
        # Total loss
        total_loss = decision_loss + lambda_mse * mse_loss
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Decision Loss: {decision_loss:.4f}, MSE Loss: {mse_loss:.4f}')
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"SIR DFL training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return decision_losses, mse_losses, training_time

def train_decision_focused_rnn_lstm(model, cvxpylayer, optimizer, customer, outage, 
                                   train_data, counties, seq_len, 
                                   lambda_mse=0.01, num_epochs=1000):
    """Train RNN/LSTM model using decision-focused learning."""
    decision_losses = []
    mse_losses = []
    
    print(f"Starting RNN/LSTM DFL training for {num_epochs} epochs...")
    start_time = time.time()
    
    # Create sequences for training
    def create_sequences(data, counties, seq_len):
        sequences = []
        targets = []
        
        for county in counties:
            county_data = data[data['county'] == county].sort_values('datetime').reset_index(drop=True)
            
            for i in range(seq_len, len(county_data)):
                seq = county_data['total_outage'].iloc[i-seq_len:i].values.reshape(-1, 1)
                target = county_data['total_outage'].iloc[i]
                
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    X_train, y_train = create_sequences(train_data, counties, seq_len)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        train_pred = model(X_train)
        
        # Create predicted outage array for SAIDI calculation
        # This is a simplified approach - in practice, you'd need to reconstruct
        # the full time series predictions for each county
        predicted_outages = train_pred.detach().numpy().flatten()
        
        # For DFL, we need to create a 2D array (counties x time_steps)
        # This is a simplified approach - you might need to modify this based on your needs
        num_counties = len(counties)
        time_steps = len(predicted_outages) // num_counties
        predicted_outages_2d = predicted_outages[:num_counties * time_steps].reshape(num_counties, time_steps)
        
        # Calculate SAIDI for decision making
        bl_SAIDI = calc_SAIDI_tensor(customer, predicted_outages_2d)
        
        # Decision layer
        x_opt_pred, _ = cvxpylayer(bl_SAIDI, solver_args={"solve_method": "ECOS"})
        
        # Calculate decision loss using true SAIDI
        bl_SAIDI_true = torch.tensor(calc_SAIDI(customer, outage), dtype=torch.float32)
        x_opt_true, _ = cvxpylayer(bl_SAIDI_true)
        
        true_obj_opt = 0
        true_obj_pred = torch.matmul(bl_SAIDI_true, x_opt_pred)
        decision_loss = true_obj_opt - true_obj_pred
        decision_losses.append(decision_loss.item())
        
        # MSE loss
        mse_loss = torch.mean((train_pred - y_train) ** 2)
        mse_losses.append(mse_loss.item())
        
        # Total loss
        total_loss = decision_loss + lambda_mse * mse_loss
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Decision Loss: {decision_loss:.4f}, MSE Loss: {mse_loss:.4f}')
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"RNN/LSTM DFL training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return decision_losses, mse_losses, training_time

# ============================================================
# Model saving and loading functions
# ============================================================
def save_model_weights(model, model_type, state, model_option, num_epochs, additional_info=""):
    """Save model weights with meaningful naming."""
    import os
    from datetime import datetime
    
    # Create weights directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)
    
    # Generate meaningful filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if additional_info:
        filename = f"weights/{model_type}_{state}_{model_option}_{num_epochs}epochs_{additional_info}_{timestamp}.pth"
    else:
        filename = f"weights/{model_type}_{state}_{model_option}_{num_epochs}epochs_{timestamp}.pth"
    
    # Save the model
    torch.save(model.state_dict(), filename)
    print(f"{model_type} model weights saved to: {filename}")
    return filename

def load_model_weights(model, model_path, model_type):
    """Load model weights from file."""
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"{model_type} model weights loaded from: {model_path}")
        return True
    except Exception as e:
        print(f"Error loading {model_type} model from {model_path}: {e}")
        return False

# ============================================================
# Model training functions
# ============================================================
def train_sir_model(model, train_data, train_counties, train_initial_conditions, 
                   county_data_dict, time_points, num_epochs=3000, 
                   model_option='sir-ode', census_data_dict=None, weather_data_dict=None,
                   demographic_dim=0, weather_dim=0):
    """Train SIR model using MSE loss."""
    print(f"Starting SIR MSE training for {num_epochs} epochs...")
    start_time = time.time()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    clip_value = 1
    batch_size = 5
    
    train_counties_list = list(train_counties)
    epoch_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        np.random.shuffle(train_counties_list)
        batches = [train_counties_list[i:i + batch_size] for i in range(0, len(train_counties_list), batch_size)]
        
        # Check model parameters for NaN values
        if epoch == 0:  # Only check on first epoch to avoid spam
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"Warning: NaN values detected in model parameter {name}")
                    print(f"Parameter shape: {param.shape}")
                    print(f"Parameter values: {param}")
                    return epoch_losses, 0  # Return early if NaN detected
        
        for batch in batches:
            y0 = torch.tensor([train_initial_conditions[county] for county in batch], dtype=torch.float32)
            
            # Check for NaN values in initial conditions
            if torch.isnan(y0).any():
                print(f"Warning: NaN values detected in initial conditions at epoch {epoch+1}")
                print(f"y0 contains NaN: {torch.isnan(y0).any()}")
                print(f"y0 values: {y0}")
                print(f"Batch counties: {batch}")
                for county in batch:
                    print(f"  {county}: {train_initial_conditions[county]}")
                continue
            
            N_values = y0.sum(dim=1)
            
            # Check for zero or negative population values
            if (N_values <= 0).any():
                print(f"Warning: Zero or negative population values detected at epoch {epoch+1}")
                print(f"N_values: {N_values}")
                print(f"Batch counties: {batch}")
                continue
            
            model.set_population(N_values)
            
            # Set demographic and weather data for weather model
            if model_option == 'sir-weather-ode' and census_data_dict and weather_data_dict:
                batch_demographic = []
                batch_weather = []
                for county in batch:
                    if county in census_data_dict:
                        batch_demographic.append(census_data_dict[county])
                    else:
                        batch_demographic.append(torch.zeros(demographic_dim))
                    if county in weather_data_dict:
                        batch_weather.append(weather_data_dict[county])
                    else:
                        batch_weather.append(torch.zeros(weather_dim))
                batch_demographic = torch.stack(batch_demographic, dim=0)
                batch_weather = torch.stack(batch_weather, dim=0)
                model.set_demographic(batch_demographic)
                model.set_weather(batch_weather)
            elif model_option == 'sir-weather-ode':
                # If weather model but no data provided, create dummy data
                batch_demographic = torch.zeros(len(batch), demographic_dim)
                batch_weather = torch.zeros(len(batch), weather_dim)
                model.set_demographic(batch_demographic)
                model.set_weather(batch_weather)
            
            optimizer.zero_grad()
            y_pred = odeint(model, y0, time_points, method='rk4')
            
            # Check for NaN values in predictions
            if torch.isnan(y_pred).any():
                print(f"Warning: NaN values detected in y_pred at epoch {epoch+1}")
                print(f"y_pred contains NaN: {torch.isnan(y_pred).any()}")
                print(f"y0 values: {y0}")
                if model_option == 'sir-weather-ode':
                    print(f"Demographic data contains NaN: {torch.isnan(batch_demographic).any() if 'batch_demographic' in locals() else 'N/A'}")
                    print(f"Weather data contains NaN: {torch.isnan(batch_weather).any() if 'batch_weather' in locals() else 'N/A'}")
                continue
            
            batch_loss = 0
            
            for i, county in enumerate(batch):
                actual_outage_values = county_data_dict[county]['total_outage'].values[:len(time_points)]
                actual_outages = torch.zeros(len(time_points), dtype=torch.float32)
                actual_outages[:len(actual_outage_values)] = torch.tensor(actual_outage_values, dtype=torch.float32)
                predicted_outages = y_pred[:, i, 1]
                batch_loss += criterion(predicted_outages, actual_outages)
            
            batch_loss /= len(batch)
            
            # Check for NaN in loss
            if torch.isnan(batch_loss):
                print(f"Warning: NaN loss detected at epoch {epoch+1}")
                print(f"batch_loss: {batch_loss}")
                continue
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            optimizer.step()
            epoch_loss += batch_loss.item()
        
        epoch_losses.append(epoch_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"SIR MSE training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return epoch_losses, training_time

def train_rnn_lstm_models(train_data, test_data, seq_len=24, hidden_dim=64, num_layers=2, num_epochs=100, lr=0.001):
    """Train RNN and LSTM models with separate models for each city."""
    # Create sequences
    def create_sequences(data, seq_len=24):
        sequences = []
        targets = []
        counties = []
        datetimes = []
        county_indices = []
        
        # Get unique counties and create mapping
        unique_counties = data['county'].unique()
        county_to_idx = {county: idx for idx, county in enumerate(unique_counties)}
        
        for county in unique_counties:
            county_data = data[data['county'] == county].sort_values('datetime').reset_index(drop=True)
            
            for i in range(seq_len, len(county_data)):
                seq = county_data['total_outage'].iloc[i-seq_len:i].values.reshape(-1, 1)
                target = county_data['total_outage'].iloc[i]
                
                sequences.append(seq)
                targets.append(target)
                counties.append(county)
                county_indices.append(county_to_idx[county])
                datetimes.append(county_data['datetime'].iloc[i])
        
        return np.array(sequences), np.array(targets), counties, county_indices, datetimes, unique_counties
    
    X_train, y_train, train_counties, train_county_indices, train_datetimes, unique_counties = create_sequences(train_data, seq_len)
    X_test, y_test, test_counties, test_county_indices, test_datetimes, _ = create_sequences(test_data, seq_len)
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    train_county_indices = torch.LongTensor(train_county_indices)
    test_county_indices = torch.LongTensor(test_county_indices)
    
    # Initialize multi-city models
    num_cities = len(unique_counties)
    rnn_model = MultiCityRNNModel(num_cities=num_cities, input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers)
    lstm_model = MultiCityLSTMModel(num_cities=num_cities, input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers)
    
    def train_model(model, X_train, y_train, X_test, y_test, train_county_indices, test_county_indices, num_epochs, lr):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_losses = []
        test_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            train_pred = model(X_train, train_county_indices)
            train_loss = criterion(train_pred, y_train)
            train_loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test, test_county_indices)
                test_loss = criterion(test_pred, y_test)
            
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
        
        return model, train_losses, test_losses
    
    # Train models
    print("Training RNN model...")
    rnn_start_time = time.time()
    rnn_model, rnn_train_losses, rnn_test_losses = train_model(rnn_model, X_train, y_train, X_test, y_test, train_county_indices, test_county_indices, num_epochs, lr)
    rnn_mse_time = time.time() - rnn_start_time
    print(f"RNN MSE training completed in {rnn_mse_time:.2f} seconds ({rnn_mse_time/60:.2f} minutes)")
    
    print("Training LSTM model...")
    lstm_start_time = time.time()
    lstm_model, lstm_train_losses, lstm_test_losses = train_model(lstm_model, X_train, y_train, X_test, y_test, train_county_indices, test_county_indices, num_epochs, lr)
    lstm_mse_time = time.time() - lstm_start_time
    print(f"LSTM MSE training completed in {lstm_mse_time:.2f} seconds ({lstm_mse_time/60:.2f} minutes)")
    
    # Get predictions
    rnn_model.eval()
    lstm_model.eval()
    
    with torch.no_grad():
        rnn_train_pred = rnn_model(X_train, train_county_indices).numpy()
        rnn_test_pred = rnn_model(X_test, test_county_indices).numpy()
        lstm_train_pred = lstm_model(X_train, train_county_indices).numpy()
        lstm_test_pred = lstm_model(X_test, test_county_indices).numpy()
    
    # Calculate MSE
    rnn_train_mse = np.mean((y_train.numpy() - rnn_train_pred)**2)
    rnn_test_mse = np.mean((y_test.numpy() - rnn_test_pred)**2)
    lstm_train_mse = np.mean((y_train.numpy() - lstm_train_pred)**2)
    lstm_test_mse = np.mean((y_test.numpy() - lstm_test_pred)**2)
    
    print(f"RNN - Train MSE: {rnn_train_mse:.4f}, Test MSE: {rnn_test_mse:.4f}")
    print(f"LSTM - Train MSE: {lstm_train_mse:.4f}, Test MSE: {lstm_test_mse:.4f}")
    
    # Print parameter information
    print(f"RNN Model - Total Parameters: {rnn_model.get_total_parameters():,}")
    print(f"LSTM Model - Total Parameters: {lstm_model.get_total_parameters():,}")
    print(f"Number of cities: {num_cities}")
    print(f"Parameters per city (RNN): {rnn_model.get_city_parameters(0):,}")
    print(f"Parameters per city (LSTM): {lstm_model.get_city_parameters(0):,}")
    
    return {
        'rnn_model': rnn_model,
        'lstm_model': lstm_model,
        'rnn_train_mse': rnn_train_mse,
        'rnn_test_mse': rnn_test_mse,
        'lstm_train_mse': lstm_train_mse,
        'lstm_test_mse': lstm_test_mse,
        'train_counties': train_counties,
        'test_counties': test_counties,
        'train_datetimes': train_datetimes,
        'test_datetimes': test_datetimes,
        'y_train': y_train.numpy(),
        'y_test': y_test.numpy(),
        'unique_counties': unique_counties,
        'num_cities': num_cities,
        'rnn_mse_time': rnn_mse_time,
        'lstm_mse_time': lstm_mse_time
    }

# ============================================================
# Evaluation functions
# ============================================================
def evaluate_sir_model(model, initial_conditions, counties, time_points, 
                      model_option='sir-ode', census_data_dict=None, weather_data_dict=None,
                      demographic_dim=0, weather_dim=0):
    """Evaluate SIR model predictions."""
    model.eval()
    with torch.no_grad():
        y0 = torch.tensor([initial_conditions[county] for county in counties], dtype=torch.float32)
        N_values = y0.sum(dim=1)
        model.set_population(N_values)
        
        if model_option == 'sir-weather-ode' and census_data_dict and weather_data_dict:
            batch_demographic = []
            batch_weather = []
            for county in counties:
                if county in census_data_dict:
                    batch_demographic.append(census_data_dict[county])
                else:
                    batch_demographic.append(torch.zeros(demographic_dim))
                if county in weather_data_dict:
                    batch_weather.append(weather_data_dict[county])
                else:
                    batch_weather.append(torch.zeros(weather_dim))
            batch_demographic = torch.stack(batch_demographic, dim=0)
            batch_weather = torch.stack(batch_weather, dim=0)
            model.set_demographic(batch_demographic)
            model.set_weather(batch_weather)
        elif model_option == 'sir-weather-ode':
            # If weather model but no data provided, create dummy data
            batch_demographic = torch.zeros(len(counties), demographic_dim)
            batch_weather = torch.zeros(len(counties), weather_dim)
            model.set_demographic(batch_demographic)
            model.set_weather(batch_weather)
        
        y_pred = odeint(model, y0, time_points, method='rk4')
        predictions = y_pred[:, :, 1].detach().cpu().numpy().T  # Infected component, transpose to (num_counties, time_steps)
    
    return predictions

def evaluate_rnn_lstm_model(model, test_data, counties, seq_len, max_time_steps=None, unique_counties=None):
    """Evaluate RNN/LSTM model predictions with multi-city support."""
    model.eval()
    predictions = []
    
    # Create county to index mapping
    if unique_counties is not None:
        county_to_idx = {county: idx for idx, county in enumerate(unique_counties)}
    else:
        # Fallback: create mapping from the counties in test_data
        unique_counties = test_data['county'].unique()
        county_to_idx = {county: idx for idx, county in enumerate(unique_counties)}
    
    # Use provided max_time_steps or find the maximum length across all counties
    if max_time_steps is not None:
        max_length = max_time_steps
    else:
        max_length = 0
        for county in counties:
            county_data = test_data[test_data['county'] == county].sort_values('datetime').reset_index(drop=True)
            max_length = max(max_length, len(county_data))
    
    with torch.no_grad():
        for county in counties:
            county_data = test_data[test_data['county'] == county].sort_values('datetime').reset_index(drop=True)
            county_predictions = []
            
            # Get county index for multi-city model
            county_idx = county_to_idx.get(county, 0)  # Default to 0 if county not found
            county_idx_tensor = torch.LongTensor([county_idx])
            
            for i in range(seq_len, len(county_data)):
                seq = county_data['total_outage'].iloc[i-seq_len:i].values.reshape(1, -1, 1)
                seq_tensor = torch.FloatTensor(seq)
                pred = model(seq_tensor, county_idx_tensor).numpy()[0, 0]
                county_predictions.append(pred)
            
            # Pad with zeros for the first seq_len time steps
            county_predictions = [0] * seq_len + county_predictions
            
            # Pad to max_length with zeros
            while len(county_predictions) < max_length:
                county_predictions.append(0)
            
            predictions.append(county_predictions)
    
    return np.array(predictions)

# ============================================================
# Main execution
# ============================================================
def main():
    """Main execution function."""
    print("="*60)
    print("OUTAGE PREDICTION WITH DECISION-FOCUSED LEARNING")
    print("="*60)
    
    # Setup configuration
    config = setup_configuration()
    print(f"Configuration: {args.state} state, {args.model} model")
    
    # Set random seed for reproducibility
    set_random_seed(args.random_seed)
    print(f"Random seed set to: {args.random_seed}")
    
    # Load and preprocess data
    data = load_and_preprocess_data(config)
    print(f"Loaded data for {len(data['outage_data']['county'].unique())} counties")
    
    # Subsample cities if specified
    if args.num_subsampled_cities is not None:
        all_counties = data['outage_data']['county'].unique()
        if args.num_subsampled_cities < len(all_counties):
            np.random.seed(args.random_seed)  # Ensure reproducible subsampling
            subsampled_counties = np.random.choice(all_counties, args.num_subsampled_cities, replace=False)
            print(f"Subsampling {args.num_subsampled_cities} cities from {len(all_counties)} total cities")
            print(f"Selected cities: {list(subsampled_counties)}")
            
            # Filter data to only include subsampled cities
            data['outage_data'] = data['outage_data'][data['outage_data']['county'].isin(subsampled_counties)]
            if 'census_data' in data and data['census_data'] is not None:
                data['census_data'] = data['census_data'][data['census_data']['county'].isin(subsampled_counties)]
            if 'weather_data' in data and data['weather_data'] is not None:
                data['weather_data'] = data['weather_data'][data['weather_data']['county'].isin(subsampled_counties)]
            
            print(f"Data filtered to {len(data['outage_data']['county'].unique())} counties")
        else:
            print(f"Number of subsampled cities ({args.num_subsampled_cities}) >= total cities ({len(all_counties)}), using all cities")
    else:
        print("Using all available cities")
    
    # Split train/test
    train_data, test_data, train_counties, test_counties = split_train_test(data['outage_data'], data['total_customer_dict'], config)
    print(f"Train counties: {len(train_counties)}, Test counties: {len(test_counties)}")
    
    # Prepare initial conditions
    train_initial_conditions = {}
    for county in train_counties:
        county_data = train_data[train_data['county'] == county].sort_values(by='datetime').reset_index(drop=True)
        initial_S = data['total_customer_dict'][county] - county_data['total_outage'].iloc[0]
        initial_I = county_data['total_outage'].iloc[0]
        initial_R = 0
        train_initial_conditions[county] = [initial_S, initial_I, initial_R]
    
    test_initial_conditions = {}
    for county in test_counties:
        county_data = test_data[test_data['county'] == county].sort_values(by='datetime').reset_index(drop=True)
        initial_S = data['total_customer_dict'][county] - county_data['total_outage'].iloc[0]
        initial_I = county_data['total_outage'].iloc[0]
        initial_R = 0
        test_initial_conditions[county] = [initial_S, initial_I, initial_R]
    
    # Prepare additional data
    census_data_dict = {}
    if data['census_data'] is not None:
        # Handle NaN values in census data by filling them with the mean of the column
        census_data_clean = data['census_data'].copy()
        numeric_columns = census_data_clean.select_dtypes(include=[np.number]).columns
        census_data_clean[numeric_columns] = census_data_clean[numeric_columns].fillna(census_data_clean[numeric_columns].mean())
        
        census_data_dict = {
            row['County']: torch.tensor(row[1:].values.tolist(), dtype=torch.float32)
            for _, row in census_data_clean.iterrows()
        }
        print("census_data loaded (NaN values filled with column means)")
        print(census_data_dict)


    
    weather_data_dict = {}
    if data['weather_data']:
        target_date = data['outage_data']['datetime'].min().date().strftime("%Y-%m-%d")
        for county, df in data['weather_data'].items():
            df['DateTime'] = pd.to_datetime(df['date'])
            filtered_df = df[df['DateTime'] == target_date]
            if not filtered_df.empty:
                numerical_values = filtered_df.values[0][4:-1].tolist()
                weather_data_dict[county] = torch.tensor(numerical_values, dtype=torch.float32)
        print("weather_data loaded")
        print(weather_data_dict)
    
    demographic_dim = len(list(census_data_dict.values())[0]) if census_data_dict else 0
    weather_dim = len(list(weather_data_dict.values())[0]) if weather_data_dict else 0
    
    # Time points
    max_time_steps = max(
        train_data.groupby('county').size().max(),
        test_data.groupby('county').size().max()
    )
    time_points = torch.linspace(1, max_time_steps, max_time_steps)
    
    # County data dictionary
    county_data_dict = {
        county: train_data[train_data['county'] == county]
        for county in train_counties
    }
    
    # ============================================================
    # Train SIR Model
    # ============================================================
    print("\n" + "="*40)
    print("TRAINING SIR MODEL")
    print("="*40)
    
    if args.model == 'sir-ode':
        sir_model = BaseSIRModel()
    else:
        sir_model = SIRWeatherModel(demographic_dim=demographic_dim, weather_dim=weather_dim)
    
    # Check if pretrained SIR model is provided
    if args.sir_pretrained_path:
        print(f"Loading pretrained SIR model from: {args.sir_pretrained_path}")
        if load_model_weights(sir_model, args.sir_pretrained_path, "SIR"):
            sir_mse_time = 0  # No training time if loading pretrained
            sir_losses = []
            print("SIR model loaded successfully, skipping MSE training")
        else:
            print("Failed to load pretrained SIR model, training from scratch...")
            sir_losses, sir_mse_time = train_sir_model(
                sir_model, train_data, train_counties, train_initial_conditions,
                county_data_dict, time_points, args.num_MSE_epochs, args.model,
                census_data_dict, weather_data_dict, demographic_dim, weather_dim
            )
            # Save the trained model
            save_model_weights(sir_model, "SIR", args.state, args.model, args.num_MSE_epochs, "MSE")
    else:
        print("Training SIR model from scratch...")
        sir_losses, sir_mse_time = train_sir_model(
            sir_model, train_data, train_counties, train_initial_conditions,
            county_data_dict, time_points, args.num_MSE_epochs, args.model,
            census_data_dict, weather_data_dict, demographic_dim, weather_dim
        )
        # Save the trained model
        save_model_weights(sir_model, "SIR", args.state, args.model, args.num_MSE_epochs, "MSE")
    
    # ============================================================
    # Train RNN and LSTM Models
    # ============================================================
    print("\n" + "="*40)
    print("TRAINING RNN AND LSTM MODELS")
    print("="*40)
    
    # Check if pretrained RNN/LSTM models are provided
    rnn_pretrained_loaded = False
    lstm_pretrained_loaded = False
    
    # Get unique counties for model initialization
    unique_counties = list(set(train_counties) | set(test_counties))
    
    if args.rnn_pretrained_path or args.lstm_pretrained_path:
        print("Loading pretrained RNN/LSTM models...")
        
        # Initialize models
        rnn_model = MultiCityRNNModel(
            num_cities=len(unique_counties),
            input_dim=1, hidden_dim=args.hidden_dim, 
            num_layers=args.num_layers, output_dim=1
        )
        lstm_model = MultiCityLSTMModel(
            num_cities=len(unique_counties),
            input_dim=1, hidden_dim=args.hidden_dim, 
            num_layers=args.num_layers, output_dim=1
        )
        
        # Load RNN model if path provided
        if args.rnn_pretrained_path:
            if load_model_weights(rnn_model, args.rnn_pretrained_path, "RNN"):
                rnn_pretrained_loaded = True
                print("RNN model loaded successfully, skipping MSE training")
            else:
                print("Failed to load pretrained RNN model, will train from scratch")
        
        # Load LSTM model if path provided
        if args.lstm_pretrained_path:
            if load_model_weights(lstm_model, args.lstm_pretrained_path, "LSTM"):
                lstm_pretrained_loaded = True
                print("LSTM model loaded successfully, skipping MSE training")
            else:
                print("Failed to load pretrained LSTM model, will train from scratch")
    
    # Train models if not loaded from pretrained
    if not (rnn_pretrained_loaded and lstm_pretrained_loaded):
        print("Training RNN and LSTM models...")
        baseline_results = train_rnn_lstm_models(
            train_data, test_data, seq_len=args.seq_len, 
            hidden_dim=args.hidden_dim, num_layers=args.num_layers, 
            num_epochs=args.num_MSE_epochs, lr=0.001
        )
        
        # Save the trained models
        if not rnn_pretrained_loaded:
            save_model_weights(baseline_results['rnn_model'], "RNN", args.state, "multi-city", args.num_MSE_epochs, "MSE")
        if not lstm_pretrained_loaded:
            save_model_weights(baseline_results['lstm_model'], "LSTM", args.state, "multi-city", args.num_MSE_epochs, "MSE")
    else:
        # Create baseline_results structure for loaded models
        baseline_results = {
            'rnn_model': rnn_model,
            'lstm_model': lstm_model,
            'rnn_mse_time': 0,
            'lstm_mse_time': 0,
            'unique_counties': unique_counties
        }
        print("Using pretrained RNN and LSTM models")
    
    # ============================================================
    # Evaluate MSE-Trained Models (Before DFL)
    # ============================================================
    print("\n" + "="*40)
    print("EVALUATING MSE-TRAINED MODELS (BEFORE DFL)")
    print("="*40)
    
    # Prepare data for evaluation
    train_customer = [data['total_customer_dict'][county] for county in train_counties]
    test_customer = [data['total_customer_dict'][county] for county in test_counties]
    
    # Create outage arrays
    train_true_outages = create_outage_array_from_df(train_data, train_counties, max_time_steps)
    test_true_outages = create_outage_array_from_df(test_data, test_counties, max_time_steps)
    
    # Evaluate MSE-trained SIR model
    print("Evaluating MSE-trained SIR model...")
    sir_mse_train_pred = evaluate_sir_model(
        sir_model, train_initial_conditions, train_counties, time_points,
        args.model, census_data_dict, weather_data_dict, demographic_dim, weather_dim
    )
    sir_mse_test_pred = evaluate_sir_model(
        sir_model, test_initial_conditions, test_counties, time_points,
        args.model, census_data_dict, weather_data_dict, demographic_dim, weather_dim
    )
    
    # Calculate MSE for MSE-trained SIR
    sir_mse_train_mse = calculate_mse(sir_mse_train_pred, train_true_outages)
    sir_mse_test_mse = calculate_mse(sir_mse_test_pred, test_true_outages)
    
    # Calculate SAIDI for MSE-trained SIR
    sir_mse_train_saidi = calculate_saidi_metrics(train_customer, sir_mse_train_pred)
    sir_mse_test_saidi = calculate_saidi_metrics(test_customer, sir_mse_test_pred)
    
    print(f"MSE-trained SIR - Train MSE: {sir_mse_train_mse:.4f}, Test MSE: {sir_mse_test_mse:.4f}")
    print(f"MSE-trained SIR - Train SAIDI: {np.mean(sir_mse_train_saidi):.4f}, Test SAIDI: {np.mean(sir_mse_test_saidi):.4f}")
    
    # Evaluate MSE-trained RNN model
    print("Evaluating MSE-trained RNN model...")
    rnn_mse_train_pred = evaluate_rnn_lstm_model(baseline_results['rnn_model'], train_data, train_counties, args.seq_len, max_time_steps, baseline_results['unique_counties'])
    rnn_mse_test_pred = evaluate_rnn_lstm_model(baseline_results['rnn_model'], test_data, test_counties, args.seq_len, max_time_steps, baseline_results['unique_counties'])
    
    rnn_mse_train_mse = calculate_mse(rnn_mse_train_pred, train_true_outages)
    rnn_mse_test_mse = calculate_mse(rnn_mse_test_pred, test_true_outages)
    
    rnn_mse_train_saidi = calculate_saidi_metrics(train_customer, rnn_mse_train_pred)
    rnn_mse_test_saidi = calculate_saidi_metrics(test_customer, rnn_mse_test_pred)
    
    print(f"MSE-trained RNN - Train MSE: {rnn_mse_train_mse:.4f}, Test MSE: {rnn_mse_test_mse:.4f}")
    print(f"MSE-trained RNN - Train SAIDI: {np.mean(rnn_mse_train_saidi):.4f}, Test SAIDI: {np.mean(rnn_mse_test_saidi):.4f}")
    
    # Evaluate MSE-trained LSTM model
    print("Evaluating MSE-trained LSTM model...")
    lstm_mse_train_pred = evaluate_rnn_lstm_model(baseline_results['lstm_model'], train_data, train_counties, args.seq_len, max_time_steps, baseline_results['unique_counties'])
    lstm_mse_test_pred = evaluate_rnn_lstm_model(baseline_results['lstm_model'], test_data, test_counties, args.seq_len, max_time_steps, baseline_results['unique_counties'])
    
    lstm_mse_train_mse = calculate_mse(lstm_mse_train_pred, train_true_outages)
    lstm_mse_test_mse = calculate_mse(lstm_mse_test_pred, test_true_outages)
    
    lstm_mse_train_saidi = calculate_saidi_metrics(train_customer, lstm_mse_train_pred)
    lstm_mse_test_saidi = calculate_saidi_metrics(test_customer, lstm_mse_test_pred)
    
    print(f"MSE-trained LSTM - Train MSE: {lstm_mse_train_mse:.4f}, Test MSE: {lstm_mse_test_mse:.4f}")
    print(f"MSE-trained LSTM - Train SAIDI: {np.mean(lstm_mse_train_saidi):.4f}, Test SAIDI: {np.mean(lstm_mse_test_saidi):.4f}")
    
    # SAIDI optimization for MSE-trained models
    print("\n--- MSE-trained Models SAIDI Optimization ---")
    
    # MSE-trained SIR optimization
    train_sir_mse_selected = select_cities_for_hardening(
        sir_mse_train_pred, train_customer, args.max_selected_cities,
        "Train MSE-trained SIR Predictions"
    )
    train_sir_mse_eval = evaluate_saidi_hardened(
        train_sir_mse_selected, train_customer, train_true_outages,
        "Train MSE-trained SIR Predictions"
    )
    
    test_sir_mse_selected = select_cities_for_hardening(
        sir_mse_test_pred, test_customer, args.max_selected_cities,
        "Test MSE-trained SIR Predictions"
    )
    test_sir_mse_eval = evaluate_saidi_hardened(
        test_sir_mse_selected, test_customer, test_true_outages,
        "Test MSE-trained SIR Predictions"
    )
    
    # MSE-trained RNN optimization
    train_rnn_mse_selected = select_cities_for_hardening(
        rnn_mse_train_pred, train_customer, args.max_selected_cities,
        "Train MSE-trained RNN Predictions"
    )
    train_rnn_mse_eval = evaluate_saidi_hardened(
        train_rnn_mse_selected, train_customer, train_true_outages,
        "Train MSE-trained RNN Predictions"
    )
    
    test_rnn_mse_selected = select_cities_for_hardening(
        rnn_mse_test_pred, test_customer, args.max_selected_cities,
        "Test MSE-trained RNN Predictions"
    )
    test_rnn_mse_eval = evaluate_saidi_hardened(
        test_rnn_mse_selected, test_customer, test_true_outages,
        "Test MSE-trained RNN Predictions"
    )
    
    # MSE-trained LSTM optimization
    train_lstm_mse_selected = select_cities_for_hardening(
        lstm_mse_train_pred, train_customer, args.max_selected_cities,
        "Train MSE-trained LSTM Predictions"
    )
    train_lstm_mse_eval = evaluate_saidi_hardened(
        train_lstm_mse_selected, train_customer, train_true_outages,
        "Train MSE-trained LSTM Predictions"
    )
    
    test_lstm_mse_selected = select_cities_for_hardening(
        lstm_mse_test_pred, test_customer, args.max_selected_cities,
        "Test MSE-trained LSTM Predictions"
    )
    test_lstm_mse_eval = evaluate_saidi_hardened(
        test_lstm_mse_selected, test_customer, test_true_outages,
        "Test MSE-trained LSTM Predictions"
    )
    
    # ============================================================
    # Decision-Focused Learning
    # ============================================================
    print("\n" + "="*40)
    print("DECISION-FOCUSED LEARNING")
    print("="*40)
    
    # Prepare data for DFL
    train_customer = [data['total_customer_dict'][county] for county in train_counties]
    test_customer = [data['total_customer_dict'][county] for county in test_counties]
    
    # Create outage arrays
    train_true_outages = create_outage_array_from_df(train_data, train_counties, max_time_steps)
    test_true_outages = create_outage_array_from_df(test_data, test_counties, max_time_steps)
    
    # DFL for SIR model
    print("Training SIR model with DFL...")
    cvxpylayer_sir = create_cvxpy_layer(args.lambda_smoothing, args.max_selected_cities, len(train_counties))
    sir_optimizer = torch.optim.Adam(sir_model.parameters(), lr=1e-2)
    
    sir_decision_losses, sir_mse_losses, sir_dfl_time = train_decision_focused_sir(
        sir_model, cvxpylayer_sir, sir_optimizer, train_customer, train_true_outages,
        train_initial_conditions, train_counties, time_points, args.lambda_mse, args.num_gdf_epochs, 
        args.model, census_data_dict, weather_data_dict, demographic_dim, weather_dim
    )
    
    # Save DFL-trained SIR model
    save_model_weights(sir_model, "SIR", args.state, args.model, args.num_gdf_epochs, "DFL")
    
    # DFL for RNN model
    print("Training RNN model with DFL...")
    cvxpylayer_rnn = create_cvxpy_layer(args.lambda_smoothing, args.max_selected_cities, len(train_counties))
    rnn_optimizer = torch.optim.Adam(baseline_results['rnn_model'].parameters(), lr=1e-2)
    
    rnn_decision_losses, rnn_mse_losses, rnn_dfl_time = train_decision_focused_rnn_lstm(
        baseline_results['rnn_model'], cvxpylayer_rnn, rnn_optimizer, 
        train_customer, train_true_outages, train_data, train_counties, args.seq_len,
        args.lambda_mse, args.num_gdf_epochs
    )
    
    # Save DFL-trained RNN model
    save_model_weights(baseline_results['rnn_model'], "RNN", args.state, "multi-city", args.num_gdf_epochs, "DFL")
    
    # DFL for LSTM model
    print("Training LSTM model with DFL...")
    cvxpylayer_lstm = create_cvxpy_layer(args.lambda_smoothing, args.max_selected_cities, len(train_counties))
    lstm_optimizer = torch.optim.Adam(baseline_results['lstm_model'].parameters(), lr=1e-2)
    
    lstm_decision_losses, lstm_mse_losses, lstm_dfl_time = train_decision_focused_rnn_lstm(
        baseline_results['lstm_model'], cvxpylayer_lstm, lstm_optimizer, 
        train_customer, train_true_outages, train_data, train_counties, args.seq_len,
        args.lambda_mse, args.num_gdf_epochs
    )
    
    # Save DFL-trained LSTM model
    save_model_weights(baseline_results['lstm_model'], "LSTM", args.state, "multi-city", args.num_gdf_epochs, "DFL")
    
    # ============================================================
    # Evaluation
    # ============================================================
    print("\n" + "="*40)
    print("MODEL EVALUATION")
    print("="*40)
    
    # Evaluate SIR model
    print("Evaluating SIR model...")
    sir_train_pred = evaluate_sir_model(
        sir_model, train_initial_conditions, train_counties, time_points,
        args.model, census_data_dict, weather_data_dict, demographic_dim, weather_dim
    )
    sir_test_pred = evaluate_sir_model(
        sir_model, test_initial_conditions, test_counties, time_points,
        args.model, census_data_dict, weather_data_dict, demographic_dim, weather_dim
    )
    
    # Calculate MSE
    sir_train_mse = calculate_mse(sir_train_pred, train_true_outages)
    sir_test_mse = calculate_mse(sir_test_pred, test_true_outages)
    
    # Calculate SAIDI
    sir_train_saidi = calculate_saidi_metrics(train_customer, sir_train_pred)
    sir_test_saidi = calculate_saidi_metrics(test_customer, sir_test_pred)
    
    print(f"SIR Model - Train MSE: {sir_train_mse:.4f}, Test MSE: {sir_test_mse:.4f}")
    print(f"SIR Model - Train SAIDI: {np.mean(sir_train_saidi):.4f}, Test SAIDI: {np.mean(sir_test_saidi):.4f}")
    
    # Evaluate RNN model
    print("Evaluating RNN model...")
    rnn_train_pred = evaluate_rnn_lstm_model(baseline_results['rnn_model'], train_data, train_counties, args.seq_len, max_time_steps, baseline_results['unique_counties'])
    rnn_test_pred = evaluate_rnn_lstm_model(baseline_results['rnn_model'], test_data, test_counties, args.seq_len, max_time_steps, baseline_results['unique_counties'])
    
    rnn_train_mse = calculate_mse(rnn_train_pred, train_true_outages)
    rnn_test_mse = calculate_mse(rnn_test_pred, test_true_outages)
    
    rnn_train_saidi = calculate_saidi_metrics(train_customer, rnn_train_pred)
    rnn_test_saidi = calculate_saidi_metrics(test_customer, rnn_test_pred)
    
    print(f"RNN Model - Train MSE: {rnn_train_mse:.4f}, Test MSE: {rnn_test_mse:.4f}")
    print(f"RNN Model - Train SAIDI: {np.mean(rnn_train_saidi):.4f}, Test SAIDI: {np.mean(rnn_test_saidi):.4f}")
    
    # Evaluate LSTM model
    print("Evaluating LSTM model...")
    lstm_train_pred = evaluate_rnn_lstm_model(baseline_results['lstm_model'], train_data, train_counties, args.seq_len, max_time_steps, baseline_results['unique_counties'])
    lstm_test_pred = evaluate_rnn_lstm_model(baseline_results['lstm_model'], test_data, test_counties, args.seq_len, max_time_steps, baseline_results['unique_counties'])
    
    lstm_train_mse = calculate_mse(lstm_train_pred, train_true_outages)
    lstm_test_mse = calculate_mse(lstm_test_pred, test_true_outages)
    
    lstm_train_saidi = calculate_saidi_metrics(train_customer, lstm_train_pred)
    lstm_test_saidi = calculate_saidi_metrics(test_customer, lstm_test_pred)
    
    print(f"LSTM Model - Train MSE: {lstm_train_mse:.4f}, Test MSE: {lstm_test_mse:.4f}")
    print(f"LSTM Model - Train SAIDI: {np.mean(lstm_train_saidi):.4f}, Test SAIDI: {np.mean(lstm_test_saidi):.4f}")
    
    # ============================================================
    # Ground Truth SAIDI Calculation
    # ============================================================
    print("\n" + "="*40)
    print("GROUND TRUTH SAIDI CALCULATION")
    print("="*40)
    
    # Calculate ground truth SAIDI for train and test periods
    train_gt_saidi = calculate_saidi_metrics(train_customer, train_true_outages)
    test_gt_saidi = calculate_saidi_metrics(test_customer, test_true_outages)
    
    print(f"Ground Truth - Train SAIDI: {np.mean(train_gt_saidi):.4f}")
    print(f"Ground Truth - Test SAIDI: {np.mean(test_gt_saidi):.4f}")
    
    # ============================================================
    # SAIDI Optimization using Gurobi (Knapsack Problem)
    # ============================================================
    print("\n" + "="*40)
    print("SAIDI OPTIMIZATION (KNAPSACK PROBLEM)")
    print("="*40)
    
    # Calculate total SAIDI for reporting
    train_total_saidi = np.sum(train_gt_saidi)
    test_total_saidi = np.sum(test_gt_saidi)
    
    print(f"Total Train SAIDI: {train_total_saidi:.4f}")
    print(f"Total Test SAIDI: {test_total_saidi:.4f}")
    
    # ============================================================
    # Ground Truth SAIDI Optimization
    # ============================================================
    print("\n--- Ground Truth SAIDI Optimization ---")
    
    # Train data ground truth optimization
    train_gt_selected = select_cities_for_hardening(
        train_true_outages, train_customer, args.max_selected_cities, 
        "Train Ground Truth"
    )
    train_gt_eval = evaluate_saidi_hardened(
        train_gt_selected, train_customer, train_true_outages,
        "Train Ground Truth"
    )
    
    # Test data ground truth optimization
    test_gt_selected = select_cities_for_hardening(
        test_true_outages, test_customer, args.max_selected_cities,
        "Test Ground Truth"
    )
    test_gt_eval = evaluate_saidi_hardened(
        test_gt_selected, test_customer, test_true_outages,
        "Test Ground Truth"
    )
    
    # ============================================================
    # SIR Model SAIDI Optimization
    # ============================================================
    print("\n--- SIR Model SAIDI Optimization ---")
    
    # Train data SIR optimization
    train_sir_selected = select_cities_for_hardening(
        sir_train_pred, train_customer, args.max_selected_cities,
        "Train SIR Predictions"
    )
    train_sir_eval = evaluate_saidi_hardened(
        train_sir_selected, train_customer, train_true_outages,
        "Train SIR Predictions"
    )
    
    # Test data SIR optimization
    test_sir_selected = select_cities_for_hardening(
        sir_test_pred, test_customer, args.max_selected_cities,
        "Test SIR Predictions"
    )
    test_sir_eval = evaluate_saidi_hardened(
        test_sir_selected, test_customer, test_true_outages,
        "Test SIR Predictions"
    )
    
    # ============================================================
    # RNN Model SAIDI Optimization
    # ============================================================
    print("\n--- RNN Model SAIDI Optimization ---")
    
    # Train data RNN optimization
    train_rnn_selected = select_cities_for_hardening(
        rnn_train_pred, train_customer, args.max_selected_cities,
        "Train RNN Predictions"
    )
    train_rnn_eval = evaluate_saidi_hardened(
        train_rnn_selected, train_customer, train_true_outages,
        "Train RNN Predictions"
    )
    
    # Test data RNN optimization
    test_rnn_selected = select_cities_for_hardening(
        rnn_test_pred, test_customer, args.max_selected_cities,
        "Test RNN Predictions"
    )
    test_rnn_eval = evaluate_saidi_hardened(
        test_rnn_selected, test_customer, test_true_outages,
        "Test RNN Predictions"
    )
    
    # ============================================================
    # LSTM Model SAIDI Optimization
    # ============================================================
    print("\n--- LSTM Model SAIDI Optimization ---")
    
    # Train data LSTM optimization
    train_lstm_selected = select_cities_for_hardening(
        lstm_train_pred, train_customer, args.max_selected_cities,
        "Train LSTM Predictions"
    )
    train_lstm_eval = evaluate_saidi_hardened(
        train_lstm_selected, train_customer, train_true_outages,
        "Train LSTM Predictions"
    )
    
    # Test data LSTM optimization
    test_lstm_selected = select_cities_for_hardening(
        lstm_test_pred, test_customer, args.max_selected_cities,
        "Test LSTM Predictions"
    )
    test_lstm_eval = evaluate_saidi_hardened(
        test_lstm_selected, test_customer, test_true_outages,
        "Test LSTM Predictions"
    )
    
    # ============================================================
    # Validation: Check Alignment and Period Consistency
    # ============================================================
    print("\n" + "="*40)
    print("VALIDATION: ALIGNMENT AND PERIOD CONSISTENCY")
    print("="*40)
    
    # Check array shapes
    print(f"Train data shapes:")
    print(f"  SIR predictions: {sir_train_pred.shape}")
    print(f"  RNN predictions: {rnn_train_pred.shape}")
    print(f"  LSTM predictions: {lstm_train_pred.shape}")
    print(f"  Ground truth: {train_true_outages.shape}")
    
    print(f"\nTest data shapes:")
    print(f"  SIR predictions: {sir_test_pred.shape}")
    print(f"  RNN predictions: {rnn_test_pred.shape}")
    print(f"  LSTM predictions: {lstm_test_pred.shape}")
    print(f"  Ground truth: {test_true_outages.shape}")
    
    # Check time period alignment
    print(f"\nTime period alignment:")
    print(f"  Max time steps: {max_time_steps}")
    print(f"  Train counties: {len(train_counties)}")
    print(f"  Test counties: {len(test_counties)}")
    
    # Validate SAIDI calculation consistency
    print(f"\nSAIDI calculation validation:")
    print(f"  All models use same customer counts: {len(train_customer) == len(test_customer) == len(train_counties) == len(test_counties)}")
    print(f"  All predictions have same shape as ground truth: {sir_train_pred.shape == train_true_outages.shape}")
    
    # Verify ground truth data period alignment
    print(f"\nGround truth data verification:")
    print(f"  Train data period: {train_data['datetime'].min()} to {train_data['datetime'].max()}")
    print(f"  Test data period: {test_data['datetime'].min()} to {test_data['datetime'].max()}")
    print(f"  Train ground truth SAIDI calculated from train period data: ✓")
    print(f"  Test ground truth SAIDI calculated from test period data: ✓")
    
    # Show sample ground truth values for verification
    print(f"\nSample ground truth values (first county):")
    print(f"  Train period total outages: {train_true_outages[0].sum():.2f}")
    print(f"  Test period total outages: {test_true_outages[0].sum():.2f}")
    print(f"  Train period SAIDI: {train_gt_saidi[0]:.4f}")
    print(f"  Test period SAIDI: {test_gt_saidi[0]:.4f}")
    
    # ============================================================
    # Training Time Summary
    # ============================================================
    print("\n" + "="*60)
    print("TRAINING TIME SUMMARY")
    print("="*60)
    
    print(f"Model Training Times:")
    print(f"  SIR MSE Training: {sir_mse_time:.2f} seconds ({sir_mse_time/60:.2f} minutes)")
    print(f"  SIR DFL Training: {sir_dfl_time:.2f} seconds ({sir_dfl_time/60:.2f} minutes)")
    print(f"  RNN MSE Training: {baseline_results['rnn_mse_time']:.2f} seconds ({baseline_results['rnn_mse_time']/60:.2f} minutes)")
    print(f"  RNN DFL Training: {rnn_dfl_time:.2f} seconds ({rnn_dfl_time/60:.2f} minutes)")
    print(f"  LSTM MSE Training: {baseline_results['lstm_mse_time']:.2f} seconds ({baseline_results['lstm_mse_time']/60:.2f} minutes)")
    print(f"  LSTM DFL Training: {lstm_dfl_time:.2f} seconds ({lstm_dfl_time/60:.2f} minutes)")
    
    # Calculate total training times
    sir_total_time = sir_mse_time + sir_dfl_time
    rnn_total_time = baseline_results['rnn_mse_time'] + rnn_dfl_time
    lstm_total_time = baseline_results['lstm_mse_time'] + lstm_dfl_time
    
    print(f"\nTotal Training Times:")
    print(f"  SIR Total: {sir_total_time:.2f} seconds ({sir_total_time/60:.2f} minutes)")
    print(f"  RNN Total: {rnn_total_time:.2f} seconds ({rnn_total_time/60:.2f} minutes)")
    print(f"  LSTM Total: {lstm_total_time:.2f} seconds ({lstm_total_time/60:.2f} minutes)")
    
    # ============================================================
    # Results Summary
    # ============================================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame({
        'Model': ['SIR (MSE)', 'SIR (DFL)', 'RNN (MSE)', 'RNN (DFL)', 'LSTM (MSE)', 'LSTM (DFL)', 'Ground Truth'],
        'Train_MSE': [sir_mse_train_mse, sir_train_mse, rnn_mse_train_mse, rnn_train_mse, lstm_mse_train_mse, lstm_train_mse, 'N/A'],
        'Test_MSE': [sir_mse_test_mse, sir_test_mse, rnn_mse_test_mse, rnn_test_mse, lstm_mse_test_mse, lstm_test_mse, 'N/A'],
        'Train_Remaining_SAIDI': [train_sir_mse_eval['remaining_saidi'], train_sir_eval['remaining_saidi'], train_rnn_mse_eval['remaining_saidi'], train_rnn_eval['remaining_saidi'], train_lstm_mse_eval['remaining_saidi'], train_lstm_eval['remaining_saidi'], train_gt_eval['remaining_saidi']],
        'Test_Remaining_SAIDI': [test_sir_mse_eval['remaining_saidi'], test_sir_eval['remaining_saidi'], test_rnn_mse_eval['remaining_saidi'], test_rnn_eval['remaining_saidi'], test_lstm_mse_eval['remaining_saidi'], test_lstm_eval['remaining_saidi'], test_gt_eval['remaining_saidi']]
    })
    
    print(results_df.to_string(index=False))
    
    # ============================================================
    # SAIDI Optimization Results Summary
    # ============================================================
    print("\n" + "="*60)
    print("SAIDI OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    
    # Create SAIDI optimization results summary
    saidi_opt_df = pd.DataFrame({
        'Dataset': ['Train', 'Test', 'Train', 'Test', 'Train', 'Test', 'Train', 'Test', 'Train', 'Test', 'Train', 'Test', 'Train', 'Test'],
        'Method': ['Ground Truth', 'Ground Truth', 'SIR (MSE)', 'SIR (MSE)', 'SIR (DFL)', 'SIR (DFL)', 
                   'RNN (MSE)', 'RNN (MSE)', 'RNN (DFL)', 'RNN (DFL)', 'LSTM (MSE)', 'LSTM (MSE)', 'LSTM (DFL)', 'LSTM (DFL)'],
        'Total_SAIDI': [train_total_saidi, test_total_saidi, train_total_saidi, test_total_saidi, train_total_saidi, test_total_saidi,
                        train_total_saidi, test_total_saidi, train_total_saidi, test_total_saidi, train_total_saidi, test_total_saidi, train_total_saidi, test_total_saidi],
        'Selected_SAIDI': [
            train_gt_eval['selected_total_saidi'],
            test_gt_eval['selected_total_saidi'],
            train_sir_mse_eval['selected_total_saidi'],
            test_sir_mse_eval['selected_total_saidi'],
            train_sir_eval['selected_total_saidi'],
            test_sir_eval['selected_total_saidi'],
            train_rnn_mse_eval['selected_total_saidi'],
            test_rnn_mse_eval['selected_total_saidi'],
            train_rnn_eval['selected_total_saidi'],
            test_rnn_eval['selected_total_saidi'],
            train_lstm_mse_eval['selected_total_saidi'],
            test_lstm_mse_eval['selected_total_saidi'],
            train_lstm_eval['selected_total_saidi'],
            test_lstm_eval['selected_total_saidi']
        ],
        'Remaining_SAIDI': [
            train_gt_eval['remaining_saidi'],
            test_gt_eval['remaining_saidi'],
            train_sir_mse_eval['remaining_saidi'],
            test_sir_mse_eval['remaining_saidi'],
            train_sir_eval['remaining_saidi'],
            test_sir_eval['remaining_saidi'],
            train_rnn_mse_eval['remaining_saidi'],
            test_rnn_mse_eval['remaining_saidi'],
            train_rnn_eval['remaining_saidi'],
            test_rnn_eval['remaining_saidi'],
            train_lstm_mse_eval['remaining_saidi'],
            test_lstm_mse_eval['remaining_saidi'],
            train_lstm_eval['remaining_saidi'],
            test_lstm_eval['remaining_saidi']
        ],
        'Selected_Cities': [
            len(train_gt_eval['selected_cities']),
            len(test_gt_eval['selected_cities']),
            len(train_sir_mse_eval['selected_cities']),
            len(test_sir_mse_eval['selected_cities']),
            len(train_sir_eval['selected_cities']),
            len(test_sir_eval['selected_cities']),
            len(train_rnn_mse_eval['selected_cities']),
            len(test_rnn_mse_eval['selected_cities']),
            len(train_rnn_eval['selected_cities']),
            len(test_rnn_eval['selected_cities']),
            len(train_lstm_mse_eval['selected_cities']),
            len(test_lstm_mse_eval['selected_cities']),
            len(train_lstm_eval['selected_cities']),
            len(test_lstm_eval['selected_cities'])
        ]
    })
    
    print(saidi_opt_df.to_string(index=False))
    
    # Print detailed optimization results
    print(f"\nDetailed SAIDI Optimization Results:")
    print(f"Train Ground Truth - Selected Cities: {train_gt_eval['selected_cities']}")
    print(f"Train Ground Truth - Remaining SAIDI: {train_gt_eval['remaining_saidi']:.4f}")
    print(f"Test Ground Truth - Selected Cities: {test_gt_eval['selected_cities']}")
    print(f"Test Ground Truth - Remaining SAIDI: {test_gt_eval['remaining_saidi']:.4f}")
    print(f"Train SIR (MSE) - Selected Cities: {train_sir_mse_eval['selected_cities']}")
    print(f"Train SIR (MSE) - Remaining SAIDI: {train_sir_mse_eval['remaining_saidi']:.4f}")
    print(f"Test SIR (MSE) - Selected Cities: {test_sir_mse_eval['selected_cities']}")
    print(f"Test SIR (MSE) - Remaining SAIDI: {test_sir_mse_eval['remaining_saidi']:.4f}")
    print(f"Train SIR (DFL) - Selected Cities: {train_sir_eval['selected_cities']}")
    print(f"Train SIR (DFL) - Remaining SAIDI: {train_sir_eval['remaining_saidi']:.4f}")
    print(f"Test SIR (DFL) - Selected Cities: {test_sir_eval['selected_cities']}")
    print(f"Test SIR (DFL) - Remaining SAIDI: {test_sir_eval['remaining_saidi']:.4f}")
    print(f"Train RNN (MSE) - Selected Cities: {train_rnn_mse_eval['selected_cities']}")
    print(f"Train RNN (MSE) - Remaining SAIDI: {train_rnn_mse_eval['remaining_saidi']:.4f}")
    print(f"Test RNN (MSE) - Selected Cities: {test_rnn_mse_eval['selected_cities']}")
    print(f"Test RNN (MSE) - Remaining SAIDI: {test_rnn_mse_eval['remaining_saidi']:.4f}")
    print(f"Train RNN (DFL) - Selected Cities: {train_rnn_eval['selected_cities']}")
    print(f"Train RNN (DFL) - Remaining SAIDI: {train_rnn_eval['remaining_saidi']:.4f}")
    print(f"Test RNN (DFL) - Selected Cities: {test_rnn_eval['selected_cities']}")
    print(f"Test RNN (DFL) - Remaining SAIDI: {test_rnn_eval['remaining_saidi']:.4f}")
    print(f"Train LSTM (MSE) - Selected Cities: {train_lstm_mse_eval['selected_cities']}")
    print(f"Train LSTM (MSE) - Remaining SAIDI: {train_lstm_mse_eval['remaining_saidi']:.4f}")
    print(f"Test LSTM (MSE) - Selected Cities: {test_lstm_mse_eval['selected_cities']}")
    print(f"Test LSTM (MSE) - Remaining SAIDI: {test_lstm_mse_eval['remaining_saidi']:.4f}")
    print(f"Train LSTM (DFL) - Selected Cities: {train_lstm_eval['selected_cities']}")
    print(f"Train LSTM (DFL) - Remaining SAIDI: {train_lstm_eval['remaining_saidi']:.4f}")
    print(f"Test LSTM (DFL) - Selected Cities: {test_lstm_eval['selected_cities']}")
    print(f"Test LSTM (DFL) - Remaining SAIDI: {test_lstm_eval['remaining_saidi']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"model_comparison_{args.state}_{args.model}_{timestamp}.csv"
    
    results_df.to_csv(results_filename, index=False)
    
    # Save SAIDI optimization results
    saidi_opt_filename = f"saidi_optimization_{args.state}_{args.model}_{timestamp}.csv"
    saidi_opt_df.to_csv(saidi_opt_filename, index=False)
    
    print(f"\nResults saved to {results_filename}")
    print(f"SAIDI optimization results saved to {saidi_opt_filename}")
    
    # Save timing data
    # MSE timing data
    mse_timing_df = pd.DataFrame({
        'Model': ['SIR', 'RNN', 'LSTM'],
        'MSE_Time_Seconds': [sir_mse_time, baseline_results['rnn_mse_time'], baseline_results['lstm_mse_time']],
        'MSE_Time_Minutes': [sir_mse_time/60, baseline_results['rnn_mse_time']/60, baseline_results['lstm_mse_time']/60],
        'Epochs': [args.num_MSE_epochs, args.num_MSE_epochs, args.num_MSE_epochs],
        'Random_Seed': [args.random_seed, args.random_seed, args.random_seed],
        'Num_Cities': [len(train_counties), len(train_counties), len(train_counties)],
        'Subsampled_Cities': [args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
                              args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
                              args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All']
    })
    
    mse_timing_filename = f"mse_timing_{args.state}_{args.model}_{timestamp}.csv"
    mse_timing_df.to_csv(mse_timing_filename, index=False)
    print(f"MSE timing data saved to {mse_timing_filename}")
    
    # DFL timing data
    dfl_timing_df = pd.DataFrame({
        'Model': ['SIR', 'RNN', 'LSTM'],
        'DFL_Time_Seconds': [sir_dfl_time, rnn_dfl_time, lstm_dfl_time],
        'DFL_Time_Minutes': [sir_dfl_time/60, rnn_dfl_time/60, lstm_dfl_time/60],
        'Epochs': [args.num_gdf_epochs, args.num_gdf_epochs, args.num_gdf_epochs],
        'Random_Seed': [args.random_seed, args.random_seed, args.random_seed],
        'Num_Cities': [len(train_counties), len(train_counties), len(train_counties)],
        'Subsampled_Cities': [args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
                              args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
                              args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All']
    })
    
    dfl_timing_filename = f"dfl_timing_{args.state}_{args.model}_{timestamp}.csv"
    dfl_timing_df.to_csv(dfl_timing_filename, index=False)
    print(f"DFL timing data saved to {dfl_timing_filename}")
    
    # Total timing data
    total_timing_df = pd.DataFrame({
        'Model': ['SIR', 'RNN', 'LSTM'],
        'Total_Time_Seconds': [sir_total_time, rnn_total_time, lstm_total_time],
        'Total_Time_Minutes': [sir_total_time/60, rnn_total_time/60, lstm_total_time/60],
        'MSE_Time_Seconds': [sir_mse_time, baseline_results['rnn_mse_time'], baseline_results['lstm_mse_time']],
        'DFL_Time_Seconds': [sir_dfl_time, rnn_dfl_time, lstm_dfl_time],
        'Random_Seed': [args.random_seed, args.random_seed, args.random_seed],
        'Num_Cities': [len(train_counties), len(train_counties), len(train_counties)],
        'Subsampled_Cities': [args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
                              args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
                              args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All']
    })
    
    total_timing_filename = f"total_timing_{args.state}_{args.model}_{timestamp}.csv"
    total_timing_df.to_csv(total_timing_filename, index=False)
    print(f"Total timing data saved to {total_timing_filename}")
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()
