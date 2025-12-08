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
parser.add_argument('--model', type=str, choices=['sir-ode', 'sir-weather-ode'], default='sir-ode',
                    help="Select the SIR model: 'sir-ode' (base) or 'sir-weather-ode' (with weather data)")
parser.add_argument('--state', type=str, choices=['MA', 'FL'], default='MA',
                    help="Select the state configuration: 'MA' or 'FL'")
parser.add_argument('--seed', type=int, default=42,
                    help="Set the random seed for reproducibility (default: 42)")
parser.add_argument('--num_MSE_epochs', type=int, default=1,
                    help="Number of epochs for MSE training (default: 3000)")
parser.add_argument('--num_gdf_epochs', type=int, default=500,
                    help="Number of epochs for decision-focused learning (default: 1000)")
parser.add_argument('--lambda_smoothing', type=float, default=0.1,
                    help="Smoothing parameter for CVX layer (default: 0.1)")
parser.add_argument('--lambda_mse', type=float, default=0.0000001,
                    help="Weight for MSE loss in DFL (default: 0.0)")
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
parser.add_argument('--sir_pretrained_path', type=str, default='weights/SIR_MA_sir-weather-ode_200epochs_MSE_20251207_211434.pth',
                    help="Path to pretrained SIR model weights (if provided, skip MSE training)")
# parser.add_argument('--sir_pretrained_path', type=str, default='weights/SIR_MA_sir-ode_200epochs_MSE_20251208_004658.pth',
#                     help="Path to pretrained SIR model weights (if provided, skip MSE training)")
parser.add_argument('--rnn_pretrained_path', type=str, default='weights/RNN_MA_multi-city_200epochs_MSE_20251207_214825.pth',
                    help="Path to pretrained RNN model weights (if provided, skip MSE training)")
parser.add_argument('--lstm_pretrained_path', type=str, default='weights/LSTM_MA_multi-city_200epochs_MSE_20251207_214825.pth',
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
# Evaluation helpers
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
    """Create CVXPY layer for decision-focused optimization using direct L2 penalty."""
    x = cp.Variable(n_cities)

    # SAIDI input (predicted outages aggregated to SAIDI)
    bl_SAIDI_param = cp.Parameter(n_cities)

    # Objective: maximize SAIDI - λ * ||x||_2^2
    saidi_cost = bl_SAIDI_param @ x
    l2_penalty = cp.sum_squares(x)

    objective = cp.Maximize(saidi_cost - lambda_smoothing * l2_penalty)

    constraints = [
        cp.sum(x) <= max_selected_cities,
        x >= 0,
        x <= 1
    ]

    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[bl_SAIDI_param], variables=[x])

    return cvxpylayer



def calc_SAIDI_tensor(customer, outage):
    """Calculate SAIDI using tensor operations."""
    if not isinstance(customer, torch.Tensor):
        customer = torch.tensor(customer, dtype=torch.float32)
    if not isinstance(outage, torch.Tensor):
        outage = torch.tensor(outage, dtype=torch.float32)
    
    # outage: (num_cities, T)
    total_outages = torch.sum(outage, dim=1)
    SAIDI = total_outages / customer
    return SAIDI

# def train_decision_focused_sir(
#     model, cvxpylayer, optimizer, customer, outage, 
#     initial_conditions, counties, time_points, 
#     lambda_mse=0.01, num_epochs=1000, model_option='sir-ode',
#     census_data_dict=None, weather_data_dict=None,
#     demographic_dim=0, weather_dim=0
# ):
#     """Train SIR model using decision-focused learning (regret-based)."""
#     decision_losses = []
#     mse_losses = []
    
#     print(f"Starting SIR DFL training for {num_epochs} epochs...")
#     start_time = time.time()
    
#     # precompute true SAIDI (no grad)
#     bl_SAIDI_true_np = calc_SAIDI(np.array(customer), outage)  # shape (n_cities,)
#     bl_SAIDI_true = torch.tensor(bl_SAIDI_true_np, dtype=torch.float32)

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
        
#         # Forward pass
#         y0 = torch.tensor([initial_conditions[county] for county in counties], dtype=torch.float32)
#         N_values = y0.sum(dim=1)
#         model.set_population(N_values)
        
#         if model_option == 'sir-weather-ode' and census_data_dict and weather_data_dict:
#             batch_demographic = []
#             batch_weather = []
#             for county in counties:
#                 if county in census_data_dict:
#                     batch_demographic.append(census_data_dict[county])
#                 else:
#                     batch_demographic.append(torch.zeros(demographic_dim))
#                 if county in weather_data_dict:
#                     batch_weather.append(weather_data_dict[county])
#                 else:
#                     batch_weather.append(torch.zeros(weather_dim))
#             batch_demographic = torch.stack(batch_demographic, dim=0)
#             batch_weather = torch.stack(batch_weather, dim=0)
#             model.set_demographic(batch_demographic)
#             model.set_weather(batch_weather)
        
#         y_pred = odeint(model, y0, time_points, method='rk4')
#         predicted_outages = y_pred[:, :, 1].T  # (num_cities, T)
        
#         # SAIDI based on predictions (this carries gradients)
#         bl_SAIDI_pred = calc_SAIDI_tensor(customer, predicted_outages)
        
#         # Decision layer: predicted SAIDI
#         x_opt_pred, = cvxpylayer(bl_SAIDI_pred, solver_args={"solve_method": "ECOS"})
        
#         # Decision layer: optimal decision under true SAIDI (no gradient)
#         with torch.no_grad():
#             x_opt_true, = cvxpylayer(bl_SAIDI_true, solver_args={"solve_method": "ECOS"})
        
#         # Regret-based decision loss:
#         # U_true(x*(y_true)) - U_true(x*(y_pred))
#         true_obj_opt = torch.dot(bl_SAIDI_true, x_opt_true)
#         true_obj_pred = torch.dot(bl_SAIDI_true, x_opt_pred)
#         decision_loss = true_obj_opt - true_obj_pred
#         decision_losses.append(decision_loss.item())
        
#         # MSE loss
#         true_outage = torch.tensor(outage, dtype=torch.float32)
#         mse_loss = torch.mean((predicted_outages - true_outage) ** 2)
#         mse_losses.append(mse_loss.item())
        
#         # Total loss
#         total_loss = decision_loss + lambda_mse * mse_loss
#         total_loss.backward()
#         optimizer.step()
        
#         if (epoch + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Decision Loss: {decision_loss:.4f}, MSE Loss: {mse_loss:.4f}')
    
#     end_time = time.time()
#     training_time = end_time - start_time
#     print(f"SIR DFL training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
#     return decision_losses, mse_losses, training_time

# def train_decision_focused_sir(
#     model, cvxpylayer, optimizer, customer, outage, 
#     initial_conditions, counties, time_points, 
#     lambda_mse=0.01, num_epochs=1000, model_option='sir-ode',
#     census_data_dict=None, weather_data_dict=None,
#     demographic_dim=0, weather_dim=0,
#     grad_clip=1.0,
# ):
#     """
#     Train SIR (or SIRWeather) model using decision-focused learning (regret-based).

#     Key fixes vs previous version:
#     - Everything moved to the same device as the model
#     - True SAIDI computed once on that device using the same tensor function
#     - model.train() called each epoch
#     - Gradient clipping
#     - No unnecessary tensor re-creation inside the loop
#     """
#     print(f"Starting SIR DFL training for {num_epochs} epochs...")
#     start_time = time.time()

#     # --------------------------------------------------------
#     # Device & static tensors
#     # --------------------------------------------------------
#     device = next(model.parameters()).device

#     # Customer and outage as tensors on the correct device
#     customer_t = torch.tensor(customer, dtype=torch.float32, device=device)           # (n_cities,)
#     outage_t   = torch.tensor(outage,   dtype=torch.float32, device=device)          # (n_cities, T)

#     # Time points on correct device
#     time_points = time_points.to(device)

#     # Precompute true SAIDI (no grad needed, but on same device / dtype)
#     with torch.no_grad():
#         bl_SAIDI_true = calc_SAIDI_tensor(customer_t, outage_t)  # (n_cities,)
#         # Detach just to be explicit that we don't want grads w.r.t. true SAIDI
#         bl_SAIDI_true = bl_SAIDI_true.detach()

#     decision_losses = []
#     mse_losses = []

#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()

#         # ----------------------------------------------------
#         # Build initial conditions batch on correct device
#         # ----------------------------------------------------
#         y0_list = [initial_conditions[county] for county in counties]
#         y0 = torch.tensor(y0_list, dtype=torch.float32, device=device)  # (n_cities, 3)

#         N_values = y0.sum(dim=1)
#         model.set_population(N_values)

#         # ----------------------------------------------------
#         # Set weather / census covariates (if using weather model)
#         # ----------------------------------------------------
#         if model_option == 'sir-weather-ode':
#             # If dicts are provided, build batches; otherwise zeros
#             if census_data_dict is not None and weather_data_dict is not None:
#                 batch_demographic = []
#                 batch_weather = []
#                 for county in counties:
#                     if county in census_data_dict:
#                         batch_demographic.append(census_data_dict[county].to(device))
#                     else:
#                         batch_demographic.append(torch.zeros(demographic_dim, device=device))

#                     if county in weather_data_dict:
#                         batch_weather.append(weather_data_dict[county].to(device))
#                     else:
#                         batch_weather.append(torch.zeros(weather_dim, device=device))

#                 batch_demographic = torch.stack(batch_demographic, dim=0)  # (n_cities, demo_dim)
#                 batch_weather = torch.stack(batch_weather, dim=0)          # (n_cities, weather_dim)
#             else:
#                 batch_demographic = torch.zeros(len(counties), demographic_dim, device=device)
#                 batch_weather     = torch.zeros(len(counties), weather_dim,     device=device)

#             model.set_demographic(batch_demographic)
#             model.set_weather(batch_weather)

#         # ----------------------------------------------------
#         # Forward ODE solve
#         # y_pred: (T, n_cities, 3)
#         # ----------------------------------------------------
#         y_pred = odeint(model, y0, time_points, method='rk4')  # uses adjoint; grad through parameters
#         predicted_outages = y_pred[:, :, 1].transpose(0, 1)    # (n_cities, T)

#         # ----------------------------------------------------
#         # SAIDI from predictions (this carries gradient)
#         # ----------------------------------------------------
#         bl_SAIDI_pred = calc_SAIDI_tensor(customer_t, predicted_outages)  # (n_cities,)

#         # ----------------------------------------------------
#         # CVX layer calls
#         # ----------------------------------------------------
#         # Decision layer with predicted SAIDI (grad flows through this)
#         x_opt_pred, = cvxpylayer(bl_SAIDI_pred)

#         # Oracle decision under true SAIDI (no grad)
#         with torch.no_grad():
#             x_opt_true, = cvxpylayer(bl_SAIDI_true)

#         # ----------------------------------------------------
#         # Regret-based decision loss:
#         # U_true(x*(y_true)) - U_true(x*(y_pred))
#         # ----------------------------------------------------
#         true_obj_opt  = (bl_SAIDI_true * x_opt_true).sum()
#         true_obj_pred = (bl_SAIDI_true * x_opt_pred).sum()
#         decision_loss = true_obj_opt - true_obj_pred
#         decision_losses.append(decision_loss.item())

#         # ----------------------------------------------------
#         # Prediction MSE loss over full outage trajectories
#         # ----------------------------------------------------
#         mse_loss = torch.mean((predicted_outages - outage_t) ** 2)
#         mse_losses.append(mse_loss.item())

#         # ----------------------------------------------------
#         # Total loss and backprop
#         # ----------------------------------------------------
#         total_loss = decision_loss + lambda_mse * mse_loss
#         total_loss.backward()

#         # Optional: inspect gradient norm in early epochs if needed
#         # if epoch < 5:
#         #     gn = 0.0
#         #     for p in model.parameters():
#         #         if p.grad is not None:
#         #             gn += p.grad.norm().item() ** 2
#         #     print(f"[Epoch {epoch+1}] total grad norm = {(gn ** 0.5):.4e}")

#         if grad_clip is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

#         optimizer.step()

#         if (epoch + 1) % 100 == 0:
#             print(
#                 f"Epoch [{epoch + 1}/{num_epochs}]  "
#                 f"Decision Loss: {decision_loss.item():.4f}  "
#                 f"MSE Loss: {mse_loss.item():.4f}  "
#                 f"Total: {total_loss.item():.4f}"
#             )

#     training_time = time.time() - start_time
#     print(f"SIR DFL training completed in {training_time:.2f} seconds "
#           f"({training_time/60:.2f} minutes)")

#     return decision_losses, mse_losses, training_time


# def train_decision_focused_sir(
#     model, cvxpylayer, optimizer, customer, outage, 
#     initial_conditions, counties, time_points, 
#     lambda_mse=0.01, num_epochs=1000, model_option='sir-ode',
#     census_data_dict=None, weather_data_dict=None,
#     demographic_dim=0, weather_dim=0,
#     grad_clip=1.0,
# ):
#     """
#     Train SIR (or SIRWeather) model using decision-focused learning (regret-based)
#     on CPU. This avoids GPU/CPU issues with CvxpyLayer and ensures gradients flow.
#     """
#     print(f"Starting SIR DFL training for {num_epochs} epochs...")

#     # --------------------------------------------------------
#     # FORCE CPU for DFL
#     # --------------------------------------------------------
#     device = torch.device("cpu")
#     model.to(device)
#     model.train()

#     # Static tensors on CPU
#     customer_t = torch.tensor(customer, dtype=torch.float32, device=device)      # (n_cities,)
#     outage_t   = torch.tensor(outage,   dtype=torch.float32, device=device)      # (n_cities, T)
#     time_points = time_points.to(device)

#     # Precompute true SAIDI (no grad)
#     with torch.no_grad():
#         bl_SAIDI_true = calc_SAIDI_tensor(customer_t, outage_t).detach()         # (n_cities,)

#     decision_losses = []
#     mse_losses = []
#     start_time = time.time()

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()

#         # ----------------------------------------------------
#         # Initial conditions
#         # ----------------------------------------------------
#         y0_list = [initial_conditions[county] for county in counties]
#         y0 = torch.tensor(y0_list, dtype=torch.float32, device=device)           # (n_cities, 3)

#         N_values = y0.sum(dim=1)
#         model.set_population(N_values)

#         # ----------------------------------------------------
#         # Set covariates (weather / census)
#         # ----------------------------------------------------
#         if model_option == 'sir-weather-ode':
#             if census_data_dict is not None and weather_data_dict is not None:
#                 batch_demographic = []
#                 batch_weather = []
#                 for county in counties:
#                     if county in census_data_dict:
#                         batch_demographic.append(census_data_dict[county].to(device))
#                     else:
#                         batch_demographic.append(torch.zeros(demographic_dim, device=device))

#                     if county in weather_data_dict:
#                         batch_weather.append(weather_data_dict[county].to(device))
#                     else:
#                         batch_weather.append(torch.zeros(weather_dim, device=device))

#                 batch_demographic = torch.stack(batch_demographic, dim=0)        # (n_cities, demo_dim)
#                 batch_weather = torch.stack(batch_weather, dim=0)                # (n_cities, weather_dim)
#             else:
#                 batch_demographic = torch.zeros(len(counties), demographic_dim, device=device)
#                 batch_weather     = torch.zeros(len(counties), weather_dim,     device=device)

#             model.set_demographic(batch_demographic)
#             model.set_weather(batch_weather)

#         # ----------------------------------------------------
#         # ODE solve
#         # y_pred: (T, n_cities, 3)
#         # ----------------------------------------------------
#         y_pred = odeint(model, y0, time_points, method='rk4')
#         predicted_outages = y_pred[:, :, 1].transpose(0, 1)                      # (n_cities, T)

#         # ----------------------------------------------------
#         # SAIDI from predictions (carries gradient)
#         # ----------------------------------------------------
#         bl_SAIDI_pred = calc_SAIDI_tensor(customer_t, predicted_outages)         # (n_cities,)

#         # ----------------------------------------------------
#         # CVX layer (CPU)
#         # ----------------------------------------------------
#         # predicted decision (grad flows through bl_SAIDI_pred)
#         x_opt_pred, = cvxpylayer(bl_SAIDI_pred)

#         # oracle decision under true SAIDI (no grad)
#         with torch.no_grad():
#             x_opt_true, = cvxpylayer(bl_SAIDI_true)

#         # ----------------------------------------------------
#         # Decision loss (regret)
#         # ----------------------------------------------------
#         true_obj_opt  = (bl_SAIDI_true * x_opt_true).sum()
#         true_obj_pred = (bl_SAIDI_true * x_opt_pred).sum()
#         decision_loss = true_obj_opt - true_obj_pred
#         decision_losses.append(decision_loss.item())

#         # ----------------------------------------------------
#         # MSE loss (prediction)
#         # ----------------------------------------------------
#         mse_loss = torch.mean((predicted_outages - outage_t) ** 2)
#         mse_losses.append(mse_loss.item())

#         # ----------------------------------------------------
#         # Total loss + backward
#         # ----------------------------------------------------
#         total_loss = decision_loss + lambda_mse * mse_loss
#         total_loss.backward()

#         if grad_clip is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

#         optimizer.step()

#         if (epoch + 1) % 20 == 0:
#             # quick sanity check: param norm should change over time
#             with torch.no_grad():
#                 param_norm = torch.sqrt(sum((p**2).sum() for p in model.parameters()))
#             print(
#                 f"[Epoch {epoch+1}/{num_epochs}]  "
#                 f"DecLoss: {decision_loss.item():.4f}  "
#                 f"MSE: {mse_loss.item():.4f}  "
#                 f"Total: {total_loss.item():.4f}  "
#                 f"||θ||: {param_norm.item():.4f}"
#             )

#     training_time = time.time() - start_time
#     print(f"SIR DFL training completed in {training_time:.2f} seconds "
#           f"({training_time/60:.2f} minutes)")

#     return decision_losses, mse_losses, training_time

def train_decision_focused_sir(
    model, cvxpylayer, optimizer, customer, outage,
    initial_conditions, counties, time_points,
    lambda_mse=0.01, num_epochs=1000, model_option='sir-ode',
    census_data_dict=None, weather_data_dict=None,
    demographic_dim=0, weather_dim=0,
    grad_clip=1.0,
):
    """
    Decision-focused training for SIR / SIRWeather.

    - Everything runs on CPU (safer with cvxpylayer on Mac).
    - Includes MSE term so params MUST move if gradients are flowing.
    """
    print(f"Starting SIR DFL training for {num_epochs} epochs...")

    device = torch.device("cpu")
    model.to(device)
    model.train()

    # Static tensors on CPU
    customer_t = torch.tensor(customer, dtype=torch.float32, device=device)     # (n_cities,)
    outage_t   = torch.tensor(outage,   dtype=torch.float32, device=device)     # (n_cities, T)
    time_points = time_points.to(device)

    # True SAIDI (no grad)
    with torch.no_grad():
        bl_SAIDI_true = calc_SAIDI_tensor(customer_t, outage_t).detach()        # (n_cities,)

    decision_losses = []
    mse_losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # ------------------------------
        # Initial conditions per county
        # ------------------------------
        y0_list = [initial_conditions[county] for county in counties]
        y0 = torch.tensor(y0_list, dtype=torch.float32, device=device)          # (B, 3)
        B = y0.shape[0]

        N_values = y0.sum(dim=1)                                                # (B,)
        model.set_population(N_values)

        # ------------------------------
        # Covariates (SIRWeather only)
        # ------------------------------
        if model_option == 'sir-weather-ode':
            if census_data_dict is not None and weather_data_dict is not None:
                demo_batch = []
                weather_batch = []
                for county in counties:
                    if county in census_data_dict:
                        demo_batch.append(census_data_dict[county].to(device))
                    else:
                        demo_batch.append(torch.zeros(demographic_dim, device=device))

                    if county in weather_data_dict:
                        weather_batch.append(weather_data_dict[county].to(device))
                    else:
                        weather_batch.append(torch.zeros(weather_dim, device=device))

                demo_batch = torch.stack(demo_batch, dim=0)      # (B, demo_dim)
                weather_batch = torch.stack(weather_batch, dim=0)  # (B, weather_dim)
            else:
                demo_batch = torch.zeros(B, demographic_dim, device=device)
                weather_batch = torch.zeros(B, weather_dim, device=device)

            model.set_demographic(demo_batch)
            model.set_weather(weather_batch)

        # ------------------------------
        # ODE solve
        # ------------------------------
        # y_pred: [T, B, 3]
        y_pred = odeint(model, y0, time_points, method='rk4')

        # Predicted outages per city/time: [B, T]
        predicted_outages = y_pred[:, :, 1].transpose(0, 1)                      # (B, T)

        # ------------------------------
        # SAIDI from predictions (grad)
        # ------------------------------
        bl_SAIDI_pred = calc_SAIDI_tensor(customer_t, predicted_outages)         # (B,)

        # ------------------------------
        # CVX layer (differentiable decision)
        # ------------------------------
        x_opt_pred, = cvxpylayer(bl_SAIDI_pred)                                  # decision for predicted SAIDI

        with torch.no_grad():
            x_opt_true, = cvxpylayer(bl_SAIDI_true)                              # oracle decision

        # ------------------------------
        # Decision loss (regret)
        # ------------------------------
        true_obj_opt  = (bl_SAIDI_true * x_opt_true).sum()
        true_obj_pred = (bl_SAIDI_true * x_opt_pred).sum()
        decision_loss = true_obj_opt - true_obj_pred
        decision_losses.append(decision_loss.item())

        # ------------------------------
        # MSE loss on full trajectories
        # ------------------------------
        mse_loss = torch.mean((predicted_outages - outage_t) ** 2)
        mse_losses.append(mse_loss.item())

        # ------------------------------
        # Total loss & backprop
        # ------------------------------
        total_loss = decision_loss + lambda_mse * mse_loss
        total_loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                param_norm = torch.sqrt(sum((p**2).sum() for p in model.parameters()))
            print(
                f"[Epoch {epoch+1}/{num_epochs}]  "
                f"DecLoss: {decision_loss.item():.4f}  "
                f"MSE: {mse_loss.item():.4f}  "
                f"Total: {total_loss.item():.4f}  "
                f"||θ||: {param_norm.item():.4f}"
            )

    training_time = time.time() - start_time
    print(f"SIR DFL training completed in {training_time:.2f} seconds "
          f"({training_time/60:.2f} minutes)")

    return decision_losses, mse_losses, training_time


def train_decision_focused_rnn_lstm(
    model, cvxpylayer, optimizer, customer, outage, 
    train_data, counties, seq_len, 
    lambda_mse=0.01, num_epochs=1000, max_time_steps=None
):
    """
    Train RNN/LSTM model using decision-focused learning (regret-based).
    Reconstructs a (num_cities x max_time_steps) outage tensor from sequence predictions.
    """
    decision_losses = []
    mse_losses = []
    
    print(f"Starting RNN/LSTM DFL training for {num_epochs} epochs...")
    start_time = time.time()
    
    # Create sequences for training, tracking (county_idx, time_idx)
    def create_sequences(data, counties, seq_len):
        sequences = []
        targets = []
        meta = []  # (county_idx, time_index_within_county)
        
        county_to_idx = {county: idx for idx, county in enumerate(counties)}
        max_len_per_county = 0
        
        for county in counties:
            county_data = data[data['county'] == county].sort_values('datetime').reset_index(drop=True)
            length = len(county_data)
            max_len_per_county = max(max_len_per_county, length)
            
            for t in range(seq_len, length):
                seq = county_data['total_outage'].iloc[t-seq_len:t].values.reshape(-1, 1)
                target = county_data['total_outage'].iloc[t]
                
                sequences.append(seq)
                targets.append(target)
                meta.append((county_to_idx[county], t))  # time index t in this county
        
        return np.array(sequences), np.array(targets), meta, county_to_idx, max_len_per_county
    
    X_train_np, y_train_np, meta, county_to_idx, max_len_per_county = create_sequences(
        train_data, counties, seq_len
    )
    
    if max_time_steps is None:
        max_time_steps = max_len_per_county
    
    X_train = torch.FloatTensor(X_train_np)
    y_train = torch.FloatTensor(y_train_np).unsqueeze(1)
    
    # Precompute true SAIDI (no grad) from full outage matrix
    bl_SAIDI_true_np = calc_SAIDI(np.array(customer), outage)
    bl_SAIDI_true = torch.tensor(bl_SAIDI_true_np, dtype=torch.float32)
    
    num_cities = len(counties)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass through RNN/LSTM
        train_pred = model(X_train)   # shape (num_sequences, 1)
        
        # Reconstruct per-city time series tensor (num_cities x max_time_steps)
        device = train_pred.device
        predicted_outages = torch.zeros(num_cities, max_time_steps, dtype=train_pred.dtype, device=device)
        
        for k, (c_idx, t_idx) in enumerate(meta):
            if t_idx < max_time_steps:
                predicted_outages[c_idx, t_idx] = train_pred[k, 0]
        
        # SAIDI from predicted outages (this carries gradients)
        bl_SAIDI_pred = calc_SAIDI_tensor(customer, predicted_outages)
        
        # Decision layer: predicted SAIDI
        x_opt_pred, = cvxpylayer(bl_SAIDI_pred, solver_args={"solve_method": "ECOS"})
        
        # Decision layer: optimal decision under true SAIDI (no grad)
        with torch.no_grad():
            x_opt_true, = cvxpylayer(bl_SAIDI_true, solver_args={"solve_method": "ECOS"})
        
        # Regret-based decision loss
        true_obj_opt = torch.dot(bl_SAIDI_true, x_opt_true)
        true_obj_pred = torch.dot(bl_SAIDI_true, x_opt_pred)
        decision_loss = true_obj_opt - true_obj_pred
        decision_losses.append(decision_loss.item())
        
        # Prediction MSE at sequence level (same as your original)
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
    os.makedirs("weights", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if additional_info:
        filename = f"weights/{model_type}_{state}_{model_option}_{num_epochs}epochs_{additional_info}_{timestamp}.pth"
    else:
        filename = f"weights/{model_type}_{state}_{model_option}_{num_epochs}epochs_{timestamp}.pth"
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
# Model training functions (MSE)
# ============================================================
def train_sir_model(
    model, train_data, train_counties, train_initial_conditions, 
    county_data_dict, time_points, num_epochs=3000, 
    model_option='sir-ode', census_data_dict=None, weather_data_dict=None,
    demographic_dim=0, weather_dim=0
):
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
        
        # NaN checks on first epoch
        if epoch == 0:
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"Warning: NaN values detected in model parameter {name}")
                    print(f"Parameter shape: {param.shape}")
                    print(f"Parameter values: {param}")
                    return epoch_losses, 0
        
        for batch in batches:
            y0 = torch.tensor([train_initial_conditions[county] for county in batch], dtype=torch.float32)
            
            if torch.isnan(y0).any():
                print(f"Warning: NaN in initial conditions at epoch {epoch+1}")
                print(f"Batch counties: {batch}")
                continue
            
            N_values = y0.sum(dim=1)
            if (N_values <= 0).any():
                print(f"Warning: non-positive population at epoch {epoch+1}")
                print(f"N_values: {N_values}")
                continue
            
            model.set_population(N_values)
            
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
                batch_demographic = torch.zeros(len(batch), demographic_dim)
                batch_weather = torch.zeros(len(batch), weather_dim)
                model.set_demographic(batch_demographic)
                model.set_weather(batch_weather)
            
            optimizer.zero_grad()
            y_pred = odeint(model, y0, time_points, method='rk4')
            
            if torch.isnan(y_pred).any():
                print(f"Warning: NaN in y_pred at epoch {epoch+1}")
                continue
            
            batch_loss = 0
            for i, county in enumerate(batch):
                actual_outage_values = county_data_dict[county]['total_outage'].values[:len(time_points)]
                actual_outages = torch.zeros(len(time_points), dtype=torch.float32)
                actual_outages[:len(actual_outage_values)] = torch.tensor(actual_outage_values, dtype=torch.float32)
                predicted_outages = y_pred[:, i, 1]
                batch_loss += criterion(predicted_outages, actual_outages)
            
            batch_loss /= len(batch)
            
            if torch.isnan(batch_loss):
                print(f"Warning: NaN loss at epoch {epoch+1}")
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

def train_rnn_lstm_models(
    train_data, test_data, seq_len=24, hidden_dim=64, num_layers=2, 
    num_epochs=100, lr=0.001
):
    """Train RNN and LSTM models with multi-city support (MSE only)."""
    # Create sequences
    def create_sequences(data, seq_len=24):
        sequences = []
        targets = []
        counties = []
        datetimes = []
        county_indices = []
        
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
    
    print("Training RNN model...")
    rnn_start_time = time.time()
    rnn_model, rnn_train_losses, rnn_test_losses = train_model(
        rnn_model, X_train, y_train, X_test, y_test, train_county_indices, test_county_indices, num_epochs, lr
    )
    rnn_mse_time = time.time() - rnn_start_time
    print(f"RNN MSE training completed in {rnn_mse_time:.2f} seconds ({rnn_mse_time/60:.2f} minutes)")
    
    print("Training LSTM model...")
    lstm_start_time = time.time()
    lstm_model, lstm_train_losses, lstm_test_losses = train_model(
        lstm_model, X_train, y_train, X_test, y_test, train_county_indices, test_county_indices, num_epochs, lr
    )
    lstm_mse_time = time.time() - lstm_start_time
    print(f"LSTM MSE training completed in {lstm_mse_time:.2f} seconds ({lstm_mse_time/60:.2f} minutes)")
    
    with torch.no_grad():
        rnn_train_pred = rnn_model(X_train, train_county_indices).numpy()
        rnn_test_pred = rnn_model(X_test, test_county_indices).numpy()
        lstm_train_pred = lstm_model(X_train, train_county_indices).numpy()
        lstm_test_pred = lstm_model(X_test, test_county_indices).numpy()
    
    rnn_train_mse = np.mean((y_train.numpy() - rnn_train_pred)**2)
    rnn_test_mse = np.mean((y_test.numpy() - rnn_test_pred)**2)
    lstm_train_mse = np.mean((y_train.numpy() - lstm_train_pred)**2)
    lstm_test_mse = np.mean((y_test.numpy() - lstm_test_pred)**2)
    
    print(f"RNN - Train MSE: {rnn_train_mse:.4f}, Test MSE: {rnn_test_mse:.4f}")
    print(f"LSTM - Train MSE: {lstm_train_mse:.4f}, Test MSE: {lstm_test_mse:.4f}")
    
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
# Evaluation functions (prediction)
# ============================================================
def evaluate_sir_model(
    model, initial_conditions, counties, time_points, 
    model_option='sir-ode', census_data_dict=None, weather_data_dict=None,
    demographic_dim=0, weather_dim=0
):
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
            batch_demographic = torch.zeros(len(counties), demographic_dim)
            batch_weather = torch.zeros(len(counties), weather_dim)
            model.set_demographic(batch_demographic)
            model.set_weather(batch_weather)
        
        y_pred = odeint(model, y0, time_points, method='rk4')
        predictions = y_pred[:, :, 1].detach().cpu().numpy().T
    
    return predictions

def evaluate_rnn_lstm_model(model, test_data, counties, seq_len, max_time_steps=None, unique_counties=None):
    """Evaluate RNN/LSTM model predictions with multi-city support."""
    model.eval()
    predictions = []
    
    if unique_counties is not None:
        county_to_idx = {county: idx for idx, county in enumerate(unique_counties)}
    else:
        unique_counties = test_data['county'].unique()
        county_to_idx = {county: idx for idx, county in enumerate(unique_counties)}
    
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
            
            county_idx = county_to_idx.get(county, 0)
            county_idx_tensor = torch.LongTensor([county_idx])
            
            for i in range(seq_len, len(county_data)):
                seq = county_data['total_outage'].iloc[i-seq_len:i].values.reshape(1, -1, 1)
                seq_tensor = torch.FloatTensor(seq)
                pred = model(seq_tensor, county_idx_tensor).numpy()[0, 0]
                county_predictions.append(pred)
            
            county_predictions = [0] * seq_len + county_predictions
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
            np.random.seed(args.random_seed)
            subsampled_counties = np.random.choice(all_counties, args.num_subsampled_cities, replace=False)
            print(f"Subsampling {args.num_subsampled_cities} cities from {len(all_counties)} total cities")
            print(f"Selected cities: {list(subsampled_counties)}")
            
            data['outage_data'] = data['outage_data'][data['outage_data']['county'].isin(subsampled_counties)]
            if 'census_data' in data and data['census_data'] is not None:
                data['census_data'] = data['census_data'][data['census_data']['county'].isin(subsampled_counties)]
            if 'weather_data' in data and data['weather_data'] is not None:
                data['weather_data'] = {k: v for k, v in data['weather_data'].items() if k in subsampled_counties}
            
            print(f"Data filtered to {len(data['outage_data']['county'].unique())} counties")
        else:
            print(f"Number of subsampled cities ({args.num_subsampled_cities}) >= total cities ({len(all_counties)}), using all cities")
    else:
        print("Using all available cities")
    
    # Split train/test
    train_data, test_data, train_counties, test_counties = split_train_test(
        data['outage_data'], data['total_customer_dict'], config
    )
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
    
    # Prepare census data
    census_data_dict = {}
    if data['census_data'] is not None:
        census_data_clean = data['census_data'].copy()
        numeric_columns = census_data_clean.select_dtypes(include=[np.number]).columns
        census_data_clean[numeric_columns] = census_data_clean[numeric_columns].fillna(census_data_clean[numeric_columns].mean())
        
        census_data_dict = {
            row['County']: torch.tensor(row[1:].values.tolist(), dtype=torch.float32)
            for _, row in census_data_clean.iterrows()
        }
        print("census_data loaded (NaN values filled with column means)")
        print(census_data_dict)
    
    # Prepare weather data
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
    
    # Time points (global max over train/test)
    max_time_steps = max(
        train_data.groupby('county').size().max(),
        test_data.groupby('county').size().max()
    )
    time_points = torch.linspace(1, max_time_steps, max_time_steps)
    
    # County data dictionary for SIR training
    county_data_dict = {
        county: train_data[train_data['county'] == county]
        for county in train_counties
    }
    
    # ============================================================
    # Train SIR Model (MSE)
    # ============================================================
    print("\n" + "="*40)
    print("TRAINING SIR MODEL")
    print("="*40)
    
    if args.model == 'sir-ode':
        sir_model = BaseSIRModel()
    else:
        sir_model = SIRWeatherModel(demographic_dim=demographic_dim, weather_dim=weather_dim)
    
    # Pretrained?
    if args.sir_pretrained_path:
        print(f"Loading pretrained SIR model from: {args.sir_pretrained_path}")
        if load_model_weights(sir_model, args.sir_pretrained_path, "SIR"):
            sir_mse_time = 0
            sir_losses = []
            print("SIR model loaded successfully, skipping MSE training")
        else:
            print("Failed to load pretrained SIR model, training from scratch...")
            sir_losses, sir_mse_time = train_sir_model(
                sir_model, train_data, train_counties, train_initial_conditions,
                county_data_dict, time_points, args.num_MSE_epochs, args.model,
                census_data_dict, weather_data_dict, demographic_dim, weather_dim
            )
            save_model_weights(sir_model, "SIR", args.state, args.model, args.num_MSE_epochs, "MSE")
    else:
        print("Training SIR model from scratch...")
        sir_losses, sir_mse_time = train_sir_model(
            sir_model, train_data, train_counties, train_initial_conditions,
            county_data_dict, time_points, args.num_MSE_epochs, args.model,
            census_data_dict, weather_data_dict, demographic_dim, weather_dim
        )
        save_model_weights(sir_model, "SIR", args.state, args.model, args.num_MSE_epochs, "MSE")
    
    # ============================================================
    # Train RNN and LSTM Models (MSE)
    # ============================================================
    print("\n" + "="*40)
    print("TRAINING RNN AND LSTM MODELS")
    print("="*40)
    
    rnn_pretrained_loaded = False
    lstm_pretrained_loaded = False
    
    unique_counties = list(set(train_counties) | set(test_counties))
    
    if args.rnn_pretrained_path or args.lstm_pretrained_path:
        print("Loading pretrained RNN/LSTM models...")
        
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
        
        if args.rnn_pretrained_path:
            if load_model_weights(rnn_model, args.rnn_pretrained_path, "RNN"):
                rnn_pretrained_loaded = True
                print("RNN model loaded successfully, skipping MSE training")
            else:
                print("Failed to load pretrained RNN model, will train from scratch")
        
        if args.lstm_pretrained_path:
            if load_model_weights(lstm_model, args.lstm_pretrained_path, "LSTM"):
                lstm_pretrained_loaded = True
                print("LSTM model loaded successfully, skipping MSE training")
            else:
                print("Failed to load pretrained LSTM model, will train from scratch")
    
    if not (rnn_pretrained_loaded and lstm_pretrained_loaded):
        print("Training RNN and LSTM models...")
        baseline_results = train_rnn_lstm_models(
            train_data, test_data, seq_len=args.seq_len, 
            hidden_dim=args.hidden_dim, num_layers=args.num_layers, 
            num_epochs=args.num_MSE_epochs, lr=0.001
        )
        
        if not rnn_pretrained_loaded:
            save_model_weights(baseline_results['rnn_model'], "RNN", args.state, "multi-city", args.num_MSE_epochs, "MSE")
        if not lstm_pretrained_loaded:
            save_model_weights(baseline_results['lstm_model'], "LSTM", args.state, "multi-city", args.num_MSE_epochs, "MSE")
    else:
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
    
    train_customer = [data['total_customer_dict'][county] for county in train_counties]
    test_customer = [data['total_customer_dict'][county] for county in test_counties]
    
    train_true_outages = create_outage_array_from_df(train_data, train_counties, max_time_steps)
    test_true_outages = create_outage_array_from_df(test_data, test_counties, max_time_steps)
    
    # SIR
    print("Evaluating MSE-trained SIR model...")
    sir_mse_train_pred = evaluate_sir_model(
        sir_model, train_initial_conditions, train_counties, time_points,
        args.model, census_data_dict, weather_data_dict, demographic_dim, weather_dim
    )
    sir_mse_test_pred = evaluate_sir_model(
        sir_model, test_initial_conditions, test_counties, time_points,
        args.model, census_data_dict, weather_data_dict, demographic_dim, weather_dim
    )
    
    sir_mse_train_mse = calculate_mse(sir_mse_train_pred, train_true_outages)
    sir_mse_test_mse = calculate_mse(sir_mse_test_pred, test_true_outages)
    
    sir_mse_train_saidi = calculate_saidi_metrics(train_customer, sir_mse_train_pred)
    sir_mse_test_saidi = calculate_saidi_metrics(test_customer, sir_mse_test_pred)
    
    print(f"MSE-trained SIR - Train MSE: {sir_mse_train_mse:.4f}, Test MSE: {sir_mse_test_mse:.4f}")
    print(f"MSE-trained SIR - Train SAIDI: {np.mean(sir_mse_train_saidi):.4f}, Test SAIDI: {np.mean(sir_mse_test_saidi):.4f}")
    
    # RNN
    print("Evaluating MSE-trained RNN model...")
    rnn_mse_train_pred = evaluate_rnn_lstm_model(
        baseline_results['rnn_model'], train_data, train_counties, args.seq_len, max_time_steps, baseline_results['unique_counties']
    )
    rnn_mse_test_pred = evaluate_rnn_lstm_model(
        baseline_results['rnn_model'], test_data, test_counties, args.seq_len, max_time_steps, baseline_results['unique_counties']
    )
    
    rnn_mse_train_mse = calculate_mse(rnn_mse_train_pred, train_true_outages)
    rnn_mse_test_mse = calculate_mse(rnn_mse_test_pred, test_true_outages)
    
    rnn_mse_train_saidi = calculate_saidi_metrics(train_customer, rnn_mse_train_pred)
    rnn_mse_test_saidi = calculate_saidi_metrics(test_customer, rnn_mse_test_pred)
    
    print(f"MSE-trained RNN - Train MSE: {rnn_mse_train_mse:.4f}, Test MSE: {rnn_mse_test_mse:.4f}")
    print(f"MSE-trained RNN - Train SAIDI: {np.mean(rnn_mse_train_saidi):.4f}, Test SAIDI: {np.mean(rnn_mse_test_saidi):.4f}")
    
    # LSTM
    print("Evaluating MSE-trained LSTM model...")
    lstm_mse_train_pred = evaluate_rnn_lstm_model(
        baseline_results['lstm_model'], train_data, train_counties, args.seq_len, max_time_steps, baseline_results['unique_counties']
    )
    lstm_mse_test_pred = evaluate_rnn_lstm_model(
        baseline_results['lstm_model'], test_data, test_counties, args.seq_len, max_time_steps, baseline_results['unique_counties']
    )
    
    lstm_mse_train_mse = calculate_mse(lstm_mse_train_pred, train_true_outages)
    lstm_mse_test_mse = calculate_mse(lstm_mse_test_pred, test_true_outages)
    
    lstm_mse_train_saidi = calculate_saidi_metrics(train_customer, lstm_mse_train_pred)
    lstm_mse_test_saidi = calculate_saidi_metrics(test_customer, lstm_mse_test_pred)
    
    print(f"MSE-trained LSTM - Train MSE: {lstm_mse_train_mse:.4f}, Test MSE: {lstm_mse_test_mse:.4f}")
    print(f"MSE-trained LSTM - Train SAIDI: {np.mean(lstm_mse_train_saidi):.4f}, Test SAIDI: {np.mean(lstm_mse_test_saidi):.4f}")
    
    # SAIDI optimization for MSE-trained models
    print("\n--- MSE-trained Models SAIDI Optimization ---")
    
    # SIR (MSE)
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
    
    # RNN (MSE)
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
    
    # LSTM (MSE)
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
    
    # Recompute (to be explicit)
    train_customer = [data['total_customer_dict'][county] for county in train_counties]
    test_customer = [data['total_customer_dict'][county] for county in test_counties]
    train_true_outages = create_outage_array_from_df(train_data, train_counties, max_time_steps)
    test_true_outages = create_outage_array_from_df(test_data, test_counties, max_time_steps)
    
    # SIR DFL
    # print("Training SIR model with DFL...")
    # cvxpylayer_sir = create_cvxpy_layer(args.lambda_smoothing, args.max_selected_cities, len(train_counties))
    # sir_optimizer = torch.optim.Adam(sir_model.parameters(), lr=1e-2)
    
    # sir_decision_losses, sir_mse_losses, sir_dfl_time = train_decision_focused_sir(
    #     sir_model, cvxpylayer_sir, sir_optimizer, train_customer, train_true_outages,
    #     train_initial_conditions, train_counties, time_points, args.lambda_mse, args.num_gdf_epochs, 
    #     args.model, census_data_dict, weather_data_dict, demographic_dim, weather_dim
    # )

    # Choose device once
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # sir_model.to(device)

    # Build these once on CPU, then train_decision_focused_sir will move them to device
    # train_customer = [data['total_customer_dict'][county] for county in train_counties]
    # train_true_outages = create_outage_array_from_df(train_data, train_counties, max_time_steps)

    print("Training SIR model with DFL...")
    cvxpylayer_sir = create_cvxpy_layer(
        args.lambda_smoothing, args.max_selected_cities, len(train_counties)
    )
    sir_optimizer = torch.optim.Adam(sir_model.parameters(), lr=1e-2)

    sir_decision_losses, sir_mse_losses, sir_dfl_time = train_decision_focused_sir(
        sir_model,
        cvxpylayer_sir,
        sir_optimizer,
        customer=train_customer,
        outage=train_true_outages,
        initial_conditions=train_initial_conditions,
        counties=train_counties,
        time_points=time_points,          # will be moved to device inside
        lambda_mse=args.lambda_mse,
        num_epochs=args.num_gdf_epochs,
        model_option=args.model,
        census_data_dict=census_data_dict,
        weather_data_dict=weather_data_dict,
        demographic_dim=demographic_dim,
        weather_dim=weather_dim,
        grad_clip=1.0,
    )


    save_model_weights(sir_model, "SIR", args.state, args.model, args.num_gdf_epochs, "DFL")
    
    # RNN DFL
    print("Training RNN model with DFL...")
    cvxpylayer_rnn = create_cvxpy_layer(args.lambda_smoothing, args.max_selected_cities, len(train_counties))
    rnn_optimizer = torch.optim.Adam(baseline_results['rnn_model'].parameters(), lr=1e-2)
    
    rnn_decision_losses, rnn_mse_losses, rnn_dfl_time = train_decision_focused_rnn_lstm(
        baseline_results['rnn_model'], cvxpylayer_rnn, rnn_optimizer, 
        train_customer, train_true_outages, train_data, train_counties, args.seq_len,
        args.lambda_mse, args.num_gdf_epochs, max_time_steps
    )
    save_model_weights(baseline_results['rnn_model'], "RNN", args.state, "multi-city", args.num_gdf_epochs, "DFL")
    
    # LSTM DFL
    print("Training LSTM model with DFL...")
    cvxpylayer_lstm = create_cvxpy_layer(args.lambda_smoothing, args.max_selected_cities, len(train_counties))
    lstm_optimizer = torch.optim.Adam(baseline_results['lstm_model'].parameters(), lr=1e-2)
    
    lstm_decision_losses, lstm_mse_losses, lstm_dfl_time = train_decision_focused_rnn_lstm(
        baseline_results['lstm_model'], cvxpylayer_lstm, lstm_optimizer, 
        train_customer, train_true_outages, train_data, train_counties, args.seq_len,
        args.lambda_mse, args.num_gdf_epochs, max_time_steps
    )
    save_model_weights(baseline_results['lstm_model'], "LSTM", args.state, "multi-city", args.num_gdf_epochs, "DFL")
    
    # ============================================================
    # Evaluation after DFL
    # ============================================================
    print("\n" + "="*40)
    print("MODEL EVALUATION")
    print("="*40)
    
    # SIR
    print("Evaluating SIR model...")
    sir_train_pred = evaluate_sir_model(
        sir_model, train_initial_conditions, train_counties, time_points,
        args.model, census_data_dict, weather_data_dict, demographic_dim, weather_dim
    )
    sir_test_pred = evaluate_sir_model(
        sir_model, test_initial_conditions, test_counties, time_points,
        args.model, census_data_dict, weather_data_dict, demographic_dim, weather_dim
    )
    
    sir_train_mse = calculate_mse(sir_train_pred, train_true_outages)
    sir_test_mse = calculate_mse(sir_test_pred, test_true_outages)
    
    sir_train_saidi = calculate_saidi_metrics(train_customer, sir_train_pred)
    sir_test_saidi = calculate_saidi_metrics(test_customer, sir_test_pred)
    
    print(f"SIR Model - Train MSE: {sir_train_mse:.4f}, Test MSE: {sir_test_mse:.4f}")
    print(f"SIR Model - Train SAIDI: {np.mean(sir_train_saidi):.4f}, Test SAIDI: {np.mean(sir_test_saidi):.4f}")
    
    # RNN
    print("Evaluating RNN model...")
    rnn_train_pred = evaluate_rnn_lstm_model(
        baseline_results['rnn_model'], train_data, train_counties, args.seq_len, max_time_steps, baseline_results['unique_counties']
    )
    rnn_test_pred = evaluate_rnn_lstm_model(
        baseline_results['rnn_model'], test_data, test_counties, args.seq_len, max_time_steps, baseline_results['unique_counties']
    )
    
    rnn_train_mse = calculate_mse(rnn_train_pred, train_true_outages)
    rnn_test_mse = calculate_mse(rnn_test_pred, test_true_outages)
    
    rnn_train_saidi = calculate_saidi_metrics(train_customer, rnn_train_pred)
    rnn_test_saidi = calculate_saidi_metrics(test_customer, rnn_test_pred)
    
    print(f"RNN Model - Train MSE: {rnn_train_mse:.4f}, Test MSE: {rnn_test_mse:.4f}")
    print(f"RNN Model - Train SAIDI: {np.mean(rnn_train_saidi):.4f}, Test SAIDI: {np.mean(rnn_test_saidi):.4f}")
    
    # LSTM
    print("Evaluating LSTM model...")
    lstm_train_pred = evaluate_rnn_lstm_model(
        baseline_results['lstm_model'], train_data, train_counties, args.seq_len, max_time_steps, baseline_results['unique_counties']
    )
    lstm_test_pred = evaluate_rnn_lstm_model(
        baseline_results['lstm_model'], test_data, test_counties, args.seq_len, max_time_steps, baseline_results['unique_counties']
    )
    
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
    
    train_gt_saidi = calculate_saidi_metrics(train_customer, train_true_outages)
    test_gt_saidi = calculate_saidi_metrics(test_customer, test_true_outages)
    
    print(f"Ground Truth - Train SAIDI: {np.mean(train_gt_saidi):.4f}")
    print(f"Ground Truth - Test SAIDI: {np.mean(test_gt_saidi):.4f}")
    
    print("\n" + "="*40)
    print("SAIDI OPTIMIZATION (KNAPSACK PROBLEM)")
    print("="*40)
    
    train_total_saidi = np.sum(train_gt_saidi)
    test_total_saidi = np.sum(test_gt_saidi)
    
    print(f"Total Train SAIDI: {train_total_saidi:.4f}")
    print(f"Total Test SAIDI: {test_total_saidi:.4f}")
    
    # Ground truth optimization
    print("\n--- Ground Truth SAIDI Optimization ---")
    train_gt_selected = select_cities_for_hardening(
        train_true_outages, train_customer, args.max_selected_cities, 
        "Train Ground Truth"
    )
    train_gt_eval = evaluate_saidi_hardened(
        train_gt_selected, train_customer, train_true_outages,
        "Train Ground Truth"
    )
    
    test_gt_selected = select_cities_for_hardening(
        test_true_outages, test_customer, args.max_selected_cities,
        "Test Ground Truth"
    )
    test_gt_eval = evaluate_saidi_hardened(
        test_gt_selected, test_customer, test_true_outages,
        "Test Ground Truth"
    )
    
    # SIR optimization after DFL
    print("\n--- SIR Model SAIDI Optimization ---")
    train_sir_selected = select_cities_for_hardening(
        sir_train_pred, train_customer, args.max_selected_cities,
        "Train SIR Predictions"
    )
    train_sir_eval = evaluate_saidi_hardened(
        train_sir_selected, train_customer, train_true_outages,
        "Train SIR Predictions"
    )
    
    test_sir_selected = select_cities_for_hardening(
        sir_test_pred, test_customer, args.max_selected_cities,
        "Test SIR Predictions"
    )
    test_sir_eval = evaluate_saidi_hardened(
        test_sir_selected, test_customer, test_true_outages,
        "Test SIR Predictions"
    )
    
    # RNN optimization after DFL
    print("\n--- RNN Model SAIDI Optimization ---")
    train_rnn_selected = select_cities_for_hardening(
        rnn_train_pred, train_customer, args.max_selected_cities,
        "Train RNN Predictions"
    )
    train_rnn_eval = evaluate_saidi_hardened(
        train_rnn_selected, train_customer, train_true_outages,
        "Train RNN Predictions"
    )
    
    test_rnn_selected = select_cities_for_hardening(
        rnn_test_pred, test_customer, args.max_selected_cities,
        "Test RNN Predictions"
    )
    test_rnn_eval = evaluate_saidi_hardened(
        test_rnn_selected, test_customer, test_true_outages,
        "Test RNN Predictions"
    )
    
    # LSTM optimization after DFL
    print("\n--- LSTM Model SAIDI Optimization ---")
    train_lstm_selected = select_cities_for_hardening(
        lstm_train_pred, train_customer, args.max_selected_cities,
        "Train LSTM Predictions"
    )
    train_lstm_eval = evaluate_saidi_hardened(
        train_lstm_selected, train_customer, train_true_outages,
        "Train LSTM Predictions"
    )
    
    test_lstm_selected = select_cities_for_hardening(
        lstm_test_pred, test_customer, args.max_selected_cities,
        "Test LSTM Predictions"
    )
    test_lstm_eval = evaluate_saidi_hardened(
        test_lstm_selected, test_customer, test_true_outages,
        "Test LSTM Predictions"
    )
    
    # ============================================================
    # Validation: Alignment and Period Consistency
    # ============================================================
    print("\n" + "="*40)
    print("VALIDATION: ALIGNMENT AND PERIOD CONSISTENCY")
    print("="*40)
    
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
    
    print(f"\nTime period alignment:")
    print(f"  Max time steps: {max_time_steps}")
    print(f"  Train counties: {len(train_counties)}")
    print(f"  Test counties: {len(test_counties)}")
    
    print(f"\nSAIDI calculation validation:")
    print(f"  All models use same customer counts: {len(train_customer) == len(test_customer) == len(train_counties) == len(test_counties)}")
    print(f"  All predictions have same shape as ground truth: {sir_train_pred.shape == train_true_outages.shape}")
    
    print(f"\nGround truth data verification:")
    print(f"  Train data period: {train_data['datetime'].min()} to {train_data['datetime'].max()}")
    print(f"  Test data period: {test_data['datetime'].min()} to {test_data['datetime'].max()}")
    print(f"  Train ground truth SAIDI calculated from train period data: ✓")
    print(f"  Test ground truth SAIDI calculated from test period data: ✓")
    
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
        'Train_Remaining_SAIDI': [
            train_sir_mse_eval['remaining_saidi'], train_sir_eval['remaining_saidi'],
            train_rnn_mse_eval['remaining_saidi'], train_rnn_eval['remaining_saidi'],
            train_lstm_mse_eval['remaining_saidi'], train_lstm_eval['remaining_saidi'],
            train_gt_eval['remaining_saidi']
        ],
        'Test_Remaining_SAIDI': [
            test_sir_mse_eval['remaining_saidi'], test_sir_eval['remaining_saidi'],
            test_rnn_mse_eval['remaining_saidi'], test_rnn_eval['remaining_saidi'],
            test_lstm_mse_eval['remaining_saidi'], test_lstm_eval['remaining_saidi'],
            test_gt_eval['remaining_saidi']
        ]
    })
    
    print(results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("SAIDI OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    
    saidi_opt_df = pd.DataFrame({
        'Dataset': ['Train', 'Test', 'Train', 'Test', 'Train', 'Test', 'Train', 'Test', 'Train', 'Test', 'Train', 'Test', 'Train', 'Test'],
        'Method': ['Ground Truth', 'Ground Truth', 'SIR (MSE)', 'SIR (MSE)', 'SIR (DFL)', 'SIR (DFL)', 
                   'RNN (MSE)', 'RNN (MSE)', 'RNN (DFL)', 'RNN (DFL)', 'LSTM (MSE)', 'LSTM (MSE)', 'LSTM (DFL)', 'LSTM (DFL)'],
        'Total_SAIDI': [train_total_saidi, test_total_saidi, train_total_saidi, test_total_saidi, train_total_saidi, test_total_saidi,
                        train_total_saidi, test_total_saidi, train_total_saidi, test_total_saidi, train_total_saidi, test_total_saidi, train_total_saidi, test_total_saidi],
        'Selected_SAIDI': [
            train_gt_eval['selected_total_saidi'], test_gt_eval['selected_total_saidi'],
            train_sir_mse_eval['selected_total_saidi'], test_sir_mse_eval['selected_total_saidi'],
            train_sir_eval['selected_total_saidi'], test_sir_eval['selected_total_saidi'],
            train_rnn_mse_eval['selected_total_saidi'], test_rnn_mse_eval['selected_total_saidi'],
            train_rnn_eval['selected_total_saidi'], test_rnn_eval['selected_total_saidi'],
            train_lstm_mse_eval['selected_total_saidi'], test_lstm_mse_eval['selected_total_saidi'],
            train_lstm_eval['selected_total_saidi'], test_lstm_eval['selected_total_saidi']
        ],
        'Remaining_SAIDI': [
            train_gt_eval['remaining_saidi'], test_gt_eval['remaining_saidi'],
            train_sir_mse_eval['remaining_saidi'], test_sir_mse_eval['remaining_saidi'],
            train_sir_eval['remaining_saidi'], test_sir_eval['remaining_saidi'],
            train_rnn_mse_eval['remaining_saidi'], test_rnn_mse_eval['remaining_saidi'],
            train_rnn_eval['remaining_saidi'], test_rnn_eval['remaining_saidi'],
            train_lstm_mse_eval['remaining_saidi'], test_lstm_mse_eval['remaining_saidi'],
            train_lstm_eval['remaining_saidi'], test_lstm_eval['remaining_saidi']
        ],
        'Selected_Cities': [
            len(train_gt_eval['selected_cities']), len(test_gt_eval['selected_cities']),
            len(train_sir_mse_eval['selected_cities']), len(test_sir_mse_eval['selected_cities']),
            len(train_sir_eval['selected_cities']), len(test_sir_eval['selected_cities']),
            len(train_rnn_mse_eval['selected_cities']), len(test_rnn_mse_eval['selected_cities']),
            len(train_rnn_eval['selected_cities']), len(test_rnn_eval['selected_cities']),
            len(train_lstm_mse_eval['selected_cities']), len(test_lstm_mse_eval['selected_cities']),
            len(train_lstm_eval['selected_cities']), len(test_lstm_eval['selected_cities'])
        ]
    })
    
    print(saidi_opt_df.to_string(index=False))
    
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
    
    # Save results and timing (unchanged from your original structure)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/model_comparison_{args.state}_{args.model}_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults saved to {results_filename}")
    
    saidi_opt_filename = f"results/saidi_optimization_{args.state}_{args.model}_{timestamp}.csv"
    saidi_opt_df.to_csv(saidi_opt_filename, index=False)
    print(f"SAIDI optimization results saved to {saidi_opt_filename}")
    
    mse_timing_df = pd.DataFrame({
        'Model': ['SIR', 'RNN', 'LSTM'],
        'MSE_Time_Seconds': [sir_mse_time, baseline_results['rnn_mse_time'], baseline_results['lstm_mse_time']],
        'MSE_Time_Minutes': [sir_mse_time/60, baseline_results['rnn_mse_time']/60, baseline_results['lstm_mse_time']/60],
        'Epochs': [args.num_MSE_epochs, args.num_MSE_epochs, args.num_MSE_epochs],
        'Random_Seed': [args.random_seed, args.random_seed, args.random_seed],
        'Num_Cities': [len(train_counties), len(train_counties), len(train_counties)],
        'Subsampled_Cities': [
            args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
            args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
            args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All'
        ]
    })
    mse_timing_filename = f"results/mse_timing_{args.state}_{args.model}_{timestamp}.csv"
    mse_timing_df.to_csv(mse_timing_filename, index=False)
    print(f"MSE timing data saved to {mse_timing_filename}")
    
    dfl_timing_df = pd.DataFrame({
        'Model': ['SIR', 'RNN', 'LSTM'],
        'DFL_Time_Seconds': [sir_dfl_time, rnn_dfl_time, lstm_dfl_time],
        'DFL_Time_Minutes': [sir_dfl_time/60, rnn_dfl_time/60, lstm_dfl_time/60],
        'Epochs': [args.num_gdf_epochs, args.num_gdf_epochs, args.num_gdf_epochs],
        'Random_Seed': [args.random_seed, args.random_seed, args.random_seed],
        'Num_Cities': [len(train_counties), len(train_counties), len(train_counties)],
        'Subsampled_Cities': [
            args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
            args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
            args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All'
        ]
    })
    dfl_timing_filename = f"results/dfl_timing_{args.state}_{args.model}_{timestamp}.csv"
    dfl_timing_df.to_csv(dfl_timing_filename, index=False)
    print(f"DFL timing data saved to {dfl_timing_filename}")
    
    total_timing_df = pd.DataFrame({
        'Model': ['SIR', 'RNN', 'LSTM'],
        'Total_Time_Seconds': [sir_total_time, rnn_total_time, lstm_total_time],
        'Total_Time_Minutes': [sir_total_time/60, rnn_total_time/60, lstm_total_time/60],
        'MSE_Time_Seconds': [sir_mse_time, baseline_results['rnn_mse_time'], baseline_results['lstm_mse_time']],
        'DFL_Time_Seconds': [sir_dfl_time, rnn_dfl_time, lstm_dfl_time],
        'Random_Seed': [args.random_seed, args.random_seed, args.random_seed],
        'Num_Cities': [len(train_counties), len(train_counties), len(train_counties)],
        'Subsampled_Cities': [
            args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
            args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All', 
            args.num_subsampled_cities if args.num_subsampled_cities is not None else 'All'
        ]
    })
    total_timing_filename = f"results/total_timing_{args.state}_{args.model}_{timestamp}.csv"
    total_timing_df.to_csv(total_timing_filename, index=False)
    print(f"Total timing data saved to {total_timing_filename}")
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()
