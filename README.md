## Synthetic Generator Deployment (Single Warehouse)

This repository accompanies the notebook `synthetic_generator_deployment_problem_single_warehouse.ipynb`. The notebook builds a fully differentiable pipeline that couples epidemic-style outage forecasting with generator deployment planning for a small set of cities supplied by a single mobile generator warehouse.

- **Synthetic grid outages** are generated with a multi-city SIR model (`torchdiffeq`), then optionally perturbed with noise.
- **Forecasting models** are trained either with pure MSE loss or with a differentiable economic objective (DFL) that backpropagates through a CVXPYLayer dispatch problem.
- **Dispatch optimization** solves a generator transportation problem with both differentiable convex layers and a Gurobi-powered baseline, and compares them with a greedy online policy and the ground-truth optimum.
- **Evaluation tooling** reports total/regret costs, SAIDI-style metrics, and produces publication-quality plots for outages, generator allocations, and cost breakdowns.

### Notebook Roadmap

- **0. Setup** – Optional `pip install` cells for `torchdiffeq`, `cvxpylayers`, `diffcp`, and `gurobipy`.
- **1. Synthetic Data Generation** – Defines population parameters, creates the true SIR dynamics (`TrueSIRModel`), and visualizes clean/noisy outbreaks.
- **2. Forecasting Models** – Implements training utilities (`train_sir_model_mse`, `train_sir_model_dfl`) and tracks cost/mse histories.
- **3. Dispatch Optimization** – Builds the CVXPYLayer, differentiable GDP solver, and Gurobi baselines; includes greedy online heuristics, regret calculations, and cost accounting (`compute_total_cost`).
- **4. Visualization & Reporting** – Provides multi-panel plotting helpers (`plot_three_columns_three_cities`, `plot_four_rows_three_cities_with_cost`, etc.) to compare GDP, prediction-only, online, and ground-truth strategies across cities.

### Prerequisites

- Python 3.10+ with JupyterLab or VS Code Notebook support
- PyTorch (tested with 2.1+) and `torchdiffeq==0.2.4`
- `cvxpylayers==0.1.6` (requires `diffcp==1.0.23`)
- `gurobipy==11.0.3` with an active Gurobi license
- NumPy, Matplotlib, SciPy (for convenience utilities)

Install core dependencies inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or pip install torchdiffeq==0.2.4 cvxpylayers==0.1.6 diffcp==1.0.23 gurobipy==11.0.3 matplotlib numpy scipy
```

> ⚠️ Gurobi requires a valid license. Ensure `grbgetkey` is configured before running the optimization cells.

### Running the Notebook

1. Launch Jupyter Lab or VS Code and open `synthetic_generator_deployment_problem_single_warehouse.ipynb`.
2. Execute the setup cells (installs are optional if the environment is already satisfied).
3. Run the notebook top-to-bottom. Key tuning knobs are grouped near the data-generation cells (population, infection rates) and the optimization configuration (economic weights `tau`, `gamma`, `transport_cost_per_gen`, generator capacity `N_g`, and training horizon).
4. The notebook saves intermediate models (`sir_model_*.pth`), plots (PDF/PNG exports), and tabular summaries to the working directory.

### Interpreting Results

- **Training curves** plot total economic loss and MSE for MSE-only vs DFL training.
- **Dispatch comparisons** highlight cost breakdowns across strategies (GDP/Two-stage/Online/Groundtruth) and compute regret relative to the optimal hindsight solver.
- **City-level dashboards** overlay predicted outages with generator allocations and true demand to visualize how well each policy hedges outages.

### Customization Tips

- Adjust the number of cities, populations, and SIR parameters to explore different outage scenarios.
- Swap in real outage data by replacing the synthetic generator cell outputs with observed time series tensors.
- Modify the economic weights to stress-test sensitivity to transportation vs unmet demand penalties.
- Extend the notebook by exporting trained models or plot figures for downstream papers/presentations.

### Repository Layout

- `synthetic_generator_deployment_problem_single_warehouse.ipynb` – Primary end-to-end experiment.
- `GDF-ODE/` – Supporting scripts for generalized differentiable forecasting (used by the notebook).
- Additional notebooks (e.g., `indianapolis_*`, `synthetic_gdp_multi_warehouse.ipynb`) demonstrate multi-warehouse and real-world scenarios.

Questions or contributions are welcome through GitHub issues or pull requests.

