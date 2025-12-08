# models.py
import torch
import torch.nn as nn

class BaseSIRModel(nn.Module):
    def __init__(self):
        super(BaseSIRModel, self).__init__()
        # Learnable scalar parameters for beta and gamma (ensuring positivity via Softplus later, if needed)
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.population = None

    def set_population(self, population):
        self.population = population

    def forward(self, t, y):
        # y: [batch, 3] containing S, I, R
        S, I, R = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        N = self.population.unsqueeze(1)
        dS = - self.beta * S * I / N
        dI = self.beta * S * I / N - self.gamma * I
        dR = self.gamma * I
        return torch.cat([dS, dI, dR], dim=1)

class SIRWeatherModel(nn.Module):
    def __init__(self, demographic_dim: int, weather_dim: int, hidden_dim: int = 32):
        """
        Initialize the SIR model that incorporates demographic and weather data.
        
        Args:
            demographic_dim (int): Dimension of the demographic input.
            weather_dim (int): Dimension of the weather input.
            hidden_dim (int): Hidden dimension for the beta and gamma networks.
        """
        super(SIRWeatherModel, self).__init__()
        input_dim = 3 + demographic_dim + weather_dim  # state (S,I,R) plus extra features
        self.beta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensures beta is positive
        )
        self.gamma_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensures gamma is positive
        )
        self.demographic = None  # To be set externally (tensor of shape [batch, demographic_dim])
        self.weather = None      # To be set externally (tensor of shape [batch, weather_dim])
        self.population = None

    def set_population(self, population):
        self.population = population

    def set_demographic(self, demographic_tensor):
        """
        Set the demographic features for the current batch.
        """
        self.demographic = demographic_tensor

    def set_weather(self, weather_tensor):
        """
        Set the weather features for the current batch.
        """
        self.weather = weather_tensor

    def forward(self, t, y):
        # y: [batch, 3] (S, I, R)
        if self.demographic is None or self.weather is None:
            raise ValueError("Demographic and weather data must be set for SIRWeatherModel.")
        # Concatenate the SIR state with demographic and weather features.
        features = torch.cat([y, self.demographic, self.weather], dim=1)
        beta = self.beta_net(features)   # shape: [batch, 1]
        gamma = self.gamma_net(features) # shape: [batch, 1]
        S, I, R = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        N = self.population.unsqueeze(1)
        dS = - beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return torch.cat([dS, dI, dR], dim=1)


# # RNN and LSTM models for baseline comparison
# class RNNModel(nn.Module):
#     def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
#         super(RNNModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, input_dim)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
#         out, _ = self.rnn(x, h0)
#         # Take the last output
#         out = self.fc(out[:, -1, :])
#         return out

# class LSTMModel(nn.Module):
#     def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
#         super(LSTMModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, input_dim)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         # Take the last output
#         out = self.fc(out[:, -1, :])
#         return out

# # Wrapper class for passing census_data and N to the model
# class STODEModelWrapper(nn.Module):
#     def __init__(self, model, census_data, weather_data, N_values):
#         super(STODEModelWrapper, self).__init__()
#         self.model = model
#         self.census_data = census_data
#         self.weather_data = weather_data
#         self.N_values = N_values

#     def forward(self, t, y):
#         return self.model(t, y, census_data=self.census_data, weather_data = self.weather_data, N=self.N_values)


class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = True  # <---- ensure batch-first
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers,
            nonlinearity='tanh',
            batch_first=True,              # <---- important
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None):
        # x: [B, T, C] if batch_first
        x = x.float()
        B = x.size(0) if self.batch_first else x.size(1)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device, dtype=x.dtype)
        out, _ = self.rnn(x, h0)  # out: [B, T, H] (batch_first=True)
        y  = self.fc(out[:, -1, :])  # last timestep
        return y


class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = True  # <---- ensure batch-first
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,              # <---- important
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hc=None):
        # x: [B, T, C] if batch_first
        x = x.float()
        B = x.size(0) if self.batch_first else x.size(1)
        if hc is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device, dtype=x.dtype)
            c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device, dtype=x.dtype)
            hc = (h0, c0)
        out, _ = self.lstm(x, hc)  # out: [B, T, H] (batch_first=True)
        y  = self.fc(out[:, -1, :])  # last timestep
        return y


class MultiCityRNNModel(nn.Module):
    """RNN model with separate networks for each city/county."""
    def __init__(self, num_cities, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.0):
        super().__init__()
        self.num_cities = num_cities
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create separate RNN models for each city
        self.city_models = nn.ModuleList([
            RNNModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
            for _ in range(num_cities)
        ])
    
    def forward(self, x, city_indices=None):
        """
        Forward pass for multi-city RNN model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            city_indices: Tensor of city indices for each sample in batch [batch_size]
        
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        if city_indices is None:
            # If no city indices provided, use the first city model for all samples
            return self.city_models[0](x)
        
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, 1, device=x.device)
        
        # Process each sample with its corresponding city model
        for i in range(batch_size):
            city_idx = city_indices[i].item()
            if 0 <= city_idx < self.num_cities:
                # Single sample forward pass
                single_x = x[i:i+1]  # Keep batch dimension
                outputs[i] = self.city_models[city_idx](single_x)
            else:
                # Fallback to first city model if index is invalid
                single_x = x[i:i+1]
                outputs[i] = self.city_models[0](single_x)
        
        return outputs
    
    def get_total_parameters(self):
        """Get total number of parameters across all city models."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_city_parameters(self, city_idx):
        """Get number of parameters for a specific city model."""
        if 0 <= city_idx < self.num_cities:
            return sum(p.numel() for p in self.city_models[city_idx].parameters() if p.requires_grad)
        return 0


class MultiCityLSTMModel(nn.Module):
    """LSTM model with separate networks for each city/county."""
    def __init__(self, num_cities, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.0):
        super().__init__()
        self.num_cities = num_cities
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create separate LSTM models for each city
        self.city_models = nn.ModuleList([
            LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
            for _ in range(num_cities)
        ])
    
    def forward(self, x, city_indices=None):
        """
        Forward pass for multi-city LSTM model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            city_indices: Tensor of city indices for each sample in batch [batch_size]
        
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        if city_indices is None:
            # If no city indices provided, use the first city model for all samples
            return self.city_models[0](x)
        
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, 1, device=x.device)
        
        # Process each sample with its corresponding city model
        for i in range(batch_size):
            city_idx = city_indices[i].item()
            if 0 <= city_idx < self.num_cities:
                # Single sample forward pass
                single_x = x[i:i+1]  # Keep batch dimension
                outputs[i] = self.city_models[city_idx](single_x)
            else:
                # Fallback to first city model if index is invalid
                single_x = x[i:i+1]
                outputs[i] = self.city_models[0](single_x)
        
        return outputs
    
    def get_total_parameters(self):
        """Get total number of parameters across all city models."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_city_parameters(self, city_idx):
        """Get number of parameters for a specific city model."""
        if 0 <= city_idx < self.num_cities:
            return sum(p.numel() for p in self.city_models[city_idx].parameters() if p.requires_grad)
        return 0