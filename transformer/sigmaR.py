import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from cctools import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train = False
test = True

# Define the parameter ranges: [min, max]
param_ranges = {
    'h': [0.6, 0.8],
    'Omega_b': [0.03, 0.06],
    'Omega_c': [0.1, 0.45],
    'A_s': [1.5e-9, 2.5e-9],
    'n_s': [0.9, 1.0],
    'mnu': [0.0, 0.2],       # Neutrino mass
    'Omega_Lambda': [0.6, 0.75], # Omega_Lambda
    'redshift': [0.0, 3.0]   # Redshift
}
parameters = list(param_ranges.keys())

# Number of samples
n_samples = 100
n_test = 5
# Bin of modes
R_values = np.logspace(-1, 1.5, 50)

class SigmaDataset(Dataset):
    def __init__(self, params_array, sigma_R_array):
        self.params = torch.tensor(params_array, dtype=torch.float32)
        self.sigma_R = torch.tensor(sigma_R_array, dtype=torch.float32)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx], self.sigma_R[idx]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, d_model=128, nhead=4):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
        x = self.decoder(x)
        return x

if train:
    params_array = get_n_samples(param_ranges, n_samples)

    power_spectra = generate_power_spectra(params_array)
    sigma_R_array = get_sigma_r(power_spectra, R_values)

    dataset = SigmaDataset(params_array, sigma_R_array)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    model = TransformerModel(input_dim=len(parameters), output_dim=len(R_values)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        for params, sigma_R in dataloader:
            params, sigma_R = params.to(device), sigma_R.to(device)
            optimizer.zero_grad()
            outputs = model(params)
            loss = criterion(outputs, sigma_R)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'transformer_sigmaR.pth')

if test:
    test_array = get_n_samples(param_ranges, n_test)

    power_spectra = generate_power_spectra(test_array)
    sigma_R = get_sigma_r(power_spectra, R_values)

    model = TransformerModel(input_dim=len(parameters), output_dim=len(R_values)).to(device)
    model.load_state_dict(torch.load('transformer_sigmaR.pth'))
    model.eval()

    test_params_tensor = torch.tensor(test_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        predicted_sigma_R = model(test_params_tensor).cpu().numpy()

    print(predicted_sigma_R)

    for i, (pred, test) in enumerate(zip(predicted_sigma_R, sigma_R)):
        plt.plot(R_values, pred, c=f"C{i}")
        plt.plot(R_values, test, c=f"C{i}", ls=":")

    plt.xscale('log')
    plt.xlabel('R')
    plt.ylabel('sigma(R)')
    plt.title('Predicted sigma(R)')
    plt.show()

    for i, (pred, test) in enumerate(zip(predicted_sigma_R, sigma_R)):
        plt.plot(R_values, (pred/test-1)*100, c=f"C{i}")

    plt.xscale('log')
    plt.xlabel('R')
    plt.ylabel('Residual sigma(R) [%]')
    plt.title('Predicted sigma(R)')
    plt.show()

