import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf
from cvxopt import matrix, solvers

# Step 1: Fetch Historical Data
etfs = ['SPUS', 'HLAL', 'SPSK', 'SPRE', 'SPTE', 'SPWO', 'UMMA']
data = yf.download(etfs, start='2020-01-01', end='2024-12-12')

# Print the columns to see what keys are available
print(data.columns)

# Step 2: Calculate Returns
returns = data['Close'].pct_change().dropna()

# Step 3: Calculate Expected Returns and Covariance Matrix
expected_returns = returns.mean() * 252  # Annualize returns
cov_matrix = returns.cov() * 252  # Annualize covariance matrix

# Add a small regularization term to the covariance matrix
reg_cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * 1e-4

# Step 4: Optimize Portfolio
n = len(etfs)
cov_matrix = matrix(reg_cov_matrix.values)
expected_returns = matrix(expected_returns.values)

# Constraints: sum of weights = 1 and weights >= 0
G = matrix(0.0, (n, n))
G[::n+1] = -1.0
h = matrix(0.0, (n, 1))
A = matrix(1.0, (1, n))
b = matrix(1.0)

# Objective: maximize expected return
solvers.options['show_progress'] = False
sol = solvers.qp(cov_matrix, -expected_returns, G, h, A, b)
weights = np.array(sol['x']).flatten()

# Print the optimized weights
print("Optimized ETF Weights:")
for etf, weight in zip(etfs, weights):
    print(f"{etf}: {weight:.4f}")

# Calculate the expected portfolio return and risk
portfolio_return = np.dot(weights, expected_returns)
portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

print(f"Expected Portfolio Return: {portfolio_return.item():.4f}")
print(f"Expected Portfolio Risk: {portfolio_risk.item():.4f}")