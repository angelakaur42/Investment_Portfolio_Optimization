import pandas as pd
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from fredapi import Fred

# Load the DataFrame from the pickle file
df = pd.read_pickle("Yahoo-Finance-Scraper.pkl")
tickers=df.columns.tolist()

#calculate the lognormal returns (daily_returns) for each ticker
log_returns=np.log(df/df.shift(1))

#drop any missing values
log_returns=log_returns.dropna()

#calculate the covariance matrix using annualized log returns
cov_matrix=log_returns.cov()*252
#print(cov_matrix)

#fetch the risk-free-rate
fred=Fred(api_key='ae17d21b6f9ea45d731a285cc3579d70')
ten_year_treasury_rate=fred.get_series_latest_release('GS10')/100
risk_free_rate=ten_year_treasury_rate.iloc[-1]
print(risk_free_rate)


#calculate the portfolio standard deviation/volatility
def standard_deviation (weights, cov_matrix):
    variance=weights.T @ cov_matrix @weights 
    return np.sqrt(variance)

#calculate the expected returns
def expected_returns(weights, log_returns):
    return np.sum(log_returns.mean()*weights*252)

#calculate the sharpe ratio
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_returns (weights, log_returns)*risk_free_rate)/standard_deviation(weights, cov_matrix)

#setting the constraints and bounds
constraints={'type':'eq','fun':lambda weights: np.sum(weights)-1}
bounds=[(0, 0.4) for _ in range(len(tickers))] #we don't want an asset to have over 40% allocation

#Initialize weights
initial_weights=np.array([1/len(tickers)]*len(tickers))

#---------------------------------Portfolio Optimization----------------------------------------

#This function optimizes portfolio weights to maximize Sharpe Ratio using Mean-Variance Optimization
#MVO aims for a balance between expected return and risk through Sharpe Ratio maximization.
def maximize_sharpe_ratio(log_returns, cov_matrix, tickers, initial_weights):

    def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
    
    #Perform Optimization
    optimized_results=minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
    optimal_weights=optimized_results.x

    return optimal_weights

#Minimum volatility prioritizes reducing risk (volatility) even if it potentially lowers expected return.
def minimize_volatility(log_returns, cov_matrix, tickers, initial_weights):
    
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

#   #Perform optimization
    optimized_results=minimize(portfolio_volatility,initial_weights, args=(cov_matrix),method='SLSQP', constraints=constraints, bounds=bounds)

    optimal_weights=optimized_results.x

    return optimal_weights

    
#---------------------------Analyze the Optimal Portfolio-------------------------------------
#obtain optimal weights and calculate the expected portfolio return, expected volatility, and sharpe ratio for the optimal portfolio

optimal_weights=minimize_volatility(log_returns, cov_matrix, tickers, initial_weights)

print("Optimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")


optimal_portfolio_return=expected_returns(optimal_weights, log_returns)
optimal_portfolio_volatility=standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio=sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return {optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

#---------------------------Display the final portfolio on a plot---------------------------------

top_10_indices=np.argsort(optimal_weights)[-10:]
top_10_weights=optimal_weights[top_10_indices]
top_10_tickers=np.array(tickers)[top_10_indices]

plt.figure(figsize=(10, 6))
plt.bar(top_10_tickers, top_10_weights)

plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')

#plt.show()


