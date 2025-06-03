import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from sklearn.cluster import KMeans

# Function to fetch historical data for S&P 500, Bonds, Gold, and Real Estate
def get_historical_data():
    st.write("Fetching historical data...\n")
    try:
        sp500_tr = yf.download("^SP500TR", start="2015-01-01", end="2023-01-01", progress=False)
        sp500_returns = sp500_tr['Adj Close'].pct_change().mean() * 252
        st.write("S&P 500 Total Return data fetched.")

        bond_data = yf.download("TLT", start="2015-01-01", end="2023-01-01", progress=False)
        bond_returns = bond_data['Adj Close'].pct_change().mean() * 252
        st.write("Bond data fetched.")

        gold_data = yf.download("GLD", start="2015-01-01", end="2023-01-01", progress=False)
        gold_returns = gold_data['Adj Close'].pct_change().mean() * 252
        st.write("Gold data fetched.")

        real_estate_data = yf.download("VNQ", start="2015-01-01", end="2023-01-01", progress=False)
        real_estate_returns = real_estate_data['Adj Close'].pct_change().mean() * 252
        st.write("Real Estate data fetched.\n")

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None, None, None, None, None, None

    return sp500_returns, bond_returns, gold_returns, real_estate_returns, sp500_tr, bond_data, gold_data, real_estate_data

# Function to gather user input on investment preferences
def get_user_inputs():
    term = st.selectbox("Choose investment term", ['Medium', 'Long', 'Short'])
    risk_tolerance = st.slider("Risk Tolerance (%)", 1, 100, 50)
    desired_return = st.slider("Desired Return (%)", 7.75, 9.0, 8.0)
    invest_real_estate = st.radio("Do you want to invest in real estate?", ['Yes', 'No'])
    real_estate_return = st.number_input("Expected Real Estate Return (%)", value=7.0) if invest_real_estate == 'Yes' else 0.0
    return term, risk_tolerance, desired_return, real_estate_return

def display_dynamic_feedback(risk_tolerance, desired_return):
    st.write("### Dynamic Feedback")
    if risk_tolerance < 30:
        st.write("Your risk tolerance is on the lower side. Consider safer, lower-return assets.")
    elif risk_tolerance < 70:
        st.write("Your risk tolerance is moderate. A balanced portfolio could suit you.")
    else:
        st.write("Your risk tolerance is high. A more aggressive portfolio could generate higher returns.")
        
    if desired_return < 8:
        st.write("Your desired return is conservative. A safer approach is recommended.")
    elif desired_return < 9:
        st.write("Your desired return is moderate. Expect a balanced portfolio mix.")
    else:
        st.write("You're aiming for a high return. Expect higher risk, with a focus on growth assets.")


# Function to calculate portfolio performance
def portfolio_performance(weights, returns):
    return np.dot(weights, returns)

# Function to calculate portfolio statistics
def calculate_portfolio_stats(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Portfolio optimization function
def optimize_portfolio(returns, cov_matrix, target_return):
    num_assets = len(returns)
    initial_weights = np.array([1 / num_assets] * num_assets)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: portfolio_performance(w, returns) - target_return}]
    bounds = [(0, 1) for _ in range(num_assets)]
    result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                      initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        st.error("Optimization failed.")
        return None

# Function to display return breakdown
def display_return_breakdown(weights, returns, asset_names):
    st.write("--- Return Breakdown by Asset ---")
    breakdown = {asset_names[i]: weights[i] * returns[i] * 100 for i in range(len(asset_names))}
    st.table(pd.DataFrame.from_dict(breakdown, orient='index', columns=['Return Contribution (%)']))

# Plot allocation
def plot_allocation(weights, asset_names):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(asset_names, weights * 100, color='skyblue')
    ax.set_title("Portfolio Allocation")
    ax.set_ylabel("Percentage of Portfolio (%)")
    st.pyplot(fig)

# Efficient Frontier plot
def plot_efficient_frontier(mean_returns, cov_matrix):
    num_portfolios = 1000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility = calculate_portfolio_stats(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = portfolio_return / portfolio_volatility
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.7)
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    fig.colorbar(scatter, label="Sharpe Ratio")
    st.pyplot(fig)

# Correlation Heatmap
def plot_correlation_heatmap(price_data):
    returns = price_data.pct_change().dropna()
    corr_matrix = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Assets")
    st.pyplot(fig)

# Monte Carlo Simulation
def monte_carlo_simulation(mean_returns, cov_matrix, num_simulations=1000):
    results = np.zeros((3, num_simulations))
    for i in range(num_simulations):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility = calculate_portfolio_stats(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = portfolio_return / portfolio_volatility
    return results

# KMeans Clustering of portfolios
def kmeans_clustering(portfolio_returns, portfolio_volatilities, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(np.column_stack((portfolio_volatilities, portfolio_returns)))
    return kmeans

# Additional Graphs

# 1. Asset Returns Distribution
def plot_returns_distribution(price_data, asset_names):
    st.write("--- Asset Returns Distribution ---")
    returns = price_data.pct_change().dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    for asset in asset_names:
        ax.hist(returns[asset], bins=50, alpha=0.5, label=asset)
    ax.set_title("Distribution of Asset Returns")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right')
    st.pyplot(fig)

# 2. Portfolio Volatility vs. Return
def plot_volatility_vs_return(returns, cov_matrix, num_portfolios=1000):
    st.write("--- Portfolio Volatility vs. Return ---")
    num_assets = len(returns)
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility = calculate_portfolio_stats(weights, returns, cov_matrix)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = portfolio_return / portfolio_volatility
    
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.7)
    ax.set_title("Portfolio Volatility vs. Return")
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Return")
    fig.colorbar(scatter, label="Sharpe Ratio")
    st.pyplot(fig)

# 3. Cumulative Portfolio Performance
def plot_cumulative_performance(weights, price_data):
    st.write("--- Cumulative Portfolio Performance ---")
    returns = price_data.pct_change().dropna()
    portfolio_returns = np.dot(returns, weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cumulative_returns, label="Cumulative Return")
    ax.set_title("Cumulative Portfolio Performance")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    st.pyplot(fig)

# 4. Correlation Matrix of Returns
def plot_correlation_matrix(price_data):
    st.write("--- Correlation Matrix of Returns ---")
    returns = price_data.pct_change().dropna()
    correlation_matrix = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Asset Returns")
    st.pyplot(fig)

# 5. Risk Contribution by Asset
def plot_risk_contribution(weights, cov_matrix, asset_names):
    st.write("--- Risk Contribution by Asset ---")
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_risk_contribution = np.dot(cov_matrix, weights) * weights / portfolio_volatility
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(asset_names, marginal_risk_contribution * 100, color='skyblue')
    ax.set_title("Risk Contribution by Asset")
    ax.set_ylabel("Risk Contribution (%)")
    st.pyplot(fig)

def calculate_value_at_risk(weights, returns, confidence_level=0.05):
    portfolio_returns = np.dot(returns, weights)
    portfolio_var = np.percentile(portfolio_returns, confidence_level * 100)  # VaR at confidence level
    return portfolio_var



# After all the visualizations are done, you can add a summary paragraph.

def optimization_summary(optimized_weights, returns, cov_matrix, asset_names):
    # Calculate portfolio statistics
    portfolio_return, portfolio_volatility = calculate_portfolio_stats(optimized_weights, returns, cov_matrix)
    sharpe_ratio = portfolio_return / portfolio_volatility
    portfolio_value_at_risk = calculate_value_at_risk(optimized_weights, returns, cov_matrix, 0.05)
    portfolio_max_drawdown = calculate_max_drawdown(optimized_weights, returns, cov_matrix)
   

    # Detailed summary with better formatting
    summary = (
        f"--- Portfolio Optimization Summary ---\n\n"
        f"Optimal Asset Allocation:\n"
        + "\n".join([f"  - {name}: {weight * 100:.2f}%" for name, weight in zip(asset_names, optimized_weights)]) + "\n\n"
        f"Key Portfolio Statistics:\n"
        f"  - Expected Annual Return: {portfolio_return * 100:.2f}%\n"
        f"  - Portfolio Volatility (Risk): {portfolio_volatility * 100:.2f}%\n"
        f"  - Sharpe Ratio: {sharpe_ratio:.2f} (indicating a {'high' if sharpe_ratio > 1 else 'low'} risk-adjusted return)\n"
        f"  - Value at Risk (5% level): ${portfolio_value_at_risk:,.2f}\n"
        f"  - Maximum Drawdown: {portfolio_max_drawdown * 100:.2f}%\n\n"
        f"Additional Insights:\n"
        f"  - The portfolio is designed to maximize returns given the constraints, with an optimal allocation to diversify risk across different asset classes.\n"
        f"  - The Sharpe ratio of {sharpe_ratio:.2f} indicates the portfolio's risk-return efficiency.\n"
        f"  - A Value at Risk of ${portfolio_value_at_risk:,.2f} means that, in the worst-case scenario, the portfolio could lose this amount or more over the next year with 95% confidence.\n"
        f"  - The Maximum Drawdown of {portfolio_max_drawdown * 100:.2f}% provides insight into the potential peak-to-trough losses in a market downturn."
    )
    portfolio_value_at_risk = calculate_value_at_risk(optimized_weights, returns, cov_matrix, 0.05)
    summary += f"  - Value at Risk (5% confidence level): ${portfolio_value_at_risk:,.2f}\n"

    return summary

def calculate_value_at_risk(weights, returns, cov_matrix, confidence_level=0.05):
    # Simulate portfolio returns and calculate Value at Risk (VaR)
    portfolio_returns = np.dot(weights, returns)
    portfolio_var = np.percentile(portfolio_returns, confidence_level * 100)  # VaR at confidence level
    return portfolio_var

def calculate_max_drawdown(weights, returns, cov_matrix):
    # Simulate portfolio returns over time to calculate max drawdown
    simulated_returns = np.dot(weights, returns)
    cumulative_returns = np.cumsum(simulated_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def plot_risk_return_scatter(returns, cov_matrix):
    st.write("--- Monte Carlo Simulations ---")
    num_portfolios = 1000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility = calculate_portfolio_stats(weights, returns, cov_matrix)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = portfolio_return / portfolio_volatility
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.7)
    ax.set_title("Portfolio Risk vs Return")
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    fig.colorbar(scatter, label="Sharpe Ratio")
    st.pyplot(fig)

def generate_csv_report(optimized_weights, asset_names, returns, portfolio_return, portfolio_volatility, sharpe_ratio):
    # Create a dictionary for the data
    data = {
        "Asset": asset_names,
        "Weight (%)": optimized_weights * 100,  # Convert to percentage
        "Expected Return (%)": returns * 100  # Convert to percentage
    }
    
    # Create the DataFrame from the data dictionary
    report_df = pd.DataFrame(data)
    
    # Add portfolio-level data (which is same for all assets)
    report_df['Portfolio Return (%)'] = portfolio_return * 100  # Convert to percentage
    report_df['Portfolio Volatility (%)'] = portfolio_volatility * 100  # Convert to percentage
    report_df['Sharpe Ratio'] = sharpe_ratio  # Same for all assets
    
    # Convert the DataFrame to a CSV (in-memory)
    csv_data = report_df.to_csv(index=False)

    risk_free_rate = 0.03  # Example risk-free rate

    ''' Add the Capital Market Line (CML)
    x_cml = np.linspace(0, max(results[1, :]), 100)  # Risk range
    y_cml = risk_free_rate + sharpe_ratio * x_cml  # Return = Rf + Sharpe * Risk

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap="viridis", alpha=0.5, label="Random Portfolios")
    ax.plot(risks, rets, 'r-', label="Efficient Frontier")
    ax.plot(x_cml, y_cml, 'g--', label="Capital Market Line (CML)")
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier, CML, and Random Portfolios")
    ax.legend()
    st.pyplot(fig)'''

    
    # Create a download button with the CSV data
    st.download_button(
        label="Download Portfolio Report",
        data=csv_data,
        file_name="portfolio_report.csv",
        mime="text/csv"
    )
    
    # Optionally display the DataFrame in the Streamlit app
    st.write(report_df)


# Integration in the main function
def main():
    st.title("Portfolio Optimization Tool")
    sp500_ret, bond_ret, gold_ret, real_est_ret, sp500_data, bond_data, gold_data, real_est_data = get_historical_data()
    term, risk_tol, desired_ret, real_est_ret = get_user_inputs()

    display_dynamic_feedback(risk_tol, desired_ret)

    returns = np.array([sp500_ret, bond_ret, gold_ret])
    asset_names = ["S&P 500", "Bonds", "Gold"]
    if real_est_ret:
        returns = np.append(returns, real_est_ret)
        asset_names.append("Real Estate")

    price_data = pd.concat([sp500_data['Adj Close'], bond_data['Adj Close'], gold_data['Adj Close'],
                            real_est_data['Adj Close'] if real_est_ret else None], axis=1).dropna()
    price_data.columns = asset_names
    cov_matrix = price_data.pct_change().dropna().cov() * 252

    optimized_weights = optimize_portfolio(returns, cov_matrix, desired_ret / 100)
    if optimized_weights is not None:
        # Display optimized portfolio allocation
        st.write("--- Optimized Portfolio Allocation ---")
        for name, weight in zip(asset_names, optimized_weights):
            st.write(f"  - {name}: {weight * 100:.2f}%")
        
        display_return_breakdown(optimized_weights, returns, asset_names)
        plot_allocation(optimized_weights, asset_names)
        plot_efficient_frontier(returns, cov_matrix)
        plot_correlation_heatmap(price_data)
        st.write("### Additional Insights")
        st.write(f"Risk-Return Tradeoff: {np.round(calculate_portfolio_stats(optimized_weights, returns, cov_matrix), 2)}")
        
        def plot_allocation_pie_chart(asset_names, weights):
            fig, ax = plt.subplots()
            ax.pie(weights, labels=asset_names, autopct="%1.1f%%", startangle=90)
            ax.set_title("Portfolio Allocation")
            st.pyplot(fig)

        plot_allocation_pie_chart(asset_names, optimized_weights)
        # New Graphs

        # Plot Asset Returns Distribution
        plot_returns_distribution(price_data, asset_names)

        # Plot Portfolio Volatility vs. Return
        plot_volatility_vs_return(returns, cov_matrix)

        # Plot Cumulative Portfolio Performance
        plot_cumulative_performance(optimized_weights, price_data)

        # Plot Correlation Matrix of Returns
        plot_correlation_matrix(price_data)

        # Plot Risk Contribution by Asset
        plot_risk_contribution(optimized_weights, cov_matrix, asset_names)

        # Plot risk-return scatter after optimization
        plot_risk_return_scatter(returns, cov_matrix)

    

        # Monte Carlo Simulations
        st.write("--- Monte Carlo Simulations ---")
        mc_results = monte_carlo_simulation(returns, cov_matrix)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(mc_results[1, :], mc_results[0, :], c=mc_results[2, :], cmap='viridis', alpha=0.7)
        ax.set_title("Monte Carlo Simulation: Portfolio Performance")
        ax.set_xlabel("Risk (Volatility)")
        ax.set_ylabel("Return")
        fig.colorbar(ax.collections[0], label="Sharpe Ratio")
        st.pyplot(fig)

        # KMeans Clustering
        st.write("--- KMeans Clustering of Portfolio Simulations ---")
        kmeans = kmeans_clustering(mc_results[0, :], mc_results[1, :])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(mc_results[1, :], mc_results[0, :], c=kmeans.labels_, cmap='viridis', alpha=0.7)
        ax.set_title("KMeans Clustering of Monte Carlo Simulations")
        ax.set_xlabel("Risk (Volatility)")
        ax.set_ylabel("Return")
        st.pyplot(fig)

        

        # Optimization Summary
        summary = optimization_summary(optimized_weights, returns, cov_matrix, asset_names)
        st.write("### Optimization Summary")
        st.write(summary)

        portfolio_return, portfolio_volatility = calculate_portfolio_stats(optimized_weights, returns, cov_matrix)
        sharpe_ratio = portfolio_return / portfolio_volatility

        generate_csv_report(optimized_weights, asset_names, returns, portfolio_return, portfolio_volatility, sharpe_ratio)


if __name__ == "__main__":
    main()
