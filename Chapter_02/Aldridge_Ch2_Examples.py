"""
Chapter 2 - Prices and Returns
Examples and code from the textbook chapter on analyzing stock prices and returns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# SECTION 1: Loading and Displaying Stock Data
# ============================================================================

def load_and_display_data(filepath='c:/example/TSLA.csv'):
    """
    Load TSLA stock data from CSV file and display basic information
    """
    df = pd.read_csv(filepath)
    print("=" * 70)
    print("TSLA Stock Data - First and Last Rows")
    print("=" * 70)
    print(df)
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nColumn names:", df.columns.tolist())
    return df


# ============================================================================
# SECTION 2: Plotting Stock Prices
# ============================================================================

def plot_close_prices(df):
    """
    Figure 2.2: Plot TSLA daily close prices
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Adj Close'])
    plt.title('TSLA Daily Close Prices')
    plt.xlabel('Trading Days')
    plt.ylabel('Price ($)')
    plt.tight_layout()
    plt.show()


def plot_close_prices_with_dates(df):
    """
    Figure 2.3: Plot closing prices with dates on x-axis
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['Adj Close'])
    plt.xlabel('Dates')
    
    # Create custom x-tick labels
    xt = np.arange(0, len(df), 365)
    plt.xticks(xt, df.iloc[xt]['Date'], rotation=90)
    plt.tight_layout()
    plt.show()


def compare_close_vs_adj_close(df):
    """
    Figure 2.4: Compare Close vs Adj Close prices (example with HD)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close')
    plt.plot(df['Adj Close'], label='Adj Close')
    plt.legend()
    plt.xlabel('Dates')
    plt.ylabel('Price ($)')
    plt.title('Close vs. Adj Close prices')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 3: Computing Price Differences and Returns
# ============================================================================

def compute_price_differences(df):
    """
    Compute price differences: ΔPt = Pt - Pt-1
    """
    df['diff'] = df['Adj Close'] - df['Adj Close'].shift(1)
    return df


def compute_returns(df):
    """
    Compute daily returns: Rt = Pt/Pt-1 - 1
    """
    df['R'] = (df['Adj Close'] / df['Adj Close'].shift(1) - 1).fillna(0)
    return df


def compute_return_differences(df):
    """
    Compute daily return differences: ΔRt = Rt - Rt-1
    """
    df['diff_R'] = df['R'] - df['R'].shift(1)
    return df


def compute_log_returns(df):
    """
    Compute daily log returns: rt = log(Pt) - log(Pt-1)
    """
    df['logR'] = (np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))).fillna(0)
    return df


def drop_last_row(df):
    """
    Alternative approach: Drop rows with missing observations instead of filling with 0
    """
    df_clean = df.dropna()
    return df_clean


# ============================================================================
# SECTION 4: Plotting Price and Return Series
# ============================================================================

def plot_price_series(df):
    """
    Figure 2.5: Plot daily adjusted close prices
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['Adj Close'])
    plt.title('Daily Adj Close Prices, TSLA')
    plt.xlabel('Dates')
    plt.ylabel('Price ($)')
    
    # Add date labels
    xt = np.arange(0, len(df), 365)
    plt.xticks(xt, df.iloc[xt]['Date'].values if 'Date' in df.columns else xt, rotation=45)
    plt.tight_layout()
    plt.savefig('adjClose_TSLA.png')
    plt.show()


def plot_price_differences(df):
    """
    Figure 2.6: Plot daily price differences
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['diff'])
    plt.title('Daily Difference of Prices, TSLA')
    plt.xlabel('Dates')
    plt.ylabel('Price Difference ($)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_returns(df):
    """
    Figure 2.7: Plot daily returns
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['R'])
    plt.title('Daily Returns, TSLA')
    plt.xlabel('Dates')
    plt.ylabel('Returns')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_return_differences(df):
    """
    Figure 2.8: Plot daily difference in returns
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['diff_R'])
    plt.title('Daily Difference of Returns, TSLA')
    plt.xlabel('Dates')
    plt.ylabel('Return Differences')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_log_returns(df):
    """
    Figure 2.9: Plot daily log returns
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['logR'])
    plt.title('Daily Log Returns, TSLA')
    plt.xlabel('Dates')
    plt.ylabel('Log Returns')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 5: Histograms for Distribution Analysis
# ============================================================================

def plot_price_histogram(df):
    """
    Figure 2.10: Histogram of daily adjusted close prices
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['Adj Close'], bins=50, edgecolor='black')
    plt.title('Histogram of Daily Adj Close Prices, TSLA')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def plot_price_diff_histogram(df):
    """
    Figure 2.11: Histogram of price differences
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['diff'].dropna(), bins=50, edgecolor='black')
    plt.title('Histogram of Daily Difference of Prices, TSLA')
    plt.xlabel('Price Difference ($)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def plot_returns_histogram(df):
    """
    Figure 2.12: Histogram of daily returns
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['R'], bins=50, edgecolor='black')
    plt.title('Histogram of Daily Returns, TSLA')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def plot_return_diff_histogram(df):
    """
    Figure 2.13: Histogram of daily difference in returns
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['diff_R'].dropna(), bins=50, edgecolor='black')
    plt.title('Histogram of Daily Difference of Returns, TSLA')
    plt.xlabel('Return Differences')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def plot_log_returns_histogram(df):
    """
    Figure 2.14: Histogram of daily log returns
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['logR'], bins=50, edgecolor='black')
    plt.title('Histogram of Daily Log Returns, TSLA')
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 6: Kolmogorov-Smirnov Tests for Normality
# ============================================================================

def perform_ks_test(data, name):
    """
    Perform Kolmogorov-Smirnov test for normality
    
    Returns:
    - KS statistic: maximum vertical distance between sample CDF and normal CDF
    - p-value: likelihood that data is normally distributed (0-1)
    """
    stat, p_value = stats.kstest(data, 'norm')
    print(f"\nKS Test for {name}:")
    print(f"  KS Statistic: {stat:.3f}")
    print(f"  p-value: {p_value:.1f}")
    return stat, p_value


def run_all_ks_tests(df):
    """
    Table 2.1: Run KS tests on all price and return variables
    """
    print("\n" + "=" * 70)
    print("Kolmogorov-Smirnov Tests for Normality")
    print("=" * 70)
    
    results = {}
    
    # Adj Close
    stat, p_val = perform_ks_test(df['Adj Close'].dropna(), 'Adj Close')
    results['Adj Close'] = {'KS Statistic': stat, 'p-value': p_val}
    
    # Price Difference
    stat, p_val = perform_ks_test(df['diff'].dropna(), 'Price Difference')
    results['Price Difference'] = {'KS Statistic': stat, 'p-value': p_val}
    
    # Returns
    stat, p_val = perform_ks_test(df['R'].dropna(), 'Returns')
    results['Returns'] = {'KS Statistic': stat, 'p-value': p_val}
    
    # Return Difference
    stat, p_val = perform_ks_test(df['diff_R'].dropna(), 'Return Difference')
    results['Return Difference'] = {'KS Statistic': stat, 'p-value': p_val}
    
    # Log Returns
    stat, p_val = perform_ks_test(df['logR'].dropna(), 'Log Returns')
    results['Log Returns'] = {'KS Statistic': stat, 'p-value': p_val}
    
    # Create summary table
    results_df = pd.DataFrame(results).T
    print("\n" + "=" * 70)
    print("Summary Table: KS Test Results")
    print("=" * 70)
    print(results_df)
    
    return results_df


# ============================================================================
# SECTION 7: Autocorrelation Analysis
# ============================================================================

def plot_autocorrelation(data, title, max_lags=3700):
    """
    Plot autocorrelation function
    """
    from pandas.plotting import autocorrelation_plot
    
    plt.figure(figsize=(12, 6))
    autocorrelation_plot(data)
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.xlim(0, max_lags)
    plt.tight_layout()
    plt.show()


def plot_price_autocorrelation(df):
    """
    Figure 2.15: Autocorrelation plot for Bitcoin (BTC-USD) prices
    Note: This function expects BTC-USD data, but demonstrates with available data
    """
    plot_autocorrelation(
        df['Adj Close'].dropna(), 
        'Autocorrelation plot, prices of BTC-USD'
    )


def plot_returns_autocorrelation(df):
    """
    Figure 2.16: Autocorrelation plot for Bitcoin daily returns
    Note: This function expects BTC-USD data, but demonstrates with available data
    """
    plot_autocorrelation(
        df['R'].dropna(), 
        'Autocorrelation plot, 1-day returns of BTC-USD'
    )
