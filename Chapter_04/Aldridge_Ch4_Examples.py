"""
Chapter 4: Cleaning Data
Examples from "Financial Engineering with Machine Learning and Python"
by Irene Aldridge (Packt, 2026)

This script demonstrates:
1. Loading and visualizing Bitcoin price data
2. Creating histograms to examine distributions
3. Calculating and plotting returns
4. Identifying outliers
5. Data cleaning techniques

Note: Using synthetic data to demonstrate concepts when real data is unavailable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# Example 1: Figure 4.1 - Bitcoin Daily Closing and Low Prices
# ============================================================================

def plot_bitcoin_prices(btc_data):
    """
    Create Figure 4.1: Time series plot of Bitcoin closing and low prices
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot closing and low prices
    ax.plot(btc_data.index, btc_data['Close'], label='BTC-USD Close', linewidth=1.5)
    ax.plot(btc_data.index, btc_data['Low'], label='BTC-USD Low', 
            linestyle='--', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Dates', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.set_title('Bitcoin Daily Closing and Low Prices\n2014-09-17 - 2025-05-16', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis to show prices clearly
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig('/example-path/figure_4_1_bitcoin_prices.png', dpi=300, bbox_inches='tight')
    print("Saved: figure_4_1_bitcoin_prices.png")
    plt.close()


# ============================================================================
# Example 2: Figure 4.2 - Histogram of Bitcoin Prices
# ============================================================================

def plot_bitcoin_price_histogram(btc_data):
    """
    Create Figure 4.2: Histogram showing distribution of Bitcoin prices
    
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    ax.hist(btc_data['Close'], bins=100, edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Price (USD)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Histogram of Bitcoin Prices\n2014-09-17 - 2025-05-16', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig('/example-path/figure_4_2_bitcoin_histogram.png', dpi=300, bbox_inches='tight')
    print("Saved: figure_4_2_bitcoin_histogram.png")
    plt.close()
    
    # Print distribution statistics
    print("\nBitcoin Price Distribution Statistics:")
    print(f"Mean: ${btc_data['Close'].mean():,.2f}")
    print(f"Median: ${btc_data['Close'].median():,.2f}")
    print(f"Std Dev: ${btc_data['Close'].std():,.2f}")
    print(f"Min: ${btc_data['Close'].min():,.2f}")
    print(f"Max: ${btc_data['Close'].max():,.2f}")
    print(f"Skewness: {btc_data['Close'].skew():.2f}")
    print(f"Kurtosis: {btc_data['Close'].kurtosis():.2f}")


# ============================================================================
# Example 3: Calculate Returns
# ============================================================================

def calculate_returns(btc_data):
    """
    Calculate daily returns for Bitcoin prices
    
    Returns are calculated as: r_t = (P_t - P_{t-1}) / P_{t-1}
    Or using log returns: r_t = log(P_t / P_{t-1})
    """
    # Calculate simple returns
    btc_data['Returns'] = btc_data['Close'].pct_change()
    
    # Calculate log returns (more commonly used in finance)
    btc_data['Log_Returns'] = np.log(btc_data['Close'] / btc_data['Close'].shift(1))
    
    print("\nReturns Statistics:")
    print(f"Mean Daily Return: {btc_data['Returns'].mean()*100:.4f}%")
    print(f"Std Dev of Returns: {btc_data['Returns'].std()*100:.4f}%")
    print(f"Min Return: {btc_data['Returns'].min()*100:.2f}%")
    print(f"Max Return: {btc_data['Returns'].max()*100:.2f}%")
    
    # Annualized metrics
    annualized_return = btc_data['Returns'].mean() * 252
    annualized_vol = btc_data['Returns'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol
    
    print(f"\nAnnualized Metrics:")
    print(f"Annualized Return: {annualized_return*100:.2f}%")
    print(f"Annualized Volatility: {annualized_vol*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return btc_data


# ============================================================================
# Example 4: Figure 4.3 - Time Series of Daily Returns
# ============================================================================

def plot_bitcoin_returns(btc_data):
    """
    Create Figure 4.3: Time series plot of Bitcoin daily returns
    Shows outliers and volatility clustering
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot returns
    ax.plot(btc_data.index, btc_data['Returns'], label='BTC-USD Daily Return', 
            linewidth=0.8, alpha=0.7)
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Dates', fontsize=12)
    ax.set_ylabel('Daily Return', fontsize=12)
    ax.set_title('Bitcoin Daily Returns\n2014-09-17 - 2025-05-16', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    plt.tight_layout()
    plt.savefig('/example-path/figure_4_3_bitcoin_returns.png', dpi=300, bbox_inches='tight')
    print("Saved: figure_4_3_bitcoin_returns.png")
    plt.close()


# ============================================================================
# Example 5: Figure 4.4 - Histogram of Returns
# ============================================================================

def plot_returns_histogram(btc_data):
    """
    Create Figure 4.4: Histogram of Bitcoin returns
    
    Key observation: Sharp peak around zero with heavy tails,
    indicating frequent small fluctuations punctuated by rare large movements
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove NaN values
    returns_clean = btc_data['Returns'].dropna()
    
    # Create histogram
    ax.hist(returns_clean, bins=100, edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Daily Return', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Histogram of Bitcoin Daily Returns\n2014-09-17 - 2025-05-16', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/example-path/figure_4_4_returns_histogram.png', dpi=300, bbox_inches='tight')
    print("Saved: figure_4_4_returns_histogram.png")
    plt.close()


# ============================================================================
# Example 6: Outlier Detection
# ============================================================================

def detect_outliers(btc_data, method='iqr', threshold=3):
    """
    Detect outliers in Bitcoin returns using various methods
    
    Parameters:
    -----------
    btc_data : pd.DataFrame
        DataFrame containing returns
    method : str
        Method to use: 'iqr' (Interquartile Range), 'zscore', or 'mad' (Median Absolute Deviation)
    threshold : float
        Threshold for outlier detection (depends on method)
    
    Returns:
    --------
    pd.Series : Boolean series indicating outliers
    """
    returns = btc_data['Returns'].dropna()
    
    if method == 'iqr':
        # Interquartile Range method
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (returns < lower_bound) | (returns > upper_bound)
        
        print(f"\nIQR Method Details:")
        print(f"Q1 (25th percentile): {Q1:.6f}")
        print(f"Q3 (75th percentile): {Q3:.6f}")
        print(f"IQR: {IQR:.6f}")
        print(f"Lower bound: {lower_bound:.6f}")
        print(f"Upper bound: {upper_bound:.6f}")
        
    elif method == 'zscore':
        # Z-score method
        mean = returns.mean()
        std = returns.std()
        z_scores = np.abs((returns - mean) / std)
        outliers = z_scores > threshold
        
        print(f"\nZ-Score Method Details:")
        print(f"Mean: {mean:.6f}")
        print(f"Std Dev: {std:.6f}")
        print(f"Threshold: {threshold} standard deviations")
        
    elif method == 'mad':
        # Median Absolute Deviation method
        median = returns.median()
        mad = np.median(np.abs(returns - median))
        modified_z_scores = 0.6745 * (returns - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        
        print(f"\nMAD Method Details:")
        print(f"Median: {median:.6f}")
        print(f"MAD: {mad:.6f}")
        print(f"Threshold: {threshold}")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\nOutlier Detection ({method.upper()} method):")
    print(f"Number of outliers: {outliers.sum()} ({outliers.sum()/len(returns)*100:.2f}%)")
    
    if outliers.sum() > 0:
        print(f"Most extreme positive outlier: {returns[outliers].max()*100:.2f}%")
        print(f"Most extreme negative outlier: {returns[outliers].min()*100:.2f}%")
    
    return outliers


def visualize_outliers(btc_data, outliers):
    """
    Visualize outliers in the returns time series
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Time series with outliers highlighted
    returns = btc_data['Returns'].dropna()
    # Align outliers with returns by using the same index
    outliers_aligned = outliers.reindex(returns.index, fill_value=False)
    
    ax1.plot(returns.index, returns, label='Returns', linewidth=0.8, alpha=0.5)
    outlier_returns = returns[outliers_aligned]
    if len(outlier_returns) > 0:
        ax1.scatter(outlier_returns.index, outlier_returns, 
                    color='red', s=50, label='Outliers', zorder=5, alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Daily Return', fontsize=12)
    ax1.set_title('Bitcoin Returns with Outliers Highlighted', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot([returns], vert=True, labels=['Returns'])
    ax2.set_ylabel('Daily Return', fontsize=12)
    ax2.set_title('Box Plot of Bitcoin Returns', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/example-path/outlier_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved: outlier_visualization.png")
    plt.close()


# ============================================================================
# Example 7: Data Cleaning Techniques
# ============================================================================

def winsorize_data(series, lower_percentile=0.01, upper_percentile=0.99):
    """
    Winsorize data by capping extreme values at specified percentiles
    
    This technique replaces outliers with less extreme values rather than removing them.
    Named after Charles P. Winsor, this method is useful when you want to reduce the 
    influence of outliers without discarding data points.
    """
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    
    winsorized = series.copy()
    winsorized[winsorized < lower_bound] = lower_bound
    winsorized[winsorized > upper_bound] = upper_bound
    
    print(f"\nWinsorization (at {lower_percentile*100:.0f}% and {upper_percentile*100:.0f}% percentiles):")
    print(f"Lower bound: {lower_bound:.6f}")
    print(f"Upper bound: {upper_bound:.6f}")
    print(f"Number of values winsorized: {(series != winsorized).sum()}")
    
    # Compare statistics
    print(f"\nBefore Winsorization:")
    print(f"  Mean: {series.mean():.6f}, Std: {series.std():.6f}")
    print(f"After Winsorization:")
    print(f"  Mean: {winsorized.mean():.6f}, Std: {winsorized.std():.6f}")
    
    return winsorized


def remove_outliers(btc_data, outliers):
    """
    Remove outliers from the dataset
    
    This is the most aggressive cleaning method and should be used with caution.
    """
    # Align outliers with the full dataframe
    outliers_aligned = outliers.reindex(btc_data.index, fill_value=False)
    cleaned_data = btc_data[~outliers_aligned].copy()
    
    print(f"\nData after outlier removal:")
    print(f"Original size: {len(btc_data)}")
    print(f"Cleaned size: {len(cleaned_data)}")
    print(f"Rows removed: {len(btc_data) - len(cleaned_data)}")
    print(f"Percentage removed: {(len(btc_data) - len(cleaned_data))/len(btc_data)*100:.2f}%")
    
    return cleaned_data


def compare_distributions(original, cleaned, winsorized):
    """
    Compare distributions before and after cleaning
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Histograms
    # Original distribution
    axes[0, 0].hist(original.dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Original Distribution', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # After outlier removal
    axes[0, 1].hist(cleaned.dropna(), bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title('After Outlier Removal', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Winsorized
    axes[0, 2].hist(winsorized.dropna(), bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 2].set_title('Winsorized Distribution', fontweight='bold', fontsize=12)
    axes[0, 2].set_xlabel('Return')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Box plots
    axes[1, 0].boxplot([original.dropna()], vert=True, labels=['Original'])
    axes[1, 0].set_ylabel('Return')
    axes[1, 0].set_title('Original Box Plot', fontweight='bold', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].boxplot([cleaned.dropna()], vert=True, labels=['Cleaned'])
    axes[1, 1].set_ylabel('Return')
    axes[1, 1].set_title('Cleaned Box Plot', fontweight='bold', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 2].boxplot([winsorized.dropna()], vert=True, labels=['Winsorized'])
    axes[1, 2].set_ylabel('Return')
    axes[1, 2].set_title('Winsorized Box Plot', fontweight='bold', fontsize=12)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/example-path/distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: distribution_comparison.png")
    plt.close()


# ============================================================================
# Example 8: Statistical Tests for Normality
# ============================================================================

def test_normality(returns):
    """
    Test whether returns are normally distributed
    
    Multiple tests are used because:
    - Jarque-Bera: Tests for skewness and kurtosis
    - Shapiro-Wilk: Most powerful test for small to medium samples
    - Kolmogorov-Smirnov: Tests overall distribution shape
    - Anderson-Darling: More sensitive to tail behavior
    """
    returns_clean = returns.dropna()
    
    print("\n" + "="*60)
    print("NORMALITY TESTS")
    print("="*60)
    
    # Jarque-Bera test
    jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
    print(f"\n1. Jarque-Bera Test:")
    print(f"   Purpose: Tests if data has normal skewness and kurtosis")
    print(f"   Statistic: {jb_stat:.2f}")
    print(f"   P-value: {jb_pvalue:.6f}")
    print(f"   Result: {'REJECT normality (H0)' if jb_pvalue < 0.05 else 'Cannot reject normality (H0)'}")
    print(f"   Interpretation: {'Data is NOT normally distributed' if jb_pvalue < 0.05 else 'Data may be normally distributed'}")
    
    # Shapiro-Wilk test (use sample if data is large)
    if len(returns_clean) > 5000:
        sample = returns_clean.sample(n=5000, random_state=42)
        sw_stat, sw_pvalue = stats.shapiro(sample)
        print(f"\n2. Shapiro-Wilk Test (on 5000 sample):")
    else:
        sw_stat, sw_pvalue = stats.shapiro(returns_clean)
        print(f"\n2. Shapiro-Wilk Test:")
    
    print(f"   Purpose: Most powerful test for detecting departures from normality")
    print(f"   Statistic: {sw_stat:.6f}")
    print(f"   P-value: {sw_pvalue:.6f}")
    print(f"   Result: {'REJECT normality (H0)' if sw_pvalue < 0.05 else 'Cannot reject normality (H0)'}")
    print(f"   Interpretation: {'Data is NOT normally distributed' if sw_pvalue < 0.05 else 'Data may be normally distributed'}")
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.kstest(returns_clean, 'norm', 
                                       args=(returns_clean.mean(), returns_clean.std()))
    print(f"\n3. Kolmogorov-Smirnov Test:")
    print(f"   Purpose: Tests maximum distance between empirical and theoretical CDF")
    print(f"   Statistic: {ks_stat:.6f}")
    print(f"   P-value: {ks_pvalue:.6f}")
    print(f"   Result: {'REJECT normality (H0)' if ks_pvalue < 0.05 else 'Cannot reject normality (H0)'}")
    print(f"   Interpretation: {'Data is NOT normally distributed' if ks_pvalue < 0.05 else 'Data may be normally distributed'}")
    
    # Anderson-Darling test
    ad_result = stats.anderson(returns_clean, dist='norm')
    print(f"\n4. Anderson-Darling Test:")
    print(f"   Purpose: More sensitive to tail behavior than K-S test")
    print(f"   Statistic: {ad_result.statistic:.6f}")
    print(f"   Critical values at significance levels:")
    for i, (cv, sl) in enumerate(zip(ad_result.critical_values, ad_result.significance_level)):
        reject = "REJECT" if ad_result.statistic > cv else "Cannot reject"
        print(f"     {sl}%: {cv:.3f} -> {reject}")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    if jb_pvalue < 0.05 or sw_pvalue < 0.05:
        print("Returns are NOT normally distributed (heavy tails, skewness)")
        print("This is typical for financial returns - use robust methods!")
    else:
        print("Returns appear to be approximately normally distributed")
    print("="*60)


def plot_qq_plot(returns):
    """
    Create Q-Q plot to visually assess normality
    
    Q-Q (Quantile-Quantile) plots compare the quantiles of your data
    against the quantiles of a theoretical normal distribution.
    
    If data is normal: points fall on the diagonal line
    If data has heavy tails: points deviate at the extremes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    returns_clean = returns.dropna()
    
    # Q-Q plot
    stats.probplot(returns_clean, dist="norm", plot=ax1)
    ax1.set_title('Q-Q Plot: Bitcoin Returns vs Normal Distribution', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram with normal distribution overlay
    ax2.hist(returns_clean, bins=50, density=True, alpha=0.7, edgecolor='black')
    
    # Fit and plot normal distribution
    mu, sigma = returns_clean.mean(), returns_clean.std()
    x = np.linspace(returns_clean.min(), returns_clean.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
             label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
    
    ax2.set_xlabel('Return', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Returns Distribution vs Normal', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/example-path/qq_plot.png', dpi=300, bbox_inches='tight')
    print("Saved: qq_plot.png")
    plt.close()


# ============================================================================
# Example 9: Additional Analysis
# ============================================================================

def analyze_volatility_clustering(btc_data):
    """
    Analyze volatility clustering - periods of high volatility tend to cluster together
    """
    returns = btc_data['Returns'].dropna()
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=30).std()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Returns with rolling volatility
    ax1.plot(returns.index, returns, alpha=0.5, linewidth=0.5, label='Daily Returns')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(rolling_vol.index, rolling_vol, color='red', linewidth=2, 
                  label='30-Day Rolling Volatility', alpha=0.7)
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Daily Returns', fontsize=12, color='blue')
    ax1_twin.set_ylabel('Rolling Volatility', fontsize=12, color='red')
    ax1.set_title('Volatility Clustering in Bitcoin Returns', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Autocorrelation of squared returns (test for ARCH effects)
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(returns**2, ax=ax2)
    ax2.set_title('Autocorrelation of Squared Returns (ARCH Effects)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Lag', fontsize=12)
    ax2.set_ylabel('Autocorrelation', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/example-path/volatility_clustering.png', dpi=300, bbox_inches='tight')
    print("Saved: volatility_clustering.png")
    plt.close()

