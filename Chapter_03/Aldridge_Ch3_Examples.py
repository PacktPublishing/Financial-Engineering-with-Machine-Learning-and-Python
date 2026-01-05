"""
Chapter 3 - Measuring Investment Performance
Examples from "Financial Engineering with Machine Learning and Python" by Irene Aldridge
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#==============================================================================
# Example 1: Computing Mean and Standard Deviation
#==============================================================================

def compute_statistics(returns):
    """
    Compute mean and standard deviation from a sample vector of returns.
    
    Parameters:
    -----------
    returns : array-like
        Array of returns
    
    Returns:
    --------
    m : float
        Mean of returns
    s : float
        Standard deviation of returns
    """
    R = np.array(returns)
    m = np.mean(R)
    s = np.std(R, ddof=1)  # Using sample standard deviation (N-1)
    return m, s


#==============================================================================
# Example 2: Figure 3.1 - Returns with Mean and Standard Deviation
#==============================================================================

def plot_returns_with_std(df, save_path='figure_3_1.png'):
    """
    Plot returns with mean and +/- 1 standard deviation bands.
    Replicates Figure 3.1 from the chapter.
    """
    returns = df['Return'].values
    m, s = compute_statistics(returns)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot returns as scatter points
    ax.scatter(df.index, returns, color='steelblue', s=50, zorder=3, label='Returns')
    
    # Plot mean line
    ax.axhline(y=m, color='black', linestyle='-', linewidth=2, label='mean')
    
    # Plot +/- 1 standard deviation lines
    ax.axhline(y=m + s, color='red', linestyle='-', linewidth=2, label='mean + 1 stdev')
    ax.axhline(y=m - s, color='red', linestyle='-', linewidth=2, label='mean - 1 stdev')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Return', fontsize=12)
    ax.set_title('AAPL Returns, Mean and Standard Deviation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 3.1 saved to {save_path}")
    plt.close()
    
    return m, s


#==============================================================================
# Example 3: Figure 3.2 - Multiple Standard Deviation Bands
#==============================================================================

def plot_returns_multiple_std(df, save_path='figure_3_2.png'):
    """
    Plot returns with mean and +/- 1, 2, 3 standard deviation bands.
    Replicates Figure 3.2 from the chapter.
    """
    returns = df['Return'].values
    m, s = compute_statistics(returns)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot returns as scatter points
    ax.scatter(df.index, returns, color='steelblue', s=50, zorder=3)
    
    # Plot mean line
    ax.axhline(y=m, color='black', linestyle='-', linewidth=2.5, label='mean')
    
    # Plot standard deviation lines
    ax.axhline(y=m + s, color='red', linestyle='-', linewidth=2, label='mean + 1 stdev')
    ax.axhline(y=m - s, color='red', linestyle='-', linewidth=2, label='mean - 1 stdev')
    
    ax.axhline(y=m + 2*s, color='green', linestyle='-', linewidth=2, label='mean + 2 stdev')
    ax.axhline(y=m - 2*s, color='green', linestyle='-', linewidth=2, label='mean - 2 stdev')
    
    ax.axhline(y=m + 3*s, color='blue', linestyle='-', linewidth=2, label='mean + 3 stdev')
    ax.axhline(y=m - 3*s, color='blue', linestyle='-', linewidth=2, label='mean - 3 stdev')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Return', fontsize=12)
    ax.set_title('AAPL Returns, Mean and Standard Deviation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 3.2 saved to {save_path}")
    plt.close()


#==============================================================================
# Example 4: Figure 3.3 - Bell Curve / Normal Distribution Histogram
#==============================================================================

def plot_normal_distribution(n_samples=5000, mean=0, std=1, save_path='figure_3_3.png'):
    """
    Plot histogram of normal distribution with standard deviation markers.
    Replicates Figure 3.3 from the chapter.
    """
    # Generate random samples from normal distribution
    data = np.random.normal(mean, std, n_samples)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot histogram
    n, bins, patches = ax.hist(data, bins=50, color='steelblue', 
                               edgecolor='black', alpha=0.8)
    
    # Add vertical lines for standard deviations
    ax.axvline(x=mean, color='black', linestyle='-', linewidth=2.5, label='m')
    ax.axvline(x=mean + std, color='red', linestyle='-', linewidth=2, label='m+s')
    ax.axvline(x=mean - std, color='red', linestyle='-', linewidth=2, label='m-s')
    ax.axvline(x=mean + 2*std, color='green', linestyle='-', linewidth=2, label="m+2*s")
    ax.axvline(x=mean - 2*std, color='green', linestyle='-', linewidth=2, label="m-2*s")
    ax.axvline(x=mean + 3*std, color='blue', linestyle='-', linewidth=2, label="m+3*s")
    ax.axvline(x=mean - 3*std, color='blue', linestyle='-', linewidth=2, label="m-3*s")
    
    ax.set_xlabel('Standard Deviations from Mean', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Histogram of Sample Data Drawn from a Normal Distribution with Mean of {mean} and Standard Deviation of {std}.', 
                 fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 3.3 saved to {save_path}")
    plt.close()


#==============================================================================
# Example 5: Cumulative Returns (Figure 4.4)
#==============================================================================

def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns from a series of returns.
    
    Parameters:
    -----------
    returns : array-like
        Array of returns
    
    Returns:
    --------
    cum_returns : np.array
        Cumulative returns
    """
    cum_returns = (1 + returns).cumprod() - 1
    return cum_returns


def plot_cumulative_returns(df, save_path='figure_4_4.png'):
    """
    Plot cumulative returns over time.
    """
    cum_returns = calculate_cumulative_returns(df['Return'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, cum_returns, color='steelblue', linewidth=2, label='Cumulative Return')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title('Cumulative Returns for AAPL', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 4.4 saved to {save_path}")
    plt.close()
    
    return cum_returns


#==============================================================================
# Example 6: Maximum Drawdown Calculation (Figure 53.5)
#==============================================================================

def calculate_drawdown_metrics(df):
    """
    Calculate cumulative returns, high watermarks, and drawdowns.
    This replicates the code shown on page 14 of the chapter.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Return' column
    
    Returns:
    --------
    results : pd.DataFrame
        DataFrame with CumR, HW, and DD columns
    """
    # Compute cumulative returns, HW and MDD
    CumR = []
    i = 0
    HW = []
    hw = 0
    DD = []
    dd = 0
    dd0 = 0
    
    returns = df['Return'].values
    
    for i in range(len(df)):
        i += 1
        cr = (1 + df['Return'].iloc[:i]).product() - 1
        CumR.append(cr)
        
        # Calculate high watermark
        if cr > hw:
            hw = cr
        HW.append(hw)
        
        # Calculate drawdown
        dd = cr - hw
        DD.append(dd)
    
    results = pd.DataFrame({
        'CumR': CumR,
        'HW': HW,
        'DD': DD
    }, index=df.index)
    
    return results


def plot_maximum_drawdown(df, save_path='figure_53_5.png'):
    """
    Plot cumulative returns, high watermark, and drawdown.
    Replicates Figure 53.5 from the chapter.
    """
    results = calculate_drawdown_metrics(df)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative returns
    ax.plot(results.index, results['CumR'], color='steelblue', 
            linewidth=2, label='Cumulative Return')
    
    # Plot high watermark
    ax.plot(results.index, results['HW'], color='black', 
            linewidth=2.5, label='High Watermark')
    
    # Plot drawdown
    ax.plot(results.index, results['DD'], color='green', 
            linewidth=2, linestyle='--', label='Drawdown')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Return', fontsize=12)
    ax.set_title('Maximum Drawdown', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 53.5 saved to {save_path}")
    plt.close()
    
    # Calculate and print maximum drawdown
    max_dd = results['DD'].min()
    max_dd_date = results['DD'].idxmin()
    
    print(f"\nMaximum Drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
    print(f"Maximum Drawdown Date: {max_dd_date}")
    
    return results, max_dd

