# eda.py
# Description: Generates and displays exploratory data analysis plots.

import matplotlib.pyplot as plt
import seaborn as sns
import config
import feature_engineering

def plot_demand_timeseries(data):
    """Plots the full demand time series."""
    data[config.TARGET_COL].plot(figsize=(15, 6), title="Electricity Demand over Time")
    plt.xlabel("Year")
    plt.ylabel("Demand (in KW)")
    plt.show()

def plot_hourly_boxplot(data):
    """Plots demand by the hour of the day."""
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=data, x='hour', y=config.TARGET_COL)
    plt.title("Demand by the Hour of the Day")
    plt.show()

def plot_monthly_boxplot(data):
    """Plots demand by the month."""
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=data, x='month', y=config.TARGET_COL)
    plt.title("Demand by the Month")
    plt.show()

def plot_temp_scatter(data):
    """Plots a scatter plot of Demand vs. Temperature."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=data, x='Temperature', y=config.TARGET_COL, alpha=0.5)
    plt.title("Demand vs Temperature")
    plt.show()

def plot_correlation_matrix(data):
    """Plots the correlation matrix heatmap."""
    plt.figure(figsize=(15, 7))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

def main():
    """Main EDA pipeline."""
    print("Loading data for EDA...")
    data = feature_engineering.get_processed_data(config.DATA_PATH)
    
    print("Generating plots...")
    plot_demand_timeseries(data)
    plot_hourly_boxplot(data)
    plot_monthly_boxplot(data)
    plot_temp_scatter(data)
    plot_correlation_matrix(data)
    print("EDA complete.")

if __name__ == "__main__":
    main()