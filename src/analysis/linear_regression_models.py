#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Regression Models

This script implements OLS regression models for each performance metric using the
model-ready datasets created during the variable selection phase. It generates
comprehensive model diagnostics, parameter estimates, and visualization of results.

Author: Daniel Tierney
Date: April 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
import logging
from datetime import datetime
import json
import matplotlib.ticker as mticker

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "output" / "modeling"
OUTPUT_DIR = BASE_DIR / "output" / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / f"regression_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger("regression_models")

# Set Matplotlib and Seaborn styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def log_section(section_name):
    """Log a section header to improve log readability."""
    logger.info(f"\n{'=' * 40}\n{section_name}\n{'=' * 40}")

def load_model_data():
    """
    Load model-ready datasets for each performance metric.
    
    Returns:
        dict: Dictionary with performance metrics as keys and DataFrames as values
    """
    log_section("Loading Model Data")
    
    performance_metrics = [
        "overall_performance", 
        "football_performance", 
        "hurling_performance", 
        "code_balance"
    ]
    
    datasets = {}
    
    for metric in performance_metrics:
        file_path = DATA_DIR / f"model_data_{metric}.csv"
        
        if file_path.exists():
            datasets[metric] = pd.read_csv(file_path)
            logger.info(f"Loaded {metric} dataset with shape {datasets[metric].shape}")
        else:
            logger.error(f"File not found: {file_path}")
    
    if not datasets:
        logger.error("No datasets could be loaded")
        raise FileNotFoundError("No model datasets found")
    
    return datasets

def inspect_datasets(datasets):
    """
    Inspect the loaded datasets for data quality and structure.
    
    Args:
        datasets: Dictionary with performance metrics as keys and DataFrames as values
    """
    log_section("Inspecting Datasets")
    
    for metric, df in datasets.items():
        logger.info(f"\nDataset: {metric}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_values}")
        
        # Check for variable correlations
        y_col = metric
        X_cols = [col for col in df.columns if col != y_col]
        
        # Compute basic statistics for the dependent variable
        y_mean = df[y_col].mean()
        y_std = df[y_col].std()
        y_min = df[y_col].min()
        y_max = df[y_col].max()
        
        logger.info(f"{y_col} statistics: mean={y_mean:.2f}, std={y_std:.2f}, min={y_min:.2f}, max={y_max:.2f}")
        
        # Log the number of predictor variables
        logger.info(f"Number of predictor variables: {len(X_cols)}")
        logger.info(f"Predictor variables: {', '.join(X_cols)}")
        
        # Compute VIF to check for remaining multicollinearity
        X = df[X_cols].copy()
        
        # Add a constant (intercept) for statsmodels
        X = sm.add_constant(X)
        
        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # Log VIF values (excluding the constant)
        vif_filtered = vif_data[vif_data["Variable"] != "const"].sort_values("VIF", ascending=False)
        logger.info(f"\nVariance Inflation Factors (VIF):")
        for _, row in vif_filtered.iterrows():
            logger.info(f"  {row['Variable']}: {row['VIF']:.2f}")
        
        # Check if any VIF values are above the threshold (excluding the constant)
        vif_threshold = 5
        high_vif = vif_filtered[vif_filtered["VIF"] > vif_threshold]
        if not high_vif.empty:
            logger.warning(f"Variables with VIF > {vif_threshold}: {', '.join(high_vif['Variable'].tolist())}")
        else:
            logger.info(f"All variables have VIF < {vif_threshold}, indicating low multicollinearity")

def build_regression_models(datasets):
    """
    Build OLS regression models for each performance metric.
    
    Args:
        datasets: Dictionary with performance metrics as keys and DataFrames as values
        
    Returns:
        dict: Dictionary with performance metrics as keys and fitted model objects as values
    """
    log_section("Building Regression Models")
    
    models = {}
    model_summaries = {}
    
    for metric, df in datasets.items():
        logger.info(f"\nBuilding model for {metric}")
        
        # Split into dependent and independent variables
        y_col = metric
        X_cols = [col for col in df.columns if col != y_col]
        
        y = df[y_col]
        X = df[X_cols]
        
        # Add a constant (intercept)
        X = sm.add_constant(X)
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Save the model and its summary
        models[metric] = model
        model_summaries[metric] = model.summary()
        
        # Log key model statistics
        logger.info(f"Model R-squared: {model.rsquared:.4f}")
        logger.info(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        logger.info(f"F-statistic: {model.fvalue:.4f} (p-value: {model.f_pvalue:.4g})")
        logger.info(f"AIC: {model.aic:.4f}")
        logger.info(f"BIC: {model.bic:.4f}")
        
        # Log coefficient estimates with their p-values
        logger.info("\nCoefficient estimates:")
        for var, coef in model.params.items():
            p_value = model.pvalues[var]
            std_err = model.bse[var]
            t_stat = model.tvalues[var]
            conf_int = model.conf_int().loc[var]
            
            significance = ""
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            elif p_value < 0.1:
                significance = "."
                
            logger.info(f"  {var}: {coef:.4f} (SE: {std_err:.4f}, t: {t_stat:.2f}, p: {p_value:.4g}) {significance}")
            logger.info(f"     95% CI: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")
    
    return models, model_summaries

def evaluate_model_diagnostics(models, datasets):
    """
    Evaluate model diagnostics including normality of residuals, 
    homoscedasticity, and linearity.
    
    Args:
        models: Dictionary with performance metrics as keys and fitted model objects as values
        datasets: Dictionary with performance metrics as keys and DataFrames as values
        
    Returns:
        dict: Dictionary with diagnostic results
    """
    log_section("Evaluating Model Diagnostics")
    
    diagnostics = {}
    
    for metric, model in models.items():
        logger.info(f"\nDiagnostics for {metric} model")
        df = datasets[metric]
        
        # Store diagnostic results
        model_diagnostics = {}
        
        # Prepare data for diagnostics
        y_col = metric
        X_cols = [col for col in df.columns if col != y_col]
        
        y = df[y_col]
        X = df[X_cols]
        X_with_const = sm.add_constant(X)
        
        # Get residuals
        residuals = model.resid
        model_diagnostics['residuals'] = residuals
        
        # Get fitted values
        fitted_values = model.fittedvalues
        model_diagnostics['fitted_values'] = fitted_values
        
        # Check normality of residuals using Jarque-Bera test
        jb_test = sm.stats.jarque_bera(residuals)
        jb_stat, jb_pval = jb_test[0], jb_test[1]
        model_diagnostics['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pval}
        
        if jb_pval < 0.05:
            logger.warning(f"Residuals may not be normally distributed (Jarque-Bera p-value: {jb_pval:.4g})")
        else:
            logger.info(f"Residuals appear to be normally distributed (Jarque-Bera p-value: {jb_pval:.4g})")
        
        # Check for heteroscedasticity using Breusch-Pagan test
        bp_test = sm.stats.diagnostic.het_breuschpagan(residuals, X_with_const)
        bp_lm_stat, bp_lm_pval = bp_test[0], bp_test[1]
        model_diagnostics['breusch_pagan'] = {'statistic': bp_lm_stat, 'p_value': bp_lm_pval}
        
        if bp_lm_pval < 0.05:
            logger.warning(f"Heteroscedasticity detected (Breusch-Pagan p-value: {bp_lm_pval:.4g})")
        else:
            logger.info(f"No significant heteroscedasticity detected (Breusch-Pagan p-value: {bp_lm_pval:.4g})")
        
        # Check for linearity: RESET test
        try:
            reset_test = sm.stats.diagnostic.linear_reset(model, power=2, test_type='fitted')
            reset_stat = reset_test.statistic
            reset_pval = reset_test.pvalue
            model_diagnostics['reset_test'] = {'statistic': reset_stat, 'p_value': reset_pval}
            
            if reset_pval < 0.05:
                logger.warning(f"Non-linearity detected (RESET p-value: {reset_pval:.4g})")
            else:
                logger.info(f"No significant non-linearity detected (RESET p-value: {reset_pval:.4g})")
        except Exception as e:
            logger.warning(f"Could not perform RESET test: {str(e)}")
            model_diagnostics['reset_test'] = {'statistic': None, 'p_value': None}
        
        # Check for omitted variables: Calculate RMSE
        rmse = np.sqrt(np.mean(residuals**2))
        model_diagnostics['rmse'] = rmse
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        # Check for influential observations using Cook's distance
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        model_diagnostics['cooks_distance'] = cooks_d
        
        # Identify potentially influential observations (Cook's distance > 4/n)
        threshold = 4 / len(y)
        influential = np.where(cooks_d > threshold)[0]
        model_diagnostics['influential_obs'] = influential
        
        if len(influential) > 0:
            logger.warning(f"Found {len(influential)} potentially influential observations")
            logger.info(f"Indices of influential observations: {influential.tolist()}")
        else:
            logger.info("No potentially influential observations detected")
        
        diagnostics[metric] = model_diagnostics 

    return diagnostics

def create_coefficient_plots(models):
    """
    Create coefficient plots to visualize the effect sizes of predictors
    across different performance metrics.
    
    Args:
        models: Dictionary with performance metrics as keys and fitted model objects as values
    """
    log_section("Creating Coefficient Plots")
    
    # Create a figure for each model
    for metric, model in models.items():
        logger.info(f"Creating coefficient plot for {metric}")
        
        # Extract coefficients and their confidence intervals (excluding intercept)
        coefs = model.params[1:]  # Exclude the intercept
        conf_int = model.conf_int().iloc[1:]  # Exclude the intercept
        
        # Create DataFrame for plotting
        coef_df = pd.DataFrame({
            'Variable': coefs.index,
            'Coefficient': coefs.values,
            'Lower CI': conf_int[0].values,
            'Upper CI': conf_int[1].values,
            'P-value': model.pvalues[1:].values
        })
        
        # Add significance indicators
        coef_df['Significance'] = 'Not significant'
        coef_df.loc[coef_df['P-value'] < 0.1, 'Significance'] = 'p < 0.1'
        coef_df.loc[coef_df['P-value'] < 0.05, 'Significance'] = 'p < 0.05'
        coef_df.loc[coef_df['P-value'] < 0.01, 'Significance'] = 'p < 0.01'
        coef_df.loc[coef_df['P-value'] < 0.001, 'Significance'] = 'p < 0.001'
        
        # Sort by coefficient value
        coef_df = coef_df.sort_values('Coefficient')
        
        # Define colors by significance level
        colors = {
            'Not significant': 'lightgray',
            'p < 0.1': 'lightblue',
            'p < 0.05': 'blue',
            'p < 0.01': 'darkblue',
            'p < 0.001': 'purple'
        }
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot the coefficients and their confidence intervals
        for i, (_, row) in enumerate(coef_df.iterrows()):
            plt.plot(
                [row['Lower CI'], row['Upper CI']], 
                [i, i], 
                'o-', 
                color=colors[row['Significance']], 
                linewidth=2, 
                markersize=8
            )
            plt.plot(
                [row['Coefficient']], 
                [i], 
                'o', 
                color='red', 
                markersize=8
            )
        
        # Add zero reference line
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Customize the plot
        plt.yticks(range(len(coef_df)), coef_df['Variable'])
        plt.xlabel('Coefficient Value (with 95% Confidence Interval)')
        plt.title(f'Regression Coefficients for {metric}\n'
                  f'Note: Negative coefficients indicate better performance (lower grade)')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=sig)
            for sig, color in colors.items()
        ]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Coefficient'))
        
        plt.legend(handles=legend_elements, loc='best')
        
        # Save the figure
        fig_path = FIGURE_DIR / f"coefficient_plot_{metric}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved coefficient plot to {fig_path}")

def create_diagnostic_plots(models, diagnostics):
    """
    Create diagnostic plots to assess model assumptions.
    
    Args:
        models: Dictionary with performance metrics as keys and fitted model objects as values
        diagnostics: Dictionary with diagnostic results
    """
    log_section("Creating Diagnostic Plots")
    
    for metric, model in models.items():
        logger.info(f"Creating diagnostic plots for {metric}")
        model_diag = diagnostics[metric]
        
        # Create a 2x2 grid of diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Residuals vs Fitted Values plot
        residuals = model_diag['residuals']
        fitted = model_diag['fitted_values']
        
        axes[0, 0].scatter(fitted, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted Values')
        
        # Add a smoothed line to help visualize patterns
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, fitted, frac=0.5)
            axes[0, 0].plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2)
        except:
            logger.warning("Could not add LOWESS smoothed line to residual plot")
        
        # 2. QQ Plot
        from scipy import stats
        
        # Sort the residuals
        sorted_residuals = np.sort(residuals)
        
        # Generate theoretical quantiles
        theoretical_quantiles = stats.norm.ppf(np.arange(1, len(residuals) + 1) / (len(residuals) + 1))
        
        # Create QQ plot
        axes[0, 1].scatter(theoretical_quantiles, sorted_residuals, alpha=0.5)
        
        # Add the reference line
        min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
        max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        axes[0, 1].set_xlabel('Theoretical Quantiles')
        axes[0, 1].set_ylabel('Sample Quantiles')
        axes[0, 1].set_title('Normal Q-Q Plot')
        
        # 3. Scale-Location Plot (sqrt of standardized residuals vs fitted values)
        std_residuals = residuals / np.std(residuals)
        sqrt_abs_std_resid = np.sqrt(np.abs(std_residuals))
        
        axes[1, 0].scatter(fitted, sqrt_abs_std_resid, alpha=0.5)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Standardized Residuals|')
        axes[1, 0].set_title('Scale-Location Plot')
        
        # Add a smoothed line
        try:
            smoothed = lowess(sqrt_abs_std_resid, fitted, frac=0.5)
            axes[1, 0].plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2)
        except:
            logger.warning("Could not add LOWESS smoothed line to scale-location plot")
        
        # 4. Cook's Distance Plot
        cooks_d = model_diag['cooks_distance']
        axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt='ro', linefmt='r-', basefmt='b-')
        axes[1, 1].set_xlabel('Observation Index')
        axes[1, 1].set_ylabel("Cook's Distance")
        axes[1, 1].set_title("Cook's Distance Plot")
        
        # Add threshold line
        threshold = 4 / len(cooks_d)
        axes[1, 1].axhline(y=threshold, color='g', linestyle='--', label=f'Threshold ({threshold:.3f})')
        axes[1, 1].legend()
        
        # Highlight influential points
        influential = model_diag['influential_obs']
        if len(influential) > 0:
            axes[1, 1].plot(influential, cooks_d[influential], 'ro', ms=10, mfc='none', label='Influential Points')
            axes[1, 1].legend()
        
        # Set an overall title for the figure
        fig.suptitle(f'Diagnostic Plots for {metric} Model', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the overall title
        
        # Save the figure
        fig_path = FIGURE_DIR / f"diagnostic_plots_{metric}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved diagnostic plots to {fig_path}")

def create_comparative_analysis(models):
    """
    Create comparative analysis of models across different performance metrics.
    
    Args:
        models: Dictionary with performance metrics as keys and fitted model objects as values
    """
    log_section("Creating Comparative Analysis")
    
    # Extract key statistics
    model_stats = {}
    
    for metric, model in models.items():
        model_stats[metric] = {
            'R-squared': model.rsquared,
            'Adjusted R-squared': model.rsquared_adj,
            'F-statistic': model.fvalue,
            'p-value': model.f_pvalue,
            'AIC': model.aic,
            'BIC': model.bic,
            'Log-likelihood': model.llf,
            'Number of observations': model.nobs
        }
    
    # Create a DataFrame for comparison
    stats_df = pd.DataFrame(model_stats)
    
    # Log the comparison
    logger.info("\nModel comparison:")
    logger.info(stats_df.to_string())
    
    # Create a bar chart for R-squared and Adjusted R-squared
    plt.figure(figsize=(10, 6))
    
    metrics = list(model_stats.keys())
    r2_values = [model_stats[m]['R-squared'] for m in metrics]
    adj_r2_values = [model_stats[m]['Adjusted R-squared'] for m in metrics]
    
    bar_width = 0.35
    index = np.arange(len(metrics))
    
    plt.bar(index, r2_values, bar_width, label='R-squared')
    plt.bar(index + bar_width, adj_r2_values, bar_width, label='Adjusted R-squared')
    
    plt.xlabel('Performance Metric')
    plt.ylabel('Value')
    plt.title('Model Fit Comparison')
    plt.xticks(index + bar_width / 2, metrics)
    plt.legend()
    
    # Save the figure
    fig_path = FIGURE_DIR / "model_r2_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved R-squared comparison plot to {fig_path}")
    
    # Create a comparison of coefficient significances
    significance_data = {}
    
    for metric, model in models.items():
        # Get p-values (excluding constant)
        p_values = model.pvalues[1:]
        
        # Count significance levels
        significance_data[metric] = {
            'p < 0.001': sum(p_values < 0.001),
            'p < 0.01': sum((p_values >= 0.001) & (p_values < 0.01)),
            'p < 0.05': sum((p_values >= 0.01) & (p_values < 0.05)),
            'p < 0.1': sum((p_values >= 0.05) & (p_values < 0.1)),
            'Not significant': sum(p_values >= 0.1)
        }
    
    # Create a DataFrame for the significance counts
    significance_df = pd.DataFrame(significance_data)
    
    # Log the significance comparison
    logger.info("\nSignificance comparison:")
    logger.info(significance_df.to_string())
    
    # Create a stacked bar chart for significance levels
    plt.figure(figsize=(10, 6))
    
    significance_levels = list(significance_df.index)
    colors = ['purple', 'darkblue', 'blue', 'lightblue', 'lightgray']
    
    bottom = np.zeros(len(metrics))
    
    for i, level in enumerate(significance_levels):
        values = [significance_df.loc[level, m] for m in metrics]
        plt.bar(metrics, values, bottom=bottom, label=level, color=colors[i])
        bottom += values
    
    plt.xlabel('Performance Metric')
    plt.ylabel('Number of Variables')
    plt.title('Significance of Variables Across Models')
    plt.legend()
    
    # Save the figure
    fig_path = FIGURE_DIR / "model_significance_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved significance comparison plot to {fig_path}")
    
    # Create a heatmap of coefficient values across models
    # First, collect all variables used in any model
    all_vars = set()
    for model in models.values():
        all_vars.update(model.params.index[1:])  # Exclude the constant
    
    all_vars = list(all_vars)
    
    # Create a DataFrame with coefficients for all variables across models
    coef_data = {}
    
    for metric, model in models.items():
        coef_data[metric] = {}
        
        for var in all_vars:
            if var in model.params:
                coef_data[metric][var] = model.params[var]
            else:
                coef_data[metric][var] = np.nan
    
    coef_df = pd.DataFrame(coef_data)
    
    # Log the coefficient comparison
    logger.info("\nCoefficient comparison:")
    logger.info(coef_df.to_string())
    
    # Create a heatmap of coefficients
    plt.figure(figsize=(12, 8))
    
    # Use a diverging colormap centered at zero
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    mask = np.isnan(coef_df.values)
    
    sns.heatmap(
        coef_df, 
        cmap=cmap, 
        center=0, 
        annot=True, 
        fmt=".3f", 
        linewidths=.5,
        mask=mask
    )
    
    plt.title('Coefficient Values Across Models\nNote: Negative coefficients indicate better performance (lower grade)')
    
    # Save the figure
    fig_path = FIGURE_DIR / "model_coefficient_heatmap.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved coefficient heatmap to {fig_path}")
    
    return stats_df, significance_df, coef_df 

def generate_model_report(models, model_summaries, diagnostics, comparative_stats):
    """
    Generate a comprehensive model report in markdown format.
    
    Args:
        models: Dictionary with performance metrics as keys and fitted model objects as values
        model_summaries: Dictionary with performance metrics as keys and model summaries as values
        diagnostics: Dictionary with diagnostic results
        comparative_stats: Tuple of DataFrames with comparative statistics
    """
    log_section("Generating Model Report")
    
    stats_df, significance_df, coef_df = comparative_stats
    
    # Get all variables from the coefficient DataFrame
    all_vars = coef_df.index.tolist()
    
    report_path = OUTPUT_DIR / f"regression_model_report_{datetime.now().strftime('%Y%m%d')}.md"
    
    with open(report_path, 'w') as f:
        # Write header
        f.write("# Linear Regression Models Report\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Write introduction
        f.write("## Introduction\n\n")
        f.write("This report presents the results of linear regression models for the Cork GAA club performance analysis. ")
        f.write("The models examine the relationship between demographic, socioeconomic, and geographic factors and club performance. ")
        f.write("Note that in the original performance metrics, lower values indicate better performance ")
        f.write("(1 is best, 6 is worst), so negative coefficients indicate factors that improve performance.\n\n")
        
        # Write comparative analysis
        f.write("## Comparative Analysis\n\n")
        
        f.write("### Model Fit Statistics\n\n")
        f.write(stats_df.to_markdown())
        f.write("\n\n")
        
        f.write("### Variable Significance Across Models\n\n")
        f.write(significance_df.to_markdown())
        f.write("\n\n")
        
        f.write("### Coefficient Comparison\n\n")
        f.write(coef_df.to_markdown())
        f.write("\n\n")
        
        f.write("### Visualizations\n\n")
        
        f.write("The following visualizations are available in the `output/models/figures` directory:\n\n")
        f.write("- Model R-squared comparison (`model_r2_comparison.png`)\n")
        f.write("- Significance of variables across models (`model_significance_comparison.png`)\n")
        f.write("- Coefficient values across models (`model_coefficient_heatmap.png`)\n")
        f.write("- Coefficient plots for each performance metric\n")
        f.write("- Diagnostic plots for each model\n\n")
        
        # Write detailed model results
        f.write("## Detailed Model Results\n\n")
        
        for metric, model in models.items():
            f.write(f"### {metric} Model\n\n")
            
            # Write model summary
            f.write("#### Model Summary\n\n")
            f.write("```\n")
            f.write(str(model_summaries[metric]))
            f.write("\n```\n\n")
            
            # Write key diagnostics
            f.write("#### Diagnostics\n\n")
            
            diag = diagnostics[metric]
            
            f.write("| Diagnostic | Value | Interpretation |\n")
            f.write("|------------|-------|---------------|\n")
            
            # Jarque-Bera test for normality
            jb_stat = diag['jarque_bera']['statistic']
            jb_pval = diag['jarque_bera']['p_value']
            jb_interp = "Residuals appear normal" if jb_pval >= 0.05 else "Residuals may not be normal"
            f.write(f"| Jarque-Bera test | Stat: {jb_stat:.4f}, p-value: {jb_pval:.4g} | {jb_interp} |\n")
            
            # Breusch-Pagan test for heteroscedasticity
            bp_stat = diag['breusch_pagan']['statistic']
            bp_pval = diag['breusch_pagan']['p_value']
            bp_interp = "Homoscedasticity assumption met" if bp_pval >= 0.05 else "Heteroscedasticity detected"
            f.write(f"| Breusch-Pagan test | Stat: {bp_stat:.4f}, p-value: {bp_pval:.4g} | {bp_interp} |\n")
            
            # RESET test for linearity
            reset_stat = diag['reset_test']['statistic']
            reset_pval = diag['reset_test']['p_value']
            if reset_stat is not None and reset_pval is not None:
                reset_interp = "Linearity assumption met" if reset_pval >= 0.05 else "Non-linearity detected"
                f.write(f"| RESET test | Stat: {reset_stat:.4f}, p-value: {reset_pval:.4g} | {reset_interp} |\n")
            else:
                f.write("| RESET test | Not performed | Could not evaluate linearity |\n")
            
            # RMSE
            rmse = diag['rmse']
            f.write(f"| RMSE | {rmse:.4f} | Model prediction error |\n")
            
            # Influential observations
            n_infl = len(diag['influential_obs'])
            f.write(f"| Influential observations | {n_infl} | Based on Cook's distance > 4/n |\n\n")
            
            # Write coefficient interpretation
            f.write("#### Coefficient Interpretation\n\n")
            f.write("*Note: In the original performance metrics, lower values indicate better performance (1 is best, 6 is worst). ")
            f.write("Therefore, negative coefficients indicate factors that improve performance (lower the grade).*\n\n")
            
            f.write("| Variable | Coefficient | Std. Error | t-value | p-value | Significance | Interpretation |\n")
            f.write("|----------|-------------|------------|---------|---------|--------------|----------------|\n")
            
            # Skip the intercept
            for var in model.params.index[1:]:
                coef = model.params[var]
                stderr = model.bse[var]
                tval = model.tvalues[var]
                pval = model.pvalues[var]
                
                # Add significance indicators
                if pval < 0.001:
                    sig = "p < 0.001 ***"
                elif pval < 0.01:
                    sig = "p < 0.01 **"
                elif pval < 0.05:
                    sig = "p < 0.05 *"
                elif pval < 0.1:
                    sig = "p < 0.1 ."
                else:
                    sig = "Not significant"
                
                # Add interpretation
                if coef < 0:
                    interp = "Improves performance"
                else:
                    interp = "Worsens performance"
                
                f.write(f"| {var} | {coef:.4f} | {stderr:.4f} | {tval:.2f} | {pval:.4g} | {sig} | {interp} |\n")
            
            f.write("\n")
        
        # Write conclusion
        f.write("## Conclusion\n\n")
        f.write("The regression models provide insights into the demographic, socioeconomic, and geographic factors ")
        f.write("that influence GAA club performance in Cork. Key findings include:\n\n")
        
        # Calculate average R-squared
        avg_r2 = stats_df.loc['R-squared'].mean()
        best_model = stats_df.loc['R-squared'].idxmax()
        
        f.write(f"- The models explain on average {avg_r2:.1%} of the variation in club performance.\n")
        f.write(f"- The best-performing model is for {best_model} (R² = {stats_df.loc['R-squared', best_model]:.1%}).\n")
        
        # Identify most consistently significant variables
        significant_vars = []
        for var in all_vars:
            significance_count = 0
            for metric, model in models.items():
                if var in model.params.index and model.pvalues[var] < 0.05:
                    significance_count += 1
            
            if significance_count >= len(models) // 2:
                direction = "positive" if sum(model.params.get(var, 0) > 0 for model in models.values()) > len(models) // 2 else "negative"
                effect = "worsening" if direction == "positive" else "improving"
                significant_vars.append((var, significance_count, direction, effect))
        
        significant_vars.sort(key=lambda x: x[1], reverse=True)
        
        if significant_vars:
            f.write("- The most consistently significant variables across models are:\n")
            for var, count, direction, effect in significant_vars[:3]:
                f.write(f"  - {var}: Significant in {count}/{len(models)} models, with a {direction} coefficient ({effect} performance)\n")
        
        # Note about limitations
        f.write("\n### Limitations\n\n")
        
        has_non_normal = any(diagnostics[m]['jarque_bera']['p_value'] < 0.05 for m in models)
        has_heteroskedasticity = any(diagnostics[m]['breusch_pagan']['p_value'] < 0.05 for m in models)
        has_non_linearity = any(diagnostics[m]['reset_test']['p_value'] < 0.05 for m in models)
        
        f.write("The models have the following limitations:\n\n")
        
        if has_non_normal:
            f.write("- Some models show non-normal residuals, which may affect the validity of confidence intervals and hypothesis tests.\n")
        
        if has_heteroskedasticity:
            f.write("- Heteroscedasticity was detected in some models, which may affect the efficiency of the estimates.\n")
        
        if has_non_linearity:
            f.write("- Some models show evidence of non-linearity, suggesting that the linear model may not fully capture the relationships.\n")
        
        f.write("- The analysis is limited by the available data and the small sample size of GAA clubs in Cork.\n")
        f.write("- The models do not establish causal relationships, only associations between variables.\n")
        
        # Recommendations
        f.write("\n### Next Steps\n\n")
        f.write("Based on the findings, the following steps are recommended:\n\n")
        f.write("1. Consider model refinements to address any issues with normality, heteroscedasticity, or non-linearity.\n")
        f.write("2. Investigate potential nonlinear relationships through polynomial terms or other transformations.\n")
        f.write("3. Test for spatial autocorrelation in the residuals to determine if spatial regression models are needed.\n")
        f.write("4. Validate model stability through cross-validation or bootstrap analysis.\n")
        f.write("5. Compare results across different performance metrics to identify common patterns.\n")
    
    logger.info(f"Model report generated: {report_path}")
    
    return report_path

def main():
    """Main execution function."""
    try:
        log_section("Starting Linear Regression Analysis")
        
        # Load model data
        datasets = load_model_data()
        
        # Inspect datasets
        inspect_datasets(datasets)
        
        # Build regression models
        models, model_summaries = build_regression_models(datasets)
        
        # Evaluate model diagnostics
        diagnostics = evaluate_model_diagnostics(models, datasets)
        
        # Create visualizations
        create_coefficient_plots(models)
        
        # Only proceed with diagnostic plots and analysis if diagnostics were successfully generated
        if diagnostics:
            create_diagnostic_plots(models, diagnostics)
            comparative_stats = create_comparative_analysis(models)
            
            # Generate model report
            report_path = generate_model_report(models, model_summaries, diagnostics, comparative_stats)
            
            # Save model results as JSON
            results = {
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_stats': {
                    metric: {
                        'r_squared': model.rsquared,
                        'adj_r_squared': model.rsquared_adj,
                        'aic': model.aic,
                        'bic': model.bic,
                        'log_likelihood': model.llf,
                        'f_statistic': model.fvalue,
                        'f_pvalue': model.f_pvalue,
                        'num_observations': int(model.nobs)
                    } for metric, model in models.items()
                }
            }
            
            # Add coefficients
            results['coefficients'] = {
                metric: {
                    var: {
                        'value': float(model.params[var]),
                        'std_error': float(model.bse[var]),
                        't_statistic': float(model.tvalues[var]),
                        'p_value': float(model.pvalues[var]),
                        'conf_int_lower': float(model.conf_int().loc[var, 0]),
                        'conf_int_upper': float(model.conf_int().loc[var, 1])
                    } for var in model.params.index
                } for metric, model in models.items()
            }
            
            # Save results
            results_path = OUTPUT_DIR / f"regression_model_results_{datetime.now().strftime('%Y%m%d')}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Model results saved: {results_path}")
            
        log_section("Linear Regression Analysis Completed")
        logger.info(f"Report generated: {report_path}")
        logger.info(f"Results saved: {results_path}")
        logger.info(f"Visualizations saved in: {FIGURE_DIR}")
        
    except Exception as e:
        logger.error(f"Error in linear regression analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 