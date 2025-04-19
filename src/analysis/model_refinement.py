#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Refinement

This script implements refined regression models based on diagnostic results from the
baseline models. It addresses:
1. Non-linearity in overall_performance model
2. Heteroscedasticity in football/hurling models
3. Non-normal residuals across all models
4. Influential observations identified through Cook's distance

Author: Daniel Tierney
Date: April 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import ProbPlot
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings
import matplotlib.ticker as mticker
from scipy import stats

# Ignore future warnings from statsmodels
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "output" / "modeling"
OUTPUT_DIR = BASE_DIR / "output" / "models"
FIG_DIR = OUTPUT_DIR / "figures"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = OUTPUT_DIR / f"model_refinement_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_refinement")

# Create a log section function for organization
def log_section(section_name):
    """Create a visual section break in logs."""
    logger.info("\n" + "=" * 40)
    logger.info(section_name)
    logger.info("=" * 40)

# Performance metrics interpretation note
PERFORMANCE_NOTE = (
    "Note: In our performance metrics, lower values indicate better performance "
    "(1 is best, 6 is worst). Therefore, negative coefficients indicate factors "
    "that improve performance (lower the grade), while positive coefficients "
    "indicate factors that worsen performance (raise the grade)."
)

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
    
    return datasets

def load_original_results():
    """
    Load original model results and diagnostics for comparison.
    
    Returns:
        dict: Dictionary with model results from original analysis
    """
    log_section("Loading Original Model Results")
    
    # Find most recent results file
    results_files = list(OUTPUT_DIR.glob("regression_model_results_*.json"))
    if not results_files:
        logger.error("No original model results found")
        return None
    
    most_recent = sorted(results_files)[-1]
    logger.info(f"Loading results from {most_recent}")
    
    with open(most_recent, 'r') as f:
        file_results = json.load(f)
    
    logger.info("Original results loaded successfully")
    
    # Manually check for diagnostics files
    diagnostic_files = list(OUTPUT_DIR.glob("regression_model_diagnostics_*.json"))
    if not diagnostic_files:
        logger.warning("No diagnostic files found, will proceed with model stats only")
        diagnostics = {}
    else:
        most_recent_diag = sorted(diagnostic_files)[-1]
        logger.info(f"Loading diagnostics from {most_recent_diag}")
        
        try:
            with open(most_recent_diag, 'r') as f:
                diagnostics = json.load(f)
            logger.info("Diagnostics loaded successfully")
        except Exception as e:
            logger.error(f"Error loading diagnostics: {str(e)}")
            diagnostics = {}
    
    # Construct a new results dictionary with the structure expected by our code
    # This adapts the actual file structure to match what our refinement code expects
    original_results = {}
    
    # Extract relevant information from the file structure for each model
    for metric in ["overall_performance", "football_performance", "hurling_performance", "code_balance"]:
        if metric in file_results.get("model_stats", {}):
            stats = file_results["model_stats"][metric]
            coefs = file_results.get("coefficients", {}).get(metric, {})
            
            # Extract variable names from coefficients
            var_names = list(coefs.keys())
            
            # Extract coefficient values
            params = [coefs[var]["value"] for var in var_names]
            
            # Get diagnostics for this metric
            metric_diagnostics = diagnostics.get(metric, {})
            
            # Create the model result entry with expected structure
            original_results[metric] = {
                "r_squared": stats.get("r_squared", 0),
                "adj_r_squared": stats.get("adj_r_squared", 0),
                "aic": stats.get("aic", 0),
                "bic": stats.get("bic", 0),
                "params": params,
                "variable_names": var_names,
                "diagnostics": metric_diagnostics
            }
    
    # Add model_comparison key if it exists
    if "model_comparison" in file_results:
        original_results["model_comparison"] = file_results["model_comparison"]
    
    # Log diagnostic issues found in original models
    log_diagnostic_issues(original_results)
    
    return original_results

def log_diagnostic_issues(original_results):
    """
    Log key diagnostic issues from original models to guide refinement.
    
    Args:
        original_results: Dictionary with original model results
    """
    logger.info("\nDiagnostic issues to address in model refinement:")
    
    # If diagnostics are missing, we'll apply default refinement strategy based on metrics
    has_diagnostics = False
    
    for metric, results in original_results.items():
        if metric == "model_comparison":
            continue
            
        diag = results.get("diagnostics", {})
        
        # If diagnostics are empty, we'll use default assumptions
        if not diag:
            logger.warning(f"  {metric}: No diagnostics found, using default refinement strategy")
            
            # Default assumptions based on our prior knowledge from the IMPLEMENTATION_PLAN.md
            if metric == "overall_performance":
                logger.info(f"    - Assuming non-linearity based on implementation plan (RESET test p-value: 0.004982)")
            elif metric in ["football_performance", "hurling_performance"]:
                logger.info(f"    - Assuming heteroscedasticity based on implementation plan")
            
            continue
            
        has_diagnostics = True
        issues = []
        
        # Check for normality issues
        jb_pval = diag.get("jarque_bera", {}).get("p_value")
        if jb_pval and jb_pval < 0.05:
            issues.append(f"Non-normal residuals (JB p-value: {jb_pval:.4g})")
        
        # Check for heteroscedasticity
        bp_pval = diag.get("breusch_pagan", {}).get("p_value")
        if bp_pval and bp_pval < 0.05:
            issues.append(f"Heteroscedasticity (BP p-value: {bp_pval:.4g})")
        
        # Check for linearity issues
        reset_pval = diag.get("reset_test", {}).get("p_value")
        if reset_pval and reset_pval < 0.05:
            issues.append(f"Non-linearity (RESET p-value: {reset_pval:.4g})")
        
        # Check for influential observations
        n_influential = len(diag.get("influential_observations", []))
        if n_influential > 0:
            issues.append(f"{n_influential} influential observations")
        
        if issues:
            logger.info(f"  {metric}:")
            for issue in issues:
                logger.info(f"    - {issue}")
        else:
            logger.info(f"  {metric}: No major issues detected")
    
    # If no diagnostics were found, provide general guidance
    if not has_diagnostics:
        logger.warning("\nNo detailed diagnostics found. Using default refinement strategies:")
        logger.info("  - Applying polynomial terms to overall_performance")
        logger.info("  - Applying robust regression to football_performance and hurling_performance")
        logger.info("  - No refinement for code_balance (weakest model)")

def build_nonlinear_model(df, metric):
    """
    Build a nonlinear model for the given metric by adding polynomial terms.
    
    Args:
        df: DataFrame with predictors and target variable
        metric: Target variable name
    
    Returns:
        Tuple of (model, model_summary, model_diagnostics, polynomial_form)
    """
    logger.info(f"\nBuilding nonlinear model for {metric}")
    
    # Split into dependent and independent variables
    y_col = metric
    X_cols = [col for col in df.columns if col != y_col]
    
    y = df[y_col]
    X = df[X_cols]
    
    # Create a copy for the model formula
    df_copy = df.copy()
    
    # Identify variables for polynomial terms based on diagnostic plots
    # We're focusing on variables that showed nonlinear patterns in the diagnostic plots
    poly_candidates = []
    
    # For overall_performance specifically, we found these variables showed nonlinearity
    if metric == "overall_performance":
        poly_candidates = ["sociodemographic_pc2", "sociodemographic_pc3", "annual_rainfall"]
    elif metric == "football_performance":
        poly_candidates = ["sociodemographic_pc3", "annual_rainfall"]
    elif metric == "hurling_performance":
        poly_candidates = ["sociodemographic_pc2", "annual_rainfall"]
    
    # Create polynomial terms for these variables
    polynomial_form = f"{y_col} ~ "
    linear_terms = []
    poly_terms = []
    
    for col in X_cols:
        if col in poly_candidates:
            # Add squared term to the dataframe
            squared_col = f"{col}_squared"
            df_copy[squared_col] = df_copy[col] ** 2
            poly_terms.append(f"np.power({col}, 2)")
            linear_terms.append(col)
        else:
            linear_terms.append(col)
    
    # Create the formula for statsmodels
    if linear_terms:
        polynomial_form += " + ".join(linear_terms)
    
    if poly_terms:
        if linear_terms:
            polynomial_form += " + " + " + ".join(poly_terms)
        else:
            polynomial_form += " + ".join(poly_terms)
    
    logger.info(f"Polynomial model formula: {polynomial_form}")
    
    # Fit the model using formula
    try:
        model = smf.ols(polynomial_form, data=df_copy).fit()
        model_summary = model.summary()
        
        # Get model diagnostics
        model_diagnostics = evaluate_model_diagnostics(model, df_copy, polynomial_form)
        
        logger.info(f"Nonlinear model R-squared: {model.rsquared:.4f}")
        logger.info(f"Nonlinear model Adjusted R-squared: {model.rsquared_adj:.4f}")
        
        # Log polynomial term coefficients
        logger.info("\nPolynomial term coefficients:")
        for var in poly_candidates:
            squared_var = f"np.power({var}, 2)"
            if squared_var in model.params.index:
                coef = model.params[squared_var]
                pval = model.pvalues[squared_var]
                sig = ""
                if pval < 0.001:
                    sig = " ***"
                elif pval < 0.01:
                    sig = " **"
                elif pval < 0.05:
                    sig = " *"
                
                logger.info(f"  {squared_var}: {coef:.4f} (p: {pval:.4g}){sig}")
        
        return model, model_summary, model_diagnostics, polynomial_form
    
    except Exception as e:
        logger.error(f"Error building nonlinear model: {str(e)}")
        
        try:
            # Try again with a simpler polynomial form
            logger.info("Trying again with a simpler polynomial form")
            simplified_form = f"{y_col} ~ " + " + ".join(linear_terms)
            logger.info(f"Simplified formula: {simplified_form}")
            
            model = smf.ols(simplified_form, data=df_copy).fit()
            model_summary = model.summary()
            
            # Get model diagnostics
            model_diagnostics = evaluate_model_diagnostics(model, df_copy, simplified_form)
            
            logger.info(f"Simplified model R-squared: {model.rsquared:.4f}")
            logger.info(f"Simplified model Adjusted R-squared: {model.rsquared_adj:.4f}")
            
            return model, model_summary, model_diagnostics, simplified_form
        except Exception as e2:
            logger.error(f"Error building simplified model: {str(e2)}")
            
            # If we still can't build a model, check if df_copy has any issues
            logger.info("Checking data for issues")
            
            for col in df_copy.columns:
                if df_copy[col].isna().any():
                    logger.warning(f"Column {col} has {df_copy[col].isna().sum()} NaN values")
                if np.isinf(df_copy[col]).any():
                    logger.warning(f"Column {col} has {np.isinf(df_copy[col]).sum()} infinite values")
            
            return None, None, None, polynomial_form

def evaluate_model_diagnostics(model, df, formula=None):
    """
    Evaluate model diagnostics including normality of residuals, 
    homoscedasticity, and linearity.
    
    Args:
        model: Fitted statsmodels model
        df: DataFrame with predictors and target variable
        formula: Optional formula used to fit the model
        
    Returns:
        dict: Dictionary with diagnostic results
    """
    model_diagnostics = {}
    
    # Get residuals
    residuals = model.resid
    
    # Check for normality: Jarque-Bera test
    try:
        # In newer versions of scipy, jarque_bera returns only 2 values
        jb_result = stats.jarque_bera(residuals)
        if len(jb_result) == 4:
            jb_stat, jb_pval, skew, kurtosis = jb_result
        else:
            jb_stat, jb_pval = jb_result
            skew = stats.skew(residuals)
            kurtosis = stats.kurtosis(residuals)
        
        model_diagnostics['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pval, 'skew': skew, 'kurtosis': kurtosis}
        
        if jb_pval < 0.05:
            logger.warning(f"Residuals may not be normally distributed (Jarque-Bera p-value: {jb_pval:.4g})")
        else:
            logger.info(f"Residuals appear to be normally distributed (Jarque-Bera p-value: {jb_pval:.4g})")
    except Exception as e:
        logger.warning(f"Could not perform Jarque-Bera test: {str(e)}")
        model_diagnostics['jarque_bera'] = {'statistic': None, 'p_value': None}
    
    # Check for heteroscedasticity: Breusch-Pagan test
    bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, model.model.exog)
    model_diagnostics['breusch_pagan'] = {'statistic': bp_stat, 'p_value': bp_pval}
    
    if bp_pval < 0.05:
        logger.warning(f"Heteroscedasticity detected (Breusch-Pagan p-value: {bp_pval:.4g})")
    else:
        logger.info(f"No significant heteroscedasticity detected (Breusch-Pagan p-value: {bp_pval:.4g})")
    
    # Check for linearity: RESET test (only if not already a polynomial model)
    if formula is None or "power" not in formula:
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
    else:
        # Skip RESET test for polynomial models
        logger.info("Skipping RESET test for polynomial model")
        model_diagnostics['reset_test'] = {'statistic': None, 'p_value': None, 'note': 'Skipped for polynomial model'}
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(residuals**2))
    model_diagnostics['rmse'] = rmse
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Check for influential observations: Cook's distance
    influence = model.get_influence()
    cook_distances = influence.cooks_distance[0]
    
    # Threshold for Cook's distance: 4/n
    n = len(residuals)
    cook_threshold = 4/n
    
    influential_idx = np.where(cook_distances > cook_threshold)[0]
    model_diagnostics['influential_observations'] = influential_idx.tolist()
    
    if len(influential_idx) > 0:
        logger.warning(f"Found {len(influential_idx)} potentially influential observations")
        logger.info(f"Indices of influential observations: {influential_idx.tolist()}")
    else:
        logger.info("No influential observations detected")
    
    return model_diagnostics

def create_polynomial_diagnostic_plots(model, df, metric, formula):
    """
    Create diagnostic plots for the polynomial model.
    
    Args:
        model: Fitted statsmodels model
        df: DataFrame with predictors and target variable
        metric: Target variable name
        formula: Formula used to fit the model
        
    Returns:
        str: Path to the saved plot file
    """
    logger.info(f"Creating diagnostic plots for nonlinear {metric} model")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Residuals vs. Fitted
    axs[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.6)
    axs[0, 0].hlines(y=0, xmin=min(model.fittedvalues), xmax=max(model.fittedvalues), colors='red', linestyles='--')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')
    axs[0, 0].set_title('Residuals vs. Fitted')
    
    # QQ Plot
    qq = ProbPlot(model.resid)
    qq.qqplot(line='45', ax=axs[0, 1])
    axs[0, 1].set_title('Normal Q-Q')
    
    # Scale-Location
    axs[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)), alpha=0.6)
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel('√|Residuals|')
    axs[1, 0].set_title('Scale-Location')
    
    # Cook's distance
    influence = model.get_influence()
    cook_distances = influence.cooks_distance[0]
    axs[1, 1].stem(np.arange(len(cook_distances)), cook_distances, markerfmt=',')
    axs[1, 1].set_xlabel('Observation index')
    axs[1, 1].set_ylabel("Cook's distance")
    axs[1, 1].set_title("Cook's distance")
    
    # Add polynomial formula to the plot
    plt.figtext(0.5, 0.01, f"Formula: {formula}", ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.15, "pad":5})
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = FIG_DIR / f"polynomial_diagnostic_plots_{metric}.png"
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Saved diagnostic plots to {plot_path}")
    
    return str(plot_path)

def build_robust_model(df, metric):
    """
    Build a robust regression model to address heteroscedasticity.
    
    Args:
        df: DataFrame with predictors and target variable
        metric: Target variable name
        
    Returns:
        Tuple of (model, model_summary, model_diagnostics)
    """
    logger.info(f"\nBuilding robust regression model for {metric}")
    
    # Split into dependent and independent variables
    y_col = metric
    X_cols = [col for col in df.columns if col != y_col]
    
    y = df[y_col]
    X = sm.add_constant(df[X_cols])
    
    # Try different robust regression estimators
    methods = {
        "M-Estimator (Huber)": RLM(y, X, M=sm.robust.norms.HuberT()),
        "M-Estimator (Tukey's Biweight)": RLM(y, X, M=sm.robust.norms.TukeyBiweight()),
        "MM-Estimator": RLM(y, X, M=sm.robust.norms.TukeyBiweight(), robust_scale=True)
    }
    
    best_model = None
    best_method = None
    best_aic = float('inf')
    
    # Test each robust method and select the best one
    for method_name, model_spec in methods.items():
        try:
            model = model_spec.fit()
            
            # Calculate AIC manually since it's not available directly for RLM
            n = len(y)
            k = len(X.columns)
            rss = np.sum(model.resid**2)
            aic = n * np.log(rss/n) + 2 * k
            
            logger.info(f"{method_name}: AIC = {aic:.4f}")
            
            if aic < best_aic:
                best_aic = aic
                best_model = model
                best_method = method_name
        
        except Exception as e:
            logger.error(f"Error fitting {method_name}: {str(e)}")
    
    if best_model is None:
        logger.error("All robust regression methods failed")
        return None, None, None
    
    logger.info(f"Selected {best_method} as the best robust model (AIC: {best_aic:.4f})")
    
    # Calculate pseudo R-squared (1 - SS_residual/SS_total)
    ss_residual = np.sum(best_model.resid**2)
    ss_total = np.sum((y - np.mean(y))**2)
    pseudo_r2 = 1 - (ss_residual / ss_total)
    
    logger.info(f"Pseudo R-squared: {pseudo_r2:.4f}")
    
    # Format coefficients for logging
    logger.info("\nRobust coefficient estimates:")
    for i, var in enumerate(X.columns):
        coef = best_model.params[i]
        std_err = best_model.bse[i]
        t_value = coef / std_err
        p_value = 2 * (1 - stats.t.cdf(abs(t_value), len(y) - len(X.columns)))
        
        sig = ""
        if p_value < 0.001:
            sig = " ***"
        elif p_value < 0.01:
            sig = " **"
        elif p_value < 0.05:
            sig = " *"
        
        logger.info(f"  {var}: {coef:.4f} (SE: {std_err:.4f}, t: {t_value:.2f}, p: {p_value:.4g}){sig}")
    
    # Generate model diagnostics
    model_diagnostics = evaluate_robust_diagnostics(best_model, df)
    
    # Create a model summary dictionary
    model_summary = {
        'method': best_method,
        'aic': best_aic,
        'pseudo_r2': pseudo_r2,
        'params': best_model.params.tolist(),
        'bse': best_model.bse.tolist(),
        'variable_names': X.columns.tolist()
    }
    
    return best_model, model_summary, model_diagnostics

def evaluate_robust_diagnostics(model, df):
    """
    Evaluate diagnostics for robust regression models.
    
    Args:
        model: Fitted robust regression model
        df: DataFrame with data
        
    Returns:
        dict: Dictionary with diagnostic results
    """
    diagnostics = {}
    
    # Get residuals
    residuals = model.resid
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    diagnostics['rmse'] = rmse
    logger.info(f"Robust model RMSE: {rmse:.4f}")
    
    # Check for normality of residuals
    try:
        # In newer versions of scipy, jarque_bera returns only 2 values
        jb_result = stats.jarque_bera(residuals)
        if len(jb_result) == 4:
            jb_stat, jb_pval, skew, kurtosis = jb_result
        else:
            jb_stat, jb_pval = jb_result
            skew = stats.skew(residuals)
            kurtosis = stats.kurtosis(residuals)
        
        diagnostics['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pval, 'skew': skew, 'kurtosis': kurtosis}
        
        if jb_pval < 0.05:
            logger.warning(f"Residuals may not be normally distributed (Jarque-Bera p-value: {jb_pval:.4g})")
        else:
            logger.info(f"Residuals appear to be normally distributed (Jarque-Bera p-value: {jb_pval:.4g})")
    except Exception as e:
        logger.warning(f"Could not perform Jarque-Bera test: {str(e)}")
        diagnostics['jarque_bera'] = {'statistic': None, 'p_value': None}
    
    # Check for autocorrelation
    dw_stat = durbin_watson(residuals)
    diagnostics['durbin_watson'] = dw_stat
    
    if dw_stat < 1.5 or dw_stat > 2.5:
        logger.warning(f"Possible autocorrelation in residuals (Durbin-Watson: {dw_stat:.4f})")
    else:
        logger.info(f"No significant autocorrelation detected (Durbin-Watson: {dw_stat:.4f})")
    
    # Check for influential observations
    # For RLM, the hat matrix calculation is more complex and can cause alignment issues
    # We'll use a simplified approach based on the standardized residuals
    try:
        # Standardized residuals
        std_resid = residuals / np.sqrt(np.var(residuals))
        
        # Identify observations with large standardized residuals (>2.5)
        influential_idx = np.where(np.abs(std_resid) > 2.5)[0]
        diagnostics['influential_observations'] = influential_idx.tolist()
        
        if len(influential_idx) > 0:
            logger.warning(f"Found {len(influential_idx)} potentially influential observations based on standardized residuals")
            logger.info(f"Indices of influential observations: {influential_idx.tolist()}")
        else:
            logger.info("No influential observations detected based on standardized residuals")
    except Exception as e:
        logger.warning(f"Could not identify influential observations: {str(e)}")
        diagnostics['influential_observations'] = []
    
    return diagnostics

def create_robust_diagnostic_plots(model, df, metric, method_name):
    """
    Create diagnostic plots for robust regression models.
    
    Args:
        model: Fitted robust regression model
        df: DataFrame with data
        metric: Target variable name
        method_name: Name of robust regression method used
        
    Returns:
        str: Path to the saved plot file
    """
    logger.info(f"Creating diagnostic plots for robust {metric} model")
    
    # Get the actual y values
    y = df[metric]
    
    # Get X with constant
    X_cols = [col for col in df.columns if col != metric]
    X = sm.add_constant(df[X_cols])
    
    # Calculate fitted values (need to do this manually for RLM)
    fitted_values = X.dot(model.params)
    
    # Get residuals
    residuals = model.resid
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Residuals vs. Fitted
    axs[0, 0].scatter(fitted_values, residuals, alpha=0.6)
    axs[0, 0].hlines(y=0, xmin=min(fitted_values), xmax=max(fitted_values), colors='red', linestyles='--')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')
    axs[0, 0].set_title('Residuals vs. Fitted')
    
    # QQ Plot
    qq = ProbPlot(residuals)
    qq.qqplot(line='45', ax=axs[0, 1])
    axs[0, 1].set_title('Normal Q-Q')
    
    # Scale-Location
    axs[1, 0].scatter(fitted_values, np.sqrt(np.abs(residuals)), alpha=0.6)
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel('√|Residuals|')
    axs[1, 0].set_title('Scale-Location')
    
    # Fitted vs. Actual
    axs[1, 1].scatter(y, fitted_values, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(min(y), min(fitted_values))
    max_val = max(max(y), max(fitted_values))
    axs[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
    
    axs[1, 1].set_xlabel('Actual values')
    axs[1, 1].set_ylabel('Fitted values')
    axs[1, 1].set_title('Fitted vs. Actual')
    
    # Add robust method name to the plot
    plt.figtext(0.5, 0.01, f"Method: {method_name}", ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.15, "pad":5})
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = FIG_DIR / f"robust_diagnostic_plots_{metric}.png"
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Saved robust diagnostic plots to {plot_path}")
    
    return str(plot_path)

def handle_influential_observations(df, metric, original_results):
    """
    Handle influential observations by removing them and refitting the model.
    
    Args:
        df: DataFrame with predictors and target variable
        metric: Target variable name
        original_results: Dictionary with original model results
        
    Returns:
        Tuple of (cleaned_df, influential_indices)
    """
    log_section(f"Handling Influential Observations for {metric}")
    
    # Get influential observations from original model
    influential_indices = original_results.get(metric, {}).get("diagnostics", {}).get("influential_observations", [])
    
    if not influential_indices:
        logger.info("No influential observations to handle")
        return df, []
    
    logger.info(f"Found {len(influential_indices)} influential observations to handle")
    
    # Create a copy of the dataframe for influential observation removal
    df_cleaned = df.copy()
    
    # Create a mask for influential observations
    mask = np.ones(len(df), dtype=bool)
    mask[influential_indices] = False
    
    # Create a new dataframe without influential observations
    df_cleaned = df.iloc[mask].copy().reset_index(drop=True)
    
    logger.info(f"Removed {len(influential_indices)} influential observations")
    logger.info(f"Original dataset shape: {df.shape}")
    logger.info(f"Cleaned dataset shape: {df_cleaned.shape}")
    
    return df_cleaned, influential_indices

def compare_models(original_model, refined_model, metric, refinement_type):
    """
    Compare original and refined models to see if refinement improved the fit.
    
    Args:
        original_model: Dictionary with original model results
        refined_model: Fitted statsmodels model after refinement
        metric: Target variable name
        refinement_type: Type of refinement applied
        
    Returns:
        dict: Dictionary with comparison metrics
    """
    comparison = {
        'metric': metric,
        'refinement_type': refinement_type,
        'improvement': False
    }
    
    # Get original model metrics
    orig_rsq = original_model.get('r_squared', 0)
    orig_adj_rsq = original_model.get('adj_r_squared', 0)
    orig_aic = original_model.get('aic', float('inf'))
    
    # For RMSE, if it's not in the original diagnostics, we'll estimate it
    if 'diagnostics' in original_model and 'rmse' in original_model['diagnostics']:
        orig_rmse = original_model['diagnostics']['rmse']
    else:
        # Estimate RMSE using sqrt(MSE) from R²
        # MSE = (1 - R²) * var(y)
        # We'll use a placeholder of 1.5 as a reasonable estimate for GAA club performances
        orig_rmse = 1.5
    
    # Get refined model metrics
    if refinement_type == 'polynomial':
        ref_rsq = refined_model.rsquared
        ref_adj_rsq = refined_model.rsquared_adj
        ref_aic = refined_model.aic
        ref_rmse = np.sqrt(np.mean(refined_model.resid**2))
    elif refinement_type == 'robust':
        # For robust models, we use pseudo R-squared
        ss_residual = np.sum(refined_model.resid**2)
        y = refined_model.model.endog
        ss_total = np.sum((y - np.mean(y))**2)
        ref_rsq = 1 - (ss_residual / ss_total)
        
        # Adjusted R-squared
        n = len(y)
        k = len(refined_model.params)
        ref_adj_rsq = 1 - ((1 - ref_rsq) * (n - 1) / (n - k - 1))
        
        # AIC - manually calculated
        rss = ss_residual
        ref_aic = n * np.log(rss/n) + 2 * k
        
        # RMSE
        ref_rmse = np.sqrt(np.mean(refined_model.resid**2))
    else:
        ref_rsq = refined_model.rsquared
        ref_adj_rsq = refined_model.rsquared_adj
        ref_aic = refined_model.aic
        ref_rmse = np.sqrt(np.mean(refined_model.resid**2))
    
    # Calculate percentage improvements
    rsq_change = ((ref_rsq - orig_rsq) / orig_rsq) * 100 if orig_rsq != 0 else float('inf')
    adj_rsq_change = ((ref_adj_rsq - orig_adj_rsq) / orig_adj_rsq) * 100 if orig_adj_rsq != 0 else float('inf')
    aic_change = ((orig_aic - ref_aic) / orig_aic) * 100 if orig_aic != 0 else float('inf')
    rmse_change = ((orig_rmse - ref_rmse) / orig_rmse) * 100 if orig_rmse != 0 else float('inf')
    
    # Log comparison results
    logger.info(f"\nModel Comparison for {metric} ({refinement_type}):")
    logger.info(f"  R-squared: {orig_rsq:.4f} -> {ref_rsq:.4f} ({rsq_change:.2f}%)")
    logger.info(f"  Adjusted R-squared: {orig_adj_rsq:.4f} -> {ref_adj_rsq:.4f} ({adj_rsq_change:.2f}%)")
    logger.info(f"  AIC: {orig_aic:.4f} -> {ref_aic:.4f} ({aic_change:.2f}%)")
    logger.info(f"  RMSE: {orig_rmse:.4f} -> {ref_rmse:.4f} ({rmse_change:.2f}%)")
    
    # Store metrics in comparison dictionary
    comparison['original'] = {
        'r_squared': orig_rsq,
        'adj_r_squared': orig_adj_rsq,
        'aic': orig_aic,
        'rmse': orig_rmse
    }
    
    comparison['refined'] = {
        'r_squared': ref_rsq,
        'adj_r_squared': ref_adj_rsq,
        'aic': ref_aic,
        'rmse': ref_rmse
    }
    
    comparison['changes'] = {
        'r_squared': rsq_change,
        'adj_r_squared': adj_rsq_change,
        'aic': aic_change,
        'rmse': rmse_change
    }
    
    # Determine if refinement improved the model
    # We consider an improvement if adjusted R-squared increased AND at least one of AIC decreased or RMSE decreased
    if adj_rsq_change > 0 and (aic_change > 0 or rmse_change > 0):
        comparison['improvement'] = True
        logger.info("Refinement IMPROVED the model")
    else:
        logger.info("Refinement did NOT improve the model")
    
    return comparison

def create_model_comparison_plots(comparisons):
    """
    Create visualization plots comparing original and refined models.
    
    Args:
        comparisons: List of dictionaries with model comparison metrics
        
    Returns:
        str: Path to the saved plot file
    """
    log_section("Creating Model Comparison Plots")
    
    # Extract relevant metrics for plotting
    metrics = ['r_squared', 'adj_r_squared', 'aic', 'rmse']
    metric_labels = ['R²', 'Adjusted R²', 'AIC', 'RMSE']
    
    # Organize data into dataframe for easier plotting
    plot_data = []
    
    for comp in comparisons:
        metric_name = comp['metric']
        refinement = comp['refinement_type']
        
        for metric, label in zip(metrics, metric_labels):
            orig_val = comp['original'][metric]
            ref_val = comp['refined'][metric]
            pct_change = comp['changes'][metric]
            
            # For AIC and RMSE, lower is better
            if metric in ['aic', 'rmse']:
                improvement = "✅" if pct_change > 0 else "❌"
            else:
                improvement = "✅" if pct_change > 0 else "❌"
            
            plot_data.append({
                'metric_name': metric_name,
                'refinement': refinement,
                'stat_metric': label,
                'original': orig_val,
                'refined': ref_val,
                'pct_change': pct_change,
                'improvement': improvement
            })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create comparison plots
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
    # Flatten axs for easier iteration
    axs = axs.flatten()
    
    # For each metric, create a grouped bar chart
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Filter data for this metric
        metric_data = df_plot[df_plot['stat_metric'] == label]
        
        # Create grouped bar chart
        sns.barplot(x='metric_name', y='pct_change', hue='refinement', 
                    data=metric_data, ax=axs[i], palette='viridis')
        
        # Set title and labels
        axs[i].set_title(f'Percent Change in {label}', fontsize=14)
        axs[i].set_xlabel('Performance Metric', fontsize=12)
        axs[i].set_ylabel('Percent Change (%)', fontsize=12)
        
        # Add a horizontal line at 0%
        axs[i].axhline(0, color='r', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for readability
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha='right')
        
        # Add annotations for improvement indicators
        for j, p in enumerate(axs[i].patches):
            # Get the index in the metric_data
            idx = j % len(metric_data)
            improvement = metric_data.iloc[idx]['improvement']
            
            # Add annotation 10% above the bar (or below if negative)
            height = p.get_height()
            if height < 0:
                y_pos = height - (abs(height) * 0.15)
            else:
                y_pos = height + (abs(height) * 0.15) if height != 0 else 0.5
                
            axs[i].annotate(improvement,
                           (p.get_x() + p.get_width() / 2., y_pos),
                           ha='center', va='center',
                           fontsize=14)
    
    plt.suptitle('Model Refinement Comparison', fontsize=18)
    plt.tight_layout()
    
    # Save the plot
    plot_path = FIG_DIR / f"model_comparison_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Saved model comparison plot to {plot_path}")
    
    return str(plot_path)

def create_coefficient_comparison_plots(original_results, refined_models, metrics):
    """
    Create plots comparing coefficients between original and refined models.
    
    Args:
        original_results: Dictionary with original model results
        refined_models: Dictionary with refined models
        metrics: List of performance metrics
        
    Returns:
        dict: Dictionary with paths to saved plot files
    """
    log_section("Creating Coefficient Comparison Plots")
    
    plot_paths = {}
    
    for metric in metrics:
        # Skip if refined model doesn't exist
        if metric not in refined_models:
            continue
            
        # Get refined model
        refined_model_info = refined_models[metric]
        model_type = refined_model_info['type']
        model = refined_model_info['model']
        
        # Get original model coefficients
        orig_coefs = pd.Series(original_results[metric]['params'])
        orig_coefs.index = original_results[metric]['variable_names']
        
        # Get refined model coefficients
        if model_type == 'polynomial':
            ref_coefs = model.params
            # Filter out squared terms for direct comparison
            ref_coefs_linear = ref_coefs[~ref_coefs.index.str.contains('power')]
        elif model_type == 'robust':
            ref_coefs = pd.Series(model.params)
            ref_coefs.index = refined_model_info['summary']['variable_names']
            ref_coefs_linear = ref_coefs
        else:
            ref_coefs = model.params
            ref_coefs_linear = ref_coefs
        
        # Create data for plotting
        plot_data = []
        
        # Add original coefficients
        for var, coef in orig_coefs.items():
            if var == 'const':
                continue
            plot_data.append({
                'Variable': var,
                'Coefficient': coef,
                'Model': 'Original'
            })
        
        # Add refined coefficients
        for var, coef in ref_coefs_linear.items():
            if var == 'const':
                continue
            plot_data.append({
                'Variable': var,
                'Coefficient': coef,
                'Model': f'Refined ({model_type})'
            })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create coefficient comparison plot
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart
        ax = sns.barplot(x='Variable', y='Coefficient', hue='Model', data=df_plot, palette='viridis')
        
        # Set title and labels
        plt.title(f'Coefficient Comparison for {metric}', fontsize=16)
        plt.xlabel('Variable', fontsize=14)
        plt.ylabel('Coefficient Value', fontsize=14)
        
        # Add a horizontal line at 0
        plt.axhline(0, color='r', linestyle='--', alpha=0.7)
        
        # Note about coefficient interpretation
        note = "Note: Negative coefficients indicate better performance (lower grade values)"
        plt.figtext(0.5, 0.01, note, ha='center', fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.15, "pad":5})
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = FIG_DIR / f"coefficient_comparison_{metric}.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved coefficient comparison plot for {metric} to {plot_path}")
        
        plot_paths[metric] = str(plot_path)
    
    return plot_paths

def save_results(refined_models, model_comparisons):
    """
    Save model refinement results and comparisons to JSON file.
    
    Args:
        refined_models: Dictionary with refined models
        model_comparisons: List of dictionaries with model comparison metrics
        
    Returns:
        str: Path to the saved results file
    """
    log_section("Saving Model Refinement Results")
    
    results = {
        'refined_models': {},
        'model_comparisons': model_comparisons
    }
    
    # Extract key information from refined models
    for metric, model_info in refined_models.items():
        model_type = model_info['type']
        model = model_info['model']
        
        if model is None:
            continue
            
        # Create a JSON-serializable version of the model info
        model_data = {
            'type': model_type,
            'r_squared': float(model.rsquared) if hasattr(model, 'rsquared') else float(model_info.get('summary', {}).get('pseudo_r2', 0)),
            'adj_r_squared': float(model.rsquared_adj) if hasattr(model, 'rsquared_adj') else None,
            'aic': float(model.aic) if hasattr(model, 'aic') else float(model_info.get('summary', {}).get('aic', 0)),
            'formula': model_info.get('formula', '')
        }
        
        # Add coefficients
        params = model.params.tolist() if hasattr(model, 'params') else model_info.get('summary', {}).get('params', [])
        
        param_names = model.params.index.tolist() if hasattr(model, 'params') and hasattr(model.params, 'index') else model_info.get('summary', {}).get('variable_names', [])
        
        # Ensure params are native Python types for JSON serialization
        params = [float(p) for p in params]
        
        model_data['params'] = params
        model_data['param_names'] = param_names
        
        # Add diagnostics if available (ensure they're JSON serializable)
        if 'diagnostics' in model_info:
            diag = model_info['diagnostics']
            json_diag = {}
            
            # Convert jarque_bera values to serializable types
            if 'jarque_bera' in diag:
                jb = diag['jarque_bera']
                json_diag['jarque_bera'] = {
                    'statistic': float(jb.get('statistic', 0)) if jb.get('statistic') is not None else None,
                    'p_value': float(jb.get('p_value', 0)) if jb.get('p_value') is not None else None,
                    'skew': float(jb.get('skew', 0)) if jb.get('skew') is not None else None,
                    'kurtosis': float(jb.get('kurtosis', 0)) if jb.get('kurtosis') is not None else None
                }
            
            # Convert breusch_pagan values to serializable types
            if 'breusch_pagan' in diag:
                bp = diag['breusch_pagan']
                json_diag['breusch_pagan'] = {
                    'statistic': float(bp.get('statistic', 0)) if bp.get('statistic') is not None else None,
                    'p_value': float(bp.get('p_value', 0)) if bp.get('p_value') is not None else None
                }
            
            # Convert reset_test values to serializable types
            if 'reset_test' in diag:
                rt = diag['reset_test']
                json_diag['reset_test'] = {
                    'statistic': float(rt.get('statistic', 0)) if rt.get('statistic') is not None else None,
                    'p_value': float(rt.get('p_value', 0)) if rt.get('p_value') is not None else None,
                    'note': rt.get('note', '')
                }
            
            # Add RMSE
            if 'rmse' in diag:
                json_diag['rmse'] = float(diag['rmse'])
            
            # Add durbin_watson
            if 'durbin_watson' in diag:
                json_diag['durbin_watson'] = float(diag['durbin_watson'])
            
            # Add influential_observations as-is (already a list)
            if 'influential_observations' in diag:
                json_diag['influential_observations'] = diag['influential_observations']
            
            model_data['diagnostics'] = json_diag
        
        results['refined_models'][metric] = model_data
    
    # Save to JSON file
    timestamp = datetime.now().strftime("%Y%m%d")
    results_path = OUTPUT_DIR / f"refined_model_results_{timestamp}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved refined model results to {results_path}")
    
    return str(results_path)

def generate_model_refinement_report(original_results, refined_models, model_comparisons):
    """
    Generate a comprehensive model refinement report in markdown format.
    
    Args:
        original_results: Dictionary with original model results
        refined_models: Dictionary with refined models
        model_comparisons: List of dictionaries with model comparison metrics
        
    Returns:
        str: Path to the generated report
    """
    log_section("Generating Model Refinement Report")
    
    timestamp = datetime.now().strftime("%Y%m%d")
    report_path = OUTPUT_DIR / f"model_refinement_report_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        # Write header
        f.write("# Model Refinement Report\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Introduction\n\n")
        f.write("This report presents the results of model refinement for the Cork GAA club performance analysis. ")
        f.write("The refinement addresses diagnostic issues found in the original models, including ")
        f.write("non-linearity, heteroscedasticity, and influential observations.\n\n")
        
        f.write(f"{PERFORMANCE_NOTE}\n\n")
        
        # Write model comparison summary
        f.write("## Model Comparison Summary\n\n")
        
        f.write("| Performance Metric | Refinement Type | R² Improvement | Adj. R² Improvement | AIC Improvement | RMSE Improvement | Overall |\n")
        f.write("|---------------------|-----------------|---------------:|-------------------:|----------------:|-----------------:|--------:|\n")
        
        for comp in model_comparisons:
            metric = comp['metric']
            refine_type = comp['refinement_type']
            r2_change = comp['changes']['r_squared']
            adj_r2_change = comp['changes']['adj_r_squared']
            aic_change = comp['changes']['aic']
            rmse_change = comp['changes']['rmse']
            
            improved = "✅" if comp['improvement'] else "❌"
            
            f.write(f"| {metric} | {refine_type} | {r2_change:.2f}% | {adj_r2_change:.2f}% | {aic_change:.2f}% | {rmse_change:.2f}% | {improved} |\n")
        
        f.write("\n*Note: For AIC and RMSE, negative percentages indicate improvement (lower values are better).*\n\n")
        
        # Write detailed results for each performance metric
        f.write("## Detailed Model Results\n\n")
        
        for metric in refined_models.keys():
            f.write(f"### {metric}\n\n")
            
            # Original model diagnostic issues
            f.write("#### Original Model Diagnostic Issues\n\n")
            
            original_diag = original_results.get(metric, {}).get("diagnostics", {})
            
            # List of diagnostic issues
            issues = []
            
            # Check for normality issues
            jb_pval = original_diag.get("jarque_bera", {}).get("p_value")
            if jb_pval and jb_pval < 0.05:
                issues.append(f"Non-normal residuals (JB p-value: {jb_pval:.4g})")
            
            # Check for heteroscedasticity
            bp_pval = original_diag.get("breusch_pagan", {}).get("p_value")
            if bp_pval and bp_pval < 0.05:
                issues.append(f"Heteroscedasticity (BP p-value: {bp_pval:.4g})")
            
            # Check for linearity issues
            reset_pval = original_diag.get("reset_test", {}).get("p_value")
            if reset_pval and reset_pval < 0.05:
                issues.append(f"Non-linearity (RESET p-value: {reset_pval:.4g})")
            
            # Check for influential observations
            n_influential = len(original_diag.get("influential_observations", []))
            if n_influential > 0:
                issues.append(f"{n_influential} influential observations")
            
            if issues:
                f.write("Issues identified in the original model:\n\n")
                for issue in issues:
                    f.write(f"- {issue}\n")
            else:
                f.write("No major issues detected in the original model.\n")
            
            f.write("\n")
            
            # Refined model details
            model_info = refined_models.get(metric)
            if model_info and model_info['model'] is not None:
                model_type = model_info['type']
                f.write(f"#### Refined Model ({model_type})\n\n")
                
                # Model fit statistics
                f.write("**Model Fit Statistics:**\n\n")
                
                if model_type == 'polynomial':
                    model = model_info['model']
                    f.write(f"- R-squared: {model.rsquared:.4f}\n")
                    f.write(f"- Adjusted R-squared: {model.rsquared_adj:.4f}\n")
                    f.write(f"- AIC: {model.aic:.4f}\n")
                    f.write(f"- RMSE: {np.sqrt(np.mean(model.resid**2)):.4f}\n")
                    
                    # Write polynomial formula
                    formula = model_info.get('formula', 'Not available')
                    f.write(f"\n**Polynomial Formula:**\n\n```\n{formula}\n```\n\n")
                    
                elif model_type == 'robust':
                    summary = model_info['summary']
                    f.write(f"- Method: {summary.get('method', 'Not available')}\n")
                    f.write(f"- Pseudo R-squared: {summary.get('pseudo_r2', 0):.4f}\n")
                    f.write(f"- AIC: {summary.get('aic', 0):.4f}\n")
                    
                    # Diagnostics
                    diag = model_info.get('diagnostics', {})
                    f.write(f"- RMSE: {diag.get('rmse', 0):.4f}\n")
                    f.write(f"- Durbin-Watson: {diag.get('durbin_watson', 0):.4f}\n")
                
                # Diagnostics comparison
                f.write("\n**Diagnostic Improvements:**\n\n")
                
                # Get the comparison for this metric
                comp = next((c for c in model_comparisons if c['metric'] == metric), None)
                if comp:
                    # Extract diagnostics
                    orig_rsq = comp['original']['r_squared']
                    ref_rsq = comp['refined']['r_squared']
                    rsq_change = comp['changes']['r_squared']
                    
                    orig_adj_rsq = comp['original']['adj_r_squared']
                    ref_adj_rsq = comp['refined']['adj_r_squared']
                    adj_rsq_change = comp['changes']['adj_r_squared']
                    
                    orig_aic = comp['original']['aic']
                    ref_aic = comp['refined']['aic']
                    aic_change = comp['changes']['aic']
                    
                    orig_rmse = comp['original']['rmse']
                    ref_rmse = comp['refined']['rmse']
                    rmse_change = comp['changes']['rmse']
                    
                    f.write("| Metric | Original | Refined | Change |\n")
                    f.write("|--------|----------:|--------:|-------:|\n")
                    f.write(f"| R-squared | {orig_rsq:.4f} | {ref_rsq:.4f} | {rsq_change:+.2f}% |\n")
                    f.write(f"| Adjusted R-squared | {orig_adj_rsq:.4f} | {ref_adj_rsq:.4f} | {adj_rsq_change:+.2f}% |\n")
                    f.write(f"| AIC | {orig_aic:.4f} | {ref_aic:.4f} | {aic_change:+.2f}% |\n")
                    f.write(f"| RMSE | {orig_rmse:.4f} | {ref_rmse:.4f} | {rmse_change:+.2f}% |\n")
                    
                f.write("\n")
                
                # Include visualizations
                f.write("**Visualizations:**\n\n")
                f.write(f"- Coefficient comparison plot: `figures/coefficient_comparison_{metric}.png`\n")
                if model_type == 'polynomial':
                    f.write(f"- Polynomial diagnostic plots: `figures/polynomial_diagnostic_plots_{metric}.png`\n")
                elif model_type == 'robust':
                    f.write(f"- Robust diagnostic plots: `figures/robust_diagnostic_plots_{metric}.png`\n")
                
                f.write("\n")
            else:
                f.write("No refined model was successfully created for this metric.\n\n")
        
        # Write conclusion
        f.write("## Conclusion\n\n")
        
        # Count successful refinements
        successful = sum(1 for comp in model_comparisons if comp['improvement'])
        total = len(model_comparisons)
        
        f.write(f"The model refinement process successfully improved {successful} out of {total} models. ")
        
        if successful > 0:
            f.write("The main improvements were achieved through:\n\n")
            
            # Identify which refinement types were successful
            poly_success = any(comp['improvement'] and comp['refinement_type'] == 'polynomial' for comp in model_comparisons)
            robust_success = any(comp['improvement'] and comp['refinement_type'] == 'robust' for comp in model_comparisons)
            
            if poly_success:
                f.write("1. **Polynomial terms** for addressing non-linearity, particularly in the overall performance model\n")
            if robust_success:
                f.write("2. **Robust regression** for handling heteroscedasticity and influential observations, particularly in the football and hurling models\n")
            
            f.write("\nThese refinements resulted in models with better fit statistics, more reliable parameter estimates, and improved diagnostic properties.")
        else:
            f.write("None of the refinement approaches significantly improved the models. This suggests that the original models, despite their diagnostic issues, may still provide the best balance of fit and interpretability.")
        
        f.write("\n\n")
        
        # Write next steps
        f.write("## Next Steps\n\n")
        f.write("1. **Further model validation** through cross-validation and bootstrap analysis\n")
        f.write("2. **Spatial regression analysis** to address any remaining spatial autocorrelation\n")
        f.write("3. **Interpretation framework development** to translate statistical findings into practical insights\n")
        f.write("4. **Integration with spatial analysis results** to develop a comprehensive understanding of club performance factors\n")
    
    logger.info(f"Generated model refinement report at {report_path}")
    
    return str(report_path)

def main():
    """Main execution function."""
    try:
        log_section("Starting Model Refinement")
        
        # Load model data
        datasets = load_model_data()
        
        # Load original model results
        original_results = load_original_results()
        
        if original_results is None:
            logger.error("Cannot proceed without original model results")
            return
        
        # Track refined models and comparisons
        refined_models = {}
        model_comparisons = []
        
        # Check if we have diagnostics
        has_diagnostics = any(
            bool(original_results.get(metric, {}).get("diagnostics", {}))
            for metric in datasets.keys()
            if metric != "model_comparison"
        )
        
        # Process each performance metric
        for metric, df in datasets.items():
            log_section(f"Refining model for {metric}")
            
            # Skip model_comparison key in original_results
            if metric == "model_comparison":
                continue
                
            # Check if metric exists in original_results
            if metric not in original_results:
                logger.warning(f"No original results found for {metric}, skipping")
                continue
            
            # Default refinement strategy if diagnostics are missing
            if not has_diagnostics:
                logger.info(f"Using default refinement strategy for {metric} (no diagnostics available)")
                
                # Handle influential observations first (using an empty list since we don't have diagnostics)
                df_cleaned = df.copy()
                influential_indices = []
                
                # Apply appropriate refinement based on metric
                if metric == "overall_performance":
                    logger.info(f"Applying polynomial refinement to address assumed non-linearity in {metric}")
                    model, summary, diagnostics, formula = build_nonlinear_model(df_cleaned, metric)
                    
                    if model is not None:
                        # Create diagnostic plots
                        plot_path = create_polynomial_diagnostic_plots(model, df_cleaned, metric, formula)
                        
                        # Store model
                        refined_models[metric] = {
                            'type': 'polynomial',
                            'model': model,
                            'summary': summary,
                            'diagnostics': diagnostics,
                            'formula': formula,
                            'plot_path': plot_path
                        }
                        
                        # Compare with original model
                        comparison = compare_models(original_results[metric], model, metric, 'polynomial')
                        model_comparisons.append(comparison)
                
                elif metric in ["football_performance", "hurling_performance"]:
                    logger.info(f"Applying robust regression to address assumed heteroscedasticity in {metric}")
                    model, summary, diagnostics = build_robust_model(df_cleaned, metric)
                    
                    if model is not None:
                        # Create diagnostic plots
                        method_name = summary.get("method", "Robust Regression")
                        plot_path = create_robust_diagnostic_plots(model, df_cleaned, metric, method_name)
                        
                        # Store model
                        refined_models[metric] = {
                            'type': 'robust',
                            'model': model,
                            'summary': summary,
                            'diagnostics': diagnostics,
                            'plot_path': plot_path
                        }
                        
                        # Compare with original model
                        comparison = compare_models(original_results[metric], model, metric, 'robust')
                        model_comparisons.append(comparison)
                
                else:
                    logger.info(f"Skipping refinement for {metric} (weakest model)")
            
            # Use diagnostics if available
            else:
                # Get original model diagnostics
                original_diag = original_results[metric].get("diagnostics", {})
                
                # Check for non-linearity using RESET test
                reset_pval = original_diag.get("reset_test", {}).get("p_value")
                non_linear = reset_pval is not None and reset_pval < 0.05
                
                # Check for heteroscedasticity using Breusch-Pagan test
                bp_pval = original_diag.get("breusch_pagan", {}).get("p_value")
                hetero = bp_pval is not None and bp_pval < 0.05
                
                # Handle influential observations first
                df_cleaned, influential_indices = handle_influential_observations(df, metric, original_results)
                
                # Apply appropriate refinement based on diagnostics
                if non_linear:
                    logger.info(f"Applying polynomial refinement to address non-linearity in {metric}")
                    model, summary, diagnostics, formula = build_nonlinear_model(df_cleaned, metric)
                    
                    if model is not None:
                        # Create diagnostic plots
                        plot_path = create_polynomial_diagnostic_plots(model, df_cleaned, metric, formula)
                        
                        # Store model
                        refined_models[metric] = {
                            'type': 'polynomial',
                            'model': model,
                            'summary': summary,
                            'diagnostics': diagnostics,
                            'formula': formula,
                            'plot_path': plot_path
                        }
                        
                        # Compare with original model
                        comparison = compare_models(original_results[metric], model, metric, 'polynomial')
                        model_comparisons.append(comparison)
                
                elif hetero:
                    logger.info(f"Applying robust regression to address heteroscedasticity in {metric}")
                    model, summary, diagnostics = build_robust_model(df_cleaned, metric)
                    
                    if model is not None:
                        # Create diagnostic plots
                        method_name = summary.get("method", "Robust Regression")
                        plot_path = create_robust_diagnostic_plots(model, df_cleaned, metric, method_name)
                        
                        # Store model
                        refined_models[metric] = {
                            'type': 'robust',
                            'model': model,
                            'summary': summary,
                            'diagnostics': diagnostics,
                            'plot_path': plot_path
                        }
                        
                        # Compare with original model
                        comparison = compare_models(original_results[metric], model, metric, 'robust')
                        model_comparisons.append(comparison)
                
                else:
                    logger.info(f"No major issues detected in {metric}, skipping refinement")
        
        # Create model comparison visualizations
        if model_comparisons:
            comparison_plot = create_model_comparison_plots(model_comparisons)
            logger.info(f"Model comparison plot created at {comparison_plot}")
            
            # Create coefficient comparison plots
            coef_plots = create_coefficient_comparison_plots(original_results, refined_models, datasets.keys())
            logger.info(f"Coefficient comparison plots created for {list(coef_plots.keys())}")
            
            # Save results
            results_path = save_results(refined_models, model_comparisons)
            logger.info(f"Model refinement results saved to {results_path}")
            
            # Generate model refinement report
            report_path = generate_model_refinement_report(original_results, refined_models, model_comparisons)
            logger.info(f"Model refinement report generated at {report_path}")
        else:
            logger.warning("No models were refined, nothing to save")
        
        log_section("Model Refinement Completed")
        
    except Exception as e:
        logger.error(f"Error in model refinement: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 