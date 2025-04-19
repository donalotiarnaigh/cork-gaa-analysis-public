#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Geographically Weighted Regression (GWR) Analysis

This script implements an improved GWR analysis approach with proper spatial autocorrelation
assessment, robust bandwidth selection, and meaningful evaluation. It addresses the limitations
of the original implementation by following best practices in spatial statistics.

Author: Daniel Tierney
Date: April 2025
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path
import sys
import traceback
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import KFold

# For GWR modeling
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from spglm.family import Gaussian
import libpysal
from libpysal.weights import Kernel, W
from esda.moran import Moran
from splot.esda import moran_scatterplot

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output" / "improved_gwr_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"

# Create output directories if they don't exist
for dir_path in [OUTPUT_DIR, FIGURE_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"improved_gwr_analysis_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger("improved_gwr_analysis")
logger.info("Improved GWR Analysis Environment Setup Complete")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Log file: {log_file}")

# Set Matplotlib and Seaborn styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def log_section(section_name):
    """Log a section header to improve log readability."""
    logger.info(f"\n{'=' * 40}\n{section_name}\n{'=' * 40}")

def load_data():
    """
    Load club and spatial data for GWR analysis.
    
    Returns:
        tuple: (club_gdf, performance_metrics) - GeoPandas DataFrame and list of performance metrics
    """
    log_section("Loading Data")
    
    # Try to load data from multiple potential locations
    potential_data_paths = [
        PROCESSED_DIR / "cork_clubs_complete_graded.csv",
        PROCESSED_DIR / "cork_clubs_graded.csv",
        OUTPUT_DIR / "modeling" / "model_data_overall_performance.csv"
    ]
    
    club_data = None
    for path in potential_data_paths:
        if path.exists():
            logger.info(f"Loading club data from {path}")
            try:
                club_data = pd.read_csv(path)
                logger.info(f"Successfully loaded {len(club_data)} clubs")
                break
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
    
    if club_data is None:
        logger.error("Could not load club data from any known location")
        return None, None
    
    # Try to load spatial data
    potential_spatial_paths = [
        PROCESSED_DIR / "cork_clubs_complete.gpkg",
        PROCESSED_DIR / "cork_clubs.gpkg",
        PROCESSED_DIR / "clubs_spatial.gpkg",
        BASE_DIR / "output" / "geopackage" / "final_analysis.gpkg"
    ]
    
    club_spatial = None
    for path in potential_spatial_paths:
        if path.exists():
            logger.info(f"Loading spatial data from {path}")
            try:
                club_spatial = gpd.read_file(path)
                logger.info(f"Successfully loaded {len(club_spatial)} spatial features")
                break
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
    
    if club_spatial is None:
        logger.error("Could not load club spatial data from any known location")
        return None, None
    
    # Identify performance metrics in the dataset
    potential_metrics = ['overall_performance', 'football_performance', 'hurling_performance', 'code_balance']
    available_metrics = [metric for metric in potential_metrics if metric in club_data.columns]
    
    if not available_metrics:
        logger.error("No performance metrics found in the dataset")
        return None, None
    
    logger.info(f"Available performance metrics: {available_metrics}")
    
    # Merge tabular and spatial data
    # First determine the common key for joining
    if 'club_name' in club_data.columns and 'club_name' in club_spatial.columns:
        join_key = 'club_name'
    elif 'name' in club_data.columns and 'name' in club_spatial.columns:
        join_key = 'name'
    elif 'Club' in club_data.columns and 'Club' in club_spatial.columns:
        join_key = 'Club'
    else:
        logger.warning("No common key found for joining. Will attempt index-based join.")
        join_key = None
    
    try:
        # Check if both datasets have the same columns and the same club names
        club_data_set = set(club_data['Club'].tolist())
        club_spatial_set = set(club_spatial['Club'].tolist())
        
        if club_data_set == club_spatial_set:
            # If the datasets have the same clubs, just copy the spatial data
            logger.info("Datasets contain the same clubs. Using spatial data as base.")
            club_gdf = club_spatial.copy()
        elif join_key:
            logger.info(f"Joining data on {join_key}")
            club_gdf = club_spatial.merge(club_data, on=join_key, how="inner")
        else:
            # Fall back to index-based join if data sizes match
            if len(club_data) == len(club_spatial):
                logger.info("Performing index-based join")
                club_gdf = club_spatial.copy()
                for col in club_data.columns:
                    if col not in club_gdf.columns:
                        club_gdf[col] = club_data[col].values
            else:
                logger.error("Cannot join data: no common key and size mismatch")
                return None, None
        
        logger.info(f"Final joined GeoDataFrame has {len(club_gdf)} clubs")
    except Exception as e:
        logger.error(f"Error joining data: {e}")
        logger.error(traceback.format_exc())
        return None, None
    
    # Ensure the CRS is set to Irish Transverse Mercator (EPSG:2157)
    if club_gdf.crs is None or club_gdf.crs.to_string() != 'EPSG:2157':
        logger.info(f"Converting CRS from {club_gdf.crs} to EPSG:2157")
        club_gdf = club_gdf.to_crs("EPSG:2157")
    
    # Save the integrated data
    club_gdf.to_file(OUTPUT_DIR / "club_data_for_gwr.gpkg", driver="GPKG")
    logger.info(f"Saved integrated club data to {OUTPUT_DIR / 'club_data_for_gwr.gpkg'}")
    
    return club_gdf, available_metrics

def select_variables(club_gdf, performance_metric):
    """
    Select variables for the GWR model with careful handling of multicollinearity.
    
    Args:
        club_gdf: GeoDataFrame with club data
        performance_metric: The performance metric to analyze
        
    Returns:
        tuple: (X, y, selected_vars) - Independent variables, dependent variable, and variable names
    """
    log_section(f"Selecting Variables for {performance_metric} Model")
    
    # Determine environmental and socioeconomic variables available in the dataset
    all_columns = club_gdf.columns.tolist()
    
    # Define variable categories - check what's actually in the data
    environmental_vars = [col for col in all_columns if col in [
        'Elevation', 'annual_rainfall', 'rain_days', 'Population_Density',
        'Latitude', 'Longitude'
    ]]
    
    # Try to load socioeconomic data from another source since it's not in the main file
    try:
        socio_path = PROCESSED_DIR / "cork_clubs_demographics.csv"
        if socio_path.exists():
            socio_df = pd.read_csv(socio_path)
            # Merge with club_gdf
            if 'Club' in socio_df.columns and 'Club' in club_gdf.columns:
                logger.info(f"Merging socioeconomic data from {socio_path}")
                for col in socio_df.columns:
                    if col != 'Club' and col not in club_gdf.columns:
                        club_gdf[col] = socio_df.set_index('Club').reindex(club_gdf['Club']).reset_index()[col]
        else:
            logger.warning(f"Socioeconomic data not found at {socio_path}")
    except Exception as e:
        logger.warning(f"Could not load socioeconomic data: {e}")
    
    # Update columns list after potential merge
    all_columns = club_gdf.columns.tolist()
    
    # Now check for socioeconomic variables
    socioeconomic_vars = [col for col in all_columns if col in [
        'third_level_rate', 'secondary_education_rate', 'basic_education_rate',
        'employment_rate', 'unemployment_rate', 'professional_rate', 'working_class_rate',
        'youth_proportion', 'school_age_rate', 'urban'
    ]]
    
    # If no socioeconomic variables, use environmental only
    if not socioeconomic_vars and not environmental_vars:
        logger.warning("No standard variables found. Using coordinates for basic model.")
        # Add coordinates if not already there
        if 'x_coord' not in club_gdf.columns:
            club_gdf['x_coord'] = club_gdf.geometry.centroid.x
            club_gdf['y_coord'] = club_gdf.geometry.centroid.y
            environmental_vars = ['x_coord', 'y_coord']
    
    # Log the available variables
    logger.info(f"Available environmental variables: {environmental_vars}")
    logger.info(f"Available socioeconomic variables: {socioeconomic_vars}")
    
    # Start with all potential predictor variables
    potential_vars = environmental_vars + socioeconomic_vars
    
    # Ensure the performance metric is available
    if performance_metric not in club_gdf.columns:
        logger.error(f"Performance metric '{performance_metric}' not found in dataset")
        return None, None, None
    
    # Extract the dependent variable
    y = club_gdf[performance_metric].values
    
    # Check for missing values in the dependent variable
    if np.isnan(y).any():
        logger.warning(f"Found {np.isnan(y).sum()} missing values in {performance_metric}")
        
        # Determine how to handle missing values based on proportion
        missing_proportion = np.isnan(y).mean()
        
        if missing_proportion > 0.2:
            logger.error(f"Too many missing values in {performance_metric} ({missing_proportion:.1%})")
            return None, None, None
        else:
            # Impute with mean for moderate missing values
            logger.info(f"Imputing {np.isnan(y).sum()} missing values with mean")
            y_mean = np.nanmean(y)
            y = np.where(np.isnan(y), y_mean, y)
    
    # Ensure all potential predictors are numeric
    numeric_vars = []
    for var in potential_vars:
        if var in club_gdf.columns:
            try:
                club_gdf[var] = pd.to_numeric(club_gdf[var], errors='coerce')
                
                # Check for missing values
                missing_count = club_gdf[var].isna().sum()
                if missing_count > 0:
                    missing_pct = missing_count / len(club_gdf)
                    if missing_pct > 0.2:
                        logger.warning(f"Excluding {var}: {missing_count} missing values ({missing_pct:.1%})")
                        continue
                    else:
                        logger.info(f"Imputing {missing_count} missing values for {var}")
                        club_gdf[var].fillna(club_gdf[var].mean(), inplace=True)
                
                numeric_vars.append(var)
            except:
                logger.warning(f"Could not convert {var} to numeric, excluding from analysis")
    
    # Check if we have enough variables
    if len(numeric_vars) < 2:
        logger.error(f"Not enough numeric variables available (found {len(numeric_vars)})")
        return None, None, None
    
    logger.info(f"Initial numeric variables: {numeric_vars}")
    
    # Check for multicollinearity using VIF
    X_initial = club_gdf[numeric_vars].values
    
    # Handle potential problems in the matrix
    if np.isnan(X_initial).any():
        logger.warning("Found NaN values in X matrix, filling with column means")
        for i in range(X_initial.shape[1]):
            col_mean = np.nanmean(X_initial[:, i])
            X_initial[:, i] = np.where(np.isnan(X_initial[:, i]), col_mean, X_initial[:, i])
    
    # Calculate VIF for each variable
    try:
        # Add a constant term for VIF calculation
        X_with_const = sm.add_constant(X_initial)
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = numeric_vars
        vif_data["VIF"] = [variance_inflation_factor(X_with_const, i+1) for i in range(len(numeric_vars))]
        
        logger.info("Initial VIF values:")
        for _, row in vif_data.iterrows():
            logger.info(f"  {row['Variable']}: {row['VIF']:.2f}")
        
        # Handle high multicollinearity
        vif_threshold = 10
        high_vif_vars = vif_data[vif_data["VIF"] > vif_threshold]["Variable"].tolist()
        
        if high_vif_vars:
            logger.warning(f"High multicollinearity detected in: {high_vif_vars}")
            
            # Start with variables below the threshold
            selected_vars = vif_data[vif_data["VIF"] <= vif_threshold]["Variable"].tolist()
            
            # If no variables below threshold, select those with lowest VIF up to a reasonable number
            if not selected_vars:
                logger.warning("All variables have high VIF. Selecting those with lowest VIF.")
                vif_data = vif_data.sort_values("VIF")
                selected_vars = vif_data["Variable"].head(min(5, len(vif_data))).tolist()
        else:
            selected_vars = numeric_vars
    except Exception as e:
        logger.warning(f"Error in VIF calculation: {e}")
        logger.warning(traceback.format_exc())
        
        # If VIF calculation fails, select a small subset of variables with low correlation
        logger.info("Using correlation-based variable selection as fallback")
        
        corr_matrix = club_gdf[numeric_vars].corr().abs()
        
        # Identify variables with correlation > 0.7
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.7:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    high_corr.append((col_i, col_j, corr_matrix.iloc[i, j]))
        
        # Log high correlations
        if high_corr:
            logger.info("High correlations found:")
            for col_i, col_j, corr in high_corr:
                logger.info(f"  {col_i} - {col_j}: {corr:.2f}")
            
            # Keep variables with lower mean correlation with others
            mean_corr = corr_matrix.mean(axis=1)
            mean_corr = mean_corr.sort_values()
            selected_vars = mean_corr.index[:min(5, len(mean_corr))].tolist()
        else:
            selected_vars = numeric_vars
    
    # Add coordinates for GWR if not already included
    if 'x_coord' not in selected_vars and 'y_coord' not in selected_vars:
        # Calculate and add centroid coordinates
        club_gdf['x_coord'] = club_gdf.geometry.centroid.x
        club_gdf['y_coord'] = club_gdf.geometry.centroid.y
        
        # Add to selected variables
        selected_vars.extend(['x_coord', 'y_coord'])
    
    logger.info(f"Final selected variables: {selected_vars}")
    
    # Create final X matrix
    X = club_gdf[selected_vars].values
    
    # Return the predictor matrix, dependent variable, and variable names
    return X, y, selected_vars

def assess_spatial_autocorrelation(club_gdf, performance_metric, residuals=None):
    """
    Assess spatial autocorrelation in the dependent variable or residuals
    to determine if GWR is appropriate.
    
    Args:
        club_gdf: GeoDataFrame with club data
        performance_metric: The performance metric to analyze
        residuals: Residuals from a global model (if available)
        
    Returns:
        tuple: (moran_result, spatial_weights) - Moran's I result and spatial weights
    """
    log_section("Assessing Spatial Autocorrelation")
    
    # Calculate centroids for distance-based weights
    if 'centroid' not in club_gdf.columns:
        club_gdf['centroid'] = club_gdf.geometry.centroid
    
    # Extract coordinates from centroids
    coords = np.array([(p.x, p.y) for p in club_gdf.centroid])
    
    # Create a distance matrix
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Calculate Euclidean distance in meters
            dist_matrix[i, j] = np.sqrt(
                (coords[i, 0] - coords[j, 0])**2 + 
                (coords[i, 1] - coords[j, 1])**2
            )
    
    # Calculate distance statistics
    mean_dist = np.mean(dist_matrix[dist_matrix > 0])
    median_dist = np.median(dist_matrix[dist_matrix > 0])
    max_dist = np.max(dist_matrix)
    
    logger.info(f"Distance statistics:")
    logger.info(f"  Mean distance: {mean_dist:.2f} meters")
    logger.info(f"  Median distance: {median_dist:.2f} meters")
    logger.info(f"  Maximum distance: {max_dist:.2f} meters")
    
    # Create spatial weights with different approaches
    # 1. Kernel weights based on median distance
    kernel_bandwidth = median_dist
    logger.info(f"Creating kernel weights with bandwidth {kernel_bandwidth:.2f}")
    
    try:
        # Create PySAL weights object
        kernel_w = Kernel(coords, fixed=True, function='gaussian', bandwidth=kernel_bandwidth)
        logger.info(f"Created kernel weights with {kernel_w.n} observations")
    except Exception as e:
        logger.warning(f"Error creating kernel weights: {e}")
        logger.warning(traceback.format_exc())
        
        # Fall back to manual weight creation
        logger.info("Falling back to manual weight creation")
        
        # Create a spatial weights matrix with Gaussian kernel
        weight_matrix = np.exp(-0.5 * (dist_matrix / kernel_bandwidth)**2)
        np.fill_diagonal(weight_matrix, 0)  # Set self-weights to 0
        
        # Row-standardize weights
        for i in range(n):
            if np.sum(weight_matrix[i]) > 0:
                weight_matrix[i] = weight_matrix[i] / np.sum(weight_matrix[i])
        
        # Convert to PySAL W object
        neighbors = {i: [j for j in range(n) if weight_matrix[i, j] > 0] for i in range(n)}
        weights = {i: [weight_matrix[i, j] for j in neighbors[i]] for i in range(n)}
        
        kernel_w = W(neighbors, weights)
        logger.info(f"Created manual weights with {kernel_w.n} observations")
    
    # Calculate Moran's I on the performance metric
    try:
        # Determine what to test: original variable or residuals
        if residuals is not None:
            logger.info("Testing for spatial autocorrelation in residuals")
            test_values = residuals
        else:
            logger.info(f"Testing for spatial autocorrelation in {performance_metric}")
            test_values = club_gdf[performance_metric].values
            
            # Handle missing values if any
            if np.isnan(test_values).any():
                logger.warning(f"Found {np.isnan(test_values).sum()} missing values, imputing with mean")
                mean_val = np.nanmean(test_values)
                test_values = np.where(np.isnan(test_values), mean_val, test_values)
        
        # Calculate Moran's I
        moran = Moran(test_values, kernel_w)
        
        logger.info(f"Moran's I results:")
        logger.info(f"  I value: {moran.I:.4f}")
        logger.info(f"  Expected I: {moran.EI:.4f}")
        logger.info(f"  p-value: {moran.p_sim:.4f}")
        logger.info(f"  z-score: {moran.z_sim:.4f}")
        
        # Create Moran scatter plot
        fig, ax = plt.subplots(figsize=(10, 10))
        moran_scatterplot(moran, ax=ax)
        
        if residuals is not None:
            ax.set_title(f"Moran's I for OLS Residuals: {moran.I:.4f} (p={moran.p_sim:.4f})")
        else:
            ax.set_title(f"Moran's I for {performance_metric}: {moran.I:.4f} (p={moran.p_sim:.4f})")
        
        # Save figure
        plt.tight_layout()
        filename = f"morans_scatterplot_{'residuals' if residuals is not None else performance_metric}.png"
        plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Moran's I scatter plot to {FIGURE_DIR / filename}")
        
        # Interpretation
        if moran.p_sim < 0.05:
            if residuals is not None:
                logger.info("Significant spatial autocorrelation detected in residuals - GWR may be appropriate")
            else:
                logger.info("Significant spatial autocorrelation detected in performance variable")
        else:
            if residuals is not None:
                logger.warning("No significant spatial autocorrelation in residuals - GWR may not offer improvement")
            else:
                logger.info("No significant spatial autocorrelation in performance variable")
        
        return moran, kernel_w
        
    except Exception as e:
        logger.error(f"Error calculating Moran's I: {e}")
        logger.error(traceback.format_exc())
        return None, kernel_w

def select_bandwidth_cross_validation(coords, y, X, fixed=True, num_folds=5):
    """
    Select the optimal bandwidth for GWR using cross-validation.
    
    Args:
        coords: Coordinate array of shape (n, 2)
        y: Dependent variable array
        X: Independent variables array
        fixed: Whether to use fixed (True) or adaptive (False) bandwidth
        num_folds: Number of cross-validation folds
        
    Returns:
        float: Optimal bandwidth value
    """
    log_section("Selecting Bandwidth with Cross-Validation")
    
    # Create K-fold cross-validation splits
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Define a range of potential bandwidths
    if fixed:
        # For fixed bandwidth, use distance-based ranges from the distance matrix we already calculated
        # Calculate pairwise distances between all points
        dist_matrix = np.zeros((len(coords), len(coords)))
        for i in range(len(coords)):
            for j in range(len(coords)):
                dist_matrix[i, j] = np.sqrt(
                    (coords[i, 0] - coords[j, 0])**2 + 
                    (coords[i, 1] - coords[j, 1])**2
                )
        
        # Get max distance
        max_dist = np.max(dist_matrix)
        bandwidths = np.linspace(max_dist * 0.05, max_dist * 0.5, 10)
        logger.info(f"Testing fixed bandwidths from {bandwidths[0]:.2f} to {bandwidths[-1]:.2f}")
    else:
        # For adaptive bandwidth, use proportion of nearest neighbors
        n = len(coords)
        min_neighbors = max(10, int(n * 0.1))
        max_neighbors = min(n - 1, int(n * 0.5))
        bandwidths = np.linspace(min_neighbors, max_neighbors, 10).astype(int)
        logger.info(f"Testing adaptive bandwidths from {bandwidths[0]} to {bandwidths[-1]} neighbors")
    
    # Store cross-validation results
    cv_scores = []
    
    # Perform cross-validation for each bandwidth
    for bw in bandwidths:
        fold_scores = []
        
        for train_idx, test_idx in kf.split(X):
            # Split data into training and testing sets
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            coords_train, coords_test = coords[train_idx], coords[test_idx]
            
            try:
                # Fit GWR model with current bandwidth
                gwr_model = GWR(coords_train, y_train, X_train, bw=bw, fixed=fixed, kernel='gaussian')
                gwr_results = gwr_model.fit()
                
                # Predict on test set
                predictions = gwr_model.predict(coords_test, X_test, gwr_results.params)
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((y_test - predictions)**2))
                fold_scores.append(rmse)
                
            except Exception as e:
                logger.warning(f"Error with bandwidth {bw}: {e}")
                fold_scores.append(np.inf)
                continue
        
        # Average RMSE across folds
        mean_rmse = np.mean(fold_scores)
        cv_scores.append(mean_rmse)
        logger.info(f"Bandwidth {bw}: Mean RMSE = {mean_rmse:.4f}")
    
    # Find optimal bandwidth (minimum RMSE)
    valid_scores = np.isfinite(cv_scores)
    if not any(valid_scores):
        logger.warning("No valid cross-validation scores. Using default bandwidth.")
        if fixed:
            optimal_bw = np.median(bandwidths)
        else:
            optimal_bw = int(len(coords) * 0.3)  # 30% of observations
    else:
        optimal_idx = np.argmin(np.array(cv_scores)[valid_scores])
        optimal_bw = bandwidths[np.where(valid_scores)[0][optimal_idx]]
    
    logger.info(f"Optimal bandwidth selected: {optimal_bw}")
    
    # Create a plot of CV results
    plt.figure(figsize=(10, 6))
    plt.plot(bandwidths, cv_scores, 'o-')
    plt.axvline(x=optimal_bw, color='red', linestyle='--', label=f'Optimal: {optimal_bw}')
    plt.xlabel('Bandwidth' + (' (neighbors)' if not fixed else ' (meters)'))
    plt.ylabel('Cross-Validation RMSE')
    plt.title('Bandwidth Selection via Cross-Validation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    filename = f"bandwidth_cv_{'fixed' if fixed else 'adaptive'}.png"
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved bandwidth selection plot to {FIGURE_DIR / filename}")
    
    return optimal_bw

def run_ols_model(X, y, variable_names):
    """
    Run an OLS model for comparison with GWR.
    
    Args:
        X: Independent variables array
        y: Dependent variable array
        variable_names: List of variable names
        
    Returns:
        tuple: (model, residuals) - OLS model and residuals
    """
    log_section("Running OLS Model")
    
    # Add a constant to X
    X_const = sm.add_constant(X)
    
    # Column names for the output
    column_names = ['const'] + variable_names
    
    # Fit OLS model
    try:
        ols_model = sm.OLS(y, X_const).fit()
        
        # Log OLS results
        logger.info(f"OLS model results:")
        logger.info(f"  R²: {ols_model.rsquared:.4f}")
        logger.info(f"  Adjusted R²: {ols_model.rsquared_adj:.4f}")
        logger.info(f"  F-statistic: {ols_model.fvalue:.4f} (p={ols_model.f_pvalue:.4g})")
        
        # Log coefficients
        logger.info("Coefficient estimates:")
        for name, coef, pval in zip(column_names, ols_model.params, ols_model.pvalues):
            sig = ""
            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
            elif pval < 0.1:
                sig = "."
            
            logger.info(f"  {name}: {coef:.4f} (p={pval:.4g}) {sig}")
        
        # Get residuals for spatial autocorrelation test
        residuals = ols_model.resid
        
        return ols_model, residuals
    
    except Exception as e:
        logger.error(f"Error fitting OLS model: {e}")
        logger.error(traceback.format_exc())
        return None, None

def run_gwr_model(coords, y, X, variable_names, bandwidth, fixed=True):
    """
    Run a GWR model with the provided bandwidth.
    
    Args:
        coords: Coordinate array of shape (n, 2)
        y: Dependent variable array
        X: Independent variables array
        variable_names: List of variable names
        bandwidth: Bandwidth parameter for GWR
        fixed: Whether to use fixed (True) or adaptive (False) bandwidth
        
    Returns:
        object: Fitted GWR model results
    """
    log_section("Running GWR Model")
    
    logger.info(f"Running GWR with {'fixed' if fixed else 'adaptive'} bandwidth: {bandwidth}")
    
    try:
        # Add constant to X if needed for GWR
        X_for_gwr = sm.add_constant(X) if 'const' not in variable_names else X
        variables_with_const = ['const'] + variable_names if 'const' not in variable_names else variable_names
        
        # Create GWR model
        gwr_model = GWR(coords, y, X_for_gwr, bw=bandwidth, fixed=fixed, 
                        kernel='gaussian', family=Gaussian())
        
        # Fit the model
        gwr_results = gwr_model.fit()
        
        # Log model summary
        logger.info(f"GWR model results:")
        logger.info(f"  Bandwidth: {gwr_results.bw} ({'fixed' if gwr_results.fixed else 'adaptive'})")
        logger.info(f"  AICc: {gwr_results.aicc:.4f}")
        logger.info(f"  R²: {gwr_results.R2:.4f}")
        logger.info(f"  Adjusted R²: {gwr_results.adj_R2:.4f}")
        
        # Calculate local coefficients summary
        logger.info("Local coefficients summary:")
        for i, var in enumerate(variables_with_const):
            min_val = np.min(gwr_results.params[:, i])
            max_val = np.max(gwr_results.params[:, i])
            mean_val = np.mean(gwr_results.params[:, i])
            std_val = np.std(gwr_results.params[:, i])
            
            logger.info(f"  {var}:")
            logger.info(f"    Range: [{min_val:.4f}, {max_val:.4f}]")
            logger.info(f"    Mean: {mean_val:.4f}, Std Dev: {std_val:.4f}")
        
        return gwr_results
    
    except Exception as e:
        logger.error(f"Error fitting GWR model: {e}")
        logger.error(traceback.format_exc())
        return None

def visualize_gwr_results(club_gdf, gwr_results, variable_names, performance_metric, ols_model=None):
    """
    Visualize the GWR model results including spatial variation in local parameters
    and model fit.
    
    Args:
        club_gdf: GeoDataFrame with club data
        gwr_results: Fitted GWR model results
        variable_names: List of variable names
        performance_metric: The performance metric being analyzed
        ols_model: OLS model for comparison (optional)
    """
    log_section("Visualizing GWR Results")
    
    try:
        # Create a copy of the GeoDataFrame for visualization
        viz_gdf = club_gdf.copy()
        
        # Add local R² values
        viz_gdf['local_r2'] = gwr_results.localR2
        
        # Add coefficient values for each variable
        variables_with_const = ['const'] + variable_names if hasattr(gwr_results, 'params') else variable_names
        
        for i, var in enumerate(variables_with_const):
            column_name = f'coef_{var}'
            viz_gdf[column_name] = gwr_results.params[:, i]
        
        # Create local R² map
        fig, ax = plt.subplots(figsize=(12, 10))
        
        viz_gdf.plot(
            column='local_r2',
            cmap='viridis',
            legend=True,
            ax=ax,
            edgecolor='gray',
            legend_kwds={'label': 'Local R²'},
            alpha=0.7
        )
        
        ax.set_title(f'Local R² Values for {performance_metric} GWR Model', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        filename = f"local_r2_map_{performance_metric}.png"
        plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved local R² map to {FIGURE_DIR / filename}")
        
        # Create coefficient maps for each variable
        for var in variable_names:
            column_name = f'coef_{var}'
            
            # Create diverging color map centered at zero
            vmin = min(0, viz_gdf[column_name].min())
            vmax = max(0, viz_gdf[column_name].max())
            vabs = max(abs(vmin), abs(vmax))
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            viz_gdf.plot(
                column=column_name,
                cmap='RdBu_r',
                legend=True,
                ax=ax,
                edgecolor='gray',
                legend_kwds={'label': f'Coefficient: {var}'},
                vmin=-vabs,
                vmax=vabs,
                alpha=0.7
            )
            
            ax.set_title(f'Local Coefficients for {var}', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.tight_layout()
            filename = f"coefficient_map_{var}_{performance_metric}.png"
            plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved coefficient map for {var} to {FIGURE_DIR / filename}")
        
        # Create a parameter summary table
        param_summary = pd.DataFrame(index=variables_with_const)
        param_summary['Mean'] = [np.mean(gwr_results.params[:, i]) for i in range(gwr_results.params.shape[1])]
        param_summary['Std Dev'] = [np.std(gwr_results.params[:, i]) for i in range(gwr_results.params.shape[1])]
        param_summary['Min'] = [np.min(gwr_results.params[:, i]) for i in range(gwr_results.params.shape[1])]
        param_summary['Max'] = [np.max(gwr_results.params[:, i]) for i in range(gwr_results.params.shape[1])]
        
        # Add OLS coefficients for comparison if available
        if ols_model is not None:
            param_summary['OLS Coefficient'] = ols_model.params
            param_summary['OLS p-value'] = ols_model.pvalues
        
        # Save the parameter summary
        param_summary.to_csv(OUTPUT_DIR / f"parameter_summary_{performance_metric}.csv")
        logger.info(f"Saved parameter summary to {OUTPUT_DIR / f'parameter_summary_{performance_metric}.csv'}")
        
        # Create coefficient comparison plot (GWR vs OLS)
        if ols_model is not None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for i, var in enumerate(variables_with_const):
                # Box plot for GWR coefficients
                box_pos = i + 0.3
                ax.boxplot(gwr_results.params[:, i], positions=[box_pos], widths=0.3,
                          showfliers=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', color='blue'),
                          medianprops=dict(color='navy'))
                
                # Point for OLS coefficient
                ols_pos = i - 0.3
                if var in ols_model.params:
                    ols_coef = ols_model.params[var]
                    ax.plot(ols_pos, ols_coef, 'ro', ms=10)
            
            # Custom legend
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            legend_elements = [
                Patch(facecolor='lightblue', edgecolor='blue', label='GWR Coefficients'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='OLS Coefficient')
            ]
            ax.legend(handles=legend_elements, loc='best')
            
            # Customize plot
            ax.set_xticks(range(len(variables_with_const)))
            ax.set_xticklabels(variables_with_const, rotation=45, ha='right')
            ax.set_ylabel('Coefficient Value')
            ax.set_title(f'Coefficient Comparison: GWR vs OLS for {performance_metric}')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            filename = f"coefficient_comparison_{performance_metric}.png"
            plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved coefficient comparison plot to {FIGURE_DIR / filename}")
        
        # Save the complete visualization data
        viz_gdf.to_file(OUTPUT_DIR / f"gwr_visualization_{performance_metric}.gpkg", driver='GPKG')
        logger.info(f"Saved visualization data to {OUTPUT_DIR / f'gwr_visualization_{performance_metric}.gpkg'}")
        
    except Exception as e:
        logger.error(f"Error visualizing GWR results: {e}")
        logger.error(traceback.format_exc())

def assess_gwr_performance(gwr_results, ols_model, y, performance_metric):
    """
    Assess the performance of the GWR model compared to OLS.
    
    Args:
        gwr_results: Fitted GWR model results
        ols_model: OLS model for comparison
        y: Dependent variable array
        performance_metric: The performance metric being analyzed
        
    Returns:
        dict: Dictionary with performance assessment results
    """
    log_section("Assessing GWR Performance")
    
    # Create a results dictionary
    assessment = {
        'metric': performance_metric,
        'gwr_r2': gwr_results.R2,
        'gwr_adj_r2': gwr_results.adj_R2,
        'gwr_aicc': gwr_results.aicc,
        'ols_r2': ols_model.rsquared,
        'ols_adj_r2': ols_model.rsquared_adj,
        'ols_aicc': ols_model.aic,
        'improvement_r2': gwr_results.R2 - ols_model.rsquared,
        'improvement_adj_r2': gwr_results.adj_R2 - ols_model.rsquared_adj,
        'bandwidth': gwr_results.bw,
        'fixed_bandwidth': gwr_results.fixed
    }
    
    # Log the assessment
    logger.info(f"GWR Model Assessment for {performance_metric}:")
    logger.info(f"  GWR R²: {assessment['gwr_r2']:.4f}, OLS R²: {assessment['ols_r2']:.4f}")
    logger.info(f"  R² Improvement: {assessment['improvement_r2']:.4f} ({assessment['improvement_r2']*100:.2f}%)")
    logger.info(f"  GWR Adj. R²: {assessment['gwr_adj_r2']:.4f}, OLS Adj. R²: {assessment['ols_adj_r2']:.4f}")
    logger.info(f"  Adj. R² Improvement: {assessment['improvement_adj_r2']:.4f} ({assessment['improvement_adj_r2']*100:.2f}%)")
    
    # Interpret the results
    if assessment['improvement_adj_r2'] > 0.05:
        logger.info("Assessment: GWR provides substantial improvement over OLS")
    elif assessment['improvement_adj_r2'] > 0.02:
        logger.info("Assessment: GWR provides moderate improvement over OLS")
    else:
        logger.info("Assessment: GWR provides minimal improvement over OLS")
    
    # Save the assessment to a CSV file
    assessment_df = pd.DataFrame([assessment])
    assessment_df.to_csv(OUTPUT_DIR / f"gwr_assessment_{performance_metric}.csv", index=False)
    logger.info(f"Saved GWR assessment to {OUTPUT_DIR / f'gwr_assessment_{performance_metric}.csv'}")
    
    return assessment

def run_gwr_analysis(performance_metric):
    """
    Run the complete GWR analysis pipeline for a specific performance metric.
    
    Args:
        performance_metric: The performance metric to analyze
        
    Returns:
        dict: Dictionary with analysis results
    """
    log_section(f"Complete GWR Analysis for {performance_metric}")
    
    # 1. Load data
    club_gdf, _ = load_data()
    if club_gdf is None:
        logger.error("Could not load data, aborting analysis")
        return None
    
    # 2. Select variables with multicollinearity handling
    X, y, selected_vars = select_variables(club_gdf, performance_metric)
    if X is None or y is None or not selected_vars:
        logger.error("Variable selection failed, aborting analysis")
        return None
    
    # 3. Run OLS model for comparison
    ols_model, residuals = run_ols_model(X, y, selected_vars)
    if ols_model is None:
        logger.error("OLS model failed, aborting analysis")
        return None
    
    # 4. Assess spatial autocorrelation in OLS residuals
    centroids = np.array([(p.x, p.y) for p in club_gdf.geometry.centroid])
    moran_result, weights = assess_spatial_autocorrelation(club_gdf, performance_metric, residuals)
    
    # Determine if GWR is warranted
    if moran_result is not None and moran_result.p_sim >= 0.05:
        logger.warning("No significant spatial autocorrelation in residuals. GWR may not be necessary.")
        logger.warning("Proceeding with GWR for comparison purposes.")
    
    # 5. Select optimal bandwidth using cross-validation
    try:
        bandwidth = select_bandwidth_cross_validation(centroids, y, X, fixed=True)
    except Exception as e:
        logger.error(f"Bandwidth selection failed: {e}")
        logger.error(traceback.format_exc())
        
        # Calculate distance matrix manually
        n = len(centroids)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.sqrt(
                    (centroids[i, 0] - centroids[j, 0])**2 + 
                    (centroids[i, 1] - centroids[j, 1])**2
                )
        
        # Fall back to simpler bandwidth selection
        logger.info("Using median distance as fallback bandwidth")
        bandwidth = np.median(dist_matrix[dist_matrix > 0])
    
    # 6. Run GWR model
    gwr_results = run_gwr_model(centroids, y, X, selected_vars, bandwidth, fixed=True)
    if gwr_results is None:
        logger.error("GWR model failed, aborting analysis")
        return None
    
    # 7. Visualize GWR results
    visualize_gwr_results(club_gdf, gwr_results, selected_vars, performance_metric, ols_model)
    
    # 8. Assess GWR performance compared to OLS
    assessment = assess_gwr_performance(gwr_results, ols_model, y, performance_metric)
    
    # 9. Return results
    results = {
        'metric': performance_metric,
        'gwr_results': gwr_results,
        'ols_model': ols_model,
        'selected_vars': selected_vars,
        'bandwidth': bandwidth,
        'moran_result': moran_result,
        'assessment': assessment
    }
    
    return results

def generate_summary_report(all_results):
    """
    Generate a summary report of all GWR analyses.
    
    Args:
        all_results: Dictionary with results for each performance metric
    """
    log_section("Generating Summary Report")
    
    # Create summary dataframe
    summary_rows = []
    
    for metric, results in all_results.items():
        if results and 'assessment' in results:
            summary_rows.append({
                'Metric': metric,
                'OLS R²': results['assessment']['ols_r2'],
                'GWR R²': results['assessment']['gwr_r2'],
                'R² Improvement': results['assessment']['improvement_r2'],
                'OLS Adj. R²': results['assessment']['ols_adj_r2'],
                'GWR Adj. R²': results['assessment']['gwr_adj_r2'],
                'Adj. R² Improvement': results['assessment']['improvement_adj_r2'],
                'Bandwidth': results['assessment']['bandwidth'],
                'Fixed Bandwidth': results['assessment']['fixed_bandwidth'],
                'Spatial Autocorrelation': results['moran_result'].p_sim < 0.05 if results['moran_result'] else 'Not tested'
            })
    
    if not summary_rows:
        logger.error("No valid results to include in summary report")
        return
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary table
    summary_df.to_csv(OUTPUT_DIR / "gwr_analysis_summary.csv", index=False)
    logger.info(f"Saved summary table to {OUTPUT_DIR / 'gwr_analysis_summary.csv'}")
    
    # Create summary markdown report
    with open(OUTPUT_DIR / "gwr_analysis_report.md", "w") as f:
        f.write("# Geographically Weighted Regression Analysis Report\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Summary of Results\n\n")
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")
        
        f.write("## Interpretation\n\n")
        
        # Add interpretation based on R² improvements
        max_improvement = summary_df['Adj. R² Improvement'].max()
        max_metric = summary_df.loc[summary_df['Adj. R² Improvement'].idxmax(), 'Metric']
        
        if max_improvement > 0.05:
            f.write(f"GWR provides substantial improvement over OLS for {max_metric} (Adj. R² improvement: {max_improvement:.4f}).\n\n")
            f.write("This indicates significant spatial heterogeneity in the relationships between predictors and performance metrics.\n\n")
        elif max_improvement > 0.02:
            f.write(f"GWR provides moderate improvement over OLS for {max_metric} (Adj. R² improvement: {max_improvement:.4f}).\n\n")
            f.write("This suggests some spatial heterogeneity in the relationships, but the improvement may not justify the additional complexity.\n\n")
        else:
            f.write(f"GWR provides minimal improvement over OLS across all metrics (max Adj. R² improvement: {max_improvement:.4f} for {max_metric}).\n\n")
            f.write("This suggests that relationships between predictors and performance metrics are relatively consistent across space.\n\n")
        
        f.write("## GWR Analysis Value Assessment\n\n")
        
        # Add value assessment
        if max_improvement > 0.05:
            f.write("**Recommendation**: Consider including GWR analysis in the final project as it provides valuable insights into spatial variations in relationships.\n\n")
        elif max_improvement > 0.02:
            f.write("**Recommendation**: GWR analysis could be included as a supplementary analysis, but its value should be carefully evaluated against its complexity.\n\n")
        else:
            f.write("**Recommendation**: The standard OLS regression is likely sufficient for this analysis. GWR does not provide substantial additional insights to justify its inclusion in the final project.\n\n")
        
        f.write("## Technical Details\n\n")
        
        f.write("### Spatial Autocorrelation\n\n")
        
        # Add spatial autocorrelation information
        autocorr_metrics = [row['Metric'] for row in summary_rows if row['Spatial Autocorrelation'] is True]
        if autocorr_metrics:
            f.write(f"Significant spatial autocorrelation was detected in the residuals for the following metrics: {', '.join(autocorr_metrics)}.\n\n")
        else:
            f.write("No significant spatial autocorrelation was detected in the residuals for any of the performance metrics.\n\n")
        
        f.write("### Bandwidth Selection\n\n")
        f.write("Bandwidth parameters were selected using cross-validation to minimize prediction error.\n\n")
        
        # Add bandwidth information
        for row in summary_rows:
            f.write(f"- {row['Metric']}: {'Fixed' if row['Fixed Bandwidth'] else 'Adaptive'} bandwidth of {row['Bandwidth']}\n")
        
        f.write("\n### Variables Used\n\n")
        
        # Add variable information for each metric
        for metric, results in all_results.items():
            if results and 'selected_vars' in results:
                f.write(f"#### {metric}\n\n")
                f.write(f"Variables included in the model: {', '.join(results['selected_vars'])}\n\n")
                
                # Add OLS coefficient information if available
                if 'ols_model' in results and results['ols_model'] is not None:
                    f.write("OLS Coefficients:\n\n")
                    coef_data = []
                    for var, coef, pval in zip(['const'] + results['selected_vars'], 
                                              results['ols_model'].params, 
                                              results['ols_model'].pvalues):
                        sig = ""
                        if pval < 0.001:
                            sig = "***"
                        elif pval < 0.01:
                            sig = "**"
                        elif pval < 0.05:
                            sig = "*"
                        elif pval < 0.1:
                            sig = "."
                        
                        coef_data.append({
                            'Variable': var,
                            'Coefficient': coef,
                            'p-value': pval,
                            'Significance': sig
                        })
                    
                    coef_df = pd.DataFrame(coef_data)
                    f.write(coef_df.to_markdown(index=False, floatfmt=".4f"))
                    f.write("\n\n")
        
        f.write("## Conclusion\n\n")
        
        # Add overall conclusion
        if max_improvement > 0.05:
            f.write("The GWR analysis reveals important spatial variations in the relationships between demographic/environmental factors and GAA club performance. ")
            f.write("This spatial heterogeneity suggests that the importance of different factors varies across Cork, which could have implications for targeted development strategies.\n\n")
        elif max_improvement > 0.02:
            f.write("The GWR analysis shows some evidence of spatial variations in relationships, but these variations are modest. ")
            f.write("While there may be some local differences in how factors influence club performance, the global relationships captured by OLS regression provide a reasonable approximation.\n\n")
        else:
            f.write("The GWR analysis indicates that relationships between demographic/environmental factors and GAA club performance are relatively consistent across Cork. ")
            f.write("The simpler OLS regression models capture these relationships adequately, and the additional complexity of GWR does not yield substantial improvements in explanatory power.\n\n")
    
    logger.info(f"Generated summary report at {OUTPUT_DIR / 'gwr_analysis_report.md'}")

def main():
    """Main function to run the improved GWR analysis."""
    log_section("Starting Improved GWR Analysis")
    
    try:
        # Load data and get available metrics
        club_gdf, available_metrics = load_data()
        
        if club_gdf is None or not available_metrics:
            logger.error("Data loading failed. Exiting.")
            return
        
        # Store results for all metrics
        all_results = {}
        
        # Run analysis for each performance metric
        for metric in available_metrics:
            logger.info(f"Running analysis for {metric}")
            results = run_gwr_analysis(metric)
            all_results[metric] = results
        
        # Generate summary report
        generate_summary_report(all_results)
        
        log_section("Improved GWR Analysis Completed")
        logger.info(f"All analyses completed successfully. Results saved to {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()