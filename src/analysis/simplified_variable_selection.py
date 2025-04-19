#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Variable Selection and Model Preparation

This script creates final predictor variable sets based on composite variables,
documents the selection methodology, and prepares model-ready datasets for each
performance metric (football, hurling, overall).

Author: Daniel Tierney
Date: April 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mticker
from datetime import datetime
import sys

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
OUTPUT_DIR = BASE_DIR / "output" / "modeling"
REPORTS_DIR = BASE_DIR / "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Define performance metrics
PERFORMANCE_METRICS = [
    "overall_performance",
    "football_performance",
    "hurling_performance",
    "code_balance"
]

# Define PCA components from variables_composites.py
PCA_COMPONENTS = [
    "sociodemographic_pc1",
    "sociodemographic_pc2",
    "sociodemographic_pc3"
]

# Define composite variables
COMPOSITE_INDICES = [
    "education_advantage_index",
    "education_diversity_index",
    "socioeconomic_advantage_index",
    "youth_potential_index"
]

# Define interaction terms
INTERACTION_TERMS = [
    "elevation_x_third_level_rate",
    "elevation_x_professional_rate",
    "elevation_x_unemployment_rate",
    "elevation_x_working_class_rate"
]

# Define geographic variables
GEOGRAPHIC_VARS = [
    "elevation",
    "annual_rainfall",
    "urban",
    "rural_distance"
]

def log_section(section_name):
    """Log a section header to improve log readability."""
    logger.info(f"\n{'=' * 40}\n{section_name}\n{'=' * 40}")

def load_data():
    """
    Load data with composite variables.
    
    Returns:
        pd.DataFrame: Data with composite variables
    """
    log_section("Loading Data")
    
    # First check if the file with composites exists
    composites_path = DATA_DIR / "cork_clubs_with_composites.csv"
    if not composites_path.exists():
        logger.error(f"File not found: {composites_path}")
        raise FileNotFoundError(f"Could not find composites file at {composites_path}")
    
    # Load data with composites
    df = pd.read_csv(composites_path)
    logger.info(f"Loaded data with {df.shape[0]} clubs and {df.shape[1]} variables")
    
    # Check for required columns
    required_metrics = [metric for metric in PERFORMANCE_METRICS if metric in df.columns]
    if not required_metrics:
        logger.error("No performance metrics found in data")
        raise ValueError("No performance metrics found in data")
    
    logger.info(f"Found {len(required_metrics)} performance metrics: {', '.join(required_metrics)}")
    
    # Check for composite variables
    found_composites = []
    for var_list, var_type in zip(
        [PCA_COMPONENTS, COMPOSITE_INDICES, INTERACTION_TERMS, GEOGRAPHIC_VARS],
        ["PCA components", "composite indices", "interaction terms", "geographic variables"]
    ):
        found = [var for var in var_list if var in df.columns]
        found_composites.extend(found)
        logger.info(f"Found {len(found)} {var_type}: {', '.join(found)}")
    
    if not found_composites:
        logger.error("No composite variables found in data")
        raise ValueError("No composite variables found in data")
    
    return df

def calculate_vif(df, variables):
    """
    Calculate Variance Inflation Factor for a set of variables.
    
    Args:
        df: DataFrame containing the variables
        variables: List of variable names
        
    Returns:
        pd.DataFrame: DataFrame with VIF values for each variable
    """
    log_section("Calculating VIF")
    
    # Create a copy of the data with only the specified variables
    X = df[variables].copy()
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    # Add a constant column for statsmodels
    X['const'] = 1
    
    # Calculate VIF for each variable
    vif_data = []
    for i, var in enumerate(variables):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({
                'Variable': var,
                'VIF': vif
            })
            logger.info(f"VIF for {var}: {vif:.2f}")
        except Exception as e:
            logger.warning(f"Error calculating VIF for {var}: {str(e)}")
    
    # Create DataFrame
    vif_df = pd.DataFrame(vif_data)
    
    # Identify variables with high VIF
    high_vif = vif_df[vif_df['VIF'] > 5]
    
    if not high_vif.empty:
        logger.warning(f"Found {len(high_vif)} variables with VIF > 5:")
        for _, row in high_vif.iterrows():
            logger.warning(f"  - {row['Variable']}: VIF = {row['VIF']:.2f}")
    else:
        logger.info("No variables with VIF > 5 found")
    
    return vif_df

def create_correlation_matrix(df, variables, performance_metrics):
    """
    Calculate correlation matrix between variables and performance metrics.
    
    Args:
        df: DataFrame containing the variables and metrics
        variables: List of predictor variables
        performance_metrics: List of performance metrics
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    log_section("Calculating Correlation Matrix")
    
    # Extract variables and metrics
    correlation_vars = variables + performance_metrics
    
    # Calculate correlation matrix
    corr_matrix = df[correlation_vars].corr()
    
    # Log correlations with performance metrics
    for metric in performance_metrics:
        logger.info(f"\nCorrelations with {metric}:")
        correlations = corr_matrix[metric].drop(performance_metrics).sort_values(ascending=False)
        for var, corr in correlations.items():
            logger.info(f"  - {var}: {corr:.3f}")
    
    return corr_matrix

def create_core_variable_set(df):
    """
    Create the core variable set using PCA components, composite indices,
    geographic variables, and key interaction terms.
    
    Args:
        df: DataFrame containing all variables
        
    Returns:
        list: Core variable set
    """
    log_section("Creating Core Variable Set")
    
    # Start with PCA components
    core_variables = [var for var in PCA_COMPONENTS if var in df.columns]
    logger.info(f"Added {len(core_variables)} PCA components to core variable set")
    
    # Add geographic variables
    geo_vars = [var for var in GEOGRAPHIC_VARS if var in df.columns]
    core_variables.extend(geo_vars)
    logger.info(f"Added {len(geo_vars)} geographic variables to core variable set")
    
    # Add key interaction terms
    key_interactions = [var for var in INTERACTION_TERMS if var in df.columns]
    core_variables.extend(key_interactions)
    logger.info(f"Added {len(key_interactions)} interaction terms to core variable set")
    
    # Add composite indices if not already represented by PCA
    if core_variables:
        # Calculate VIF to check for multicollinearity
        vif_results = calculate_vif(df, core_variables)
        
        # If we have variables with high VIF, try to reduce
        high_vif = vif_results[vif_results['VIF'] > 5]
        if not high_vif.empty:
            logger.warning(f"Core variable set has {len(high_vif)} variables with VIF > 5")
            logger.info("Attempting to reduce multicollinearity by removing high VIF variables")
            
            # Sort by VIF in descending order
            high_vif = high_vif.sort_values('VIF', ascending=False)
            
            # Remove highest VIF variables until all VIF values are below 5
            removed_vars = []
            for var in high_vif['Variable']:
                if var in core_variables:
                    core_variables.remove(var)
                    removed_vars.append(var)
                    
                    # Recalculate VIF with updated variable set
                    if core_variables:
                        updated_vif = calculate_vif(df, core_variables)
                        if (updated_vif['VIF'] > 5).sum() == 0:
                            break
            
            logger.info(f"Removed {len(removed_vars)} high VIF variables: {', '.join(removed_vars)}")
    
    logger.info(f"Final core variable set has {len(core_variables)} variables: {', '.join(core_variables)}")
    
    return core_variables

def create_specialized_variable_sets(df, core_variables, performance_metrics):
    """
    Create specialized variable sets optimized for each performance metric.
    
    Args:
        df: DataFrame containing all variables
        core_variables: Core variable set
        performance_metrics: List of performance metrics
        
    Returns:
        dict: Dictionary mapping performance metrics to specialized variable sets
    """
    log_section("Creating Specialized Variable Sets")
    
    # Calculate correlation matrix
    corr_matrix = create_correlation_matrix(df, core_variables + COMPOSITE_INDICES, performance_metrics)
    
    # Create specialized sets
    specialized_sets = {}
    
    for metric in performance_metrics:
        logger.info(f"\nCreating specialized set for {metric}")
        
        # Start with core variables
        specialized_set = core_variables.copy()
        
        # Get correlations with this metric
        correlations = corr_matrix[metric].drop(performance_metrics)
        
        # Add composite indices with strong correlations (|r| > 0.15) if not already in core set
        strong_composites = []
        for var in COMPOSITE_INDICES:
            if var in correlations and abs(correlations[var]) > 0.15 and var not in specialized_set:
                strong_composites.append((var, correlations[var]))
        
        # Sort by absolute correlation
        strong_composites.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Add top variables to specialized set
        for var, corr in strong_composites:
            specialized_set.append(var)
            logger.info(f"  - Added {var} (correlation: {corr:.3f})")
        
        # Check for multicollinearity in the specialized set
        if specialized_set:
            vif_results = calculate_vif(df, specialized_set)
            high_vif = vif_results[vif_results['VIF'] > 5]
            
            # If we have high VIF variables, remove them
            if not high_vif.empty:
                logger.warning(f"Specialized set for {metric} has {len(high_vif)} variables with VIF > 5")
                
                # Sort by VIF in descending order
                high_vif = high_vif.sort_values('VIF', ascending=False)
                
                # Remove highest VIF variables until all VIF values are below 5
                removed_vars = []
                for var in high_vif['Variable']:
                    if var in specialized_set:
                        specialized_set.remove(var)
                        removed_vars.append(var)
                        
                        # Recalculate VIF with updated variable set
                        if specialized_set:
                            updated_vif = calculate_vif(df, specialized_set)
                            if (updated_vif['VIF'] > 5).sum() == 0:
                                break
                
                logger.info(f"  - Removed {len(removed_vars)} high VIF variables: {', '.join(removed_vars)}")
        
        specialized_sets[metric] = specialized_set
        logger.info(f"  - Final set has {len(specialized_set)} variables")
    
    return specialized_sets

def prepare_model_datasets(df, specialized_sets):
    """
    Prepare model-ready datasets for each performance metric.
    
    Args:
        df: DataFrame containing all variables
        specialized_sets: Dictionary mapping performance metrics to variable sets
        
    Returns:
        dict: Dictionary mapping performance metrics to prepared datasets
    """
    log_section("Preparing Model-Ready Datasets")
    
    # Dictionary to store prepared datasets
    model_datasets = {}
    
    for metric, variables in specialized_sets.items():
        logger.info(f"\nPreparing dataset for {metric}")
        
        # Extract variables and metric
        model_vars = variables + [metric]
        model_df = df[model_vars].copy()
        
        # Handle missing values
        missing_count = model_df.isnull().sum()
        if missing_count.sum() > 0:
            logger.warning(f"Dataset for {metric} has {missing_count.sum()} missing values")
            logger.info("Filling missing values with mean")
            model_df = model_df.fillna(model_df.mean())
        
        # Standardize predictor variables
        predictors = variables.copy()
        scaler = StandardScaler()
        model_df[predictors] = scaler.fit_transform(model_df[predictors])
        logger.info(f"Standardized {len(predictors)} predictor variables")
        
        # Add to dictionary
        model_datasets[metric] = model_df
        
        # Export dataset
        output_path = OUTPUT_DIR / f"model_data_{metric}.csv"
        model_df.to_csv(output_path, index=False)
        logger.info(f"Exported model-ready dataset to {output_path}")
    
    return model_datasets

def create_data_dictionary(specialized_sets):
    """
    Create a data dictionary documenting all variables in the specialized sets.
    
    Args:
        specialized_sets: Dictionary mapping performance metrics to variable sets
    """
    log_section("Creating Data Dictionary")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Path for data dictionary
    dict_path = OUTPUT_DIR / "variable_sets_dictionary.md"
    
    # Open file for writing
    with open(dict_path, 'w') as f:
        f.write("# Model Variable Sets Dictionary\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Overview\n\n")
        f.write("This document describes the variable sets selected for each performance metric ")
        f.write("after addressing multicollinearity through the use of composite variables. ")
        f.write("Each variable set has been optimized for a specific performance metric and ")
        f.write("has been validated to have low multicollinearity (VIF < 5).\n\n")
        
        # Document variable sets for each metric
        for metric, variables in specialized_sets.items():
            f.write(f"## {metric}\n\n")
            f.write(f"Variable set optimized for {metric} with {len(variables)} variables.\n\n")
            
            f.write("| Variable | Type | Description |\n")
            f.write("|----------|------|-------------|\n")
            
            for var in variables:
                # Determine variable type
                if var in PCA_COMPONENTS:
                    var_type = "PCA Component"
                    description = pca_component_description(var)
                elif var in COMPOSITE_INDICES:
                    var_type = "Composite Index"
                    description = composite_index_description(var)
                elif var in INTERACTION_TERMS:
                    var_type = "Interaction Term"
                    description = interaction_term_description(var)
                elif var in GEOGRAPHIC_VARS:
                    var_type = "Geographic Variable"
                    description = geographic_variable_description(var)
                else:
                    var_type = "Other"
                    description = "Variable not categorized"
                
                f.write(f"| {var} | {var_type} | {description} |\n")
            
            f.write("\n")
        
        # Add validation information
        f.write("## Validation\n\n")
        f.write("All variable sets have been validated to ensure low multicollinearity ")
        f.write("with all variables having Variance Inflation Factor (VIF) < 5. ")
        f.write("The variables have been selected based on their theoretical importance, ")
        f.write("correlation with the performance metric, and multicollinearity considerations.\n\n")
        
        f.write("## Usage\n\n")
        f.write("The model-ready datasets are available in the `output/modeling` directory ")
        f.write("with standardized variables and complete cases (no missing values).")
    
    logger.info(f"Data dictionary created at {dict_path}")

def pca_component_description(component):
    """Return description for a PCA component."""
    descriptions = {
        "sociodemographic_pc1": "Educational and professional advantage component (37.16% of variance)",
        "sociodemographic_pc2": "Youth demographics component (25.73% of variance)",
        "sociodemographic_pc3": "Employment disadvantage component (14.14% of variance)"
    }
    return descriptions.get(component, "PCA component")

def composite_index_description(index):
    """Return description for a composite index."""
    descriptions = {
        "education_advantage_index": "Weighted index of education variables with emphasis on third-level education",
        "education_diversity_index": "Shannon diversity index applied to education variables",
        "socioeconomic_advantage_index": "Weighted index of socioeconomic variables (employment, class)",
        "youth_potential_index": "Weighted combination of youth proportion and school age rate"
    }
    return descriptions.get(index, "Composite index")

def interaction_term_description(term):
    """Return description for an interaction term."""
    if "third_level_rate" in term:
        return "Interaction between elevation and third-level education rate"
    elif "professional_rate" in term:
        return "Interaction between elevation and professional employment rate"
    elif "unemployment_rate" in term:
        return "Interaction between elevation and unemployment rate"
    elif "working_class_rate" in term:
        return "Interaction between elevation and working class rate"
    return "Interaction term"

def geographic_variable_description(var):
    """Return description for a geographic variable."""
    descriptions = {
        "elevation": "Elevation in meters above sea level",
        "annual_rainfall": "Annual rainfall in millimeters",
        "urban": "Binary indicator for urban (1) or rural (0)",
        "rural_distance": "Distance to nearest rural area in kilometers"
    }
    return descriptions.get(var, "Geographic variable")

def create_correlation_heatmap(correlation_matrix, output_path):
    """
    Create a correlation heatmap visualization.
    
    Args:
        correlation_matrix: Correlation matrix
        output_path: Path to save the heatmap
    """
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )
    
    # Set title and labels
    plt.title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Correlation heatmap saved to {output_path}")

def generate_documentation(df, core_variables, specialized_sets, corr_matrix, vif_data):
    """
    Generate documentation on selection methodology and multicollinearity reduction.
    
    Args:
        df: DataFrame with all variables
        core_variables: Core variable set
        specialized_sets: Specialized variable sets
        corr_matrix: Correlation matrix
        vif_data: VIF results for core variables
    """
    log_section("Generating Documentation")
    
    doc_path = OUTPUT_DIR / "variable_selection_methodology.md"
    
    with open(doc_path, 'w') as f:
        f.write("# Variable Selection Methodology\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## 1. Introduction\n\n")
        f.write("This document describes the methodology used to select variables for ")
        f.write("statistical modeling in the GAA club analysis project. ")
        f.write("The variable selection process focuses on:")
        f.write("\n\n")
        f.write("1. Creating a core set of predictor variables using PCA components, geographic variables, and interaction terms\n")
        f.write("2. Developing specialized variable sets optimized for each performance metric\n")
        f.write("3. Ensuring low multicollinearity in all variable sets (VIF < 5)\n")
        f.write("4. Preparing model-ready datasets for analysis\n\n")
        
        f.write("## 2. Core Variable Set\n\n")
        f.write("The core variable set forms the foundation for all specialized sets and ")
        f.write("includes variables that address multicollinearity while preserving ")
        f.write("important relationships with performance metrics.\n\n")
        
        f.write("### 2.1 Components of the Core Set\n\n")
        f.write("The core variable set includes:\n\n")
        
        # List variables by type
        pca_vars = [var for var in core_variables if var in PCA_COMPONENTS]
        geo_vars = [var for var in core_variables if var in GEOGRAPHIC_VARS]
        interaction_vars = [var for var in core_variables if var in INTERACTION_TERMS]
        composite_vars = [var for var in core_variables if var in COMPOSITE_INDICES]
        
        if pca_vars:
            f.write("**PCA Components:**\n")
            for var in pca_vars:
                f.write(f"- {var}: {pca_component_description(var)}\n")
            f.write("\n")
        
        if geo_vars:
            f.write("**Geographic Variables:**\n")
            for var in geo_vars:
                f.write(f"- {var}: {geographic_variable_description(var)}\n")
            f.write("\n")
        
        if interaction_vars:
            f.write("**Interaction Terms:**\n")
            for var in interaction_vars:
                f.write(f"- {var}: {interaction_term_description(var)}\n")
            f.write("\n")
        
        if composite_vars:
            f.write("**Composite Indices:**\n")
            for var in composite_vars:
                f.write(f"- {var}: {composite_index_description(var)}\n")
            f.write("\n")
        
        f.write("### 2.2 Multicollinearity in Core Set\n\n")
        f.write("The core variable set was designed to have low multicollinearity ")
        f.write("with all variables having Variance Inflation Factor (VIF) < 5.\n\n")
        
        f.write("**Final VIF Values:**\n\n")
        f.write("| Variable | VIF |\n")
        f.write("|----------|-----|\n")
        for _, row in vif_data.iterrows():
            f.write(f"| {row['Variable']} | {row['VIF']:.2f} |\n")
        f.write("\n")
        
        f.write("## 3. Specialized Variable Sets\n\n")
        f.write("For each performance metric, a specialized variable set was created ")
        f.write("to optimize the prediction of that specific metric.\n\n")
        
        for metric, variables in specialized_sets.items():
            f.write(f"### 3.{list(specialized_sets.keys()).index(metric) + 1} {metric}\n\n")
            
            # Get correlation with this metric
            if corr_matrix is not None and metric in corr_matrix.columns:
                correlations = corr_matrix[metric].drop(PERFORMANCE_METRICS).sort_values(ascending=False)
                
                f.write(f"**Selected Variables ({len(variables)}):**\n\n")
                f.write("| Variable | Correlation |\n")
                f.write("|----------|------------|\n")
                for var in variables:
                    if var in correlations:
                        f.write(f"| {var} | {correlations[var]:.3f} |\n")
                    else:
                        f.write(f"| {var} | N/A |\n")
                f.write("\n")
                
                # Highlight top correlations
                f.write("**Top Correlations (Absolute Value):**\n\n")
                top_corr = correlations.abs().sort_values(ascending=False).head(5)
                for var, corr in top_corr.items():
                    original_corr = correlations[var]
                    direction = "positive" if original_corr > 0 else "negative"
                    f.write(f"- {var}: {original_corr:.3f} ({direction})\n")
                f.write("\n")
            else:
                f.write(f"Variable set includes {len(variables)} variables.\n\n")
        
        f.write("## 4. Addressing Multicollinearity\n\n")
        f.write("### 4.1 Composite Variables Approach\n\n")
        f.write("The project addresses multicollinearity through several approaches:\n\n")
        f.write("1. **PCA Components**: Using principal components that capture underlying patterns while being uncorrelated\n")
        f.write("2. **Composite Indices**: Creating weighted indices for related variables (education, socioeconomic status)\n")
        f.write("3. **Interaction Terms**: Creating standardized interaction terms to address confounding relationships\n")
        f.write("4. **VIF-Based Selection**: Removing variables with high VIF values from variable sets\n\n")
        
        f.write("### 4.2 Multicollinearity Reduction Results\n\n")
        f.write("The variable selection approach successfully reduced multicollinearity:")
        f.write("\n\n")
        f.write("- All variables in the final sets have VIF < 5\n")
        f.write("- PCA components are orthogonal by design\n")
        f.write("- Interaction terms address confounding between elevation and socioeconomic factors\n\n")
        
        f.write("## 5. Model-Ready Datasets\n\n")
        f.write("The final step in the variable selection process was the preparation of ")
        f.write("model-ready datasets for each performance metric:\n\n")
        
        f.write("1. **Variable Selection**: Each dataset includes the specialized variable set for its metric\n")
        f.write("2. **Standardization**: All predictor variables are standardized (mean=0, std=1)\n")
        f.write("3. **Missing Value Handling**: Any missing values are imputed with the mean\n")
        f.write("4. **Data Export**: Each dataset is exported as a CSV file in the modeling directory\n\n")
        
        f.write("## 6. Conclusion\n\n")
        f.write("The variable selection methodology successfully created optimized variable sets ")
        f.write("for each performance metric while addressing multicollinearity. ")
        f.write("The resulting datasets are ready for statistical modeling with reduced risk ")
        f.write("of model instability due to multicollinearity.")
    
    logger.info(f"Methodology documentation created at {doc_path}")
    
    # Create correlation heatmap for visualization
    if corr_matrix is not None:
        # Select only predictors and performance metrics for clarity
        predictors = []
        for var_set in specialized_sets.values():
            predictors.extend(var for var in var_set if var not in predictors)
        
        plot_vars = predictors + PERFORMANCE_METRICS
        plot_corr = corr_matrix.loc[plot_vars, plot_vars]
        
        heatmap_path = OUTPUT_DIR / "correlation_heatmap.png"
        create_correlation_heatmap(plot_corr, heatmap_path)

def validate_final_datasets(model_datasets, specialized_sets):
    """
    Validate the final model-ready datasets.
    
    Args:
        model_datasets: Dictionary of model datasets
        specialized_sets: Dictionary of specialized variable sets
        
    Returns:
        bool: True if validation is successful
    """
    log_section("Validating Final Datasets")
    
    validation_success = True
    
    # Check each dataset
    for metric, dataset in model_datasets.items():
        logger.info(f"\nValidating dataset for {metric}")
        
        # Check column count
        expected_cols = len(specialized_sets[metric]) + 1  # +1 for the metric itself
        if len(dataset.columns) != expected_cols:
            logger.error(f"Column count mismatch: expected {expected_cols}, got {len(dataset.columns)}")
            validation_success = False
        else:
            logger.info(f"Column count: {len(dataset.columns)} (correct)")
        
        # Check for missing values
        missing_values = dataset.isnull().sum().sum()
        if missing_values > 0:
            logger.error(f"Found {missing_values} missing values")
            validation_success = False
        else:
            logger.info("No missing values found (correct)")
        
        # Check for infinite values
        inf_count = np.isinf(dataset.select_dtypes(include=['number'])).sum().sum()
        if inf_count > 0:
            logger.error(f"Found {inf_count} infinite values")
            validation_success = False
        else:
            logger.info("No infinite values found (correct)")
        
        # Check standardization of predictor variables
        predictors = specialized_sets[metric]
        for var in predictors:
            mean = dataset[var].mean()
            std = dataset[var].std()
            if abs(mean) > 0.01 or abs(std - 1) > 0.01:
                logger.warning(f"Variable {var} not properly standardized: mean={mean:.4f}, std={std:.4f}")
            else:
                logger.info(f"Variable {var} properly standardized")
    
    if validation_success:
        logger.info("All datasets passed validation")
    else:
        logger.error("Validation failed for one or more datasets")
    
    return validation_success

def main():
    """Main function to execute the variable selection process."""
    try:
        log_section("Starting Simplified Variable Selection and Model Preparation")
        
        # Load data
        df = load_data()
        
        # Get available performance metrics
        available_metrics = [metric for metric in PERFORMANCE_METRICS if metric in df.columns]
        
        if not available_metrics:
            logger.error("No performance metrics found in data")
            return False
        
        # Create core variable set
        core_variables = create_core_variable_set(df)
        
        if not core_variables:
            logger.error("Failed to create core variable set")
            return False
        
        # Calculate VIF for core variables
        vif_results = calculate_vif(df, core_variables)
        
        # Calculate correlation matrix
        all_variables = core_variables + COMPOSITE_INDICES
        all_variables = list(set([var for var in all_variables if var in df.columns]))
        corr_matrix = create_correlation_matrix(df, all_variables, available_metrics)
        
        # Create specialized variable sets
        specialized_sets = create_specialized_variable_sets(df, core_variables, available_metrics)
        
        if not specialized_sets:
            logger.error("Failed to create specialized variable sets")
            return False
        
        # Prepare model-ready datasets
        model_datasets = prepare_model_datasets(df, specialized_sets)
        
        if not model_datasets:
            logger.error("Failed to prepare model-ready datasets")
            return False
        
        # Validate final datasets
        validation_result = validate_final_datasets(model_datasets, specialized_sets)
        
        # Generate documentation
        generate_documentation(df, core_variables, specialized_sets, corr_matrix, vif_results)
        
        # Create data dictionary
        create_data_dictionary(specialized_sets)
        
        log_section("Simplified Variable Selection Completed")
        logger.info(f"Created specialized variable sets for {len(specialized_sets)} performance metrics")
        logger.info(f"Exported {len(model_datasets)} model-ready datasets")
        logger.info("Documentation and data dictionary created")
        
        return validation_result
    
    except Exception as e:
        logger.error(f"Error in variable selection process: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 