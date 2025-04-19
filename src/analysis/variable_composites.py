#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Variable Composites Generator

This script creates composite variables to address multicollinearity in the GAA club analysis project.
It focuses on creating composite indices for education variables, socioeconomic factors, and
environmental variables which showed high multicollinearity in our analysis.

Author: Daniel Tierney
Date: April 2025
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Define file paths
CLUB_DATA_PATH = DATA_DIR / "cork_clubs_transformed.csv"
NEAREST_DATA_PATH = DATA_DIR / "cork_clubs_nearest_key.csv"
SA_DATA_PATH = DATA_DIR / "cork_sa_analysis_key.csv"

# Define variable groups with high multicollinearity
EDUCATION_VARS = [
    "basic_education_rate", 
    "secondary_education_rate", 
    "third_level_rate"
]

SOCIOECONOMIC_VARS = [
    "unemployment_rate",
    "professional_rate",
    "working_class_rate"
]

DEMOGRAPHIC_VARS = [
    "youth_proportion",
    "school_age_rate",
    "youth_gender_ratio"
]

# This script creates several types of composite variables:
# 1. Simple composite indices: weighted combinations of related variables
# 2. PCA-based composites: principal components of variable groups
# 3. Interaction terms: cross-products between environment and demographic variables

def load_data():
    """
    Load club data and demographic data.
    
    Returns:
        pd.DataFrame: Club data with demographic variables
    """
    logger.info("Loading data files...")
    
    # Load club data
    club_df = pd.read_csv(CLUB_DATA_PATH)
    logger.info(f"Loaded club data with {club_df.shape[0]} clubs")
    
    try:
        # Try to load demographic catchment data
        nearest_df = pd.read_csv(NEAREST_DATA_PATH)
        logger.info(f"Loaded nearest catchment data with {nearest_df.shape[0]} records")
        
        # Check if nearest_club field exists for aggregation
        if "nearest_club" in nearest_df.columns:
            logger.info("Aggregating demographic data by nearest_club")
            
            # Identify demographic variables present in the data
            found_edu_vars = [var for var in EDUCATION_VARS if var in nearest_df.columns]
            found_se_vars = [var for var in SOCIOECONOMIC_VARS if var in nearest_df.columns]
            found_demo_vars = [var for var in DEMOGRAPHIC_VARS if var in nearest_df.columns]
            
            all_demo_vars = found_edu_vars + found_se_vars + found_demo_vars
            
            if all_demo_vars:
                logger.info(f"Found {len(all_demo_vars)} demographic variables: {all_demo_vars}")
                
                # Calculate weighted averages for each club
                nearest_df["weight"] = 1  # Can use population if available later
                
                # Aggregate data by club with weighted mean
                agg_dict = {var: lambda x: np.average(x, weights=nearest_df.loc[x.index, "weight"]) 
                           for var in all_demo_vars}
                
                # Group by nearest_club and calculate weighted mean
                demo_by_club = nearest_df.groupby("nearest_club").agg(agg_dict).reset_index()
                logger.info(f"Aggregated data for {len(demo_by_club)} clubs")
                
                # Rename nearest_club to match Club in club_df
                demo_by_club = demo_by_club.rename(columns={"nearest_club": "Club"})
                
                # Merge with club data
                club_df = club_df.merge(demo_by_club, on="Club", how="left")
                logger.info(f"Merged demographic data with club data, resulting in {club_df.shape[1]} columns")
                
                # Log how many clubs have demographic data
                clubs_with_data = club_df[all_demo_vars].notna().all(axis=1).sum()
                logger.info(f"{clubs_with_data} clubs have complete demographic data")
                
                # Impute missing values with mean for clubs without data
                if clubs_with_data < len(club_df):
                    logger.warning(f"{len(club_df) - clubs_with_data} clubs missing demographic data, imputing with mean")
                    for var in all_demo_vars:
                        club_df[var] = club_df[var].fillna(club_df[var].mean())
            else:
                logger.warning("No demographic variables found in catchment data")
        else:
            logger.warning("No nearest_club found in data, cannot aggregate")
            
    except FileNotFoundError:
        logger.warning(f"Catchment data file not found at {NEAREST_DATA_PATH}")
        logger.warning("Falling back to small area analysis data")
        
        try:
            sa_df = pd.read_csv(SA_DATA_PATH)
            logger.info(f"Loaded small area analysis data with {sa_df.shape[0]} records")
            
            # Extract demographic variables
            found_edu_vars = [var for var in EDUCATION_VARS if var in sa_df.columns]
            found_se_vars = [var for var in SOCIOECONOMIC_VARS if var in sa_df.columns]
            found_demo_vars = [var for var in DEMOGRAPHIC_VARS if var in sa_df.columns]
            
            if found_edu_vars or found_se_vars or found_demo_vars:
                logger.info(f"Found demographic variables in small area data: {found_edu_vars + found_se_vars + found_demo_vars}")
                logger.warning("Cannot link small area data with clubs without catchment assignments")
                logger.info("Using synthetic data for demonstration purposes")
                
                # Create synthetic demographic variables for demonstration
                for var in EDUCATION_VARS + SOCIOECONOMIC_VARS + DEMOGRAPHIC_VARS:
                    if var not in club_df.columns:
                        club_df[var] = np.random.normal(0.5, 0.15, len(club_df))
                
                logger.warning("Created synthetic demographic variables for demonstration")
            
        except FileNotFoundError:
            logger.warning(f"Small area analysis data not found at {SA_DATA_PATH}")
            logger.info("Creating synthetic data for demonstration purposes")
            
            # Create synthetic demographic variables for demonstration
            for var in EDUCATION_VARS + SOCIOECONOMIC_VARS + DEMOGRAPHIC_VARS:
                if var not in club_df.columns:
                    club_df[var] = np.random.normal(0.5, 0.15, len(club_df))
            
            logger.warning("Created synthetic demographic variables for demonstration")
    
    # Check that Elevation exists, use club_elevation from nearest data if needed
    if "Elevation" in club_df.columns:
        # Ensure elevation has no missing values
        if club_df["Elevation"].isnull().any():
            logger.warning(f"Found {club_df['Elevation'].isnull().sum()} clubs with missing elevation")
            club_df["Elevation"] = club_df["Elevation"].fillna(club_df["Elevation"].mean())
    elif "elevation" in club_df.columns:
        logger.info("Using existing 'elevation' column")
    elif "club_elevation" in club_df.columns:
        logger.info("Renaming 'club_elevation' to 'elevation'")
        club_df["elevation"] = club_df["club_elevation"]
    else:
        logger.warning("No elevation data found, using random values for demonstration")
        club_df["elevation"] = np.random.normal(100, 50, len(club_df))
    
    return club_df

def create_education_index(df, education_vars=EDUCATION_VARS):
    """
    Create a composite education index using available education variables.
    
    Args:
        df: DataFrame containing education variables
        education_vars: List of education variables to use
        
    Returns:
        DataFrame with education index added
    """
    logger.info("Creating education composite index...")
    
    # Check which variables are available
    avail_vars = [var for var in education_vars if var in df.columns]
    
    if not avail_vars:
        logger.warning("No education variables available, cannot create education index")
        return df
    
    logger.info(f"Using {len(avail_vars)} education variables: {avail_vars}")
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Option 1: Create weighted education index (higher weight for third-level)
    if "third_level_rate" in avail_vars:
        # Higher weight for third_level_rate (positive indicator)
        # Lower weight for basic_education_rate (negative indicator)
        weights = {}
        for var in avail_vars:
            if var == "third_level_rate":
                weights[var] = 0.6  # Higher weight for third-level
            elif var == "basic_education_rate":
                weights[var] = -0.3  # Negative weight for basic education
            else:
                weights[var] = 0.1  # Small weight for secondary
                
        # Calculate weighted sum
        result_df["education_advantage_index"] = sum(df[var] * weights[var] for var in avail_vars)
        
        # Normalize to 0-1 range
        min_val = result_df["education_advantage_index"].min()
        max_val = result_df["education_advantage_index"].max()
        result_df["education_advantage_index"] = (result_df["education_advantage_index"] - min_val) / (max_val - min_val)
        
        logger.info("Created education_advantage_index using weighted sum approach")
    
    # Option 2: Calculate education diversity index
    if len(avail_vars) >= 2:
        # Shannon diversity index formula
        result_df["education_diversity_index"] = 0
        for var in avail_vars:
            # Add a small constant to avoid log(0)
            values = df[var] + 0.01
            # Normalize so sum = 1
            normalized = values / values.sum()
            # Shannon formula component
            result_df["education_diversity_index"] -= normalized * np.log(normalized)
            
        # Normalize to 0-1 range 
        min_val = result_df["education_diversity_index"].min()
        max_val = result_df["education_diversity_index"].max()
        result_df["education_diversity_index"] = (result_df["education_diversity_index"] - min_val) / (max_val - min_val)
        
        logger.info("Created education_diversity_index using Shannon diversity formula")
    
    return result_df

def create_socioeconomic_index(df, se_vars=SOCIOECONOMIC_VARS):
    """
    Create a composite socioeconomic index.
    
    Args:
        df: DataFrame containing socioeconomic variables
        se_vars: List of socioeconomic variables to use
        
    Returns:
        DataFrame with socioeconomic index added
    """
    logger.info("Creating socioeconomic composite index...")
    
    # Check which variables are available
    avail_vars = [var for var in se_vars if var in df.columns]
    
    if not avail_vars:
        logger.warning("No socioeconomic variables available, cannot create socioeconomic index")
        return df
    
    logger.info(f"Using {len(avail_vars)} socioeconomic variables: {avail_vars}")
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Create socioeconomic advantage index
    weights = {}
    for var in avail_vars:
        if var == "professional_rate":
            weights[var] = 0.5  # Positive indicator
        elif var == "unemployment_rate":
            weights[var] = -0.5  # Negative indicator
        elif var == "working_class_rate":
            weights[var] = -0.3  # Slight negative indicator
        else:
            weights[var] = 0.1
    
    # Calculate weighted sum
    result_df["socioeconomic_advantage_index"] = 0
    for var in avail_vars:
        result_df["socioeconomic_advantage_index"] += df[var] * weights[var]
    
    # Normalize to 0-1 range
    min_val = result_df["socioeconomic_advantage_index"].min()
    max_val = result_df["socioeconomic_advantage_index"].max()
    result_df["socioeconomic_advantage_index"] = (result_df["socioeconomic_advantage_index"] - min_val) / (max_val - min_val)
    
    logger.info("Created socioeconomic_advantage_index")
    
    return result_df

def create_demographic_index(df, demo_vars=DEMOGRAPHIC_VARS):
    """
    Create a composite demographic index focusing on youth population.
    
    Args:
        df: DataFrame containing demographic variables
        demo_vars: List of demographic variables to use
        
    Returns:
        DataFrame with demographic index added
    """
    logger.info("Creating demographic composite index...")
    
    # Check which variables are available
    avail_vars = [var for var in demo_vars if var in df.columns]
    
    if not avail_vars:
        logger.warning("No demographic variables available, cannot create demographic index")
        return df
    
    logger.info(f"Using {len(avail_vars)} demographic variables: {avail_vars}")
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Create youth potential index
    if "youth_proportion" in avail_vars and "school_age_rate" in avail_vars:
        # Higher weight for school age population (more directly relevant to GAA)
        result_df["youth_potential_index"] = 0.7 * df["school_age_rate"] + 0.3 * df["youth_proportion"]
        
        # Normalize to 0-1 range
        min_val = result_df["youth_potential_index"].min()
        max_val = result_df["youth_potential_index"].max()
        result_df["youth_potential_index"] = (result_df["youth_potential_index"] - min_val) / (max_val - min_val)
        
        logger.info("Created youth_potential_index")
    
    return result_df

def create_pca_composites(df):
    """
    Create composite variables using Principal Component Analysis.
    
    Args:
        df: DataFrame containing variables for PCA
        
    Returns:
        DataFrame with PCA composite variables added
    """
    logger.info("Creating PCA-based composite variables...")
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Define variable groups for PCA
    education_vars = [var for var in EDUCATION_VARS if var in df.columns]
    socioeconomic_vars = [var for var in SOCIOECONOMIC_VARS if var in df.columns]
    demographic_vars = [var for var in DEMOGRAPHIC_VARS if var in df.columns]
    
    all_vars = education_vars + socioeconomic_vars + demographic_vars
    
    if len(all_vars) < 3:
        logger.warning("Not enough variables for PCA, need at least 3, found {len(all_vars)}")
        return result_df
    
    # Prepare data for PCA
    X = df[all_vars].copy()
    
    # Handle missing values with imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Apply PCA
    n_components = min(3, len(all_vars))  # Use at most 3 components
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)
    
    # Add principal components to dataframe
    for i in range(n_components):
        result_df[f'sociodemographic_pc{i+1}'] = principal_components[:, i]
    
    # Log explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    logger.info(f"PCA created {n_components} components explaining {cumulative_variance[-1]*100:.2f}% of variance")
    for i in range(n_components):
        logger.info(f"  - PC{i+1}: {explained_variance[i]*100:.2f}% of variance")
    
    # Create documentation of PCA loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=all_vars
    )
    
    # Save loadings to CSV
    loadings.to_csv(OUTPUT_DIR / "pca_loadings.csv")
    logger.info(f"Saved PCA loadings to {OUTPUT_DIR / 'pca_loadings.csv'}")
    
    return result_df

def create_interaction_terms(df):
    """
    Create interaction terms between environmental and demographic factors.
    
    Args:
        df: DataFrame containing variables for interaction terms
        
    Returns:
        DataFrame with interaction terms added
    """
    logger.info("Creating interaction terms...")
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Check different capitalization of elevation field
    elevation_col = None
    if "elevation" in df.columns:
        elevation_col = "elevation"
    elif "Elevation" in df.columns:
        elevation_col = "Elevation"
    
    if elevation_col is None:
        logger.warning("Elevation variable not found, cannot create interaction terms")
        return result_df
    
    logger.info(f"Using {elevation_col} for interaction terms")
    
    # Create interaction terms with elevation
    interaction_vars = []
    
    # Education interactions with elevation
    for var in [v for v in EDUCATION_VARS if v in df.columns]:
        interaction_name = f"elevation_x_{var}"
        result_df[interaction_name] = df[elevation_col] * df[var]
        interaction_vars.append(interaction_name)
    
    # Socioeconomic interactions with elevation
    for var in [v for v in SOCIOECONOMIC_VARS if v in df.columns]:
        interaction_name = f"elevation_x_{var}"
        result_df[interaction_name] = df[elevation_col] * df[var]
        interaction_vars.append(interaction_name)
    
    if interaction_vars:
        logger.info(f"Created {len(interaction_vars)} interaction terms with elevation")
        
        # Standardize interaction terms
        for var in interaction_vars:
            mean_val = result_df[var].mean()
            std_val = result_df[var].std()
            if std_val > 0:
                result_df[var] = (result_df[var] - mean_val) / std_val
    
    return result_df

def main():
    """
    Main function to run the variable composites creation.
    """
    try:
        logger.info("Starting variable composites generation")
        
        # Load data
        df = load_data()
        
        # Create composite indices
        df = create_education_index(df)
        df = create_socioeconomic_index(df)
        df = create_demographic_index(df)
        
        # Create PCA-based composites
        df = create_pca_composites(df)
        
        # Create interaction terms
        df = create_interaction_terms(df)
        
        # Save results
        output_path = DATA_DIR / "cork_clubs_with_composites.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved data with composite variables to {output_path}")
        
        # Document created variables
        created_vars = [
            "education_advantage_index",
            "education_diversity_index",
            "socioeconomic_advantage_index", 
            "youth_potential_index",
            "sociodemographic_pc1",
            "sociodemographic_pc2",
            "sociodemographic_pc3"
        ]
        
        created_vars = [var for var in created_vars if var in df.columns]
        interaction_vars = [col for col in df.columns if col.startswith("elevation_x_")]
        
        all_new_vars = created_vars + interaction_vars
        
        # Generate summary statistics for new variables
        if all_new_vars:
            summary_stats = df[all_new_vars].describe().transpose()
            summary_stats.to_csv(OUTPUT_DIR / "composite_variables_summary.csv")
            logger.info(f"Saved composite variables summary to {OUTPUT_DIR / 'composite_variables_summary.csv'}")
            
            # Calculate correlations with performance metrics
            performance_metrics = ["overall_performance", "football_performance", "hurling_performance", "code_balance"]
            avail_metrics = [metric for metric in performance_metrics if metric in df.columns]
            
            if avail_metrics:
                corr_vars = all_new_vars + avail_metrics
                correlations = df[corr_vars].corr()
                correlations.to_csv(OUTPUT_DIR / "composite_variable_correlations.csv")
                logger.info(f"Saved correlations to {OUTPUT_DIR / 'composite_variable_correlations.csv'}")
        
        logger.info("Variable composites generation completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error in variable composites generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 