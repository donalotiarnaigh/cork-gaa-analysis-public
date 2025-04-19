import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import logging
from datetime import datetime
import scipy.stats as stats
from typing import Dict, List, Tuple
from collections import Counter
from sklearn.metrics import cohen_kappa_score

# Set up constants
REPORTS_DIR = Path('reports')
OUTPUT_DIR = Path('output/statistics/catchment_comparison')
DATA_DIR = Path('data/processed')

# Ensure directories exist
REPORTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_section(title):
    """Log a section title with decoration."""
    logger.info("=" * 80)
    logger.info(f" {title} ".center(80, "="))
    logger.info("=" * 80)

def load_club_data():
    """
    Load basic club data with performance metrics.
    
    Returns:
        pd.DataFrame: Club data with performance metrics
    """
    clubs_df = pd.read_csv(DATA_DIR / 'cork_clubs_transformed.csv')
    logger.info(f"Loaded {len(clubs_df)} clubs with transformed metrics")
    return clubs_df

def load_voronoi_data():
    """
    Load data using Voronoi catchment methodology.
    
    Returns:
        pd.DataFrame: Club data with demographic variables from Voronoi catchment
    """
    log_section("Loading Voronoi Data")
    
    # Load club data
    clubs_df = load_club_data()
    
    # Load Voronoi demographic data
    voronoi_gdf = gpd.read_file(DATA_DIR / 'voronoi_demographics.gpkg')
    logger.info(f"Loaded {len(voronoi_gdf)} small areas with Voronoi assignments")
    
    # The column containing club names is 'assigned_club_name'
    club_col = 'assigned_club_name'
    
    if club_col not in voronoi_gdf.columns:
        logger.error(f"Column '{club_col}' not found in Voronoi data")
        raise ValueError(f"Column '{club_col}' not found in Voronoi data")
    
    logger.info(f"Using '{club_col}' as club identifier in Voronoi data")
    
    # Identify demographic columns (filter out geometry and ID columns)
    demographic_cols = [
        col for col in voronoi_gdf.columns 
        if ('rate' in col or 'proportion' in col or 'ratio' in col or 'T1_1' in col or 'T4_4' in col) 
        and 'voronoi' in col
    ]
    
    logger.info(f"Found {len(demographic_cols)} demographic columns in Voronoi data")
    
    # Aggregate demographic data by club using weighted averages based on population
    pop_col = [col for col in voronoi_gdf.columns if 'AGE0T' in col and 'voronoi' in col]
    if not pop_col:
        logger.warning("No population column found in Voronoi data, using equal weights")
        voronoi_gdf['weight'] = 1
    else:
        pop_col = pop_col[0]
        logger.info(f"Using '{pop_col}' as population weight in Voronoi data")
        voronoi_gdf['weight'] = voronoi_gdf[pop_col]
    
    # Group by club and calculate weighted averages
    voronoi_clubs = {}
    
    for club in clubs_df['Club'].unique():
        # Try to match club name, handle potential case differences and formatting
        club_sa = None
        
        # Try exact match first
        club_sa = voronoi_gdf[voronoi_gdf[club_col] == club]
        
        # If no exact match, try case-insensitive match
        if len(club_sa) == 0:
            # Get all unique club names from the dataset
            unique_clubs = voronoi_gdf[club_col].dropna().unique()
            
            # Try to find a case-insensitive match
            matched_club = None
            for uc in unique_clubs:
                if isinstance(uc, str) and uc.lower() == club.lower():
                    matched_club = uc
                    break
            
            if matched_club:
                club_sa = voronoi_gdf[voronoi_gdf[club_col] == matched_club]
                logger.info(f"Found case-insensitive match for club '{club}': '{matched_club}'")
        
        if len(club_sa) == 0:
            logger.warning(f"No small areas found for club '{club}' in Voronoi data")
            continue
        
        # Calculate weighted average for each demographic variable
        club_demo = {}
        total_weight = club_sa['weight'].sum()
        
        for col in demographic_cols:
            # Skip columns with all NaN values
            if club_sa[col].isna().all():
                continue
                
            # Calculate weighted average
            if total_weight > 0:
                weighted_avg = (club_sa[col] * club_sa['weight']).sum() / total_weight
            else:
                weighted_avg = club_sa[col].mean()
                
            # Store in results with standardized column name (remove _voronoi suffix)
            std_col = col.replace('_voronoi', '')
            club_demo[std_col] = weighted_avg
        
        voronoi_clubs[club] = club_demo
    
    # Convert to dataframe and merge with club data
    voronoi_df = pd.DataFrame.from_dict(voronoi_clubs, orient='index')
    voronoi_df.index.name = 'Club'
    voronoi_df.reset_index(inplace=True)
    
    # Merge with club data
    voronoi_result = pd.merge(clubs_df, voronoi_df, on='Club', how='left')
    
    logger.info(f"Final Voronoi dataset has {len(voronoi_result)} clubs with {len(voronoi_df.columns) - 1} demographic variables")
    
    return voronoi_result

def load_nearest_data():
    """
    Load data using nearest club catchment methodology.
    
    Returns:
        pd.DataFrame: Club data with demographic variables from nearest club catchment
    """
    log_section("Loading Nearest Club Data")
    
    # Load club data
    clubs_df = load_club_data()
    
    # Load nearest club demographic data
    nearest_gdf = gpd.read_file(DATA_DIR / 'nearest_demographics.gpkg')
    logger.info(f"Loaded {len(nearest_gdf)} small areas with nearest club assignments")
    
    # The column containing club names is 'nearest_club'
    club_col = 'nearest_club'
    
    if club_col not in nearest_gdf.columns:
        logger.error(f"Column '{club_col}' not found in nearest club data")
        raise ValueError(f"Column '{club_col}' not found in nearest club data")
    
    logger.info(f"Using '{club_col}' as club identifier in nearest club data")
    
    # Identify demographic columns (filter out geometry and ID columns)
    demographic_cols = [
        col for col in nearest_gdf.columns 
        if ('rate' in col or 'proportion' in col or 'ratio' in col or 'T1_1' in col or 'T4_4' in col) 
        and 'nearest' in col
    ]
    
    logger.info(f"Found {len(demographic_cols)} demographic columns in nearest club data")
    
    # Aggregate demographic data by club using weighted averages based on population
    pop_col = [col for col in nearest_gdf.columns if 'AGE0T' in col and 'nearest' in col]
    if not pop_col:
        logger.warning("No population column found in nearest club data, using equal weights")
        nearest_gdf['weight'] = 1
    else:
        pop_col = pop_col[0]
        logger.info(f"Using '{pop_col}' as population weight in nearest club data")
        nearest_gdf['weight'] = nearest_gdf[pop_col]
    
    # Group by club and calculate weighted averages
    nearest_clubs = {}
    
    for club in clubs_df['Club'].unique():
        # Try to match club name, handle potential case differences and formatting
        club_sa = None
        
        # Try exact match first
        club_sa = nearest_gdf[nearest_gdf[club_col] == club]
        
        # If no exact match, try case-insensitive match
        if len(club_sa) == 0:
            # Get all unique club names from the dataset
            unique_clubs = nearest_gdf[club_col].dropna().unique()
            
            # Try to find a case-insensitive match
            matched_club = None
            for uc in unique_clubs:
                if isinstance(uc, str) and uc.lower() == club.lower():
                    matched_club = uc
                    break
            
            if matched_club:
                club_sa = nearest_gdf[nearest_gdf[club_col] == matched_club]
                logger.info(f"Found case-insensitive match for club '{club}': '{matched_club}'")
        
        if len(club_sa) == 0:
            logger.warning(f"No small areas found for club '{club}' in nearest club data")
            continue
        
        # Calculate weighted average for each demographic variable
        club_demo = {}
        total_weight = club_sa['weight'].sum()
        
        for col in demographic_cols:
            # Skip columns with all NaN values
            if club_sa[col].isna().all():
                continue
                
            # Calculate weighted average
            if total_weight > 0:
                weighted_avg = (club_sa[col] * club_sa['weight']).sum() / total_weight
            else:
                weighted_avg = club_sa[col].mean()
                
            # Store in results with standardized column name (remove _nearest suffix)
            std_col = col.replace('_nearest', '')
            club_demo[std_col] = weighted_avg
        
        nearest_clubs[club] = club_demo
    
    # Convert to dataframe and merge with club data
    nearest_df = pd.DataFrame.from_dict(nearest_clubs, orient='index')
    nearest_df.index.name = 'Club'
    nearest_df.reset_index(inplace=True)
    
    # Merge with club data
    nearest_result = pd.merge(clubs_df, nearest_df, on='Club', how='left')
    
    logger.info(f"Final nearest club dataset has {len(nearest_result)} clubs with {len(nearest_df.columns) - 1} demographic variables")
    
    return nearest_result

def identify_common_variables(voronoi_df, nearest_df):
    """
    Identify common demographic variables between Voronoi and nearest catchment methods.
    
    Args:
        voronoi_df: DataFrame with Voronoi catchment data
        nearest_df: DataFrame with nearest club catchment data
    
    Returns:
        List of common demographic variables
    """
    # Get demographic columns (exclude basic club info and performance metrics)
    club_info_cols = ['Club', 'Latitude', 'Longitude', 'Elevation', 'annual_rainfall', 'rain_days'] + \
                     [col for col in voronoi_df.columns if 'Grade' in col or 'performance' in col or 'improvement' in col]
    
    voronoi_demo = [col for col in voronoi_df.columns if col not in club_info_cols]
    nearest_demo = [col for col in nearest_df.columns if col not in club_info_cols]
    
    # Find common columns
    common_vars = list(set(voronoi_demo).intersection(set(nearest_demo)))
    
    logger.info(f"Identified {len(common_vars)} common demographic variables between catchment methods")
    
    return common_vars

def perform_correlation_analysis(voronoi_df, nearest_df, common_vars):
    """
    Perform correlation analysis for both catchment methods and compare results.
    
    Args:
        voronoi_df: DataFrame with Voronoi catchment data
        nearest_df: DataFrame with nearest club catchment data
        common_vars: List of common demographic variables
        
    Returns:
        Dictionary containing correlation results and comparison metrics
    """
    log_section("Performing Correlation Analysis")
    
    # Performance metrics to analyze
    perf_metrics = [
        'transformed_performance',
        'transformed_football',
        'transformed_hurling',
        'transformed_code_balance'
    ]
    
    # Initialize results dictionary
    results = {
        'voronoi_correlations': {},
        'nearest_correlations': {},
        'voronoi_significance': {},
        'nearest_significance': {},
        'correlation_differences': {},
        'method_comparison': {}
    }
    
    # Function to calculate p-values for correlation
    def calculate_pvalues(df, x_vars, y_vars):
        p_values = pd.DataFrame(index=x_vars, columns=y_vars)
        
        for x in x_vars:
            for y in y_vars:
                if x in df.columns and y in df.columns:
                    # Skip if either column has all NaN values
                    if df[x].isna().all() or df[y].isna().all():
                        p_values.loc[x, y] = np.nan
                        continue
                    
                    # Calculate correlation and p-value    
                    corr, p = stats.pearsonr(df[x].dropna(), df[y].dropna())
                    p_values.loc[x, y] = p
                else:
                    p_values.loc[x, y] = np.nan
                
        return p_values
    
    # Calculate correlations for Voronoi data
    logger.info("Calculating correlations for Voronoi catchment data")
    voronoi_corr = voronoi_df[common_vars + perf_metrics].corr()
    voronoi_p = calculate_pvalues(voronoi_df, common_vars, perf_metrics)
    
    # Calculate correlations for nearest club data
    logger.info("Calculating correlations for nearest club catchment data")
    nearest_corr = nearest_df[common_vars + perf_metrics].corr()
    nearest_p = calculate_pvalues(nearest_df, common_vars, perf_metrics)
    
    # Extract correlation matrices relevant to demographic vs performance
    results['voronoi_correlations'] = voronoi_corr.loc[common_vars, perf_metrics]
    results['nearest_correlations'] = nearest_corr.loc[common_vars, perf_metrics]
    results['voronoi_significance'] = voronoi_p
    results['nearest_significance'] = nearest_p
    
    # Debug logs
    logger.info(f"Voronoi correlations shape: {results['voronoi_correlations'].shape}")
    logger.info(f"Voronoi correlations type: {type(results['voronoi_correlations'])}")
    for metric in perf_metrics:
        logger.info(f"Voronoi correlations[{metric}] type: {type(results['voronoi_correlations'][metric])}")
    
    # Calculate differences between catchment methods
    logger.info("Comparing correlation results between catchment methods")
    correlation_diff = results['voronoi_correlations'] - results['nearest_correlations']
    results['correlation_differences'] = correlation_diff
    
    # Identify strongest correlations for each method and metric
    method_comparison = {
        'strongest_correlations': {},
        'agreement_metrics': {}
    }
    
    for metric in perf_metrics:
        # Top correlations for Voronoi method
        metric_data = results['voronoi_correlations'][metric]
        # Check if metric_data is a Series or DataFrame
        if isinstance(metric_data, pd.Series):
            voronoi_top = metric_data.abs().sort_values(ascending=False).head(5)
        else:
            # If it's a DataFrame and has duplicate column names, sort by the first column
            voronoi_top = metric_data.abs().iloc[:, 0].sort_values(ascending=False).head(5)
        voronoi_top_vars = voronoi_top.index.tolist()
        
        # Top correlations for nearest method
        metric_data = results['nearest_correlations'][metric]
        # Check if metric_data is a Series or DataFrame
        if isinstance(metric_data, pd.Series):
            nearest_top = metric_data.abs().sort_values(ascending=False).head(5)
        else:
            # If it's a DataFrame and has duplicate column names, sort by the first column
            nearest_top = metric_data.abs().iloc[:, 0].sort_values(ascending=False).head(5)
        nearest_top_vars = nearest_top.index.tolist()
        
        # Combine results
        method_comparison['strongest_correlations'][metric] = {
            'voronoi': [(var, results['voronoi_correlations'].loc[var, metric], 
                        results['voronoi_significance'].loc[var, metric]) 
                       for var in voronoi_top_vars],
            'nearest': [(var, results['nearest_correlations'].loc[var, metric], 
                        results['nearest_significance'].loc[var, metric]) 
                       for var in nearest_top_vars]
        }
        
        # Calculate agreement metrics
        common_top_vars = set(voronoi_top_vars).intersection(set(nearest_top_vars))
        method_comparison['agreement_metrics'][metric] = {
            'overlap_count': len(common_top_vars),
            'overlap_percentage': len(common_top_vars) / 5 * 100 if len(voronoi_top_vars) > 0 else 0,
            'common_variables': list(common_top_vars)
        }
        
        # Calculate Cohen's Kappa for agreement between methods
        # Create binary vectors indicating if each variable is in top 5
        all_vars = list(set(voronoi_top_vars) | set(nearest_top_vars))
        voronoi_indicator = [1 if var in voronoi_top_vars else 0 for var in all_vars]
        nearest_indicator = [1 if var in nearest_top_vars else 0 for var in all_vars]
        
        # Calculate Cohen's Kappa if possible
        if len(all_vars) > 1 and (sum(voronoi_indicator) > 0 or sum(nearest_indicator) > 0):
            kappa = cohen_kappa_score(voronoi_indicator, nearest_indicator)
        else:
            kappa = 0.0
            
        method_comparison['agreement_metrics'][metric]['cohen_kappa'] = kappa
    
    results['method_comparison'] = method_comparison
    
    return results

def generate_visualizations(voronoi_df, nearest_df, common_vars, results):
    """
    Generate visualizations to compare Voronoi and nearest club catchment methodologies.
    
    Args:
        voronoi_df: DataFrame with Voronoi catchment data
        nearest_df: DataFrame with nearest club catchment data
        common_vars: List of common demographic variables
        results: Dictionary of correlation analysis results
        
    Returns:
        Dictionary of visualization file paths
    """
    log_section("Generating Visualizations")
    
    # Create output directory if it doesn't exist
    output_dir = "output/statistics/catchment_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_paths = {}
    
    # Performance metrics to analyze
    perf_metrics = [
        'transformed_performance',
        'transformed_football',
        'transformed_hurling',
        'transformed_code_balance'
    ]
    
    # 1. Create correlation heatmap comparing both methods
    logger.info("Creating correlation difference heatmap")
    try:
        plt.figure(figsize=(14, 12))
        
        # Get top 20 variables with the largest absolute difference in correlations
        abs_diff_by_row = results['correlation_differences'].abs().max(axis=1)
        if isinstance(abs_diff_by_row, pd.Series):
            abs_diff = abs_diff_by_row.sort_values(ascending=False).head(20).index
        else:
            # If it's a DataFrame with duplicate columns, use iloc to get the first column
            abs_diff = abs_diff_by_row.iloc[:, 0].sort_values(ascending=False).head(20).index
        diff_subset = results['correlation_differences'].loc[abs_diff]
        
        # Create heatmap
        sns.heatmap(diff_subset, cmap='coolwarm', center=0, 
                   annot=True, fmt=".2f", linewidths=0.5)
        
        plt.title('Differences in Correlations: Voronoi - Nearest Club Method', fontsize=16)
        plt.ylabel('Demographic Variables', fontsize=12)
        plt.xlabel('Performance Metrics', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        diff_heatmap_path = f"{output_dir}/correlation_difference_heatmap.png"
        plt.savefig(diff_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['correlation_difference_heatmap'] = diff_heatmap_path
        logger.info(f"Saved correlation difference heatmap to {diff_heatmap_path}")
    except Exception as e:
        logger.error(f"Error creating correlation difference heatmap: {e}")
    
    # 2. Create side-by-side correlation heatmaps for each method
    logger.info("Creating side-by-side correlation heatmaps")
    try:
        # Get top 20 variables with highest correlations in either method
        voronoi_max_by_row = results['voronoi_correlations'].abs().max(axis=1)
        if isinstance(voronoi_max_by_row, pd.Series):
            voronoi_top = voronoi_max_by_row.sort_values(ascending=False).head(20).index
        else:
            # If it's a DataFrame with duplicate columns, use iloc to get the first column
            voronoi_top = voronoi_max_by_row.iloc[:, 0].sort_values(ascending=False).head(20).index
            
        nearest_max_by_row = results['nearest_correlations'].abs().max(axis=1)
        if isinstance(nearest_max_by_row, pd.Series):
            nearest_top = nearest_max_by_row.sort_values(ascending=False).head(20).index
        else:
            # If it's a DataFrame with duplicate columns, use iloc to get the first column
            nearest_top = nearest_max_by_row.iloc[:, 0].sort_values(ascending=False).head(20).index
            
        top_vars = list(set(voronoi_top) | set(nearest_top))
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 14))
        
        # Voronoi heatmap
        sns.heatmap(results['voronoi_correlations'].loc[top_vars], 
                   cmap='coolwarm', center=0, 
                   annot=True, fmt=".2f", linewidths=0.5,
                   ax=axes[0])
        axes[0].set_title('Voronoi Method Correlations', fontsize=16)
        axes[0].set_ylabel('Demographic Variables', fontsize=12)
        
        # Nearest club heatmap
        sns.heatmap(results['nearest_correlations'].loc[top_vars], 
                   cmap='coolwarm', center=0, 
                   annot=True, fmt=".2f", linewidths=0.5,
                   ax=axes[1])
        axes[1].set_title('Nearest Club Method Correlations', fontsize=16)
        
        plt.tight_layout()
        
        # Save figure
        heatmap_path = f"{output_dir}/method_comparison_heatmaps.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['method_comparison_heatmaps'] = heatmap_path
        logger.info(f"Saved side-by-side correlation heatmaps to {heatmap_path}")
    except Exception as e:
        logger.error(f"Error creating side-by-side correlation heatmaps: {e}")
    
    # 3. Create bar charts comparing top correlations for each performance metric
    logger.info("Creating bar charts comparing top correlations")
    
    for metric in perf_metrics:
        try:
            metric_name = metric.replace('transformed_', '').replace('_', ' ').title()
            
            # Debugging
            if metric not in results['method_comparison']['strongest_correlations']:
                logger.error(f"Missing data for {metric} in results")
                continue
                
            # Get data for plot
            voronoi_data = results['method_comparison']['strongest_correlations'][metric]['voronoi']
            nearest_data = results['method_comparison']['strongest_correlations'][metric]['nearest']
            
            # Extract variable names and proper scalar correlations
            v_vars = []
            v_corrs = []
            for i, (var, corr, p_val) in enumerate(voronoi_data):
                try:
                    v_vars.append(var)
                    # Handle different types of correlation values
                    if isinstance(corr, pd.DataFrame):
                        corr_val = corr.iloc[0, 0] if not corr.empty else 0.0
                    elif isinstance(corr, pd.Series):
                        corr_val = corr.iloc[0] if not corr.empty else 0.0
                    elif hasattr(corr, 'item'):
                        corr_val = corr.item()
                    else:
                        corr_val = float(corr)
                    v_corrs.append(float(corr_val))
                except Exception as e:
                    logger.error(f"Error processing Voronoi correlation {i} for {metric}: {e}")
                    v_corrs.append(0.0)
            
            n_vars = []
            n_corrs = []
            for i, (var, corr, p_val) in enumerate(nearest_data):
                try:
                    n_vars.append(var)
                    # Handle different types of correlation values
                    if isinstance(corr, pd.DataFrame):
                        corr_val = corr.iloc[0, 0] if not corr.empty else 0.0
                    elif isinstance(corr, pd.Series):
                        corr_val = corr.iloc[0] if not corr.empty else 0.0
                    elif hasattr(corr, 'item'):
                        corr_val = corr.item()
                    else:
                        corr_val = float(corr)
                    n_corrs.append(float(corr_val))
                except Exception as e:
                    logger.error(f"Error processing Nearest correlation {i} for {metric}: {e}")
                    n_corrs.append(0.0)
            
            # If we couldn't extract any variables, skip this chart
            if not v_vars:
                logger.error(f"No valid variables extracted for {metric}, skipping chart")
                continue
                
            # Create figure
            plt.figure(figsize=(14, 8))
            
            # Create positions for bars
            y_pos = np.arange(len(v_vars))
            width = 0.35
            
            # Create legend labels with significance indicators
            v_labels = []
            for i, (var, corr, p_val) in enumerate(voronoi_data):
                try:
                    # Convert p-value to float
                    if isinstance(p_val, pd.DataFrame):
                        p_val_float = p_val.iloc[0, 0] if not p_val.empty else 1.0
                    elif isinstance(p_val, pd.Series):
                        p_val_float = p_val.iloc[0] if not p_val.empty else 1.0
                    elif hasattr(p_val, 'item'):
                        p_val_float = p_val.item()
                    else:
                        p_val_float = float(p_val)
                        
                    sig = ''
                    if p_val_float < 0.05:
                        sig = '*'
                    if p_val_float < 0.01:
                        sig += '*'
                    if p_val_float < 0.001:
                        sig += '*'
                    v_labels.append(f"{var}{sig}")
                except Exception as e:
                    logger.error(f"Error processing p-value {i} for {metric}: {e}")
                    v_labels.append(var)
            
            # Create bar chart for Voronoi data
            plt.barh(y_pos - width/2, v_corrs, width, alpha=0.6, color='royalblue', label='Voronoi')
            
            # Align nearest club data with Voronoi variables
            n_corrs_aligned = []
            for var in v_vars:
                try:
                    if var in n_vars:
                        idx = n_vars.index(var)
                        n_corrs_aligned.append(n_corrs[idx])
                    else:
                        n_corrs_aligned.append(0.0)
                except Exception as e:
                    logger.error(f"Error aligning nearest correlations for {var}: {e}")
                    n_corrs_aligned.append(0.0)
            
            # Create bar chart for nearest club data
            plt.barh(y_pos + width/2, n_corrs_aligned, width, alpha=0.6, color='tomato', label='Nearest Club')
            
            plt.yticks(y_pos, v_labels)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f'Top 5 Correlations for {metric_name}', fontsize=16)
            plt.xlabel('Correlation Coefficient', fontsize=12)
            plt.ylabel('Demographic Variables', fontsize=12)
            plt.legend(loc='best')
            
            # Add significance legend
            plt.figtext(0.01, 0.01, "* p<0.05, ** p<0.01, *** p<0.001", ha="left", fontsize=10)
            
            plt.tight_layout()
            
            # Save figure
            bar_path = f"{output_dir}/{metric}_top_correlations.png"
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths[f'{metric}_top_correlations'] = bar_path
            logger.info(f"Saved {metric} top correlations chart to {bar_path}")
        except Exception as e:
            logger.error(f"Error creating bar chart for {metric}: {e}")
    
    # 4. Create method agreement visualization
    logger.info("Creating method agreement visualization")
    try:
        plt.figure(figsize=(10, 6))
        
        # Extract agreement metrics
        metrics = list(results['method_comparison']['agreement_metrics'].keys())
        metric_names = [m.replace('transformed_', '').replace('_', ' ').title() for m in metrics]
        overlap_counts = [results['method_comparison']['agreement_metrics'][m]['overlap_count'] for m in metrics]
        overlap_percentages = [results['method_comparison']['agreement_metrics'][m]['overlap_percentage'] for m in metrics]
        
        # Create bar positions
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        # Create bar chart
        ax = plt.gca()
        ax.bar(x_pos, overlap_counts, width, alpha=0.6, color='mediumseagreen', label='Common Variables Count')
        
        # Add percentage line
        ax2 = ax.twinx()
        ax2.plot(x_pos, overlap_percentages, 'ro-', linewidth=2, markersize=8, label='Agreement Percentage')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Agreement Percentage (%)', color='r')
        ax2.tick_params('y', colors='r')
        
        # Set labels and title
        ax.set_xlabel('Performance Metric')
        ax.set_ylabel('Number of Common Variables')
        ax.set_title('Agreement Between Catchment Methods', fontsize=16)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_names, rotation=45)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        agreement_path = f"{output_dir}/method_agreement.png"
        plt.savefig(agreement_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['method_agreement'] = agreement_path
        logger.info(f"Saved method agreement visualization to {agreement_path}")
    except Exception as e:
        logger.error(f"Error creating method agreement visualization: {e}")
    
    return visualization_paths

def generate_report(results, visualization_paths):
    """
    Generate a comprehensive report comparing the Voronoi and nearest club catchment methodologies.
    
    Args:
        results: Dictionary of correlation analysis results
        visualization_paths: Dictionary of visualization file paths
        
    Returns:
        Path to the generated report
    """
    log_section("Generating Report")
    
    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = f"{reports_dir}/catchment_methodology_comparison.md"
    
    logger.info(f"Generating report at {report_path}")
    
    # Performance metrics to analyze
    perf_metrics = [
        'transformed_performance',
        'transformed_football',
        'transformed_hurling',
        'transformed_code_balance'
    ]
    
    try:
        with open(report_path, 'w') as f:
            # Report header
            f.write("# Catchment Methodology Comparison: Voronoi vs. Nearest Club\n\n")
            f.write("## Overview\n\n")
            f.write("This report compares two methodologies for assigning demographic data to GAA clubs in Cork:\n\n")
            f.write("1. **Voronoi Diagram Method**: Creates territories based on equidistant boundaries between clubs\n")
            f.write("2. **Nearest Club Method**: Assigns small areas to the club that is geographically closest\n\n")
            
            f.write("The analysis compares how demographic correlations with club performance differ between these methods.\n\n")
            
            # Data Summary
            f.write("## Data Summary\n\n")
            
            # Calculate variable counts
            total_vars_voronoi = len(results['voronoi_correlations'].index) if hasattr(results['voronoi_correlations'], 'index') else 0
            total_vars_nearest = len(results['nearest_correlations'].index) if hasattr(results['nearest_correlations'], 'index') else 0
            common_vars = total_vars_voronoi  # This is the same as the count of common variables
            
            f.write(f"- Total variables in Voronoi dataset: {total_vars_voronoi}\n")
            f.write(f"- Total variables in Nearest Club dataset: {total_vars_nearest}\n")
            f.write(f"- Common variables between methods: {common_vars}\n\n")
            
            # Overall Method Agreement
            f.write("## Method Agreement Analysis\n\n")
            
            # Include method agreement visualization
            if 'method_agreement' in visualization_paths:
                rel_path = os.path.relpath(visualization_paths['method_agreement'], start=reports_dir)
                f.write(f"![Method Agreement Analysis]({rel_path})\n\n")
            
            f.write("### Method Agreement Metrics\n\n")
            
            # Table header
            f.write("| Performance Metric | Common Variables (Top 5) | Agreement Percentage | Cohen's Kappa |\n")
            f.write("|--------------------|--------------------------|--------------------|---------------|\n")
            
            # Table data
            for metric in perf_metrics:
                if metric not in results['method_comparison']['agreement_metrics']:
                    logger.warning(f"Missing agreement metrics for {metric}, skipping")
                    continue
                    
                metric_name = metric.replace('transformed_', '').replace('_', ' ').title()
                agreement = results['method_comparison']['agreement_metrics'][metric]
                
                overlap_count = agreement['overlap_count']
                overlap_percentage = agreement['overlap_percentage']
                
                # Handle potential pandas types for Cohen's Kappa
                if 'cohen_kappa' in agreement:
                    kappa = agreement['cohen_kappa']
                    if hasattr(kappa, 'item'):
                        kappa = float(kappa.item())
                    else:
                        kappa = float(kappa)
                else:
                    kappa = 0.0
                
                f.write(f"| {metric_name} | {overlap_count} | {overlap_percentage:.1f}% | {kappa:.3f} |\n")
            
            f.write("\n")
            
            # Interpretation of agreement
            f.write("### Agreement Interpretation\n\n")
            
            # Calculate average agreement metrics
            available_metrics = [m for m in perf_metrics if m in results['method_comparison']['agreement_metrics']]
            
            if available_metrics:
                # Safe conversion to Python scalars for percentages
                percentages = []
                for m in available_metrics:
                    pct = results['method_comparison']['agreement_metrics'][m]['overlap_percentage']
                    if hasattr(pct, 'item'):
                        percentages.append(float(pct.item()))
                    else:
                        percentages.append(float(pct))
                
                # Safe conversion to Python scalars for kappa values
                kappas = []
                for m in available_metrics:
                    if 'cohen_kappa' in results['method_comparison']['agreement_metrics'][m]:
                        k = results['method_comparison']['agreement_metrics'][m]['cohen_kappa']
                        if hasattr(k, 'item'):
                            kappas.append(float(k.item()))
                        else:
                            kappas.append(float(k))
                
                # Calculate averages
                avg_overlap = np.mean(percentages) if percentages else 0.0
                avg_kappa = np.mean(kappas) if kappas else 0.0
            else:
                avg_overlap = 0.0
                avg_kappa = 0.0
            
            f.write(f"The average agreement between methodologies is {avg_overlap:.1f}% with an average Cohen's Kappa of {avg_kappa:.3f}. ")
            
            # Interpretation based on kappa values
            if avg_kappa < 0.2:
                f.write("This indicates poor agreement between the two catchment methodologies, suggesting they produce substantially different results.\n\n")
            elif avg_kappa < 0.4:
                f.write("This indicates fair agreement between the two catchment methodologies, with moderate differences in results.\n\n")
            elif avg_kappa < 0.6:
                f.write("This indicates moderate agreement between the two catchment methodologies, suggesting they produce somewhat similar results.\n\n")
            elif avg_kappa < 0.8:
                f.write("This indicates substantial agreement between the two catchment methodologies, suggesting they produce similar results.\n\n")
            else:
                f.write("This indicates almost perfect agreement between the two catchment methodologies, suggesting they produce very similar results.\n\n")
            
            # Correlation Differences
            f.write("## Correlation Differences Between Methods\n\n")
            
            # Include correlation difference heatmap
            if 'correlation_difference_heatmap' in visualization_paths:
                rel_path = os.path.relpath(visualization_paths['correlation_difference_heatmap'], start=reports_dir)
                f.write(f"![Correlation Differences]({rel_path})\n\n")
            
            f.write("The heatmap above shows the differences in correlation coefficients (Voronoi minus Nearest Club) for the top 20 variables with the largest absolute differences. ")
            f.write("Positive values (red) indicate stronger correlation in the Voronoi method, while negative values (blue) indicate stronger correlation in the Nearest Club method.\n\n")
            
            # Include side-by-side heatmaps
            if 'method_comparison_heatmaps' in visualization_paths:
                rel_path = os.path.relpath(visualization_paths['method_comparison_heatmaps'], start=reports_dir)
                f.write(f"![Method Comparison Heatmaps]({rel_path})\n\n")
            
            f.write("The side-by-side heatmaps above show the correlation coefficients for the top variables in each method. ")
            f.write("This allows for direct comparison of how demographic variables correlate with performance metrics across both methodologies.\n\n")
            
            # Performance Metrics Analysis
            f.write("## Performance Metrics Analysis\n\n")
            
            for metric in perf_metrics:
                metric_name = metric.replace('transformed_', '').replace('_', ' ').title()
                
                if metric not in results['method_comparison']['strongest_correlations']:
                    logger.warning(f"Missing strongest correlations data for {metric}, skipping")
                    continue
                
                f.write(f"### {metric_name}\n\n")
                
                # Include top correlations chart
                if f'{metric}_top_correlations' in visualization_paths:
                    rel_path = os.path.relpath(visualization_paths[f'{metric}_top_correlations'], start=reports_dir)
                    f.write(f"![{metric_name} Top Correlations]({rel_path})\n\n")
                
                f.write("#### Top 5 Correlations: Voronoi Method\n\n")
                
                # Table header
                f.write("| Variable | Correlation | P-Value | Significance |\n")
                f.write("|----------|------------|---------|-------------|\n")
                
                # Table data for Voronoi
                for var, corr, p_val in results['method_comparison']['strongest_correlations'][metric]['voronoi']:
                    # Safe conversion to Python scalars
                    try:
                        if isinstance(corr, pd.DataFrame):
                            corr_val = corr.iloc[0, 0] if not corr.empty else 0.0
                        elif isinstance(corr, pd.Series):
                            corr_val = corr.iloc[0] if not corr.empty else 0.0
                        elif hasattr(corr, 'item'):
                            corr_val = float(corr.item())
                        else:
                            corr_val = float(corr)
                            
                        if isinstance(p_val, pd.DataFrame):
                            p_val_float = p_val.iloc[0, 0] if not p_val.empty else 1.0
                        elif isinstance(p_val, pd.Series):
                            p_val_float = p_val.iloc[0] if not p_val.empty else 1.0
                        elif hasattr(p_val, 'item'):
                            p_val_float = float(p_val.item())
                        else:
                            p_val_float = float(p_val)
                        
                        # Add significance markers
                        sig = ""
                        if p_val_float < 0.001:
                            sig = "***"
                        elif p_val_float < 0.01:
                            sig = "**"
                        elif p_val_float < 0.05:
                            sig = "*"
                        
                        f.write(f"| {var} | {corr_val:.3f} | {p_val_float:.3f} | {sig} |\n")
                    except Exception as e:
                        logger.error(f"Error formatting Voronoi correlation for {var}: {e}")
                        f.write(f"| {var} | N/A | N/A | |\n")
                
                f.write("\n")
                
                f.write("#### Top 5 Correlations: Nearest Club Method\n\n")
                
                # Table header
                f.write("| Variable | Correlation | P-Value | Significance |\n")
                f.write("|----------|------------|---------|-------------|\n")
                
                # Table data for Nearest Club
                for var, corr, p_val in results['method_comparison']['strongest_correlations'][metric]['nearest']:
                    # Safe conversion to Python scalars
                    try:
                        if isinstance(corr, pd.DataFrame):
                            corr_val = corr.iloc[0, 0] if not corr.empty else 0.0
                        elif isinstance(corr, pd.Series):
                            corr_val = corr.iloc[0] if not corr.empty else 0.0
                        elif hasattr(corr, 'item'):
                            corr_val = float(corr.item())
                        else:
                            corr_val = float(corr)
                            
                        if isinstance(p_val, pd.DataFrame):
                            p_val_float = p_val.iloc[0, 0] if not p_val.empty else 1.0
                        elif isinstance(p_val, pd.Series):
                            p_val_float = p_val.iloc[0] if not p_val.empty else 1.0
                        elif hasattr(p_val, 'item'):
                            p_val_float = float(p_val.item())
                        else:
                            p_val_float = float(p_val)
                        
                        # Add significance markers
                        sig = ""
                        if p_val_float < 0.001:
                            sig = "***"
                        elif p_val_float < 0.01:
                            sig = "**"
                        elif p_val_float < 0.05:
                            sig = "*"
                        
                        f.write(f"| {var} | {corr_val:.3f} | {p_val_float:.3f} | {sig} |\n")
                    except Exception as e:
                        logger.error(f"Error formatting Nearest correlation for {var}: {e}")
                        f.write(f"| {var} | N/A | N/A | |\n")
                
                f.write("\n")
                
                f.write("_Note: * p<0.05, ** p<0.01, *** p<0.001_\n\n")
                
                # Add interpretation for this metric
                f.write("#### Interpretation\n\n")
                
                # Get the agreement metrics for this metric
                if metric in results['method_comparison']['agreement_metrics']:
                    agreement = results['method_comparison']['agreement_metrics'][metric]
                    
                    # Safe conversion to Python scalars
                    overlap_percentage = agreement['overlap_percentage']
                    if hasattr(overlap_percentage, 'item'):
                        overlap_percentage = float(overlap_percentage.item())
                    
                    kappa = agreement.get('cohen_kappa', 0.0)
                    if hasattr(kappa, 'item'):
                        kappa = float(kappa.item())
                    
                    f.write(f"For {metric_name.lower()}, the agreement between methods is {overlap_percentage:.1f}% with Cohen's Kappa of {kappa:.3f}. ")
                    
                    # Get the unique variable lists
                    voronoi_vars = [item[0] for item in results['method_comparison']['strongest_correlations'][metric]['voronoi']]
                    nearest_vars = [item[0] for item in results['method_comparison']['strongest_correlations'][metric]['nearest']]
                    
                    # Add variable-specific insights
                    common_vars = set(voronoi_vars).intersection(set(nearest_vars))
                    voronoi_only = set(voronoi_vars) - set(nearest_vars)
                    nearest_only = set(nearest_vars) - set(voronoi_vars)
                    
                    if common_vars:
                        common_list = ", ".join(list(common_vars))
                        f.write(f"Both methods identify {common_list} as important demographic variables. ")
                    
                    if voronoi_only:
                        voronoi_list = ", ".join(list(voronoi_only))
                        f.write(f"The Voronoi method uniquely identifies {voronoi_list} as important. ")
                    
                    if nearest_only:
                        nearest_list = ", ".join(list(nearest_only))
                        f.write(f"The Nearest Club method uniquely identifies {nearest_list} as important. ")
                else:
                    f.write(f"No agreement metrics available for {metric_name.lower()}.")
                
                f.write("\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            
            # Determine which method shows stronger correlations overall
            # Safely convert pandas objects to Python scalars
            voronoi_abs_corrs = []
            nearest_abs_corrs = []
            
            for metric in perf_metrics:
                if metric not in results['method_comparison']['strongest_correlations']:
                    continue
                
                try:
                    # Extract top 5 correlations for each method and metric
                    voronoi_vals = []
                    for _, corr, _ in results['method_comparison']['strongest_correlations'][metric]['voronoi']:
                        if isinstance(corr, pd.DataFrame):
                            val = corr.iloc[0, 0] if not corr.empty else 0.0
                        elif isinstance(corr, pd.Series):
                            val = corr.iloc[0] if not corr.empty else 0.0
                        elif hasattr(corr, 'item'):
                            val = corr.item()
                        else:
                            val = float(corr)
                        voronoi_vals.append(abs(float(val)))
                    
                    nearest_vals = []
                    for _, corr, _ in results['method_comparison']['strongest_correlations'][metric]['nearest']:
                        if isinstance(corr, pd.DataFrame):
                            val = corr.iloc[0, 0] if not corr.empty else 0.0
                        elif isinstance(corr, pd.Series):
                            val = corr.iloc[0] if not corr.empty else 0.0
                        elif hasattr(corr, 'item'):
                            val = corr.item()
                        else:
                            val = float(corr)
                        nearest_vals.append(abs(float(val)))
                    
                    # Calculate means of absolute correlations
                    voronoi_abs_corrs.append(np.mean(voronoi_vals) if voronoi_vals else 0.0)
                    nearest_abs_corrs.append(np.mean(nearest_vals) if nearest_vals else 0.0)
                
                except Exception as e:
                    logger.error(f"Error calculating average correlations for {metric}: {e}")
            
            voronoi_avg = np.mean(voronoi_abs_corrs) if voronoi_abs_corrs else 0.0
            nearest_avg = np.mean(nearest_abs_corrs) if nearest_abs_corrs else 0.0
            
            f.write("### Key Findings\n\n")
            
            # Method strength comparison
            if voronoi_avg > nearest_avg * 1.1:
                f.write("- The **Voronoi method** generally produces stronger correlations between demographic variables and club performance metrics.\n")
            elif nearest_avg > voronoi_avg * 1.1:
                f.write("- The **Nearest Club method** generally produces stronger correlations between demographic variables and club performance metrics.\n")
            else:
                f.write("- Both methods produce similar strength correlations between demographic variables and club performance metrics.\n")
            
            # Agreement level
            if avg_kappa < 0.4:
                f.write("- There is **low agreement** between the two methods in identifying important demographic variables.\n")
            elif avg_kappa < 0.7:
                f.write("- There is **moderate agreement** between the two methods in identifying important demographic variables.\n")
            else:
                f.write("- There is **high agreement** between the two methods in identifying important demographic variables.\n")
            
            # Variable consistency
            consistent_vars = []
            for metric in perf_metrics:
                if metric not in results['method_comparison']['strongest_correlations']:
                    continue
                    
                voronoi_vars = [item[0] for item in results['method_comparison']['strongest_correlations'][metric]['voronoi']]
                nearest_vars = [item[0] for item in results['method_comparison']['strongest_correlations'][metric]['nearest']]
                common = set(voronoi_vars).intersection(set(nearest_vars))
                consistent_vars.extend(list(common))
            
            consistent_vars = [var for var, count in Counter(consistent_vars).items() if count >= 2]
            
            if consistent_vars:
                vars_list = ", ".join(consistent_vars)
                f.write(f"- Key demographic variables that consistently appear across both methods include: **{vars_list}**.\n")
            
            f.write("\n")
            
            # Recommendation
            f.write("### Recommendation\n\n")
            
            if avg_kappa < 0.4:
                if voronoi_avg > nearest_avg:
                    f.write("Given the significant differences between methods and the stronger correlations found in the Voronoi approach, ")
                    f.write("we recommend **primarily using the Voronoi method** for demographic analysis while using the Nearest Club method ")
                    f.write("as a complementary approach to validate findings.\n\n")
                else:
                    f.write("Given the significant differences between methods and the stronger correlations found in the Nearest Club approach, ")
                    f.write("we recommend **primarily using the Nearest Club method** for demographic analysis while using the Voronoi method ")
                    f.write("as a complementary approach to validate findings.\n\n")
            elif avg_kappa >= 0.7:
                f.write("Given the high agreement between methods, either approach can be used with confidence. ")
                f.write("We recommend using the **Voronoi method** for its theoretical advantages in representing club territories ")
                f.write("or the **Nearest Club method** for its simplicity and intuitive interpretation, ")
                f.write("based on specific research needs and computational constraints.\n\n")
            else:
                f.write("Given the moderate agreement between methods, we recommend using **both approaches in parallel** ")
                f.write("and focusing on demographic variables that show consistent relationships across both methods ")
                f.write("to ensure robust conclusions about demographic influences on club performance.\n\n")
            
            # Metadata
            f.write("---\n\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Report generated successfully at {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return None

def main():
    """Main function to execute catchment methodology comparison."""
    log_section("Catchment Methodology Comparison")
    
    logger.info("Script initialized. Starting catchment methodology comparison...")
    
    # Load data using both catchment methodologies
    voronoi_df = load_voronoi_data()
    nearest_df = load_nearest_data()
    
    # Identify common variables for comparison
    common_vars = identify_common_variables(voronoi_df, nearest_df)
    
    # Perform correlation analysis
    results = perform_correlation_analysis(voronoi_df, nearest_df, common_vars)
    
    # Generate visualizations
    visualization_paths = generate_visualizations(voronoi_df, nearest_df, common_vars, results)
    
    # Generate report
    report_path = generate_report(results, visualization_paths)
    
    if report_path:
        logger.info(f"Catchment methodology comparison completed successfully. Report saved to {report_path}")
    else:
        logger.error("Failed to generate catchment methodology comparison report")
    
    return report_path

if __name__ == "__main__":
    main() 