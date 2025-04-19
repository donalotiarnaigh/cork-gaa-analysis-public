#!/usr/bin/env python3
"""
Distribution analysis for GAA club performance metrics.

This script:
1. Analyzes urban/rural performance differences
2. Creates demographic stratification comparisons
3. Generates performance boxplots by categorical variables
4. Tests for significant group differences (t-tests, ANOVA)
5. Analyzes dual vs. single-code club performance patterns
6. Tests for normality in distribution of performance metrics
7. Evaluates if transformed metrics show more normal distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy import stats
import matplotlib.ticker as mtick
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from matplotlib.patches import Patch
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'output' / 'statistics'
REPORTS_DIR = BASE_DIR / 'reports'

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Performance metrics to analyze
PERFORMANCE_METRICS = [
    'overall_performance',
    'football_performance',
    'hurling_performance',
    'code_balance'
]

TRANSFORMED_METRICS = [
    'transformed_overall_performance',
    'transformed_football_performance', 
    'transformed_hurling_performance',
    'transformed_code_balance'
]

def load_data():
    """Load club data and demographic data."""
    logger.info("Loading data...")
    
    try:
        # Load club data
        club_filepath = DATA_DIR / 'cork_clubs_complete_graded.csv'
        club_df = pd.read_csv(club_filepath)
        logger.info(f"Loaded club data for {len(club_df)} clubs")
        
        # Add transformed metrics
        club_df = add_transformed_metrics(club_df)
        
        # Load demographic data from nearest and Voronoi analysis
        nearest_filepath = DATA_DIR / 'nearest_demographics.gpkg'
        voronoi_filepath = DATA_DIR / 'voronoi_demographics.gpkg'
        
        try:
            import geopandas as gpd
            nearest_df = gpd.read_file(nearest_filepath)
            voronoi_df = gpd.read_file(voronoi_filepath)
            logger.info(f"Loaded demographic data: {len(nearest_df)} nearest, {len(voronoi_df)} Voronoi")
            
            # Identify demographic variables
            nearest_demo_vars = [col for col in nearest_df.columns if any(x in col for x in 
                               ['rate', 'proportion', 'education', 'class', 'employ', 'housing', 'income'])]
            voronoi_demo_vars = [col for col in voronoi_df.columns if any(x in col for x in 
                                ['rate', 'proportion', 'education', 'class', 'employ', 'housing', 'income'])]
            
            logger.info(f"Found {len(nearest_demo_vars)} demographic variables in nearest dataset")
            logger.info(f"Found {len(voronoi_demo_vars)} demographic variables in voronoi dataset")
            
            # Extract column name mappings
            nearest_club_col = 'nearest_club'
            voronoi_club_col = 'assigned_club_id' if 'assigned_club_id' in voronoi_df.columns else 'assigned_club_name'
            
            # Add column mappings to the data dictionary
            nearest_df.columns_info = {
                'club_id': nearest_club_col,
                'demo_vars': nearest_demo_vars
            }
            
            voronoi_df.columns_info = {
                'club_id': voronoi_club_col,
                'demo_vars': voronoi_demo_vars
            }
            
        except Exception as e:
            logger.warning(f"Could not load demographic data as GeoPackage: {e}")
            logger.warning("Trying to load from CSV files if available...")
            
            nearest_csv = DATA_DIR / 'nearest_demographics.csv'
            voronoi_csv = DATA_DIR / 'voronoi_demographics.csv'
            
            if nearest_csv.exists() and voronoi_csv.exists():
                nearest_df = pd.read_csv(nearest_csv)
                voronoi_df = pd.read_csv(voronoi_csv)
                logger.info(f"Loaded demographic data from CSV: {len(nearest_df)} nearest, {len(voronoi_df)} Voronoi")
            else:
                logger.warning("Could not load demographic data. Proceeding with club data only.")
                nearest_df = None
                voronoi_df = None
        
        return {
            'club_df': club_df,
            'nearest_df': nearest_df,
            'voronoi_df': voronoi_df
        }
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def add_transformed_metrics(df):
    """Add transformed versions of performance metrics for analysis."""
    logger.info("Adding transformed performance metrics...")
    
    # Create transformed versions (higher = better)
    df['transformed_football_performance'] = 7 - df['football_performance']
    df['transformed_hurling_performance'] = 7 - df['hurling_performance']
    df['transformed_overall_performance'] = 7 - df['overall_performance']
    
    # Code balance remains the same (lower is still better)
    df['transformed_code_balance'] = df['code_balance']
    
    # Add categorization for stratification analysis
    df['performance_category'] = pd.cut(
        df['overall_performance'], 
        bins=[0, 1.5, 2.5, 3.5, 4.5, 6],
        labels=['Elite', 'Strong', 'Medium', 'Developing', 'No Grade']
    )
    
    return df

def analyze_urban_rural_differences(data):
    """
    Analyzes performance differences between urban and rural clubs.
    
    Args:
        data: Dictionary containing club data and demographic data
    
    Returns:
        Dictionary of analysis results
    """
    logger.info("Analyzing urban/rural performance differences...")
    club_df = data['club_df']
    nearest_df = data['nearest_df']
    voronoi_df = data['voronoi_df']
    
    results = {}
    
    # Choose which demographic dataset to use
    if nearest_df is not None and hasattr(nearest_df, 'columns_info'):
        demo_df = nearest_df
        club_id_col = demo_df.columns_info['club_id']
        urban_flag_col = next((col for col in demo_df.columns if 'URBAN_AREA_FLAG' in col), None)
        data_source = 'nearest'
        logger.info(f"Using nearest demographic data for urban/rural analysis")
    elif voronoi_df is not None and hasattr(voronoi_df, 'columns_info'):
        demo_df = voronoi_df
        club_id_col = demo_df.columns_info['club_id']
        urban_flag_col = next((col for col in demo_df.columns if 'URBAN_AREA_FLAG' in col), None)
        data_source = 'voronoi'
        logger.info(f"Using voronoi demographic data for urban/rural analysis")
    else:
        logger.warning("No demographic data available. Urban/rural classification not performed.")
        results['error'] = "No urban/rural data available"
        return results
    
    # Check if urban/rural flag is available
    if urban_flag_col is None:
        logger.warning("No urban/rural flag found in demographic data.")
        results['error'] = "No urban/rural data available"
        return results
    
    logger.info(f"Found urban/rural flag column: {urban_flag_col}")
    
    # Aggregate by club and calculate urban ratio
    population_col = next((col for col in demo_df.columns if 'T1_1AGE0T' in col), None)
    
    if population_col is None:
        logger.warning("No population column found. Using simple average for urban ratio.")
        agg_dict = {urban_flag_col: 'mean'}
    else:
        agg_dict = {
            urban_flag_col: 'mean',
            population_col: 'sum'  # Total population
        }
        logger.info(f"Using {population_col} as population weight")
    
    urban_stats = demo_df.groupby(club_id_col).agg(agg_dict).reset_index()
    
    # Rename for clarity
    rename_dict = {club_id_col: 'Club', urban_flag_col: 'urban_ratio'}
    if population_col:
        rename_dict[population_col] = 'total_population'
    
    urban_stats.rename(columns=rename_dict, inplace=True)
    
    # Merge with club data
    merged_df = pd.merge(club_df, urban_stats, on='Club', how='left')
    logger.info(f"Merged data contains {len(merged_df)} clubs")
    
    # Create urban/rural classification based on ratio
    merged_df['is_urban'] = merged_df['urban_ratio'] > 0.5
    
    # Count urban/rural clubs
    urban_count = merged_df['is_urban'].sum()
    rural_count = (~merged_df['is_urban']).sum()
    logger.info(f"Urban/rural classification: {urban_count} urban clubs, {rural_count} rural clubs")
    
    # Calculate performance statistics by urban/rural classification
    urban_perf = merged_df[merged_df['is_urban']].describe()[PERFORMANCE_METRICS + TRANSFORMED_METRICS]
    rural_perf = merged_df[~merged_df['is_urban']].describe()[PERFORMANCE_METRICS + TRANSFORMED_METRICS]
    
    # Perform t-tests to check for significant differences
    ttest_results = {}
    mw_results = {}  # Mann-Whitney U test (non-parametric)
    
    for metric in PERFORMANCE_METRICS + TRANSFORMED_METRICS:
        urban_data = merged_df[merged_df['is_urban']][metric].dropna()
        rural_data = merged_df[~merged_df['is_urban']][metric].dropna()
        
        # Skip if not enough data
        if len(urban_data) < 2 or len(rural_data) < 2:
            logger.warning(f"Not enough data for {metric} in urban/rural comparison")
            continue
            
        # T-test
        t_stat, p_val = stats.ttest_ind(urban_data, rural_data, equal_var=False)
        ttest_results[metric] = {
            'statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'urban_mean': urban_data.mean(),
            'rural_mean': rural_data.mean()
        }
        
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(urban_data, rural_data)
        mw_results[metric] = {
            'statistic': u_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
        
        if p_val < 0.05:
            logger.info(f"Significant urban/rural difference found for {metric}: p={p_val:.4f}")
    
    results['urban_stats'] = urban_perf
    results['rural_stats'] = rural_perf
    results['ttest_results'] = ttest_results
    results['mannwhitney_results'] = mw_results
    results['urban_count'] = urban_count
    results['rural_count'] = rural_count
    results['merged_df'] = merged_df
    results['data_source'] = data_source
    
    return results

def analyze_dual_single_differences(data):
    """
    Analyzes performance differences between dual-code and single-code clubs.
    
    Args:
        data: Dictionary containing club data and demographic data
    
    Returns:
        Dictionary of analysis results
    """
    logger.info("Analyzing dual vs. single-code club differences...")
    club_df = data['club_df']
    
    results = {}
    
    # Separate dual and single-code clubs
    dual_clubs = club_df[club_df['is_dual_2024'] == True]
    single_clubs = club_df[club_df['is_dual_2024'] == False]
    
    # Calculate descriptive statistics
    dual_stats = dual_clubs.describe()[PERFORMANCE_METRICS + TRANSFORMED_METRICS]
    single_stats = single_clubs.describe()[PERFORMANCE_METRICS + TRANSFORMED_METRICS]
    
    # Perform t-tests to check for significant differences
    ttest_results = {}
    mw_results = {}  # Mann-Whitney U test (non-parametric)
    
    for metric in PERFORMANCE_METRICS + TRANSFORMED_METRICS:
        # Skip code balance for single clubs
        if metric in ['code_balance', 'transformed_code_balance'] and len(single_clubs) > 0:
            continue
            
        dual_data = dual_clubs[metric].dropna()
        
        # For single clubs, we need to handle football and hurling metrics differently
        if metric in ['football_performance', 'transformed_football_performance']:
            single_data = single_clubs[single_clubs['football_performance'] < 6][metric].dropna()
        elif metric in ['hurling_performance', 'transformed_hurling_performance']:
            single_data = single_clubs[single_clubs['hurling_performance'] < 6][metric].dropna()
        elif metric in ['overall_performance', 'transformed_overall_performance']:
            # For overall performance, use all single clubs with either football or hurling
            single_data = single_clubs[
                (single_clubs['football_performance'] < 6) | 
                (single_clubs['hurling_performance'] < 6)
            ][metric].dropna()
        else:
            # Skip code balance
            continue
        
        # Skip if not enough data
        if len(dual_data) < 2 or len(single_data) < 2:
            continue
            
        # T-test
        t_stat, p_val = stats.ttest_ind(dual_data, single_data, equal_var=False)
        ttest_results[metric] = {
            'statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'dual_mean': dual_data.mean(),
            'single_mean': single_data.mean()
        }
        
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(dual_data, single_data)
        mw_results[metric] = {
            'statistic': u_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
    
    results['dual_stats'] = dual_stats
    results['single_stats'] = single_stats
    results['ttest_results'] = ttest_results
    results['mannwhitney_results'] = mw_results
    results['dual_count'] = len(dual_clubs)
    results['single_count'] = len(single_clubs)
    results['dual_clubs'] = dual_clubs
    results['single_clubs'] = single_clubs
    
    return results

def analyze_demographic_stratification(data):
    """
    Analyzes performance metrics stratified by demographic variables.
    
    Args:
        data: Dictionary containing club data and demographic data
    
    Returns:
        Dictionary of analysis results
    """
    logger.info("Analyzing demographic stratification...")
    club_df = data['club_df']
    nearest_df = data['nearest_df']
    voronoi_df = data['voronoi_df']
    
    results = {}
    
    # Choose which demographic dataset to use (prefer nearest)
    if nearest_df is not None and hasattr(nearest_df, 'columns_info'):
        demo_df = nearest_df
        club_id_col = demo_df.columns_info['club_id']
        demo_vars = demo_df.columns_info['demo_vars']
        data_source = 'nearest'
        logger.info(f"Using nearest demographic data with {len(demo_vars)} variables")
    elif voronoi_df is not None and hasattr(voronoi_df, 'columns_info'):
        demo_df = voronoi_df
        club_id_col = demo_df.columns_info['club_id']
        demo_vars = demo_df.columns_info['demo_vars']
        data_source = 'voronoi'
        logger.info(f"Using voronoi demographic data with {len(demo_vars)} variables")
    else:
        logger.warning("No demographic data available. Stratification not performed.")
        results['error'] = "No demographic data available"
        return results
    
    # Filter to key demographic variables
    key_prefixes = [
        'basic_education_rate',
        'third_level_rate',
        'employment_rate',
        'professional_rate',
        'working_class_rate',
        'youth_proportion'
    ]
    
    # Find demographic variables that match the key prefixes
    key_demographics = []
    for prefix in key_prefixes:
        matching_vars = [var for var in demo_vars if var.startswith(prefix)]
        if matching_vars:
            key_demographics.append(matching_vars[0])  # Take the first match
    
    if not key_demographics:
        # Try with less strict matching
        for demo_var in demo_vars:
            for prefix in key_prefixes:
                if prefix.split('_')[0] in demo_var and prefix.split('_')[-1] in demo_var:
                    key_demographics.append(demo_var)
                    break
    
    logger.info(f"Found {len(key_demographics)} key demographic variables: {', '.join(key_demographics)}")
    
    if not key_demographics:
        logger.warning("No key demographic variables found. Stratification not performed.")
        results['error'] = "No key demographic variables found"
        return results
    
    # Aggregate demographics by club
    agg_dict = {d: 'mean' for d in key_demographics}
    population_col = next((col for col in demo_df.columns if 'T1_1AGE0T' in col), None)
    
    if population_col:
        agg_dict[population_col] = 'sum'  # Add population sum
        logger.info(f"Using {population_col} as population weight")
    
    club_demographics = demo_df.groupby(club_id_col).agg(agg_dict).reset_index()
    club_demographics.rename(columns={club_id_col: 'Club'}, inplace=True)
    
    logger.info(f"Created club demographics with {len(club_demographics)} clubs")
    
    # Merge with club data
    merged_df = pd.merge(club_df, club_demographics, on='Club', how='left')
    logger.info(f"Merged data contains {len(merged_df)} clubs with {merged_df['Club'].isna().sum()} missing club matches")
    
    # Stratify each demographic into high, medium, and low categories
    strata_results = {}
    
    for demo in key_demographics:
        # Create 3 equal-sized strata
        merged_df[f'{demo}_strata'] = pd.qcut(
            merged_df[demo], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        # Calculate performance statistics by strata
        strata_stats = {}
        for stratum in ['Low', 'Medium', 'High']:
            stratum_data = merged_df[merged_df[f'{demo}_strata'] == stratum]
            if len(stratum_data) > 0:
                strata_stats[stratum] = stratum_data[PERFORMANCE_METRICS + TRANSFORMED_METRICS].describe()
        
        # Perform ANOVA to test for significant differences
        anova_results = {}
        kruskal_results = {}  # Non-parametric alternative to ANOVA
        
        for metric in PERFORMANCE_METRICS + TRANSFORMED_METRICS:
            # Filter data
            if metric in ['football_performance', 'transformed_football_performance']:
                data_low = merged_df[(merged_df[f'{demo}_strata'] == 'Low') & 
                                     (merged_df[metric] < 6)][metric].dropna()
                data_med = merged_df[(merged_df[f'{demo}_strata'] == 'Medium') & 
                                     (merged_df[metric] < 6)][metric].dropna()
                data_high = merged_df[(merged_df[f'{demo}_strata'] == 'High') & 
                                      (merged_df[metric] < 6)][metric].dropna()
            elif metric in ['hurling_performance', 'transformed_hurling_performance']:
                data_low = merged_df[(merged_df[f'{demo}_strata'] == 'Low') & 
                                     (merged_df[metric] < 6)][metric].dropna()
                data_med = merged_df[(merged_df[f'{demo}_strata'] == 'Medium') & 
                                     (merged_df[metric] < 6)][metric].dropna()
                data_high = merged_df[(merged_df[f'{demo}_strata'] == 'High') & 
                                      (merged_df[metric] < 6)][metric].dropna()
            else:
                data_low = merged_df[merged_df[f'{demo}_strata'] == 'Low'][metric].dropna()
                data_med = merged_df[merged_df[f'{demo}_strata'] == 'Medium'][metric].dropna()
                data_high = merged_df[merged_df[f'{demo}_strata'] == 'High'][metric].dropna()
            
            # Skip if not enough data
            if len(data_low) < 2 or len(data_med) < 2 or len(data_high) < 2:
                continue
                
            # ANOVA
            f_stat, p_val = stats.f_oneway(data_low, data_med, data_high)
            anova_results[metric] = {
                'statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            
            # Kruskal-Wallis H test (non-parametric)
            h_stat, p_val = stats.kruskal(data_low, data_med, data_high)
            kruskal_results[metric] = {
                'statistic': h_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
        
        strata_results[demo] = {
            'strata_stats': strata_stats,
            'anova_results': anova_results,
            'kruskal_results': kruskal_results,
            'low_count': sum(merged_df[f'{demo}_strata'] == 'Low'),
            'med_count': sum(merged_df[f'{demo}_strata'] == 'Medium'),
            'high_count': sum(merged_df[f'{demo}_strata'] == 'High'),
        }
    
    results['strata_results'] = strata_results
    results['merged_df'] = merged_df
    results['demographics'] = key_demographics
    results['data_source'] = data_source
    
    return results

def test_normality(data):
    """
    Tests for normality in the distribution of performance metrics.
    
    Args:
        data: Dictionary containing club data
    
    Returns:
        Dictionary of normality test results
    """
    logger.info("Testing for normality in performance metrics...")
    club_df = data['club_df']
    
    results = {}
    
    # Test original and transformed metrics
    shapiro_results = {}
    dagostino_results = {}
    
    for metric in PERFORMANCE_METRICS + TRANSFORMED_METRICS:
        # Filter out NA values (6) for cleaner analysis
        if metric in ['football_performance', 'hurling_performance', 
                     'transformed_football_performance', 'transformed_hurling_performance']:
            
            # Skip NA/6 values
            if metric.startswith('transformed'):
                # For transformed, NA would be 1
                data_filtered = club_df[club_df[metric] != 1][metric].dropna()
            else:
                data_filtered = club_df[club_df[metric] < 6][metric].dropna()
        else:
            data_filtered = club_df[metric].dropna()
        
        # Skip if not enough data
        if len(data_filtered) < 3:
            continue
            
        # Shapiro-Wilk test (better for smaller samples)
        w_stat, p_val = stats.shapiro(data_filtered)
        shapiro_results[metric] = {
            'statistic': w_stat,
            'p_value': p_val,
            'normal': p_val >= 0.05  # Normal if p >= 0.05
        }
        
        # D'Agostino-Pearson test (better for larger samples)
        if len(data_filtered) >= 20:  # Minimum sample size for this test
            k2_stat, p_val = stats.normaltest(data_filtered)
            dagostino_results[metric] = {
                'statistic': k2_stat,
                'p_value': p_val,
                'normal': p_val >= 0.05  # Normal if p >= 0.05
            }
    
    results['shapiro_results'] = shapiro_results
    results['dagostino_results'] = dagostino_results
    
    # Compare original vs. transformed metrics
    normality_comparison = {}
    
    for orig, trans in zip(PERFORMANCE_METRICS, TRANSFORMED_METRICS):
        if orig in shapiro_results and trans in shapiro_results:
            normality_comparison[orig] = {
                'original_p': shapiro_results[orig]['p_value'],
                'transformed_p': shapiro_results[trans]['p_value'],
                'original_normal': shapiro_results[orig]['normal'],
                'transformed_normal': shapiro_results[trans]['normal'],
                'improvement': shapiro_results[trans]['p_value'] > shapiro_results[orig]['p_value']
            }
    
    results['normality_comparison'] = normality_comparison
    
    return results

def generate_urban_rural_visualizations(results):
    """
    Generates visualizations for urban/rural performance differences.
    
    Args:
        results: Dictionary of urban/rural analysis results
    
    Returns:
        List of paths to generated visualizations
    """
    logger.info("Generating urban/rural performance visualizations...")
    visualization_paths = []
    
    if 'error' in results:
        logger.warning(f"Cannot generate visualizations: {results['error']}")
        return visualization_paths
    
    # Create boxplots for performance by urban/rural classification
    merged_df = results.get('merged_df')
    if merged_df is None:
        logger.warning("No merged dataframe available for visualization.")
        return visualization_paths
    
    # Set a common style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Urban/Rural Performance Boxplots (Original Metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    metric_titles = ['Overall Performance', 'Football Performance', 
                    'Hurling Performance', 'Code Balance']
    
    for i, (metric, title) in enumerate(zip(PERFORMANCE_METRICS, metric_titles)):
        # Filter data for hurling/football 
        if metric in ['football_performance', 'hurling_performance']:
            plot_data = merged_df[merged_df[metric] < 6].copy()
        else:
            plot_data = merged_df.copy()
        
        ax = axes[i]
        sns.boxplot(
            x='is_urban', 
            y=metric, 
            data=plot_data,
            palette=['green', 'orange'],
            ax=ax
        )
        
        # Add individual points for more detail
        sns.stripplot(
            x='is_urban', 
            y=metric, 
            data=plot_data,
            color='black', 
            alpha=0.5, 
            jitter=True,
            ax=ax
        )
        
        # Add mean lines
        means = plot_data.groupby('is_urban')[metric].mean()
        for j, is_urban in enumerate([False, True]):
            if is_urban in means.index:
                ax.axhline(
                    means[is_urban], 
                    color=['green', 'orange'][j], 
                    linestyle='--', 
                    alpha=0.8,
                    label=f"Mean ({'Urban' if is_urban else 'Rural'})"
                )
        
        # Check if there's a significant difference
        if metric in results['ttest_results'] and results['ttest_results'][metric]['significant']:
            p_val = results['ttest_results'][metric]['p_value']
            ax.set_title(f"{title} by Urban/Rural Classification\n*Significant Difference (p={p_val:.3f})*")
        else:
            ax.set_title(f"{title} by Urban/Rural Classification")
        
        ax.set_xlabel("")
        ax.set_xticklabels(['Rural', 'Urban'])
        ax.set_ylabel(f"{title} (Lower = Better)")
        ax.legend()
    
    plt.tight_layout()
    urban_rural_boxplot_path = OUTPUT_DIR / 'urban_rural_performance_boxplots.png'
    plt.savefig(urban_rural_boxplot_path, dpi=300)
    plt.close(fig)
    visualization_paths.append(urban_rural_boxplot_path)
    logger.info(f"Saved urban/rural boxplots to {urban_rural_boxplot_path}")
    
    # 2. Urban/Rural Performance Boxplots (Transformed Metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    metric_titles = ['Overall Performance', 'Football Performance', 
                     'Hurling Performance', 'Code Balance']
    
    for i, (metric, title) in enumerate(zip(TRANSFORMED_METRICS, metric_titles)):
        # Filter data for hurling/football 
        if metric in ['transformed_football_performance', 'transformed_hurling_performance']:
            plot_data = merged_df[merged_df[metric] != 1].copy()
        else:
            plot_data = merged_df.copy()
        
        ax = axes[i]
        sns.boxplot(
            x='is_urban', 
            y=metric, 
            data=plot_data,
            palette=['green', 'orange'],
            ax=ax
        )
        
        # Add individual points for more detail
        sns.stripplot(
            x='is_urban', 
            y=metric, 
            data=plot_data,
            color='black', 
            alpha=0.5, 
            jitter=True,
            ax=ax
        )
        
        # Add mean lines
        means = plot_data.groupby('is_urban')[metric].mean()
        for j, is_urban in enumerate([False, True]):
            if is_urban in means.index:
                ax.axhline(
                    means[is_urban], 
                    color=['green', 'orange'][j], 
                    linestyle='--', 
                    alpha=0.8,
                    label=f"Mean ({'Urban' if is_urban else 'Rural'})"
                )
        
        # Check if there's a significant difference
        if metric in results['ttest_results'] and results['ttest_results'][metric]['significant']:
            p_val = results['ttest_results'][metric]['p_value']
            ax.set_title(f"Transformed {title} by Urban/Rural Classification\n*Significant Difference (p={p_val:.3f})*")
        else:
            ax.set_title(f"Transformed {title} by Urban/Rural Classification")
        
        ax.set_xlabel("")
        ax.set_xticklabels(['Rural', 'Urban'])
        ax.set_ylabel(f"Transformed {title} (Higher = Better)")
        ax.legend()
    
    plt.tight_layout()
    urban_rural_transformed_path = OUTPUT_DIR / 'urban_rural_transformed_boxplots.png'
    plt.savefig(urban_rural_transformed_path, dpi=300)
    plt.close(fig)
    visualization_paths.append(urban_rural_transformed_path)
    logger.info(f"Saved urban/rural transformed boxplots to {urban_rural_transformed_path}")
    
    return visualization_paths

def generate_dual_single_visualizations(results):
    """
    Generates visualizations for dual vs. single-code club differences.
    
    Args:
        results: Dictionary of dual/single analysis results
    
    Returns:
        List of paths to generated visualizations
    """
    logger.info("Generating dual vs. single-code club visualizations...")
    visualization_paths = []
    
    dual_clubs = results.get('dual_clubs')
    single_clubs = results.get('single_clubs')
    
    if dual_clubs is None or single_clubs is None:
        logger.warning("No club data available for visualization.")
        return visualization_paths
    
    # Set a common style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Dual vs. Single-code Performance Boxplots (Original Metrics)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['overall_performance', 'football_performance', 'hurling_performance']
    metric_titles = ['Overall Performance', 'Football Performance', 'Hurling Performance']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i]
        
        # Get data for the plot
        dual_data = dual_clubs[metric]
        
        if metric == 'football_performance':
            single_data = single_clubs[single_clubs['football_performance'] < 6][metric]
        elif metric == 'hurling_performance':
            single_data = single_clubs[single_clubs['hurling_performance'] < 6][metric]
        else:
            # For overall performance, use all single clubs with either football or hurling
            single_data = single_clubs[
                (single_clubs['football_performance'] < 6) | 
                (single_clubs['hurling_performance'] < 6)
            ][metric]
        
        # Create a dataframe for plotting
        plot_df = pd.DataFrame({
            'Performance': pd.concat([dual_data, single_data]),
            'Club Type': ['Dual'] * len(dual_data) + ['Single'] * len(single_data)
        })
        
        # Create the boxplot
        sns.boxplot(
            x='Club Type', 
            y='Performance', 
            data=plot_df,
            palette=['purple', 'teal'],
            ax=ax
        )
        
        # Add individual points for more detail
        sns.stripplot(
            x='Club Type', 
            y='Performance', 
            data=plot_df,
            color='black', 
            alpha=0.5, 
            jitter=True,
            ax=ax
        )
        
        # Add mean lines
        means = plot_df.groupby('Club Type')['Performance'].mean()
        for j, club_type in enumerate(['Dual', 'Single']):
            if club_type in means.index:
                ax.axhline(
                    means[club_type], 
                    color=['purple', 'teal'][j], 
                    linestyle='--', 
                    alpha=0.8,
                    label=f"Mean ({club_type})"
                )
        
        # Check if there's a significant difference
        if metric in results['ttest_results'] and results['ttest_results'][metric]['significant']:
            p_val = results['ttest_results'][metric]['p_value']
            ax.set_title(f"{title} by Club Type\n*Significant Difference (p={p_val:.3f})*")
        else:
            ax.set_title(f"{title} by Club Type")
        
        ax.set_xlabel("")
        ax.set_ylabel(f"{title} (Lower = Better)")
        ax.legend()
    
    plt.tight_layout()
    dual_single_boxplot_path = OUTPUT_DIR / 'dual_single_performance_boxplots.png'
    plt.savefig(dual_single_boxplot_path, dpi=300)
    plt.close(fig)
    visualization_paths.append(dual_single_boxplot_path)
    logger.info(f"Saved dual vs. single-code boxplots to {dual_single_boxplot_path}")
    
    # 2. Dual vs. Single-code Transformed Performance Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['transformed_overall_performance', 'transformed_football_performance', 'transformed_hurling_performance']
    metric_titles = ['Overall Performance', 'Football Performance', 'Hurling Performance']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i]
        
        # Get data for the plot
        dual_data = dual_clubs[metric]
        
        if metric == 'transformed_football_performance':
            single_data = single_clubs[single_clubs['transformed_football_performance'] != 1][metric]
        elif metric == 'transformed_hurling_performance':
            single_data = single_clubs[single_clubs['transformed_hurling_performance'] != 1][metric]
        else:
            # For overall performance, use all single clubs with either football or hurling
            single_data = single_clubs[
                (single_clubs['transformed_football_performance'] != 1) | 
                (single_clubs['transformed_hurling_performance'] != 1)
            ][metric]
        
        # Create a dataframe for plotting
        plot_df = pd.DataFrame({
            'Performance': pd.concat([dual_data, single_data]),
            'Club Type': ['Dual'] * len(dual_data) + ['Single'] * len(single_data)
        })
        
        # Create the boxplot
        sns.boxplot(
            x='Club Type', 
            y='Performance', 
            data=plot_df,
            palette=['purple', 'teal'],
            ax=ax
        )
        
        # Add individual points for more detail
        sns.stripplot(
            x='Club Type', 
            y='Performance', 
            data=plot_df,
            color='black', 
            alpha=0.5, 
            jitter=True,
            ax=ax
        )
        
        # Add mean lines
        means = plot_df.groupby('Club Type')['Performance'].mean()
        for j, club_type in enumerate(['Dual', 'Single']):
            if club_type in means.index:
                ax.axhline(
                    means[club_type], 
                    color=['purple', 'teal'][j], 
                    linestyle='--', 
                    alpha=0.8,
                    label=f"Mean ({club_type})"
                )
        
        # Check if there's a significant difference
        if metric in results['ttest_results'] and results['ttest_results'][metric]['significant']:
            p_val = results['ttest_results'][metric]['p_value']
            ax.set_title(f"Transformed {title} by Club Type\n*Significant Difference (p={p_val:.3f})*")
        else:
            ax.set_title(f"Transformed {title} by Club Type")
        
        ax.set_xlabel("")
        ax.set_ylabel(f"Transformed {title} (Higher = Better)")
        ax.legend()
    
    plt.tight_layout()
    dual_single_transformed_path = OUTPUT_DIR / 'dual_single_transformed_boxplots.png'
    plt.savefig(dual_single_transformed_path, dpi=300)
    plt.close(fig)
    visualization_paths.append(dual_single_transformed_path)
    logger.info(f"Saved dual vs. single-code transformed boxplots to {dual_single_transformed_path}")
    
    return visualization_paths

def generate_demographic_stratification_visualizations(results):
    """
    Generates visualizations for demographic stratification analysis.
    
    Args:
        results: Dictionary of demographic stratification analysis results
    
    Returns:
        List of paths to generated visualizations
    """
    logger.info("Generating demographic stratification visualizations...")
    visualization_paths = []
    
    if 'error' in results:
        logger.warning(f"Cannot generate visualizations: {results['error']}")
        return visualization_paths
    
    strata_results = results.get('strata_results')
    merged_df = results.get('merged_df')
    demographics = results.get('demographics')
    
    if strata_results is None or merged_df is None or demographics is None:
        logger.warning("No demographic stratification data available for visualization.")
        return visualization_paths
    
    # Set a common style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create visualizations for each demographic variable
    for demo in demographics:
        # 1. Performance by Demographic Strata (Original Metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (metric, title) in enumerate(zip(PERFORMANCE_METRICS, 
                                                ['Overall Performance', 'Football Performance', 
                                                 'Hurling Performance', 'Code Balance'])):
            ax = axes[i]
            
            # Filter data for hurling/football 
            if metric in ['football_performance', 'hurling_performance']:
                plot_data = merged_df[merged_df[metric] < 6].copy()
            else:
                plot_data = merged_df.copy()
            
            # Create the boxplot
            sns.boxplot(
                x=f'{demo}_strata', 
                y=metric, 
                data=plot_data,
                palette=['green', 'gray', 'orange'],
                ax=ax,
                order=['Low', 'Medium', 'High']
            )
            
            # Add individual points for more detail
            sns.stripplot(
                x=f'{demo}_strata', 
                y=metric, 
                data=plot_data,
                color='black', 
                alpha=0.5, 
                jitter=True,
                ax=ax,
                order=['Low', 'Medium', 'High']
            )
            
            # Check if there's a significant difference
            if metric in strata_results[demo]['anova_results'] and strata_results[demo]['anova_results'][metric]['significant']:
                p_val = strata_results[demo]['anova_results'][metric]['p_value']
                ax.set_title(f"{title} by {demo.replace('_', ' ').title()}\n*Significant Difference (p={p_val:.3f})*")
            else:
                ax.set_title(f"{title} by {demo.replace('_', ' ').title()}")
            
            ax.set_xlabel(f"{demo.replace('_', ' ').title()} Level")
            ax.set_ylabel(f"{title} (Lower = Better)")
        
        plt.tight_layout()
        demo_boxplot_path = OUTPUT_DIR / f'{demo}_performance_boxplots.png'
        plt.savefig(demo_boxplot_path, dpi=300)
        plt.close(fig)
        visualization_paths.append(demo_boxplot_path)
        logger.info(f"Saved {demo} boxplots to {demo_boxplot_path}")
        
        # 2. Performance by Demographic Strata (Transformed Metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (metric, title) in enumerate(zip(TRANSFORMED_METRICS, 
                                                ['Overall Performance', 'Football Performance', 
                                                 'Hurling Performance', 'Code Balance'])):
            ax = axes[i]
            
            # Filter data for hurling/football 
            if metric in ['transformed_football_performance', 'transformed_hurling_performance']:
                plot_data = merged_df[merged_df[metric] != 1].copy()
            else:
                plot_data = merged_df.copy()
            
            # Create the boxplot
            sns.boxplot(
                x=f'{demo}_strata', 
                y=metric, 
                data=plot_data,
                palette=['green', 'gray', 'orange'],
                ax=ax,
                order=['Low', 'Medium', 'High']
            )
            
            # Add individual points for more detail
            sns.stripplot(
                x=f'{demo}_strata', 
                y=metric, 
                data=plot_data,
                color='black', 
                alpha=0.5, 
                jitter=True,
                ax=ax,
                order=['Low', 'Medium', 'High']
            )
            
            # Check if there's a significant difference
            if metric in strata_results[demo]['anova_results'] and strata_results[demo]['anova_results'][metric]['significant']:
                p_val = strata_results[demo]['anova_results'][metric]['p_value']
                ax.set_title(f"Transformed {title} by {demo.replace('_', ' ').title()}\n*Significant Difference (p={p_val:.3f})*")
            else:
                ax.set_title(f"Transformed {title} by {demo.replace('_', ' ').title()}")
            
            ax.set_xlabel(f"{demo.replace('_', ' ').title()} Level")
            ax.set_ylabel(f"Transformed {title} (Higher = Better)")
        
        plt.tight_layout()
        demo_transformed_path = OUTPUT_DIR / f'{demo}_transformed_boxplots.png'
        plt.savefig(demo_transformed_path, dpi=300)
        plt.close(fig)
        visualization_paths.append(demo_transformed_path)
        logger.info(f"Saved {demo} transformed boxplots to {demo_transformed_path}")
    
    return visualization_paths

def generate_normality_visualizations(data, results):
    """
    Generates visualizations for normality test results.
    
    Args:
        data: Dictionary containing club data
        results: Dictionary of normality test results
    
    Returns:
        List of paths to generated visualizations
    """
    logger.info("Generating normality test visualizations...")
    visualization_paths = []
    
    club_df = data['club_df']
    shapiro_results = results.get('shapiro_results')
    normality_comparison = results.get('normality_comparison')
    
    if shapiro_results is None or normality_comparison is None:
        logger.warning("No normality test results available for visualization.")
        return visualization_paths
    
    # Set a common style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. QQ Plots for Original Metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(PERFORMANCE_METRICS, 
                                            ['Overall Performance', 'Football Performance', 
                                             'Hurling Performance', 'Code Balance'])):
        ax = axes[i]
        
        # Filter data
        if metric in ['football_performance', 'hurling_performance']:
            data_filtered = club_df[club_df[metric] < 6][metric].dropna()
        else:
            data_filtered = club_df[metric].dropna()
        
        # Skip if not enough data
        if len(data_filtered) < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', fontsize=14)
            ax.set_title(f"QQ Plot: {title}")
            continue
            
        # Create QQ plot
        stats.probplot(data_filtered, dist="norm", plot=ax)
        
        # Show normality test result
        if metric in shapiro_results:
            p_val = shapiro_results[metric]['p_value']
            is_normal = shapiro_results[metric]['normal']
            ax.set_title(f"QQ Plot: {title}\nShapiro-Wilk p={p_val:.4f} ({'Normal' if is_normal else 'Non-Normal'})")
        else:
            ax.set_title(f"QQ Plot: {title}")
    
    plt.tight_layout()
    qq_plot_path = OUTPUT_DIR / 'normality_qq_plots.png'
    plt.savefig(qq_plot_path, dpi=300)
    plt.close(fig)
    visualization_paths.append(qq_plot_path)
    logger.info(f"Saved normality QQ plots to {qq_plot_path}")
    
    # 2. QQ Plots for Transformed Metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(TRANSFORMED_METRICS, 
                                            ['Overall Performance', 'Football Performance', 
                                             'Hurling Performance', 'Code Balance'])):
        ax = axes[i]
        
        # Filter data
        if metric in ['transformed_football_performance', 'transformed_hurling_performance']:
            data_filtered = club_df[club_df[metric] != 1][metric].dropna()
        else:
            data_filtered = club_df[metric].dropna()
        
        # Skip if not enough data
        if len(data_filtered) < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', fontsize=14)
            ax.set_title(f"QQ Plot: Transformed {title}")
            continue
            
        # Create QQ plot
        stats.probplot(data_filtered, dist="norm", plot=ax)
        
        # Show normality test result
        if metric in shapiro_results:
            p_val = shapiro_results[metric]['p_value']
            is_normal = shapiro_results[metric]['normal']
            ax.set_title(f"QQ Plot: Transformed {title}\nShapiro-Wilk p={p_val:.4f} ({'Normal' if is_normal else 'Non-Normal'})")
        else:
            ax.set_title(f"QQ Plot: Transformed {title}")
    
    plt.tight_layout()
    transformed_qq_path = OUTPUT_DIR / 'transformed_normality_qq_plots.png'
    plt.savefig(transformed_qq_path, dpi=300)
    plt.close(fig)
    visualization_paths.append(transformed_qq_path)
    logger.info(f"Saved transformed normality QQ plots to {transformed_qq_path}")
    
    # 3. Normality Comparison Chart
    metrics = list(normality_comparison.keys())
    
    if metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create data for the plot
        metric_names = [m.replace('_performance', '').replace('_', ' ').title() for m in metrics]
        orig_p_values = [normality_comparison[m]['original_p'] for m in metrics]
        trans_p_values = [normality_comparison[m]['transformed_p'] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create the bars
        ax.bar(x - width/2, orig_p_values, width, label='Original Metrics', color='blue', alpha=0.7)
        ax.bar(x + width/2, trans_p_values, width, label='Transformed Metrics', color='orange', alpha=0.7)
        
        # Add significance threshold line
        ax.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold (p=0.05)')
        
        # Customize the chart
        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('p-value (Shapiro-Wilk Test)')
        ax.set_title('Normality Test Comparison: Original vs. Transformed Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        
        # Add text indicating normal/non-normal
        for i, p in enumerate(orig_p_values):
            text = "Normal" if p >= 0.05 else "Non-Normal"
            ax.text(i - width/2, p + 0.01, text, ha='center', va='bottom', fontsize=9, rotation=90)
        
        for i, p in enumerate(trans_p_values):
            text = "Normal" if p >= 0.05 else "Non-Normal"
            ax.text(i + width/2, p + 0.01, text, ha='center', va='bottom', fontsize=9, rotation=90)
        
        plt.tight_layout()
        comparison_path = OUTPUT_DIR / 'normality_comparison_chart.png'
        plt.savefig(comparison_path, dpi=300)
        plt.close(fig)
        visualization_paths.append(comparison_path)
        logger.info(f"Saved normality comparison chart to {comparison_path}")
    
    return visualization_paths 

def generate_report(data, urban_rural_results, dual_single_results, demo_strat_results, normality_results,
                  urban_rural_viz_paths, dual_single_viz_paths, demo_viz_paths, normality_viz_paths):
    """
    Generate a comprehensive report of the distribution analysis findings.
    
    Args:
        data: Dictionary containing data frames
        urban_rural_results: Results from urban/rural analysis
        dual_single_results: Results from dual/single-code club analysis
        demo_strat_results: Results from demographic stratification analysis
        normality_results: Results from normality tests
        urban_rural_viz_paths: Paths to urban/rural visualizations
        dual_single_viz_paths: Paths to dual/single visualizations
        demo_viz_paths: Paths to demographic visualizations
        normality_viz_paths: Paths to normality test visualizations
        
    Returns:
        Path to the generated report
    """
    logger.info("Generating distribution analysis report...")
    
    report_path = REPORTS_DIR / 'performance_distribution_analysis.md'
    
    with open(report_path, 'w') as f:
        f.write("# Performance Metrics Distribution Analysis\n\n")
        f.write("This report analyzes the distribution of performance metrics across various factors.\n\n")
        
        # Urban/Rural Analysis
        f.write("## Urban vs. Rural Club Performance\n\n")
        
        if 'error' not in urban_rural_results:
            urban_count = urban_rural_results.get('urban_count', 0)
            rural_count = urban_rural_results.get('rural_count', 0)
            total = urban_count + rural_count
            
            f.write(f"Analysis of {total} clubs ({urban_count} urban, {rural_count} rural).\n\n")
            
            # Significant findings
            f.write("### Significant Differences\n\n")
            
            significant_metrics = [m for m in PERFORMANCE_METRICS + TRANSFORMED_METRICS 
                                 if m in urban_rural_results.get('ttest_results', {}) and 
                                 urban_rural_results['ttest_results'][m]['significant']]
            
            if significant_metrics:
                f.write("The following metrics showed statistically significant differences between urban and rural clubs:\n\n")
                
                for metric in significant_metrics:
                    pretty_name = metric.replace('_performance', '').replace('transformed_', 'transformed ').replace('_', ' ').title()
                    t_stat = urban_rural_results['ttest_results'][metric]['statistic']
                    p_val = urban_rural_results['ttest_results'][metric]['p_value']
                    
                    f.write(f"- **{pretty_name}**: t={t_stat:.2f}, p={p_val:.4f}\n")
                
                f.write("\n")
            else:
                f.write("No statistically significant differences were found between urban and rural clubs.\n\n")
            
            # Visualizations
            if urban_rural_viz_paths:
                f.write("### Visualizations\n\n")
                
                for path in urban_rural_viz_paths:
                    rel_path = os.path.relpath(path, BASE_DIR)
                    f.write(f"- [{os.path.basename(path)}]({rel_path})\n")
                
                f.write("\n")
        else:
            f.write(f"Urban/Rural analysis could not be performed: {urban_rural_results['error']}\n\n")
        
        # Dual/Single Analysis
        f.write("## Dual vs. Single-Code Club Performance\n\n")
        
        dual_count = dual_single_results.get('dual_count', 0)
        single_count = dual_single_results.get('single_count', 0)
        total = dual_count + single_count
        
        f.write(f"Analysis of {total} clubs ({dual_count} dual-code, {single_count} single-code).\n\n")
        
        # Significant findings
        f.write("### Significant Differences\n\n")
        
        significant_metrics = [m for m in PERFORMANCE_METRICS + TRANSFORMED_METRICS 
                             if m in dual_single_results.get('ttest_results', {}) and 
                             dual_single_results['ttest_results'][m]['significant']]
        
        if significant_metrics:
            f.write("The following metrics showed statistically significant differences between dual-code and single-code clubs:\n\n")
            
            for metric in significant_metrics:
                pretty_name = metric.replace('_performance', '').replace('transformed_', 'transformed ').replace('_', ' ').title()
                t_stat = dual_single_results['ttest_results'][metric]['statistic']
                p_val = dual_single_results['ttest_results'][metric]['p_value']
                dual_mean = dual_single_results['ttest_results'][metric].get('dual_mean', 'N/A')
                single_mean = dual_single_results['ttest_results'][metric].get('single_mean', 'N/A')
                
                mean_diff = ""
                if isinstance(dual_mean, (int, float)) and isinstance(single_mean, (int, float)):
                    if 'transformed' in metric:
                        better = "dual-code" if dual_mean > single_mean else "single-code"
                    else:
                        better = "dual-code" if dual_mean < single_mean else "single-code"
                    mean_diff = f" (better performance by {better} clubs)"
                
                f.write(f"- **{pretty_name}**: t={t_stat:.2f}, p={p_val:.4f}{mean_diff}\n")
            
            f.write("\n")
        else:
            f.write("No statistically significant differences were found between dual-code and single-code clubs.\n\n")
        
        # Visualizations
        if dual_single_viz_paths:
            f.write("### Visualizations\n\n")
            
            for path in dual_single_viz_paths:
                rel_path = os.path.relpath(path, BASE_DIR)
                f.write(f"- [{os.path.basename(path)}]({rel_path})\n")
            
            f.write("\n")
        
        # Demographic Stratification
        f.write("## Performance by Demographic Factors\n\n")
        
        if 'error' not in demo_strat_results:
            demographics = demo_strat_results.get('demographics', [])
            
            if demographics:
                f.write(f"Analysis of performance metrics stratified by {len(demographics)} demographic variables.\n\n")
                f.write("### Significant Differences\n\n")
                
                significant_found = False
                
                for demo in demographics:
                    strata_results = demo_strat_results.get('strata_results', {}).get(demo, {})
                    anova_results = strata_results.get('anova_results', {})
                    
                    significant_metrics = [m for m in PERFORMANCE_METRICS + TRANSFORMED_METRICS 
                                         if m in anova_results and anova_results[m]['significant']]
                    
                    if significant_metrics:
                        significant_found = True
                        pretty_demo = demo.replace('_', ' ').title()
                        f.write(f"#### {pretty_demo}\n\n")
                        
                        for metric in significant_metrics:
                            pretty_name = metric.replace('_performance', '').replace('transformed_', 'transformed ').replace('_', ' ').title()
                            f_stat = anova_results[metric]['statistic']
                            p_val = anova_results[metric]['p_value']
                            
                            f.write(f"- **{pretty_name}**: F={f_stat:.2f}, p={p_val:.4f}\n")
                        
                        f.write("\n")
                
                if not significant_found:
                    f.write("No statistically significant differences were found across demographic strata.\n\n")
                
                # Visualizations
                if demo_viz_paths:
                    f.write("### Visualizations\n\n")
                    
                    for path in demo_viz_paths:
                        rel_path = os.path.relpath(path, BASE_DIR)
                        f.write(f"- [{os.path.basename(path)}]({rel_path})\n")
                    
                    f.write("\n")
            else:
                f.write("No demographic variables were available for stratification analysis.\n\n")
        else:
            f.write(f"Demographic stratification analysis could not be performed: {demo_strat_results['error']}\n\n")
        
        # Normality Tests
        f.write("## Normality of Performance Metrics\n\n")
        
        shapiro_results = normality_results.get('shapiro_results', {})
        normality_comparison = normality_results.get('normality_comparison', {})
        
        if shapiro_results:
            f.write("### Shapiro-Wilk Test Results\n\n")
            f.write("| Metric | p-value | Normal Distribution? |\n")
            f.write("| ------ | ------- | ------------------- |\n")
            
            for metric in PERFORMANCE_METRICS + TRANSFORMED_METRICS:
                if metric in shapiro_results:
                    pretty_name = metric.replace('_performance', '').replace('transformed_', 'transformed ').replace('_', ' ').title()
                    p_val = shapiro_results[metric]['p_value']
                    is_normal = shapiro_results[metric]['normal']
                    
                    f.write(f"| {pretty_name} | {p_val:.4f} | {'Yes' if is_normal else 'No'} |\n")
            
            f.write("\n")
        
        if normality_comparison:
            f.write("### Original vs. Transformed Metrics\n\n")
            f.write("| Metric | Original p-value | Transformed p-value | Improvement? |\n")
            f.write("| ------ | --------------- | ------------------- | ------------ |\n")
            
            for orig_metric, comparison in normality_comparison.items():
                pretty_name = orig_metric.replace('_performance', '').replace('_', ' ').title()
                orig_p = comparison['original_p']
                trans_p = comparison['transformed_p']
                improvement = comparison['improvement']
                
                f.write(f"| {pretty_name} | {orig_p:.4f} | {trans_p:.4f} | {'Yes' if improvement else 'No'} |\n")
            
            f.write("\n")
        
        # Visualizations
        if normality_viz_paths:
            f.write("### Visualizations\n\n")
            
            for path in normality_viz_paths:
                rel_path = os.path.relpath(path, BASE_DIR)
                f.write(f"- [{os.path.basename(path)}]({rel_path})\n")
            
            f.write("\n")
        
        # Summary of Key Findings
        f.write("## Key Findings\n\n")
        
        # Urban/Rural findings
        urban_rural_sig = [m for m in PERFORMANCE_METRICS + TRANSFORMED_METRICS 
                         if m in urban_rural_results.get('ttest_results', {}) and 
                         urban_rural_results['ttest_results'][m]['significant']]
        
        # Dual/Single findings
        dual_single_sig = [m for m in PERFORMANCE_METRICS + TRANSFORMED_METRICS 
                         if m in dual_single_results.get('ttest_results', {}) and 
                         dual_single_results['ttest_results'][m]['significant']]
        
        # Demographic findings
        demo_sig = {}
        if 'error' not in demo_strat_results:
            demographics = demo_strat_results.get('demographics', [])
            for demo in demographics:
                strata_results = demo_strat_results.get('strata_results', {}).get(demo, {})
                anova_results = strata_results.get('anova_results', {})
                
                sig_metrics = [m for m in PERFORMANCE_METRICS + TRANSFORMED_METRICS 
                             if m in anova_results and anova_results[m]['significant']]
                
                if sig_metrics:
                    demo_sig[demo] = sig_metrics
        
        # Normality findings
        normal_metrics = [m for m in PERFORMANCE_METRICS + TRANSFORMED_METRICS 
                        if m in shapiro_results and shapiro_results[m]['normal']]
        
        transformed_improvement = [orig for orig, comp in normality_comparison.items() 
                                 if comp['improvement']]
        
        # Write findings
        f.write("1. **Urban vs. Rural Performance**: ")
        if urban_rural_sig:
            pretty_metrics = [m.replace('_performance', '').replace('transformed_', 'transformed ').replace('_', ' ').title() 
                            for m in urban_rural_sig]
            f.write(f"Significant differences found in {', '.join(pretty_metrics)}.\n")
        else:
            f.write("No significant differences found between urban and rural clubs.\n")
        
        f.write("2. **Dual vs. Single-Code Performance**: ")
        if dual_single_sig:
            pretty_metrics = [m.replace('_performance', '').replace('transformed_', 'transformed ').replace('_', ' ').title() 
                            for m in dual_single_sig]
            f.write(f"Significant differences found in {', '.join(pretty_metrics)}.\n")
        else:
            f.write("No significant differences found between dual-code and single-code clubs.\n")
        
        f.write("3. **Demographic Stratification**: ")
        if demo_sig:
            demo_list = []
            for demo, metrics in demo_sig.items():
                pretty_demo = demo.replace('_', ' ').title()
                pretty_metrics = [m.replace('_performance', '').replace('transformed_', 'transformed ').replace('_', ' ').title() 
                                for m in metrics]
                demo_list.append(f"{pretty_demo} ({', '.join(pretty_metrics)})")
            
            f.write(f"Significant differences found for: {'; '.join(demo_list)}.\n")
        else:
            f.write("No significant differences found across demographic strata.\n")
        
        f.write("4. **Normality of Distributions**: ")
        if normal_metrics:
            pretty_metrics = [m.replace('_performance', '').replace('transformed_', 'transformed ').replace('_', ' ').title() 
                            for m in normal_metrics]
            f.write(f"Normally distributed metrics: {', '.join(pretty_metrics)}. ")
        else:
            f.write("No metrics follow a normal distribution. ")
        
        if transformed_improvement:
            pretty_metrics = [m.replace('_performance', '').replace('_', ' ').title() 
                            for m in transformed_improvement]
            f.write(f"Transformation improved normality for: {', '.join(pretty_metrics)}.\n")
        else:
            f.write("Transformations did not improve normality for any metrics.\n")
    
    logger.info(f"Distribution analysis report saved to {report_path}")
    return report_path

def main():
    """Run the distribution analysis."""
    logger.info("Starting distribution analysis...")
    
    # Load data
    data = load_data()
    
    # Analyze urban/rural differences
    urban_rural_results = analyze_urban_rural_differences(data)
    
    # Analyze dual/single-code differences
    dual_single_results = analyze_dual_single_differences(data)
    
    # Analyze demographic stratification
    demo_strat_results = analyze_demographic_stratification(data)
    
    # Test normality
    normality_results = test_normality(data)
    
    # Generate visualizations
    urban_rural_viz_paths = generate_urban_rural_visualizations(urban_rural_results)
    dual_single_viz_paths = generate_dual_single_visualizations(dual_single_results)
    demo_viz_paths = generate_demographic_stratification_visualizations(demo_strat_results)
    normality_viz_paths = generate_normality_visualizations(data, normality_results)
    
    # Generate report
    report_path = generate_report(
        data,
        urban_rural_results,
        dual_single_results,
        demo_strat_results,
        normality_results,
        urban_rural_viz_paths,
        dual_single_viz_paths,
        demo_viz_paths,
        normality_viz_paths
    )
    
    logger.info(f"Distribution analysis completed. Report saved to {report_path}")
    
    # Return paths for reference
    return {
        'report': report_path,
        'visualizations': {
            'urban_rural': urban_rural_viz_paths,
            'dual_single': dual_single_viz_paths,
            'demographic': demo_viz_paths,
            'normality': normality_viz_paths
        }
    }

if __name__ == "__main__":
    main() 