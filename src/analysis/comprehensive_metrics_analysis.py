#!/usr/bin/env python3
"""
Comprehensive Metrics Analysis for Cork GAA Club Performance.

This script analyzes relationships between all metrics in the consolidated metrics file,
performs bivariate analysis, calculates correlation matrices, generates descriptive
statistics by club type, analyzes performance stability over time, creates trend
visualizations, identifies clubs with significant performance changes, develops
a typology of performance patterns, and documents key insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime

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
OUTPUT_DIR = BASE_DIR / 'output' / 'analysis'
REPORTS_DIR = BASE_DIR / 'reports'

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'metrics').mkdir(parents=True, exist_ok=True)
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

# Time-based metrics for trend analysis
TIME_METRICS = [
    'football_improvement',
    'hurling_improvement',
    'Grade_2022_football_value', 
    'Grade_2022_hurling_value',
    'Grade_2024_football_value', 
    'Grade_2024_hurling_value'
]

# Custom color palettes
CORRELATION_CMAP = sns.diverging_palette(230, 20, as_cmap=True)
PERFORMANCE_CMAP = sns.color_palette("RdYlGn", as_cmap=True)
CLUB_TYPE_COLORS = {'Dual': 'purple', 'Single': 'teal'}
TREND_COLORS = {'Improvement': 'green', 'Decline': 'red', 'Stable': 'gray'}

def load_data():
    """Load club data with all metrics."""
    logger.info("Loading club data...")
    
    try:
        # Load graded club data
        club_filepath = DATA_DIR / 'cork_clubs_complete_graded.csv'
        if not club_filepath.exists():
            logger.error(f"Club data file not found: {club_filepath}")
            raise FileNotFoundError(f"Club data file not found: {club_filepath}")
            
        club_df = pd.read_csv(club_filepath)
        logger.info(f"Loaded club data for {len(club_df)} clubs")
        
        # Add transformed metrics if they don't exist
        if 'transformed_overall_performance' not in club_df.columns:
            club_df['transformed_overall_performance'] = 7 - club_df['overall_performance']
            club_df['transformed_football_performance'] = 7 - club_df['football_performance']
            club_df['transformed_hurling_performance'] = 7 - club_df['hurling_performance']
            club_df['transformed_code_balance'] = 7 - club_df['code_balance']
            logger.info("Added transformed metrics to club data")
        
        # Create club type column (dual vs single-code) based on 2024
        club_df['club_type'] = club_df['is_dual_2024'].apply(lambda x: 'Dual' if x else 'Single')
        
        # Create performance change category
        club_df['overall_change'] = club_df.apply(
            lambda row: 'Improvement' if (row['football_improvement'] < 0 and row['hurling_improvement'] < 0) else
                        'Decline' if (row['football_improvement'] > 0 and row['hurling_improvement'] > 0) else
                        'Mixed' if (row['football_improvement'] != 0 or row['hurling_improvement'] != 0) else
                        'Stable',
            axis=1
        )
        
        # Create absolute performance change magnitude
        club_df['change_magnitude'] = club_df.apply(
            lambda row: np.sqrt((row['football_improvement'] ** 2) + (row['hurling_improvement'] ** 2)),
            axis=1
        )
        
        logger.info("Added derived metrics to club data")
        
        return club_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def check_data_completeness(club_df):
    """Check data completeness and report metrics."""
    logger.info("Checking data completeness...")
    
    # Check for missing values
    missing_data = club_df.isnull().sum()
    missing_pct = (missing_data / len(club_df)) * 100
    
    # Check for NA grades (value of 6)
    na_grades = {
        'football_2022': sum(club_df['Grade_2022_football_value'] == 6),
        'hurling_2022': sum(club_df['Grade_2022_hurling_value'] == 6),
        'football_2024': sum(club_df['Grade_2024_football_value'] == 6),
        'hurling_2024': sum(club_df['Grade_2024_hurling_value'] == 6)
    }
    
    # Count clubs by type
    club_types = club_df['club_type'].value_counts()
    
    # Count clubs by performance change category
    performance_changes = club_df['overall_change'].value_counts()
    
    # Report results
    logger.info(f"Missing data analysis:")
    for col in missing_data[missing_data > 0].index:
        logger.info(f"  - {col}: {missing_data[col]} missing values ({missing_pct[col]:.2f}%)")
    
    logger.info(f"NA grades analysis:")
    for grade, count in na_grades.items():
        logger.info(f"  - {grade}: {count} clubs with NA grade ({(count/len(club_df))*100:.2f}%)")
    
    logger.info(f"Club types:")
    for club_type, count in club_types.items():
        logger.info(f"  - {club_type}: {count} clubs ({(count/len(club_df))*100:.2f}%)")
    
    logger.info(f"Performance change categories:")
    for change, count in performance_changes.items():
        logger.info(f"  - {change}: {count} clubs ({(count/len(club_df))*100:.2f}%)")
    
    # Calculate completeness metrics
    metrics = {
        'total_clubs': len(club_df),
        'complete_records': len(club_df.dropna()),
        'completeness_ratio': len(club_df.dropna()) / len(club_df),
        'na_grades': na_grades,
        'club_types': dict(club_types),
        'performance_changes': dict(performance_changes)
    }
    
    return metrics

def generate_descriptive_statistics(club_df):
    """Generate comprehensive descriptive statistics for all metrics."""
    logger.info("Generating descriptive statistics...")
    
    # Define all metrics to analyze
    all_metrics = PERFORMANCE_METRICS + TRANSFORMED_METRICS + TIME_METRICS + ['change_magnitude']
    
    # Generate overall statistics
    overall_stats = club_df[all_metrics].describe().transpose()
    
    # Add skewness and kurtosis
    overall_stats['skewness'] = club_df[all_metrics].skew()
    overall_stats['kurtosis'] = club_df[all_metrics].kurt()
    
    # Generate statistics by club type
    dual_stats = club_df[club_df['club_type'] == 'Dual'][all_metrics].describe().transpose()
    single_stats = club_df[club_df['club_type'] == 'Single'][all_metrics].describe().transpose()
    
    # Add type-specific skewness and kurtosis
    dual_stats['skewness'] = club_df[club_df['club_type'] == 'Dual'][all_metrics].skew()
    dual_stats['kurtosis'] = club_df[club_df['club_type'] == 'Dual'][all_metrics].kurt()
    single_stats['skewness'] = club_df[club_df['club_type'] == 'Single'][all_metrics].skew()
    single_stats['kurtosis'] = club_df[club_df['club_type'] == 'Single'][all_metrics].kurt()
    
    # Generate statistics by performance change category
    change_stats = {}
    for category in club_df['overall_change'].unique():
        category_df = club_df[club_df['overall_change'] == category]
        change_stats[category] = category_df[all_metrics].describe().transpose()
        change_stats[category]['skewness'] = category_df[all_metrics].skew()
        change_stats[category]['kurtosis'] = category_df[all_metrics].kurt()
    
    # Create statistical tests between dual and single clubs
    ttest_results = {}
    for metric in all_metrics:
        # For football/hurling metrics, filter out clubs with no grade (value 6 or transformed value 1)
        if 'football' in metric.lower():
            dual_data = club_df[(club_df['club_type'] == 'Dual')]
            if 'transformed' in metric.lower():
                dual_data = dual_data[dual_data[metric] != 1]
                single_data = club_df[(club_df['club_type'] == 'Single') & 
                                     (club_df[metric] != 1)]
            else:
                dual_data = dual_data[dual_data[metric] < 6]
                single_data = club_df[(club_df['club_type'] == 'Single') & 
                                     (club_df[metric] < 6)]
            dual_values = dual_data[metric]
            single_values = single_data[metric]
        elif 'hurling' in metric.lower():
            dual_data = club_df[(club_df['club_type'] == 'Dual')]
            if 'transformed' in metric.lower():
                dual_data = dual_data[dual_data[metric] != 1]
                single_data = club_df[(club_df['club_type'] == 'Single') & 
                                     (club_df[metric] != 1)]
            else:
                dual_data = dual_data[dual_data[metric] < 6]
                single_data = club_df[(club_df['club_type'] == 'Single') & 
                                     (club_df[metric] < 6)]
            dual_values = dual_data[metric]
            single_values = single_data[metric]
        else:
            dual_values = club_df[club_df['club_type'] == 'Dual'][metric]
            single_values = club_df[club_df['club_type'] == 'Single'][metric]
        
        # Skip if not enough data
        if len(dual_values) < 2 or len(single_values) < 2:
            continue
            
        # Perform t-test
        tstat, pval = stats.ttest_ind(dual_values, single_values, equal_var=False)
        
        # Store results
        ttest_results[metric] = {
            'dual_mean': dual_values.mean(),
            'single_mean': single_values.mean(),
            'difference': dual_values.mean() - single_values.mean(),
            'percent_difference': ((dual_values.mean() - single_values.mean()) / single_values.mean()) * 100,
            't_statistic': tstat,
            'p_value': pval,
            'significant': pval < 0.05
        }
    
    stats_output = {
        'overall': overall_stats,
        'dual': dual_stats,
        'single': single_stats,
        'by_change': change_stats,
        'dual_vs_single': ttest_results
    }
    
    logger.info("Descriptive statistics generated successfully")
    return stats_output 

def calculate_correlation_matrices(club_df):
    """Calculate correlation matrices for all metrics."""
    logger.info("Calculating correlation matrices...")
    
    # Define groups of metrics for correlation analysis
    performance_group = PERFORMANCE_METRICS + TRANSFORMED_METRICS
    time_group = TIME_METRICS + ['change_magnitude']
    all_metrics = performance_group + time_group
    
    # Calculate correlation matrices
    correlation_matrices = {}
    
    # All metrics correlation
    correlation_matrices['all'] = club_df[all_metrics].corr()
    
    # Original vs transformed metrics
    correlation_matrices['original_vs_transformed'] = pd.DataFrame(
        np.corrcoef(
            [club_df[m].values for m in PERFORMANCE_METRICS] + 
            [club_df[m].values for m in TRANSFORMED_METRICS]
        ),
        index=PERFORMANCE_METRICS + TRANSFORMED_METRICS,
        columns=PERFORMANCE_METRICS + TRANSFORMED_METRICS
    )
    
    # Time metrics correlation
    correlation_matrices['time'] = club_df[time_group].corr()
    
    # Create subgroup correlation matrices
    # For football clubs only (excluding clubs with no football grade)
    football_clubs = club_df[club_df['Grade_2024_football_value'] < 6]
    correlation_matrices['football'] = football_clubs[performance_group + ['Grade_2024_football_value', 'football_improvement']].corr()
    
    # For hurling clubs only (excluding clubs with no hurling grade)
    hurling_clubs = club_df[club_df['Grade_2024_hurling_value'] < 6]
    correlation_matrices['hurling'] = hurling_clubs[performance_group + ['Grade_2024_hurling_value', 'hurling_improvement']].corr()
    
    # For dual clubs only
    dual_clubs = club_df[club_df['club_type'] == 'Dual']
    correlation_matrices['dual'] = dual_clubs[all_metrics].corr()
    
    # For single-code clubs only
    single_clubs = club_df[club_df['club_type'] == 'Single']
    correlation_matrices['single'] = single_clubs[all_metrics].corr()
    
    # Calculate p-values for correlations
    pvalue_matrices = {}
    
    for matrix_name, corr_matrix in correlation_matrices.items():
        pvals = np.zeros_like(corr_matrix, dtype=float)
        
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if i != j:
                    # Get column names
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    
                    # Get data based on the matrix name
                    if matrix_name == 'football':
                        x = football_clubs[col_i].values
                        y = football_clubs[col_j].values
                    elif matrix_name == 'hurling':
                        x = hurling_clubs[col_i].values
                        y = hurling_clubs[col_j].values
                    elif matrix_name == 'dual':
                        x = dual_clubs[col_i].values
                        y = dual_clubs[col_j].values
                    elif matrix_name == 'single':
                        x = single_clubs[col_i].values
                        y = single_clubs[col_j].values
                    else:
                        x = club_df[col_i].values
                        y = club_df[col_j].values
                    
                    # Remove NaN values
                    valid = ~(np.isnan(x) | np.isnan(y))
                    x = x[valid]
                    y = y[valid]
                    
                    # Calculate p-value if enough data
                    if len(x) > 2:
                        _, p = stats.pearsonr(x, y)
                        pvals[i, j] = p
                    else:
                        pvals[i, j] = np.nan
                else:
                    pvals[i, j] = 0.0  # Diagonal elements
        
        # Create DataFrame for p-values
        pvalue_matrices[matrix_name] = pd.DataFrame(
            pvals,
            index=corr_matrix.index,
            columns=corr_matrix.columns
        )
    
    # Identify significant correlations
    significant_correlations = {}
    
    for matrix_name, corr_matrix in correlation_matrices.items():
        pval_matrix = pvalue_matrices[matrix_name]
        
        # Create mask of significant correlations (p < 0.05)
        sig_mask = (pval_matrix < 0.05) & (corr_matrix.abs() > 0.3)
        
        # Get significant correlations
        sig_pairs = []
        
        for i in range(sig_mask.shape[0]):
            for j in range(i+1, sig_mask.shape[1]):
                if sig_mask.iloc[i, j]:
                    row = sig_mask.index[i]
                    col = sig_mask.columns[j]
                    corr = corr_matrix.iloc[i, j]
                    pval = pval_matrix.iloc[i, j]
                    
                    sig_pairs.append({
                        'variable_1': row,
                        'variable_2': col,
                        'correlation': corr,
                        'p_value': pval,
                        'strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.5 else 'Weak'
                    })
        
        # Sort by absolute correlation
        sig_pairs = sorted(sig_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        significant_correlations[matrix_name] = sig_pairs
    
    # Prepare output
    correlation_output = {
        'matrices': correlation_matrices,
        'p_values': pvalue_matrices,
        'significant': significant_correlations
    }
    
    logger.info("Correlation matrices calculated successfully")
    return correlation_output

def perform_bivariate_analysis(club_df):
    """Perform bivariate analysis between all performance metrics combinations."""
    logger.info("Performing bivariate analysis...")
    
    # Select metrics for bivariate analysis
    bivariate_metrics = PERFORMANCE_METRICS + TRANSFORMED_METRICS
    
    # Prepare results storage
    bivariate_results = {}
    
    # Analyze each pair of metrics
    for i in range(len(bivariate_metrics)):
        for j in range(i+1, len(bivariate_metrics)):
            metric1 = bivariate_metrics[i]
            metric2 = bivariate_metrics[j]
            
            # Filter out clubs with no grades (6 or 1 for transformed) if football/hurling metrics
            if 'football' in metric1.lower() or 'football' in metric2.lower():
                if 'transformed' in metric1.lower() and 'football' in metric1.lower():
                    temp_df = club_df[club_df[metric1] != 1]
                elif 'football' in metric1.lower():
                    temp_df = club_df[club_df[metric1] < 6]
                else:
                    temp_df = club_df
                    
                if 'transformed' in metric2.lower() and 'football' in metric2.lower():
                    temp_df = temp_df[temp_df[metric2] != 1]
                elif 'football' in metric2.lower():
                    temp_df = temp_df[temp_df[metric2] < 6]
            elif 'hurling' in metric1.lower() or 'hurling' in metric2.lower():
                if 'transformed' in metric1.lower() and 'hurling' in metric1.lower():
                    temp_df = club_df[club_df[metric1] != 1]
                elif 'hurling' in metric1.lower():
                    temp_df = club_df[club_df[metric1] < 6]
                else:
                    temp_df = club_df
                    
                if 'transformed' in metric2.lower() and 'hurling' in metric2.lower():
                    temp_df = temp_df[temp_df[metric2] != 1]
                elif 'hurling' in metric2.lower():
                    temp_df = temp_df[temp_df[metric2] < 6]
            else:
                temp_df = club_df
                
            # Calculate correlation and statistical significance
            if len(temp_df) > 2:
                x = temp_df[metric1]
                y = temp_df[metric2]
                
                corr, p_value = stats.pearsonr(x, y)
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Store results
                bivariate_results[f"{metric1}_vs_{metric2}"] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'std_error': std_err,
                    'significant': p_value < 0.05,
                    'n_clubs': len(temp_df)
                }
    
    # Identify strongest relationships
    strongest_relationships = sorted(
        bivariate_results.items(), 
        key=lambda item: abs(item[1]['correlation']), 
        reverse=True
    )
    
    # Store top relationships
    top_relationships = {
        'strongest_positive': [pair for pair in strongest_relationships if pair[1]['correlation'] > 0][:5],
        'strongest_negative': [pair for pair in strongest_relationships if pair[1]['correlation'] < 0][:5]
    }
    
    # Add top relationships to results
    bivariate_results['top_relationships'] = top_relationships
    
    logger.info("Bivariate analysis completed successfully")
    return bivariate_results 

def analyze_performance_stability(club_df):
    """Analyze performance stability over time (2022-2024)."""
    logger.info("Analyzing performance stability over time...")
    
    # Define metrics for time analysis
    time_metrics = ['Grade_2022_football_value', 'Grade_2024_football_value', 
                  'Grade_2022_hurling_value', 'Grade_2024_hurling_value',
                  'football_improvement', 'hurling_improvement']
    
    # Create subsets of clubs
    football_clubs = club_df[(club_df['Grade_2022_football_value'] < 6) & 
                           (club_df['Grade_2024_football_value'] < 6)]
    hurling_clubs = club_df[(club_df['Grade_2022_hurling_value'] < 6) & 
                          (club_df['Grade_2024_hurling_value'] < 6)]
    dual_clubs = club_df[club_df['club_type'] == 'Dual']
    
    # Calculate improvement statistics
    improvement_stats = {
        'football': {
            'improved': sum(football_clubs['football_improvement'] < 0),
            'declined': sum(football_clubs['football_improvement'] > 0),
            'stable': sum(football_clubs['football_improvement'] == 0),
            'total_clubs': len(football_clubs),
            'avg_improvement': -1 * football_clubs['football_improvement'].mean(),  # Negative for clarity
            'max_improvement': -1 * football_clubs['football_improvement'].min(),   # Min becomes max improvement
            'max_decline': football_clubs['football_improvement'].max(),
            'std_dev': football_clubs['football_improvement'].std()
        },
        'hurling': {
            'improved': sum(hurling_clubs['hurling_improvement'] < 0),
            'declined': sum(hurling_clubs['hurling_improvement'] > 0),
            'stable': sum(hurling_clubs['hurling_improvement'] == 0),
            'total_clubs': len(hurling_clubs),
            'avg_improvement': -1 * hurling_clubs['hurling_improvement'].mean(),  # Negative for clarity
            'max_improvement': -1 * hurling_clubs['hurling_improvement'].min(),   # Min becomes max improvement
            'max_decline': hurling_clubs['hurling_improvement'].max(),
            'std_dev': hurling_clubs['hurling_improvement'].std()
        }
    }
    
    # Calculate combined improvement
    overall_improved = club_df['overall_change'].value_counts().get('Improvement', 0)
    overall_declined = club_df['overall_change'].value_counts().get('Decline', 0)
    overall_mixed = club_df['overall_change'].value_counts().get('Mixed', 0)
    overall_stable = club_df['overall_change'].value_counts().get('Stable', 0)
    
    improvement_stats['overall'] = {
        'improved': overall_improved,
        'declined': overall_declined,
        'mixed': overall_mixed,
        'stable': overall_stable,
        'total_clubs': len(club_df),
        'avg_magnitude': club_df['change_magnitude'].mean(),
        'max_magnitude': club_df['change_magnitude'].max(),
        'std_dev': club_df['change_magnitude'].std()
    }
    
    # Identify clubs with significant performance changes
    significant_changes = {
        'improvement': club_df[
            (club_df['overall_change'] == 'Improvement') & 
            (club_df['change_magnitude'] > club_df['change_magnitude'].quantile(0.75))
        ][['Club', 'football_improvement', 'hurling_improvement', 'change_magnitude']].sort_values('change_magnitude', ascending=False),
        
        'decline': club_df[
            (club_df['overall_change'] == 'Decline') & 
            (club_df['change_magnitude'] > club_df['change_magnitude'].quantile(0.75))
        ][['Club', 'football_improvement', 'hurling_improvement', 'change_magnitude']].sort_values('change_magnitude', ascending=False)
    }
    
    # Analyze dual clubs stability
    if len(dual_clubs) > 0:
        improvement_stats['dual'] = {
            'football_improved': sum(dual_clubs['football_improvement'] < 0),
            'football_declined': sum(dual_clubs['football_improvement'] > 0),
            'football_stable': sum(dual_clubs['football_improvement'] == 0),
            'hurling_improved': sum(dual_clubs['hurling_improvement'] < 0),
            'hurling_declined': sum(dual_clubs['hurling_improvement'] > 0),
            'hurling_stable': sum(dual_clubs['hurling_improvement'] == 0),
            'overall_improved': dual_clubs['overall_change'].value_counts().get('Improvement', 0),
            'overall_declined': dual_clubs['overall_change'].value_counts().get('Decline', 0),
            'overall_mixed': dual_clubs['overall_change'].value_counts().get('Mixed', 0),
            'overall_stable': dual_clubs['overall_change'].value_counts().get('Stable', 0),
            'total_clubs': len(dual_clubs)
        }
    
    # Analyze transition patterns
    transitions = {
        'football': {},
        'hurling': {}
    }
    
    # Football grade transitions
    football_grade_count = football_clubs.groupby(['Grade_2022_football', 'Grade_2024_football']).size()
    football_transitions = football_grade_count.reset_index()
    football_transitions.columns = ['2022_Grade', '2024_Grade', 'Count']
    transitions['football']['counts'] = football_transitions
    
    # Football promotion/relegation rates
    for grade in football_clubs['Grade_2022_football'].unique():
        grade_clubs = football_clubs[football_clubs['Grade_2022_football'] == grade]
        if len(grade_clubs) > 0:
            promoted = sum(grade_clubs['football_improvement'] < 0)
            relegated = sum(grade_clubs['football_improvement'] > 0)
            stable = sum(grade_clubs['football_improvement'] == 0)
            
            transitions['football'][grade] = {
                'promotion_rate': promoted / len(grade_clubs) if len(grade_clubs) > 0 else 0,
                'relegation_rate': relegated / len(grade_clubs) if len(grade_clubs) > 0 else 0,
                'stability_rate': stable / len(grade_clubs) if len(grade_clubs) > 0 else 0,
                'total_clubs': len(grade_clubs)
            }
    
    # Hurling grade transitions
    hurling_grade_count = hurling_clubs.groupby(['Grade_2022_hurling', 'Grade_2024_hurling']).size()
    hurling_transitions = hurling_grade_count.reset_index()
    hurling_transitions.columns = ['2022_Grade', '2024_Grade', 'Count']
    transitions['hurling']['counts'] = hurling_transitions
    
    # Hurling promotion/relegation rates
    for grade in hurling_clubs['Grade_2022_hurling'].unique():
        grade_clubs = hurling_clubs[hurling_clubs['Grade_2022_hurling'] == grade]
        if len(grade_clubs) > 0:
            promoted = sum(grade_clubs['hurling_improvement'] < 0)
            relegated = sum(grade_clubs['hurling_improvement'] > 0)
            stable = sum(grade_clubs['hurling_improvement'] == 0)
            
            transitions['hurling'][grade] = {
                'promotion_rate': promoted / len(grade_clubs) if len(grade_clubs) > 0 else 0,
                'relegation_rate': relegated / len(grade_clubs) if len(grade_clubs) > 0 else 0,
                'stability_rate': stable / len(grade_clubs) if len(grade_clubs) > 0 else 0,
                'total_clubs': len(grade_clubs)
            }
    
    stability_results = {
        'improvement_stats': improvement_stats,
        'significant_changes': significant_changes,
        'transitions': transitions
    }
    
    logger.info("Performance stability analysis completed successfully")
    return stability_results

def calculate_performance_stability(club_df):
    """
    Calculate performance stability metrics and add them to the club DataFrame.
    
    Args:
        club_df (pd.DataFrame): DataFrame containing club data
    
    Returns:
        pd.DataFrame: DataFrame with added performance stability metrics
    """
    logger.info("Calculating performance stability metrics")
    
    try:
        # Create performance change category
        if 'performance_change_category' not in club_df.columns:
            club_df['performance_change_category'] = club_df.apply(
                lambda row: 'Improved' if (row['football_improvement'] < -0.5 or row['hurling_improvement'] < -0.5) else
                            'Declined' if (row['football_improvement'] > 0.5 or row['hurling_improvement'] > 0.5) else
                            'Stable',
                axis=1
            )
        
        # Add 2022 overall performance (reversed from football and hurling improvements)
        if 'overall_performance_2022' not in club_df.columns:
            # Get current performance values
            fb_2024 = club_df['football_performance']
            hu_2024 = club_df['hurling_performance']
            
            # Calculate 2022 values using improvement (which is grade_2024 - grade_2022)
            fb_2022 = fb_2024 - club_df['football_improvement']
            hu_2022 = hu_2024 - club_df['hurling_improvement']
            
            # Calculate overall 2022 performance (using same formula as current overall performance)
            # Assuming overall_performance is the average of football and hurling performances
            # for clubs that have both, and the single performance for clubs that have only one
            club_df['overall_performance_2022'] = club_df.apply(
                lambda row: (
                    (fb_2022[row.name] + hu_2022[row.name]) / 2 
                    if (fb_2022[row.name] < 6 and hu_2022[row.name] < 6)
                    else fb_2022[row.name] if fb_2022[row.name] < 6
                    else hu_2022[row.name] if hu_2022[row.name] < 6
                    else None
                ),
                axis=1
            )
        
        # Analyze hurling grade transitions
        hurling_clubs = club_df[(club_df['Grade_2022_hurling'].notna()) & 
                              (club_df['Grade_2024_hurling'].notna())]
        
        if not hurling_clubs.empty:
            hurling_transitions = hurling_clubs.groupby(['Grade_2022_hurling', 'Grade_2024_hurling']).size()
            logger.info(f"Calculated {len(hurling_transitions)} hurling grade transitions")
        
        # Analyze football grade transitions
        football_clubs = club_df[(club_df['Grade_2022_football'].notna()) & 
                               (club_df['Grade_2024_football'].notna())]
        
        if not football_clubs.empty:
            football_transitions = football_clubs.groupby(['Grade_2022_football', 'Grade_2024_football']).size()
            logger.info(f"Calculated {len(football_transitions)} football grade transitions")
        
        # Calculate promotion/relegation rates
        for sport in ['football', 'hurling']:
            grade_2022_col = f"Grade_2022_{sport}"
            grade_2024_col = f"Grade_2024_{sport}"
            improvement_col = f"{sport}_improvement"
            
            # Only process clubs with valid grades
            valid_clubs = club_df[(club_df[grade_2022_col].notna()) & 
                                (club_df[improvement_col].notna())]
            
            if not valid_clubs.empty:
                promotion_rate = (valid_clubs[improvement_col] < 0).mean()
                relegation_rate = (valid_clubs[improvement_col] > 0).mean()
                stability_rate = (valid_clubs[improvement_col] == 0).mean()
                
                logger.info(f"{sport.capitalize()} stability metrics: "
                           f"Promotion rate: {promotion_rate:.2f}, "
                           f"Relegation rate: {relegation_rate:.2f}, "
                           f"Stability rate: {stability_rate:.2f}")
        
        logger.info("Performance stability metrics calculated successfully")
        return club_df
    except Exception as e:
        logger.error(f"Error calculating performance stability metrics: {e}")
        raise

def develop_club_typology(club_df):
    """Develop a typology of performance patterns using clustering."""
    logger.info("Developing club typology...")
    
    # Define metrics for clustering
    cluster_metrics = ['overall_performance', 'football_performance', 'hurling_performance', 
                     'code_balance', 'football_improvement', 'hurling_improvement']
    
    # Create a dataframe for clustering with complete records only
    # We'll filter out clubs that don't have both football and hurling grades
    cluster_df = club_df.copy()
    
    # Filter for clubs with both football and hurling metrics
    cluster_df = cluster_df[
        (cluster_df['football_performance'] < 6) & 
        (cluster_df['hurling_performance'] < 6)
    ]
    
    # Handle any remaining NaN values by filling with the mean
    for metric in cluster_metrics:
        if cluster_df[metric].isna().sum() > 0:
            cluster_df[metric].fillna(cluster_df[metric].mean(), inplace=True)
    
    # Prepare data for clustering
    X = cluster_df[cluster_metrics].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using the elbow method
    inertia = []
    for k in range(1, min(11, len(X_scaled))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Find elbow point (largest second derivative)
    deltas = np.diff(inertia)
    accelerations = np.diff(deltas)
    
    if len(accelerations) >= 2:
        optimal_k = np.argmax(accelerations) + 2  # +2 because diff reduces length twice
    else:
        optimal_k = min(4, len(X_scaled))  # Default to 4 clusters or fewer if less data
    
    logger.info(f"Determined optimal number of clusters: {optimal_k}")
    
    # Apply KMeans clustering with optimal_k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the dataframe
    cluster_df['cluster'] = cluster_labels
    
    # Calculate cluster characteristics
    cluster_stats = {}
    
    for i in range(optimal_k):
        cluster_stats[i] = {
            'count': sum(cluster_df['cluster'] == i),
            'percentage': sum(cluster_df['cluster'] == i) / len(cluster_df) * 100
        }
        
        for metric in cluster_metrics:
            cluster_data = cluster_df[cluster_df['cluster'] == i][metric]
            cluster_stats[i][f"{metric}_mean"] = cluster_data.mean()
            cluster_stats[i][f"{metric}_std"] = cluster_data.std()
        
        # Calculate percentage of dual clubs in cluster
        dual_count = sum((cluster_df['cluster'] == i) & (cluster_df['club_type'] == 'Dual'))
        cluster_stats[i]['dual_percentage'] = dual_count / sum(cluster_df['cluster'] == i) * 100 if sum(cluster_df['cluster'] == i) > 0 else 0
        
        # Calculate percentage of each performance change category
        for change in cluster_df['overall_change'].unique():
            change_count = sum((cluster_df['cluster'] == i) & (cluster_df['overall_change'] == change))
            cluster_stats[i][f"{change.lower()}_percentage"] = change_count / sum(cluster_df['cluster'] == i) * 100 if sum(cluster_df['cluster'] == i) > 0 else 0
    
    # Name the clusters based on their characteristics
    cluster_names = {}
    
    for i in range(optimal_k):
        # Extract key metrics
        overall = cluster_stats[i]['overall_performance_mean']
        fb_perf = cluster_stats[i]['football_performance_mean']
        hu_perf = cluster_stats[i]['hurling_performance_mean']
        balance = cluster_stats[i]['code_balance_mean']
        fb_imp = cluster_stats[i]['football_improvement_mean']
        hu_imp = cluster_stats[i]['hurling_improvement_mean']
        dual_pct = cluster_stats[i]['dual_percentage']
        
        # Determine performance level
        if overall < 2.5:
            performance = "Elite"
        elif overall < 3.5:
            performance = "Strong"
        elif overall < 4.5:
            performance = "Developing"
        else:
            performance = "Struggling"
        
        # Determine code balance
        if balance < 1.0:
            code_type = "Balanced"
        elif fb_perf < hu_perf:
            code_type = "Football-Focused"
        else:
            code_type = "Hurling-Focused"
        
        # Determine trajectory
        trajectory = ""
        if fb_imp < -0.5 and hu_imp < -0.5:
            trajectory = "Rapidly Improving"
        elif fb_imp < 0 and hu_imp < 0:
            trajectory = "Improving"
        elif fb_imp > 0.5 and hu_imp > 0.5:
            trajectory = "Declining"
        elif fb_imp > 0 and hu_imp > 0:
            trajectory = "Slightly Declining"
        else:
            trajectory = "Stable"
        
        # Create name
        if dual_pct > 50:
            dual_status = "Dual"
        else:
            dual_status = "Single-Code"
        
        cluster_names[i] = f"{performance} {code_type} {dual_status} Clubs ({trajectory})"
    
    # Add names to cluster stats
    for i in range(optimal_k):
        cluster_stats[i]['name'] = cluster_names[i]
    
    # Create cluster membership list
    cluster_members = {}
    for i in range(optimal_k):
        members = cluster_df[cluster_df['cluster'] == i]['Club'].tolist()
        cluster_members[i] = members
    
    # Return results
    typology_results = {
        'optimal_clusters': optimal_k,
        'inertia': inertia,
        'cluster_stats': cluster_stats,
        'cluster_centers': kmeans.cluster_centers_,
        'feature_names': cluster_metrics,
        'cluster_names': cluster_names,
        'cluster_members': cluster_members,
        'cluster_df': cluster_df
    }
    
    logger.info("Club typology development completed successfully")
    return typology_results 

def create_visualizations(club_df, correlation_matrices):
    """
    Create visualizations for comprehensive metrics analysis.
    
    Args:
        club_df (pd.DataFrame): DataFrame containing club data
        correlation_matrices (dict): Dictionary of correlation matrices
    """
    logger.info("Creating visualizations for comprehensive metrics analysis")
    
    try:
        # Create output directory for visualizations
        visualizations_dir = os.path.join(OUTPUT_DIR, "comprehensive_metrics")
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Distribution of performance metrics
        create_performance_distributions(club_df, visualizations_dir)
        
        # Correlation heatmaps
        create_correlation_heatmaps(correlation_matrices, visualizations_dir)
        
        # Performance stability visualizations
        create_stability_visualizations(club_df, visualizations_dir)
        
        # Club typology visualizations
        create_typology_visualizations(club_df, visualizations_dir)
        
        logger.info(f"All visualizations saved to {visualizations_dir}")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise

def create_performance_distributions(club_df, output_dir):
    """
    Create visualizations for distributions of performance metrics.
    
    Args:
        club_df (pd.DataFrame): DataFrame containing club data
        output_dir (str): Directory to save visualizations
    """
    logger.info("Creating performance distribution visualizations")
    
    try:
        # Box plots of performance metrics by club type
        plt.figure(figsize=(12, 8))
        performance_metrics = PERFORMANCE_METRICS + TRANSFORMED_METRICS
        
        # Prepare data for plotting
        plot_data = pd.melt(
            club_df, 
            id_vars=['club_type'], 
            value_vars=performance_metrics,
            var_name='Metric', 
            value_name='Value'
        )
        
        # Create the boxplot
        sns.boxplot(x='Metric', y='Value', hue='club_type', data=plot_data)
        plt.xticks(rotation=45, ha='right')
        plt.title('Performance Metrics by Club Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_by_club_type.png'))
        plt.close()
        
        # Histogram grid for all performance metrics
        metrics_to_plot = PERFORMANCE_METRICS + TRANSFORMED_METRICS
        n_metrics = len(metrics_to_plot)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes):
                sns.histplot(club_df[metric].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {metric}')
                
        # Hide any unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_distributions.png'))
        plt.close()
        
        logger.info("Performance distribution visualizations created successfully")
    except Exception as e:
        logger.error(f"Error creating performance distribution visualizations: {e}")
        raise

def create_correlation_heatmaps(correlation_matrices, output_dir):
    """
    Create correlation heatmaps for different correlation matrices.
    
    Args:
        correlation_matrices (dict): Dictionary of correlation matrices
        output_dir (str): Directory to save visualizations
    """
    logger.info("Creating correlation heatmap visualizations")
    
    try:
        # Create heatmaps for each correlation matrix
        for matrix_name, matrix in correlation_matrices.items():
            plt.figure(figsize=(12, 10))
            
            # Generate mask for upper triangle to create a lower triangle heatmap
            mask = np.triu(np.ones_like(matrix, dtype=bool))
            
            # Create heatmap
            sns.heatmap(
                matrix, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                mask=mask,
                fmt='.2f', 
                linewidths=0.5
            )
            
            plt.title(f'Correlation Matrix: {matrix_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'correlation_{matrix_name}.png'))
            plt.close()
            
        # Create a comparative visualization for original vs transformed metrics
        if 'original_metrics' in correlation_matrices and 'transformed_metrics' in correlation_matrices:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Original metrics
            sns.heatmap(
                correlation_matrices['original_metrics'], 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                fmt='.2f', 
                linewidths=0.5,
                ax=axes[0]
            )
            axes[0].set_title('Original Metrics Correlations')
            
            # Transformed metrics
            sns.heatmap(
                correlation_matrices['transformed_metrics'], 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                fmt='.2f', 
                linewidths=0.5,
                ax=axes[1]
            )
            axes[1].set_title('Transformed Metrics Correlations')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'original_vs_transformed_correlations.png'))
            plt.close()
            
        logger.info("Correlation heatmap visualizations created successfully")
    except Exception as e:
        logger.error(f"Error creating correlation heatmap visualizations: {e}")
        raise

def create_stability_visualizations(club_df, output_dir):
    """
    Create visualizations for performance stability analysis.
    
    Args:
        club_df (pd.DataFrame): DataFrame containing club data
        output_dir (str): Directory to save visualizations
    """
    logger.info("Creating performance stability visualizations")
    
    try:
        # Performance change categories distribution
        plt.figure(figsize=(10, 6))
        performance_change_counts = club_df['performance_change_category'].value_counts()
        performance_change_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Performance Change Categories')
        plt.xlabel('Performance Change Category')
        plt.ylabel('Number of Clubs')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_change_distribution.png'))
        plt.close()
        
        # Scatter plot of 2022 vs 2024 performance
        plt.figure(figsize=(10, 8))
        scatter = sns.scatterplot(
            data=club_df,
            x='overall_performance_2022',
            y='overall_performance',
            hue='club_type',
            style='performance_change_category',
            s=100,
            alpha=0.7
        )
        
        # Add diagonal line (y=x)
        min_val = min(club_df['overall_performance_2022'].min(), club_df['overall_performance'].min())
        max_val = max(club_df['overall_performance_2022'].max(), club_df['overall_performance'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.title('Performance Stability: 2022 vs 2024')
        plt.xlabel('Overall Performance 2022')
        plt.ylabel('Overall Performance 2024')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_stability_scatter.png'))
        plt.close()
        
        # Football grade transitions
        football_transitions = club_df.groupby(['Grade_2022_football', 'Grade_2024_football']).size().reset_index()
        football_transitions.columns = ['From', 'To', 'Count']
        
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for heatmap
        pivot_table = football_transitions.pivot(index='From', columns='To', values='Count')
        pivot_table = pivot_table.fillna(0)
        
        # Plot heatmap
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='g')
        plt.title('Football Grade Transitions: 2022 to 2024')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'football_grade_transitions_heatmap.png'))
        plt.close()
        
        # Hurling grade transitions
        hurling_transitions = club_df.groupby(['Grade_2022_hurling', 'Grade_2024_hurling']).size().reset_index()
        hurling_transitions.columns = ['From', 'To', 'Count']
        
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for heatmap
        pivot_table = hurling_transitions.pivot(index='From', columns='To', values='Count')
        pivot_table = pivot_table.fillna(0)
        
        # Plot heatmap
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='g')
        plt.title('Hurling Grade Transitions: 2022 to 2024')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hurling_grade_transitions_heatmap.png'))
        plt.close()
        
        logger.info("Performance stability visualizations created successfully")
    except Exception as e:
        logger.error(f"Error creating performance stability visualizations: {e}")
        raise

def create_typology_visualizations(club_df, output_dir):
    """
    Create visualizations for club typology analysis.
    
    Args:
        club_df (pd.DataFrame): DataFrame containing club data with cluster assignments
        output_dir (str): Directory to save visualizations
    """
    logger.info("Creating club typology visualizations")
    
    try:
        # Check if cluster column exists
        if 'cluster' not in club_df.columns:
            logger.warning("No cluster column found in club_df. Skipping typology visualizations.")
            return
        
        # Cluster distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = club_df['cluster'].value_counts().sort_index()
        cluster_counts.plot(kind='bar', color='lightgreen')
        plt.title('Distribution of Club Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Clubs')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'))
        plt.close()
        
        # Cluster characteristics
        metrics_for_radar = ['overall_performance', 'football_performance', 'hurling_performance', 
                           'code_balance']
        
        # Calculate mean values for each cluster
        cluster_profiles = club_df.groupby('cluster')[metrics_for_radar].mean()
        
        # Radar charts for each cluster
        n_clusters = len(cluster_profiles)
        n_metrics = len(metrics_for_radar)
        
        # Compute angle for each metric
        angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, axes = plt.subplots(1, n_clusters, figsize=(n_clusters*5, 5), subplot_kw=dict(polar=True))
        
        # Handle case where there's only one cluster
        if n_clusters == 1:
            axes = [axes]
            
        for i, (idx, row) in enumerate(cluster_profiles.iterrows()):
            # Prepare data
            values = row.values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot data
            axes[i].plot(angles, values, linewidth=2, linestyle='solid')
            axes[i].fill(angles, values, alpha=0.25)
            
            # Set labels
            axes[i].set_xticks(angles[:-1])
            axes[i].set_xticklabels(metrics_for_radar)
            
            axes[i].set_title(f'Cluster {idx} Profile', size=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_profiles_radar.png'))
        plt.close()
        
        # 2D scatter plot of clusters
        plt.figure(figsize=(12, 10))
        
        # Choose two key metrics for visualization
        sns.scatterplot(
            data=club_df,
            x='overall_performance',
            y='code_balance',
            hue='cluster',
            palette='viridis',
            s=100,
            alpha=0.7
        )
        
        # Add club names as annotations
        for idx, row in club_df.iterrows():
            plt.annotate(
                row['Club'],
                (row['overall_performance'], row['code_balance']),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title('Club Clusters: Overall Performance vs Code Balance')
        plt.xlabel('Overall Performance')
        plt.ylabel('Code Balance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_scatter_plot.png'))
        plt.close()
        
        logger.info("Club typology visualizations created successfully")
    except Exception as e:
        logger.error(f"Error creating club typology visualizations: {e}")
        raise 

def generate_report(club_df, correlation_matrices, cluster_profiles=None, stability_stats=None):
    """
    Generate a comprehensive report summarizing the metrics analysis.
    
    Args:
        club_df (pd.DataFrame): DataFrame containing club data
        correlation_matrices (dict): Dictionary of correlation matrices
        cluster_profiles (pd.DataFrame, optional): Cluster profiles if clustering was performed
        stability_stats (dict, optional): Dictionary of stability statistics
    """
    logger.info("Generating comprehensive metrics analysis report")
    
    try:
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(BASE_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Create output directory for visualizations
        visualizations_dir = os.path.join(OUTPUT_DIR, "comprehensive_metrics")
        
        # Start building the report
        report_path = os.path.join(reports_dir, "comprehensive_metrics_analysis.md")
        
        with open(report_path, 'w') as f:
            # Report header
            f.write("# Comprehensive GAA Club Performance Metrics Analysis\n\n")
            f.write(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # Introduction
            f.write("## Introduction\n\n")
            f.write("This report presents a comprehensive analysis of GAA club performance metrics, ")
            f.write("including performance distributions, correlations between metrics, ")
            f.write("performance stability over time, and club typologies.\n\n")
            
            # Dataset overview
            f.write("## Dataset Overview\n\n")
            f.write(f"- Total number of clubs analyzed: {len(club_df)}\n")
            f.write(f"- Number of football clubs: {len(club_df[club_df['Grade_2024_football'].notna()])}\n")
            f.write(f"- Number of hurling clubs: {len(club_df[club_df['Grade_2024_hurling'].notna()])}\n")
            f.write(f"- Number of dual clubs: {len(club_df[club_df['is_dual_2024'] == True])}\n")
            f.write("\n")
            
            # Performance Metrics Summary
            f.write("## Performance Metrics Summary\n\n")
            f.write("### Original Performance Metrics\n\n")
            
            # Create a summary statistics table for original metrics
            original_metrics_stats = club_df[PERFORMANCE_METRICS].describe().T
            original_metrics_stats = original_metrics_stats.round(2)
            
            f.write(original_metrics_stats.to_markdown() + "\n\n")
            
            f.write("### Transformed Performance Metrics\n\n")
            
            # Create a summary statistics table for transformed metrics
            transformed_metrics_stats = club_df[TRANSFORMED_METRICS].describe().T
            transformed_metrics_stats = transformed_metrics_stats.round(2)
            
            f.write(transformed_metrics_stats.to_markdown() + "\n\n")
            
            # Performance distributions
            f.write("## Performance Distributions\n\n")
            f.write("The following visualizations show the distributions of performance metrics ")
            f.write("across all clubs and by club type.\n\n")
            
            f.write("![Performance Metrics by Club Type]")
            f.write(f"(../output/analysis/comprehensive_metrics/performance_by_club_type.png)\n\n")
            
            f.write("![Metrics Distributions]")
            f.write(f"(../output/analysis/comprehensive_metrics/metrics_distributions.png)\n\n")
            
            # Correlation Analysis
            f.write("## Correlation Analysis\n\n")
            f.write("### Correlation Matrices\n\n")
            f.write("The following heatmaps show correlations between different performance metrics.\n\n")
            
            for matrix_name in correlation_matrices.keys():
                f.write(f"![{matrix_name.replace('_', ' ').title()} Correlation Matrix]")
                f.write(f"(../output/analysis/comprehensive_metrics/correlation_{matrix_name}.png)\n\n")
            
            if 'original_metrics' in correlation_matrices and 'transformed_metrics' in correlation_matrices:
                f.write("### Comparison of Original vs. Transformed Metrics Correlations\n\n")
                f.write("![Original vs Transformed Correlations]")
                f.write(f"(../output/analysis/comprehensive_metrics/original_vs_transformed_correlations.png)\n\n")
            
            # Performance Stability Analysis
            f.write("## Performance Stability Analysis\n\n")
            f.write("This section analyzes how club performance has changed between 2022 and 2024.\n\n")
            
            f.write("### Performance Change Distribution\n\n")
            f.write("![Performance Change Distribution]")
            f.write(f"(../output/analysis/comprehensive_metrics/performance_change_distribution.png)\n\n")
            
            f.write("### Performance Stability Scatter Plot\n\n")
            f.write("The scatter plot below shows the relationship between 2022 and 2024 performance metrics, ")
            f.write("highlighting stable and changing clubs.\n\n")
            
            f.write("![Performance Stability Scatter]")
            f.write(f"(../output/analysis/comprehensive_metrics/performance_stability_scatter.png)\n\n")
            
            f.write("### Grade Transitions\n\n")
            f.write("The following heatmaps show how clubs have transitioned between grades from 2022 to 2024.\n\n")
            
            f.write("![Football Grade Transitions]")
            f.write(f"(../output/analysis/comprehensive_metrics/football_grade_transitions_heatmap.png)\n\n")
            
            f.write("![Hurling Grade Transitions]")
            f.write(f"(../output/analysis/comprehensive_metrics/hurling_grade_transitions_heatmap.png)\n\n")
            
            # Add stability statistics if provided
            if stability_stats:
                f.write("### Performance Stability Statistics\n\n")
                f.write("| Statistic | Value |\n")
                f.write("| --- | --- |\n")
                
                for stat, value in stability_stats.items():
                    f.write(f"| {stat} | {value} |\n")
                
                f.write("\n")
            
            # Club Typology Analysis (if clustering was performed)
            if 'cluster' in club_df.columns:
                f.write("## Club Typology Analysis\n\n")
                f.write("This section presents a typology of GAA clubs based on their performance characteristics.\n\n")
                
                f.write("### Cluster Distribution\n\n")
                f.write("![Cluster Distribution]")
                f.write(f"(../output/analysis/comprehensive_metrics/cluster_distribution.png)\n\n")
                
                f.write("### Cluster Profiles\n\n")
                f.write("The radar charts below show the characteristic profiles of each identified club cluster.\n\n")
                
                f.write("![Cluster Profiles Radar]")
                f.write(f"(../output/analysis/comprehensive_metrics/cluster_profiles_radar.png)\n\n")
                
                f.write("### Cluster Scatter Plot\n\n")
                f.write("![Cluster Scatter Plot]")
                f.write(f"(../output/analysis/comprehensive_metrics/cluster_scatter_plot.png)\n\n")
                
                # Add cluster profiles table if provided
                if cluster_profiles is not None:
                    f.write("### Detailed Cluster Profiles\n\n")
                    f.write("The table below provides the mean values of key metrics for each cluster.\n\n")
                    
                    cluster_profiles_rounded = cluster_profiles.round(2)
                    f.write(cluster_profiles_rounded.to_markdown() + "\n\n")
            
            # Key Findings and Conclusions
            f.write("## Key Findings and Conclusions\n\n")
            
            # Automatically generate some key findings based on the data
            findings = []
            
            # Finding about average performance
            avg_performance = round(club_df['overall_performance'].mean(), 2)
            findings.append(f"The average overall performance score across all clubs is {avg_performance}.")
            
            # Finding about dual clubs
            if 'is_dual_2024' in club_df.columns:
                dual_avg = round(club_df[club_df['is_dual_2024'] == True]['overall_performance'].mean(), 2)
                single_avg = round(club_df[club_df['is_dual_2024'] == False]['overall_performance'].mean(), 2)
                if dual_avg > single_avg:
                    findings.append(f"Dual clubs have a higher average performance ({dual_avg}) compared to single-code clubs ({single_avg}).")
                else:
                    findings.append(f"Single-code clubs have a higher average performance ({single_avg}) compared to dual clubs ({dual_avg}).")
            
            # Finding about correlation
            if 'original_metrics' in correlation_matrices:
                corr_matrix = correlation_matrices['original_metrics']
                highest_corr = 0
                highest_pair = ('', '')
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        if abs(corr_matrix.iloc[i, j]) > abs(highest_corr):
                            highest_corr = corr_matrix.iloc[i, j]
                            highest_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                
                correlation_type = "positive" if highest_corr > 0 else "negative"
                findings.append(f"The strongest {correlation_type} correlation ({round(highest_corr, 2)}) is between {highest_pair[0]} and {highest_pair[1]}.")
            
            # Finding about performance stability
            if 'performance_change_category' in club_df.columns:
                improved = club_df['performance_change_category'].value_counts().get('Improved', 0)
                declined = club_df['performance_change_category'].value_counts().get('Declined', 0)
                stable = club_df['performance_change_category'].value_counts().get('Stable', 0)
                
                findings.append(f"{improved} clubs improved, {stable} remained stable, and {declined} declined in performance from 2022 to 2024.")
            
            # Write findings
            for finding in findings:
                f.write(f"- {finding}\n")
            
            f.write("\n")
            f.write("### Recommendations\n\n")
            f.write("Based on the analysis, the following recommendations are suggested:\n\n")
            f.write("1. Further investigate the factors contributing to performance differences between club types\n")
            f.write("2. Develop targeted support programs for clubs showing performance decline\n")
            f.write("3. Consider the identified club typologies when designing development initiatives\n")
            
        logger.info(f"Comprehensive metrics analysis report generated successfully: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating comprehensive metrics analysis report: {e}")
        raise 

def run_comprehensive_metrics_analysis():
    """Run a comprehensive analysis of club performance metrics."""
    logger.info("Starting comprehensive metrics analysis")
    
    try:
        # Load club data
        club_df = load_data()
        
        # Create output directories
        output_dir = os.path.join(OUTPUT_DIR, "comprehensive_metrics")
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate correlation matrices
        correlation_matrices = {}
        
        # Original metrics correlation
        original_metrics_corr = club_df[PERFORMANCE_METRICS].corr()
        correlation_matrices["original_metrics"] = original_metrics_corr
        
        # Transformed metrics correlation
        transformed_metrics_corr = club_df[TRANSFORMED_METRICS].corr()
        correlation_matrices["transformed_metrics"] = transformed_metrics_corr
        
        # Calculate performance stability metrics
        club_df = calculate_performance_stability(club_df)
        
        # Develop club typology using clustering
        typology_results = develop_club_typology(club_df)
        
        # Add cluster labels to the main dataframe
        if 'cluster_df' in typology_results and 'optimal_clusters' in typology_results:
            # Create a mapping from Club to cluster
            cluster_mapping = dict(zip(
                typology_results['cluster_df']['Club'], 
                typology_results['cluster_df']['cluster']
            ))
            
            # Add cluster column to main dataframe
            club_df['cluster'] = club_df['Club'].map(cluster_mapping)
            
            # Extract cluster profiles
            cluster_profiles = typology_results['cluster_df'].groupby('cluster')[
                ['overall_performance', 'football_performance', 'hurling_performance', 'code_balance']
            ].mean()
            
            logger.info(f"Added cluster assignments to club data. Found {typology_results['optimal_clusters']} clusters")
        else:
            cluster_profiles = None
            logger.warning("Club typology development did not produce valid clusters")
        
        # Generate visualizations
        create_visualizations(club_df, correlation_matrices)
        
        # Generate report
        report_path = generate_report(
            club_df, 
            correlation_matrices,
            cluster_profiles=cluster_profiles
        )
        
        logger.info(f"Comprehensive metrics analysis completed. Report generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error in comprehensive metrics analysis: {e}")
        raise

if __name__ == "__main__":
    try:
        run_comprehensive_metrics_analysis()
    except Exception as e:
        logger.error(f"Failed to run comprehensive metrics analysis: {e}") 