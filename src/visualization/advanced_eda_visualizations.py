#!/usr/bin/env python3
"""
Advanced EDA Visualizations for GAA Cork Club Analysis.

This script creates advanced visualizations for exploratory data analysis:
1. Scatterplots of key relationships with trend lines
2. Boxplots for categorical variables
3. Spatial distribution maps
4. Violin plots for distribution comparisons

The visualizations highlight relationships between performance metrics
and demographic/environmental factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
import logging
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
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
OUTPUT_DIR = BASE_DIR / 'output' / 'visualizations'
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

# Key demographic variables (based on previous analyses)
KEY_DEMOGRAPHIC_VARS = [
    'basic_education_rate',
    'third_level_rate',
    'employment_rate',
    'professional_rate',
    'working_class_rate',
    'youth_proportion'
]

# Environmental factors
ENVIRONMENTAL_VARS = [
    'Elevation',
    'annual_rainfall',
    'rain_days'
]

# Custom color palettes
PERFORMANCE_COLORS = {
    'Elite': '#1a9850',       # Green
    'Strong': '#91cf60',      # Light green
    'Medium': '#d9ef8b',      # Pale green-yellow
    'Developing': '#fee08b',  # Pale yellow
    'No Grade': '#d73027'     # Red
}

def load_data():
    """Load club and demographic data."""
    logger.info("Loading data...")
    
    try:
        # Load club data
        club_filepath = DATA_DIR / 'cork_clubs_complete_graded.csv'
        club_df = pd.read_csv(club_filepath)
        logger.info(f"Loaded club data for {len(club_df)} clubs")
        
        # Add transformed metrics to club data
        club_df['transformed_football_performance'] = 7 - club_df['football_performance']
        club_df['transformed_hurling_performance'] = 7 - club_df['hurling_performance']
        club_df['transformed_overall_performance'] = 7 - club_df['overall_performance']
        club_df['transformed_code_balance'] = 7 - club_df['code_balance']  # higher is better for consistency
        logger.info("Added transformed metrics to club data")
        
        # Load demographic data from nearest assignment (shown to be more reliable in catchment analysis)
        demographic_filepath = DATA_DIR / 'nearest_demographics.gpkg'
        try:
            demographic_df = gpd.read_file(demographic_filepath)
            logger.info(f"Loaded demographic data for {len(demographic_df)} small areas")
        except Exception as e:
            logger.warning(f"Could not load demographic data as GeoPackage: {e}")
            demographic_df = None
        
        # Load spatial data
        spatial_filepath = DATA_DIR / 'cork_clubs_spatial.geojson'
        try:
            spatial_df = gpd.read_file(spatial_filepath)
            logger.info(f"Loaded spatial data for {len(spatial_df)} clubs")
        except Exception as e:
            logger.warning(f"Could not load spatial data: {e}")
            spatial_df = None
        
        # Add performance category for better visualization
        club_df['performance_category'] = pd.cut(
            club_df['overall_performance'], 
            bins=[0, 1.5, 2.5, 3.5, 4.5, 6],
            labels=['Elite', 'Strong', 'Medium', 'Developing', 'No Grade']
        )
        
        # Prepare merged dataset for demographic analysis
        if demographic_df is not None:
            # Find demographic columns
            demo_cols = [col for col in demographic_df.columns if any(
                key in col for key in ['rate', 'proportion', 'education', 'class'])]
                
            # Identify club column
            club_col = 'nearest_club' if 'nearest_club' in demographic_df.columns else None
            if club_col:
                # Aggregate by club
                demo_agg = demographic_df.groupby(club_col)[demo_cols].mean().reset_index()
                demo_agg.rename(columns={club_col: 'Club'}, inplace=True)
                
                # Merge with club data
                merged_df = pd.merge(club_df, demo_agg, on='Club', how='left')
                logger.info(f"Created merged dataset with {len(merged_df)} clubs and {len(demo_cols)} demographic variables")
            else:
                logger.warning("Could not identify club column in demographic data")
                merged_df = club_df
        else:
            merged_df = club_df
        
        # Return all datasets
        return {
            'club_df': club_df,
            'demographic_df': demographic_df,
            'spatial_df': spatial_df,
            'merged_df': merged_df
        }
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def generate_scatterplots_with_trendlines(data):
    """
    Generate scatterplots of key relationships with trend lines.
    
    Args:
        data: Dictionary containing dataframes
        
    Returns:
        List of paths to generated visualizations
    """
    logger.info("Generating scatterplots with trend lines...")
    visualization_paths = []
    
    merged_df = data.get('merged_df')
    if merged_df is None:
        logger.warning("No merged data available for visualization.")
        return visualization_paths
    
    # Set common style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Performance vs Environmental Factors
    for env_var in ENVIRONMENTAL_VARS:
        if env_var not in merged_df.columns:
            logger.warning(f"Environmental variable {env_var} not found in data")
            continue
            
        for i, (metric, t_metric) in enumerate(zip(PERFORMANCE_METRICS, TRANSFORMED_METRICS)):
            # Skip code balance for cleaner analysis (focus on performance)
            if 'code_balance' in metric:
                continue
                
            # Create figure with two subplots (original and transformed metrics)
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            
            # Plot original metric
            if metric in ['football_performance', 'hurling_performance']:
                # Filter out NA values (6)
                plot_data = merged_df[merged_df[metric] < 6].copy()
            else:
                plot_data = merged_df.copy()
                
            # Original metric (lower is better)
            ax = axes[0]
            
            # Create scatter plot
            sns.scatterplot(
                x=env_var,
                y=metric,
                data=plot_data,
                hue='performance_category',
                palette=PERFORMANCE_COLORS,
                s=80,
                alpha=0.7,
                ax=ax
            )
            
            # Add trend line
            x = plot_data[env_var]
            y = plot_data[metric]
            
            # Add polynomial trend line (degree=2 for better fit)
            if len(x) > 3:
                try:
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    x_range = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
                    
                    # Calculate and display correlation
                    corr = plot_data[[metric, env_var]].corr().iloc[0, 1]
                    ax.annotate(f"Correlation: {corr:.2f}", 
                                xy=(0.05, 0.95), 
                                xycoords='axes fraction', 
                                fontsize=12, 
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
                except Exception as e:
                    logger.warning(f"Could not generate trend line for {metric} vs {env_var}: {e}")
            
            # Labels and title
            metric_title = metric.replace('_', ' ').title()
            env_var_title = env_var.replace('_', ' ').title()
            ax.set_title(f"{metric_title} vs {env_var_title}")
            ax.set_xlabel(env_var_title)
            ax.set_ylabel(f"{metric_title} (Lower = Better)")
            
            # Transformed metric (higher is better)
            ax = axes[1]
            
            # Filter data for transformed metric
            if t_metric in ['transformed_football_performance', 'transformed_hurling_performance']:
                # Filter out NA values (1)
                plot_data = merged_df[merged_df[t_metric] != 1].copy()
            else:
                plot_data = merged_df.copy()
            
            # Create scatter plot
            sns.scatterplot(
                x=env_var,
                y=t_metric,
                data=plot_data,
                hue='performance_category',
                palette=PERFORMANCE_COLORS,
                s=80,
                alpha=0.7,
                ax=ax
            )
            
            # Add trend line
            x = plot_data[env_var]
            y = plot_data[t_metric]
            
            # Add polynomial trend line (degree=2 for better fit)
            if len(x) > 3:
                try:
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    x_range = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
                    
                    # Calculate and display correlation
                    corr = plot_data[[t_metric, env_var]].corr().iloc[0, 1]
                    ax.annotate(f"Correlation: {corr:.2f}", 
                                xy=(0.05, 0.95), 
                                xycoords='axes fraction', 
                                fontsize=12, 
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
                except Exception as e:
                    logger.warning(f"Could not generate trend line for {t_metric} vs {env_var}: {e}")
                    
            # Labels and title
            t_metric_title = t_metric.replace('transformed_', '').replace('_', ' ').title()
            ax.set_title(f"Transformed {t_metric_title} vs {env_var_title}")
            ax.set_xlabel(env_var_title)
            ax.set_ylabel(f"Transformed {t_metric_title} (Higher = Better)")
            
            # Adjust layout and save
            plt.tight_layout()
            metric_name = metric.replace('_performance', '')
            output_path = OUTPUT_DIR / f"scatter_{metric_name}_vs_{env_var}.png"
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            
            visualization_paths.append(output_path)
            logger.info(f"Saved scatter plot to {output_path}")
    
    # 2. Performance vs Key Demographic Variables
    # Find available demographic variables
    demo_vars = []
    for prefix in KEY_DEMOGRAPHIC_VARS:
        matching = [col for col in merged_df.columns if col.startswith(prefix)]
        if matching:
            demo_vars.extend(matching)
    
    logger.info(f"Found {len(demo_vars)} demographic variables for scatter plots")
    
    # For each demographic variable, create scatter plots with trend lines
    for demo_var in demo_vars:
        # Skip if the variable doesn't exist
        if demo_var not in merged_df.columns:
            continue
            
        # Plot against transformed metrics (which show clearer relationships)
        for t_metric in TRANSFORMED_METRICS:
            # Skip code balance for cleaner analysis
            if 'code_balance' in t_metric:
                continue
                
            # Filter data
            if t_metric in ['transformed_football_performance', 'transformed_hurling_performance']:
                plot_data = merged_df[merged_df[t_metric] != 1].copy()
            else:
                plot_data = merged_df.copy()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create scatter plot
            sns.scatterplot(
                x=demo_var,
                y=t_metric,
                data=plot_data,
                hue='performance_category',
                palette=PERFORMANCE_COLORS,
                s=80,
                alpha=0.7,
                ax=ax
            )
            
            # Add trend line
            x = plot_data[demo_var]
            y = plot_data[t_metric]
            
            # Add polynomial trend line (degree=2 for better fit)
            if len(x) > 3:
                try:
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    x_range = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
                    
                    # Calculate and display correlation
                    corr = plot_data[[t_metric, demo_var]].corr().iloc[0, 1]
                    ax.annotate(f"Correlation: {corr:.2f}", 
                                xy=(0.05, 0.95), 
                                xycoords='axes fraction', 
                                fontsize=12, 
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
                except Exception as e:
                    logger.warning(f"Could not generate trend line for {t_metric} vs {demo_var}: {e}")
            
            # Labels and title
            metric_title = t_metric.replace('transformed_', '').replace('_', ' ').title()
            demo_var_title = demo_var.replace('_', ' ').title()
            ax.set_title(f"{metric_title} vs {demo_var_title}")
            ax.set_xlabel(demo_var_title)
            ax.set_ylabel(f"{metric_title} (Higher = Better)")
            
            # Adjust layout and save
            plt.tight_layout()
            metric_name = t_metric.replace('transformed_', '')
            # Shorten the demo var name for filename
            demo_var_short = demo_var.split('_')[0] + "_" + demo_var.split('_')[-1]
            output_path = OUTPUT_DIR / f"scatter_{metric_name}_vs_{demo_var_short}.png"
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            
            visualization_paths.append(output_path)
            logger.info(f"Saved scatter plot to {output_path}")
    
    return visualization_paths

def generate_categorical_boxplots(data):
    """
    Generate boxplots for categorical variables.
    
    Args:
        data: Dictionary containing dataframes
        
    Returns:
        List of paths to generated visualizations
    """
    logger.info("Generating boxplots for categorical variables...")
    visualization_paths = []
    
    club_df = data.get('club_df')
    merged_df = data.get('merged_df')
    
    if club_df is None:
        logger.warning("No club data available for visualization.")
        return visualization_paths
    
    # Set common style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Performance by Grade Levels
    # Create a figure for each year
    for year in [2022, 2024]:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Create boxplots for football grades
        ax = axes[0]
        football_col = f'Grade_{year}_football'
        
        if football_col in club_df.columns:
            # Filter out clubs with no football grade
            football_data = club_df[club_df[football_col] != 'No Grade'].copy()
            
            # Create boxplot
            sns.boxplot(
                x=football_col,
                y='transformed_football_performance',
                data=football_data,
                palette='Reds_r',  # Reversed so darker = better
                ax=ax
            )
            
            # Add individual points
            sns.stripplot(
                x=football_col,
                y='transformed_football_performance',
                data=football_data,
                color='black',
                alpha=0.5,
                jitter=True,
                ax=ax
            )
            
            # Labels and title
            ax.set_title(f"Football Performance by Grade ({year})")
            ax.set_xlabel("Football Grade")
            ax.set_ylabel("Transformed Football Performance (Higher = Better)")
        else:
            logger.warning(f"Football grade column for {year} not found")
            
        # Create boxplots for hurling grades
        ax = axes[1]
        hurling_col = f'Grade_{year}_hurling'
        
        if hurling_col in club_df.columns:
            # Filter out clubs with no hurling grade
            hurling_data = club_df[club_df[hurling_col] != 'No Grade'].copy()
            
            # Create boxplot
            sns.boxplot(
                x=hurling_col,
                y='transformed_hurling_performance',
                data=hurling_data,
                palette='Blues_r',  # Reversed so darker = better
                ax=ax
            )
            
            # Add individual points
            sns.stripplot(
                x=hurling_col,
                y='transformed_hurling_performance',
                data=hurling_data,
                color='black',
                alpha=0.5,
                jitter=True,
                ax=ax
            )
            
            # Labels and title
            ax.set_title(f"Hurling Performance by Grade ({year})")
            ax.set_xlabel("Hurling Grade")
            ax.set_ylabel("Transformed Hurling Performance (Higher = Better)")
        else:
            logger.warning(f"Hurling grade column for {year} not found")
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = OUTPUT_DIR / f"boxplot_performance_by_grade_{year}.png"
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        
        visualization_paths.append(output_path)
        logger.info(f"Saved grade boxplot to {output_path}")
    
    # 2. Performance by urban/rural classification
    if merged_df is not None and 'is_urban' in merged_df.columns:
        # Create figure with subplots for each performance metric
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        axes = axes.flatten()
        
        for i, (metric, title) in enumerate(zip(
            TRANSFORMED_METRICS, 
            ['Overall Performance', 'Football Performance', 'Hurling Performance', 'Code Balance']
        )):
            ax = axes[i]
            
            # Filter data
            if metric in ['transformed_football_performance', 'transformed_hurling_performance']:
                plot_data = merged_df[merged_df[metric] != 1].copy()
            else:
                plot_data = merged_df.copy()
            
            # Create boxplot
            sns.boxplot(
                x='is_urban',
                y=metric,
                data=plot_data,
                palette=['green', 'orange'],
                ax=ax
            )
            
            # Add individual points
            sns.stripplot(
                x='is_urban',
                y=metric,
                data=plot_data,
                color='black',
                alpha=0.5,
                jitter=True,
                ax=ax
            )
            
            # Labels and title
            ax.set_title(f"{title} by Urban/Rural Classification")
            ax.set_xlabel("")
            ax.set_xticklabels(['Rural', 'Urban'])
            ax.set_ylabel(f"{title} (Higher = Better)")
            
            # Calculate and show statistics
            urban_mean = plot_data[plot_data['is_urban']][metric].mean()
            rural_mean = plot_data[~plot_data['is_urban']][metric].mean()
            
            # T-test for significance
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(
                plot_data[plot_data['is_urban']][metric].dropna(),
                plot_data[~plot_data['is_urban']][metric].dropna(),
                equal_var=False
            )
            
            # Add significance annotation
            if p_val < 0.05:
                ax.annotate(f"Significant difference (p={p_val:.4f})", 
                           xy=(0.5, 0.05), 
                           xycoords='axes fraction', 
                           ha='center',
                           fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = OUTPUT_DIR / f"boxplot_urban_rural_all_metrics.png"
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        
        visualization_paths.append(output_path)
        logger.info(f"Saved urban/rural boxplot to {output_path}")
    
    # 3. Performance by dual/single club type
    if 'is_dual_2024' in club_df.columns:
        # Create figure with subplots for each performance metric
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['transformed_overall_performance', 'transformed_football_performance', 'transformed_hurling_performance']
        metric_titles = ['Overall Performance', 'Football Performance', 'Hurling Performance']
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[i]
            
            # Prepare data for dual clubs
            dual_data = club_df[club_df['is_dual_2024'] == True][metric]
            
            # For single clubs, filter appropriately
            if metric == 'transformed_football_performance':
                single_data = club_df[(club_df['is_dual_2024'] == False) & 
                                      (club_df[metric] != 1)][metric]
            elif metric == 'transformed_hurling_performance':
                single_data = club_df[(club_df['is_dual_2024'] == False) & 
                                      (club_df[metric] != 1)][metric]
            else:
                single_data = club_df[(club_df['is_dual_2024'] == False) & 
                                     ((club_df['transformed_football_performance'] != 1) | 
                                      (club_df['transformed_hurling_performance'] != 1))][metric]
            
            # Create dataframe for plotting
            plot_df = pd.DataFrame({
                'Performance': pd.concat([dual_data, single_data]),
                'Club Type': ['Dual'] * len(dual_data) + ['Single'] * len(single_data)
            })
            
            # Create boxplot
            sns.boxplot(
                x='Club Type',
                y='Performance',
                data=plot_df,
                palette=['purple', 'teal'],
                ax=ax
            )
            
            # Add individual points
            sns.stripplot(
                x='Club Type',
                y='Performance',
                data=plot_df,
                color='black',
                alpha=0.5,
                jitter=True,
                ax=ax
            )
            
            # Labels and title
            ax.set_title(f"{title} by Club Type")
            ax.set_xlabel("")
            ax.set_ylabel(f"{title} (Higher = Better)")
            
            # T-test for significance
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(
                dual_data.dropna(),
                single_data.dropna(),
                equal_var=False
            )
            
            # Add significance annotation
            if p_val < 0.05:
                ax.annotate(f"Significant difference (p={p_val:.4f})", 
                           xy=(0.5, 0.05), 
                           xycoords='axes fraction', 
                           ha='center',
                           fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = OUTPUT_DIR / f"boxplot_dual_single_advanced.png"
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        
        visualization_paths.append(output_path)
        logger.info(f"Saved dual/single club boxplot to {output_path}")
    
    return visualization_paths

def main():
    """
    Run the advanced EDA visualization generation process.
    """
    logger.info("Starting advanced EDA visualization generation...")
    
    # Load data
    data = load_data()
    
    # Generate scatterplots with trend lines
    scatter_paths = generate_scatterplots_with_trendlines(data)
    
    # Generate boxplots for categorical variables
    boxplot_paths = generate_categorical_boxplots(data)
    
    # Collect all visualization paths
    all_paths = scatter_paths + boxplot_paths
    
    # Log summary
    logger.info(f"Generated {len(all_paths)} visualizations:")
    logger.info(f"- {len(scatter_paths)} scatterplots with trend lines")
    logger.info(f"- {len(boxplot_paths)} boxplots for categorical variables")
    
    # Return paths for reference
    return {
        'scatter_paths': scatter_paths,
        'boxplot_paths': boxplot_paths,
        'all_paths': all_paths
    }

if __name__ == "__main__":
    main() 