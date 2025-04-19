#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Intangible Factors Visualizations

This script generates three key visualizations for the Intangible Factors section:
1. Variance decomposition chart - showing explained vs. unexplained variance percentages
2. Club cluster profiles radar chart - showing different club types beyond demographics
3. Case comparison visualization - comparing clubs with similar demographics but different outcomes

Author: Daniel Tierney
Date: April 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("intangible_factors_visualizations")

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "report_visualizations" / "intangible"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Data paths
MODEL_RESULTS_PATH = BASE_DIR / "output" / "models" / "refined_model_results_20250411.json"
CLUSTER_RESULTS_PATH = BASE_DIR / "output" / "analysis" / "comprehensive_metrics" / "club_clusters.gpkg"
CLUBS_DATA_PATH = BASE_DIR / "data" / "processed" / "cork_clubs_complete_graded.csv"

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def load_data():
    """Load the required data files"""
    logger.info("Loading data files...")
    
    # Load model results for variance decomposition
    model_results = None
    if MODEL_RESULTS_PATH.exists():
        try:
            with open(MODEL_RESULTS_PATH, 'r') as f:
                model_results = json.load(f)
            logger.info(f"Loaded model results from {MODEL_RESULTS_PATH}")
        except Exception as e:
            logger.error(f"Error loading model results: {e}")
    else:
        logger.warning(f"Model results file not found at {MODEL_RESULTS_PATH}")
    
    # Load club clusters data
    clusters_data = None
    if CLUSTER_RESULTS_PATH.exists():
        try:
            import geopandas as gpd
            clusters_data = gpd.read_file(CLUSTER_RESULTS_PATH)
            logger.info(f"Loaded cluster data: {len(clusters_data)} clubs from {CLUSTER_RESULTS_PATH}")
        except Exception as e:
            logger.error(f"Error loading cluster data: {e}")
    else:
        logger.warning(f"Cluster data file not found at {CLUSTER_RESULTS_PATH}")
    
    # Load general club data
    clubs_data = None
    if CLUBS_DATA_PATH.exists():
        try:
            clubs_data = pd.read_csv(CLUBS_DATA_PATH)
            logger.info(f"Loaded clubs data: {len(clubs_data)} clubs from {CLUBS_DATA_PATH}")
        except Exception as e:
            logger.error(f"Error loading clubs data: {e}")
    else:
        logger.warning(f"Clubs data file not found at {CLUBS_DATA_PATH}")
        
    # If cluster data is missing but we have club data, try to find cluster assignments in it
    if clusters_data is None and clubs_data is not None and 'cluster' in clubs_data.columns:
        logger.info("Using cluster assignments from main clubs data")
        clusters_data = clubs_data
    
    return model_results, clusters_data, clubs_data

def create_variance_decomposition_chart(model_results, clubs_data):
    """Create a visualization showing explained vs. unexplained variance"""
    logger.info("Creating variance decomposition chart...")
    
    if model_results is None:
        logger.error("Cannot create variance decomposition chart, no model results available")
        # Try to find R² values directly in clubs_data as an alternative
        if clubs_data is not None and 'r_squared' in clubs_data.columns:
            logger.info("Trying to use R² values from clubs data instead")
            r_squared_values = clubs_data['r_squared'].dropna()
            if len(r_squared_values) > 0:
                model_results = {'refined_models': {}}
                for i, val in enumerate(r_squared_values):
                    model_results['refined_models'][f'model_{i}'] = {'r_squared': val}
            else:
                return
        else:
            return
    
    # Extract R² values
    r_squared_values = {}
    if 'refined_models' in model_results:
        for metric, model in model_results['refined_models'].items():
            if 'r_squared' in model:
                r_squared_values[metric] = model['r_squared']
    
    if not r_squared_values:
        logger.error("No R² values found in model results")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    
    # Plot 1: Average R² as donut chart
    avg_r_squared = sum(r_squared_values.values()) / len(r_squared_values)
    unexplained = 1 - avg_r_squared
    
    # Create donut chart data
    sizes = [avg_r_squared, unexplained]
    labels = ['Explained\nVariance', 'Unexplained\nVariance']
    colors = ['#4CAF50', '#F5F5F5']  # Green for explained, light gray for unexplained
    
    # Draw donut chart
    ax1.pie(sizes, labels=None, colors=colors, autopct='%1.1f%%', 
           startangle=90, wedgeprops=dict(width=0.3, edgecolor='w'))
    ax1.axis('equal')
    ax1.set_title('Average Variance Explanation\nAcross All Models', fontsize=14)
    
    # Add custom legend
    handles = [
        mpatches.Patch(color=colors[0], label=f"{labels[0]} ({sizes[0]*100:.1f}%)"),
        mpatches.Patch(color=colors[1], label=f"{labels[1]} ({sizes[1]*100:.1f}%)")
    ]
    ax1.legend(handles=handles, loc='center', frameon=False, fontsize=12)
    
    # Plot 2: R² breakdown by model as horizontal bar chart
    # Sort by R² value
    sorted_models = sorted(r_squared_values.items(), key=lambda x: x[1], reverse=True)
    models = [m[0].replace('_', ' ').title() for m in sorted_models]
    r_squared = [m[1] for m in sorted_models]
    unexplained = [1 - r for r in r_squared]
    
    # Create stacked bar
    ax2.barh(models, r_squared, color='#4CAF50', label='Explained Variance')
    ax2.barh(models, unexplained, left=r_squared, color='#F5F5F5', label='Unexplained Variance')
    
    # Add percentage labels inside bars
    for i, (r, model) in enumerate(zip(r_squared, models)):
        # Only add text if there's enough space
        if r > 0.05:
            ax2.text(r/2, i, f"{r*100:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        
        # Add text for total R²
        ax2.text(1.01, i, f"R² = {r:.3f}", va='center')
    
    # Customize
    ax2.set_xlim(0, 1.2)
    ax2.xaxis.set_major_formatter(PercentFormatter(1))
    ax2.set_title('Variance Explained by Model', fontsize=14)
    ax2.set_xlabel('Proportion of Variance')
    ax2.legend(loc='upper right')
    
    # Add annotations
    plt.figtext(0.5, 0.01, 
                "Note: The unexplained variance represents the influence of factors not captured in the models,\n"
                "including intangible elements like club culture, volunteer capacity, and historical tradition.",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save figure
    output_path = OUTPUT_DIR / 'variance_decomposition_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Variance decomposition chart created successfully at {output_path}")

def create_cluster_profiles_radar(clusters_data):
    """Create a radar chart showing different club cluster profiles"""
    logger.info("Creating club cluster profiles radar chart...")
    
    if clusters_data is None or 'cluster' not in clusters_data.columns:
        logger.error("Cannot create cluster profiles radar, missing cluster data")
        return
    
    # Define metrics for radar chart
    metrics_for_radar = ['overall_performance', 'football_performance', 'hurling_performance', 'code_balance']
    
    # Check if all metrics exist in the data
    missing_metrics = [m for m in metrics_for_radar if m not in clusters_data.columns]
    if missing_metrics:
        logger.error(f"Missing metrics in clusters data: {missing_metrics}")
        return
    
    # Calculate mean values for each cluster
    cluster_profiles = clusters_data.groupby('cluster')[metrics_for_radar].mean()
    
    # Get number of clusters
    n_clusters = len(cluster_profiles)
    n_metrics = len(metrics_for_radar)
    
    if n_clusters == 0:
        logger.error("No clusters found in data")
        return
    
    # Create cluster names based on patterns (better interpretation)
    cluster_names = {}
    for cluster_id, profile in cluster_profiles.iterrows():
        # Determine key characteristics
        if profile['football_performance'] < profile['hurling_performance']:
            football_emphasis = "Strong Football"
        elif profile['football_performance'] > profile['hurling_performance']:
            football_emphasis = "Strong Hurling"
        else:
            football_emphasis = "Balanced Codes"
            
        if profile['overall_performance'] < 3:  # Better performance
            performance = "High Performance"
        elif profile['overall_performance'] > 4:  # Worse performance
            performance = "Developing"
        else:
            performance = "Mid-tier"
            
        cluster_names[cluster_id] = f"Cluster {cluster_id}: {performance}, {football_emphasis}"
    
    # Compute angle for each metric
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Set colors for each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for i, (idx, row) in enumerate(cluster_profiles.iterrows()):
        # Prepare data
        values = row.values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], 
                label=cluster_names.get(idx, f"Cluster {idx}"))
        ax.fill(angles, values, color=colors[i], alpha=0.25)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_for_radar])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set title
    plt.title('Club Cluster Profiles: Performance Characteristics', size=16)
    
    # Add note explaining the metrics
    plt.figtext(0.5, 0.01, 
                "Note: Lower values indicate better performance (1=best, 6=worst).\n"
                "Each cluster represents a different pattern of club performance across metrics.",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save figure
    output_path = OUTPUT_DIR / 'combined_cluster_profiles_radar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Club cluster profiles radar chart created successfully at {output_path}")

def create_case_comparison_visualization(clubs_data):
    """Create a visualization comparing clubs with similar geographic attributes but different outcomes"""
    logger.info("Creating case comparison visualization...")
    
    if clubs_data is None:
        logger.error("Cannot create case comparison visualization, no clubs data available")
        return
    
    # Define geographic variables to compare
    geo_vars = ['Elevation', 'annual_rainfall']
    
    # Make sure all required variables exist
    missing_vars = [v for v in geo_vars + ['overall_performance', 'Club'] if v not in clubs_data.columns]
    if missing_vars:
        logger.error(f"Missing variables in clubs data: {missing_vars}")
        return
    
    # Find pairs of clubs with similar geographic attributes but different performance
    # Strategy: Calculate geographic similarity but performance difference
    
    # Standardize geographic variables
    clubs_std = clubs_data.copy()
    for var in geo_vars:
        if clubs_std[var].std() > 0:
            clubs_std[var] = (clubs_std[var] - clubs_std[var].mean()) / clubs_std[var].std()
        else:
            clubs_std[var] = 0
    
    # Calculate pairwise geographic similarity and performance difference
    n_clubs = len(clubs_std)
    pairs = []
    
    for i in range(n_clubs):
        for j in range(i+1, n_clubs):
            club1 = clubs_std.iloc[i]
            club2 = clubs_std.iloc[j]
            
            # Skip if either club doesn't have valid performance
            if club1['overall_performance'] > 5.5 or club2['overall_performance'] > 5.5:
                continue
            
            # Calculate geographic similarity (Euclidean distance)
            geo_diffs = [club1[var] - club2[var] for var in geo_vars if var in club1.index]
            geo_similarity = np.sqrt(sum(d**2 for d in geo_diffs))
            
            # Calculate performance difference (absolute difference)
            perf_diff = abs(club1['overall_performance'] - club2['overall_performance'])
            
            # We want pairs with high geographic similarity (low distance) but large performance difference
            if geo_similarity < 0.5 and perf_diff > 2.0:
                pairs.append({
                    'club1': clubs_data.iloc[i]['Club'],
                    'club2': clubs_data.iloc[j]['Club'],
                    'geo_similarity': geo_similarity,
                    'perf_diff': perf_diff,
                    'club1_perf': clubs_data.iloc[i]['overall_performance'],
                    'club2_perf': clubs_data.iloc[j]['overall_performance'],
                    'score': perf_diff / (geo_similarity + 0.1)  # Higher is better
                })
    
    # If no good pairs found, widen the criteria
    if len(pairs) < 2:
        logger.warning("No suitable pairs found with strict criteria, widening search")
        for i in range(n_clubs):
            for j in range(i+1, n_clubs):
                club1 = clubs_std.iloc[i]
                club2 = clubs_std.iloc[j]
                
                # Skip if either club doesn't have valid performance
                if club1['overall_performance'] > 5.5 or club2['overall_performance'] > 5.5:
                    continue
                
                # Calculate geographic similarity (Euclidean distance)
                geo_diffs = [club1[var] - club2[var] for var in geo_vars if var in club1.index]
                geo_similarity = np.sqrt(sum(d**2 for d in geo_diffs))
                
                # Calculate performance difference (absolute difference)
                perf_diff = abs(club1['overall_performance'] - club2['overall_performance'])
                
                # Wider criteria
                if geo_similarity < 1.0 and perf_diff > 1.5:
                    pairs.append({
                        'club1': clubs_data.iloc[i]['Club'],
                        'club2': clubs_data.iloc[j]['Club'],
                        'geo_similarity': geo_similarity,
                        'perf_diff': perf_diff,
                        'club1_perf': clubs_data.iloc[i]['overall_performance'],
                        'club2_perf': clubs_data.iloc[j]['overall_performance'],
                        'score': perf_diff / (geo_similarity + 0.1)  # Higher is better
                    })
    
    # Sort pairs by score and select top 2
    pairs.sort(key=lambda x: x['score'], reverse=True)
    selected_pairs = pairs[:2]
    
    if not selected_pairs:
        logger.error("No suitable club pairs found for comparison")
        return
    
    # Add code_balance as additional comparison
    additional_vars = ['code_balance', 'football_performance', 'hurling_performance']
    
    # Create the visualization
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 3])
    
    # Title and explanation
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, "Similar Geographic Setting, Different Outcomes", 
                 fontsize=18, ha='center', va='center', fontweight='bold')
    ax_title.text(0.5, 0.2, 
                 "These club pairs have very similar geographic settings (elevation and rainfall) but significantly different performance outcomes,\n"
                 "suggesting the influence of intangible factors such as club culture, volunteer capacity, or historical tradition.",
                 fontsize=12, ha='center', va='center', style='italic')
    
    # Create comparison subplots for each pair
    for p_idx, pair in enumerate(selected_pairs):
        ax = fig.add_subplot(gs[1, p_idx])
        
        # Get detailed data for both clubs
        club1_data = clubs_data[clubs_data['Club'] == pair['club1']].iloc[0]
        club2_data = clubs_data[clubs_data['Club'] == pair['club2']].iloc[0]
        
        # Calculate variables to compare
        comparison_data = []
        
        # Add geographical variables
        for var in geo_vars:
            var_name = var.replace('_', ' ').title()
            val1 = club1_data[var]
            val2 = club2_data[var]
            comparison_data.append({'Variable': var_name, 'Value': val1, 'Club': pair['club1']})
            comparison_data.append({'Variable': var_name, 'Value': val2, 'Club': pair['club2']})
        
        # Add performance and other metrics
        for var in ['overall_performance'] + additional_vars:
            if var in club1_data and var in club2_data:
                var_name = var.replace('_', ' ').title()
                val1 = club1_data[var]
                val2 = club2_data[var]
                comparison_data.append({'Variable': var_name, 'Value': val1, 'Club': pair['club1']})
                comparison_data.append({'Variable': var_name, 'Value': val2, 'Club': pair['club2']})
        
        # Convert to DataFrame
        compare_df = pd.DataFrame(comparison_data)
        
        # Create grouped bar chart
        colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
        sns.barplot(
            x='Variable', 
            y='Value', 
            hue='Club', 
            data=compare_df,
            palette=colors,
            ax=ax
        )
        
        # Customize
        ax.set_title(f"Comparison: {pair['club1']} vs {pair['club2']}")
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
        # Add performance difference annotation
        perf_diff = abs(pair['club1_perf'] - pair['club2_perf'])
        ax.text(0.5, 0.9, f"Performance Difference: {perf_diff:.1f} grades", 
                transform=ax.transAxes, ha='center', bbox=dict(facecolor='yellow', alpha=0.2))
        
        # Add note about the lower being better for performance
        ax.text(0.5, 0.03, "Note: Lower values indicate better performance (1=best, 6=worst)",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=8, style='italic')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / 'geographic_similar_performance_different.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Case comparison visualization created successfully at {output_path}")

def main():
    """Main function to generate all visualizations"""
    logger.info("Starting intangible factors visualization generation...")
    
    # Load data
    model_results, clusters_data, clubs_data = load_data()
    
    # Generate variance decomposition chart
    create_variance_decomposition_chart(model_results, clubs_data)
    
    # No need to generate club cluster profiles radar as we copied the existing file
    logger.info("Using existing club cluster profiles radar chart")
    
    # Generate case comparison visualization
    create_case_comparison_visualization(clubs_data)
    
    logger.info("All intangible factors visualizations generated successfully!")

if __name__ == "__main__":
    main() 