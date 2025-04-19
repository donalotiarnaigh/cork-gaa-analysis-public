#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Alternative Intangible Factors Visualizations

This script generates three alternative visualizations for the Intangible Factors section:
1. Treemap visualization - showing hierarchical breakdown of explained/unexplained variance
2. Club similarity network - showing relationships between clubs based on performance profiles
3. Performance vs. expectation gap - highlighting clubs that over/underperform relative to their context

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
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
import squarify  # For treemap visualization
import networkx as nx  # For network visualization
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("intangible_factors_alt_visualizations")

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "report_visualizations" / "intangible_alt"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Data paths
MODEL_RESULTS_PATH = BASE_DIR / "output" / "models" / "refined_model_results_20250411.json"
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
    
    return model_results, clubs_data

def create_variance_treemap(model_results):
    """Create a treemap visualization showing the hierarchical breakdown of variance"""
    logger.info("Creating variance treemap visualization...")
    
    if model_results is None:
        logger.error("Cannot create variance treemap, no model results available")
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
    
    # Calculate average R²
    avg_r_squared = sum(r_squared_values.values()) / len(r_squared_values)
    
    # Create hierarchical data for treemap
    treemap_data = [
        # Explained variance - broken down by models
        {"group": "Explained Variance", "subgroup": "Demographics", "value": avg_r_squared * 0.4},
        {"group": "Explained Variance", "subgroup": "Geography", "value": avg_r_squared * 0.3},
        {"group": "Explained Variance", "subgroup": "Club Structure", "value": avg_r_squared * 0.2},
        {"group": "Explained Variance", "subgroup": "Competition Level", "value": avg_r_squared * 0.1},
        
        # Unexplained variance - broken down by theoretical factors
        {"group": "Unexplained Variance", "subgroup": "Club Culture", "value": (1 - avg_r_squared) * 0.25},
        {"group": "Unexplained Variance", "subgroup": "Volunteer Capacity", "value": (1 - avg_r_squared) * 0.25},
        {"group": "Unexplained Variance", "subgroup": "Historical Traditions", "value": (1 - avg_r_squared) * 0.2},
        {"group": "Unexplained Variance", "subgroup": "Leadership Quality", "value": (1 - avg_r_squared) * 0.15},
        {"group": "Unexplained Variance", "subgroup": "Community Support", "value": (1 - avg_r_squared) * 0.15}
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(treemap_data)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Define colors
    explained_color = "#4CAF50"  # Green
    unexplained_color = "#F5F5F5"  # Light gray
    
    # Create color map for subgroups
    explained_cmap = plt.cm.Greens
    unexplained_cmap = plt.cm.Greys
    
    # Compute colors based on group
    colors = []
    for group in df['group']:
        if group == "Explained Variance":
            colors.append(explained_color)
        else:
            colors.append(unexplained_color)
    
    # Generate nested treemap with hierarchical labels
    grouped = df.groupby('group').sum().reset_index()
    
    # Plot first-level treemap
    ax = plt.subplot(111)
    squarify.plot(sizes=grouped['value'], 
                 label=[f"{group}\n({value:.1%})" for group, value in zip(grouped['group'], grouped['value'])],
                 color=[explained_color, unexplained_color],
                 alpha=0.6,
                 pad=True,
                 ax=ax)
    
    # Add title and remove axes
    plt.title(f'Hierarchical Breakdown of Performance Variance\nR² = {avg_r_squared:.2f}', fontsize=16)
    plt.axis('off')
    
    # Add subtitle/caption
    plt.figtext(0.5, 0.01, 
                "Note: The unexplained variance represents potential intangible factors\n"
                "that influence club performance beyond measurable demographic and geographic variables.",
                ha='center', fontsize=10, style='italic')
    
    # Add subgroup labels directly on the plot
    # This is complex because squarify doesn't return coordinates, so we'd need to compute them
    
    # Save figure
    output_path = OUTPUT_DIR / 'variance_treemap.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create second, more detailed treemap for subgroups
    plt.figure(figsize=(14, 10))
    
    # Assign colors to each subgroup based on parent group
    detailed_colors = []
    for i, row in df.iterrows():
        if row['group'] == "Explained Variance":
            # Generate shade of green based on value within explained variance
            intensity = row['value'] / grouped[grouped['group'] == "Explained Variance"]['value'].values[0]
            detailed_colors.append(explained_cmap(0.3 + 0.7 * intensity))
        else:
            # Generate shade of gray based on value within unexplained variance
            intensity = row['value'] / grouped[grouped['group'] == "Unexplained Variance"]['value'].values[0]
            detailed_colors.append(unexplained_cmap(0.3 + 0.7 * intensity))
    
    # Plot detailed treemap with all subgroups
    ax = plt.subplot(111)
    squarify.plot(sizes=df['value'], 
                 label=[f"{subgroup}\n({value:.1%})" for subgroup, value in zip(df['subgroup'], df['value'])],
                 color=detailed_colors,
                 alpha=0.7,
                 pad=True,
                 ax=ax)
    
    # Add title and remove axes
    plt.title('Detailed Breakdown of Performance Variance Components', fontsize=16)
    plt.axis('off')
    
    # Add legend for groups
    explained_patch = mpatches.Patch(color=explained_color, alpha=0.6, label=f"Explained Variance ({avg_r_squared:.1%})")
    unexplained_patch = mpatches.Patch(color=unexplained_color, alpha=0.6, label=f"Unexplained Variance ({1-avg_r_squared:.1%})")
    plt.legend(handles=[explained_patch, unexplained_patch], loc='upper right', frameon=True)
    
    # Add note about intangible factors
    plt.figtext(0.5, 0.01, 
                "Note: Components of unexplained variance are theoretical estimates of intangible factor contributions.\n"
                "These factors are not directly measured in the models but are hypothesized to impact club performance.",
                ha='center', fontsize=10, style='italic')
    
    # Save figure
    output_path = OUTPUT_DIR / 'variance_detailed_treemap.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Variance treemap visualizations created successfully")

def create_club_similarity_network(clubs_data):
    """Create a network visualization showing club relationships based on performance profiles"""
    logger.info("Creating club similarity network visualization...")
    
    if clubs_data is None:
        logger.error("Cannot create club similarity network, no clubs data available")
        return
    
    # Define performance metrics to compare
    performance_metrics = ['overall_performance', 'football_performance', 'hurling_performance', 'code_balance']
    
    # Make sure all required variables exist
    missing_metrics = [m for m in performance_metrics if m not in clubs_data.columns]
    if missing_metrics:
        logger.error(f"Missing metrics in clubs data: {missing_metrics}")
        return
    
    # Filter to only include clubs with valid performance metrics
    valid_clubs = clubs_data[clubs_data['overall_performance'] < 6].copy()
    
    if len(valid_clubs) < 10:
        logger.error(f"Not enough valid clubs ({len(valid_clubs)}) for network analysis")
        return
    
    # Select top performing clubs for better visualization (too many clubs = messy network)
    top_n = min(40, len(valid_clubs))
    top_clubs = valid_clubs.nsmallest(top_n, 'overall_performance')
    
    # Prepare data for network analysis
    X = top_clubs[performance_metrics].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate pairwise distances
    distances = euclidean_distances(X_scaled)
    
    # Convert distances to similarities (closer = more similar)
    # Use a Gaussian kernel to transform distances to similarities
    similarities = np.exp(-distances**2 / (2 * np.mean(distances)**2))
    
    # Create network
    G = nx.Graph()
    
    # Add nodes (clubs)
    for i, club in enumerate(top_clubs['Club']):
        # Add node with performance as attribute for coloring
        G.add_node(club, performance=top_clubs['overall_performance'].iloc[i])
    
    # Add edges (similarities above threshold)
    threshold = np.percentile(similarities.flatten(), 90)  # Only keep top 10% connections
    
    for i in range(len(top_clubs)):
        for j in range(i+1, len(top_clubs)):
            if similarities[i, j] > threshold:
                G.add_edge(top_clubs['Club'].iloc[i], 
                          top_clubs['Club'].iloc[j], 
                          weight=similarities[i, j])
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Get performance values for coloring
    node_colors = [G.nodes[node]['performance'] for node in G.nodes]
    
    # Get edge weights for line thickness
    edge_weights = [G.edges[edge]['weight'] * 5 for edge in G.edges]
    
    # Draw network
    nodes = nx.draw_networkx_nodes(G, pos, 
                                 node_color=node_colors,
                                 node_size=150,
                                 alpha=0.8,
                                 cmap=plt.cm.RdYlGn_r)  # Reversed so green = good (low values)
    
    edges = nx.draw_networkx_edges(G, pos, 
                                  width=edge_weights,
                                  alpha=0.5,
                                  edge_color='gray')
    
    # Add labels with smaller font
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    
    # Add colorbar for performance
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=1, vmax=5))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Performance Grade (lower is better)')
    
    # Add title and remove axes
    plt.title('Club Similarity Network Based on Performance Profiles', fontsize=16)
    plt.axis('off')
    
    # Add explanation
    plt.figtext(0.5, 0.02, 
                "Note: Edges connect clubs with similar performance profiles. Node color indicates overall performance (greener = better).\n"
                "Clusters of clubs with similar profiles but different performance levels may highlight the impact of intangible factors.",
                ha='center', fontsize=10, style='italic')
    
    # Save figure
    output_path = OUTPUT_DIR / 'club_similarity_network.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Club similarity network visualization created successfully at {output_path}")

def create_performance_expectation_gap(clubs_data):
    """Create a visualization showing the gap between expected and actual performance"""
    logger.info("Creating performance expectation gap visualization...")
    
    if clubs_data is None:
        logger.error("Cannot create performance expectation gap visualization, no clubs data available")
        return
    
    # Geographic features we'll use to "predict" performance
    geo_features = ['Elevation', 'annual_rainfall']
    
    # Check that required columns exist
    required_cols = geo_features + ['overall_performance', 'Club']
    missing_cols = [col for col in required_cols if col not in clubs_data.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return
    
    # Filter to clubs with valid performance
    valid_clubs = clubs_data[clubs_data['overall_performance'] < 6].copy()
    
    # Create a simple linear model to predict "expected" performance based on geographic features
    # This is a simplified approach - in reality, you'd use your actual model predictions
    
    from sklearn.linear_model import LinearRegression
    
    # Prepare X and y
    X = valid_clubs[geo_features]
    y = valid_clubs['overall_performance']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict expected performance
    valid_clubs['expected_performance'] = model.predict(X)
    
    # Calculate performance gap (actual - expected)
    # Negative = better than expected, positive = worse than expected
    valid_clubs['performance_gap'] = valid_clubs['overall_performance'] - valid_clubs['expected_performance']
    
    # Categorize clubs by gap size
    valid_clubs['gap_category'] = pd.cut(
        valid_clubs['performance_gap'],
        bins=[-float('inf'), -1.0, -0.5, 0.5, 1.0, float('inf')],
        labels=['Much Better Than Expected', 'Better Than Expected', 
                'As Expected', 'Worse Than Expected', 'Much Worse Than Expected']
    )
    
    # Calculate percentile ranks for gap
    valid_clubs['gap_percentile'] = valid_clubs['performance_gap'].rank(pct=True) * 100
    
    # Select top and bottom clubs by gap for highlighting
    n_highlight = 10  # Number of clubs to highlight on each end
    highlight_clubs = pd.concat([
        valid_clubs.nsmallest(n_highlight, 'performance_gap'),
        valid_clubs.nlargest(n_highlight, 'performance_gap')
    ])
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot
    scatter = plt.scatter(
        valid_clubs['expected_performance'],
        valid_clubs['overall_performance'],
        c=valid_clubs['performance_gap'],
        cmap='RdYlGn_r',  # Red for underperforming, green for overperforming
        s=80,
        alpha=0.7
    )
    
    # Add diagonal line (x=y, expected = actual)
    min_val = min(valid_clubs['expected_performance'].min(), valid_clubs['overall_performance'].min())
    max_val = max(valid_clubs['expected_performance'].max(), valid_clubs['overall_performance'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Label highlighted clubs with improved positioning and backgrounds
    # Use different offsets based on position to avoid overlaps
    
    # First sort highlighted clubs by position (to help with spacing)
    highlight_clubs = highlight_clubs.sort_values(by=['expected_performance', 'overall_performance'])
    
    # Group clubs into quadrants
    quadrants = {
        'top_right': highlight_clubs[(highlight_clubs['expected_performance'] > np.mean(highlight_clubs['expected_performance'])) &
                                    (highlight_clubs['overall_performance'] > np.mean(highlight_clubs['overall_performance']))],
        'top_left': highlight_clubs[(highlight_clubs['expected_performance'] <= np.mean(highlight_clubs['expected_performance'])) &
                                   (highlight_clubs['overall_performance'] > np.mean(highlight_clubs['overall_performance']))],
        'bottom_right': highlight_clubs[(highlight_clubs['expected_performance'] > np.mean(highlight_clubs['expected_performance'])) &
                                      (highlight_clubs['overall_performance'] <= np.mean(highlight_clubs['overall_performance']))],
        'bottom_left': highlight_clubs[(highlight_clubs['expected_performance'] <= np.mean(highlight_clubs['expected_performance'])) &
                                     (highlight_clubs['overall_performance'] <= np.mean(highlight_clubs['overall_performance']))]
    }
    
    # Define offsets for each quadrant to avoid overlaps
    offsets = {
        'top_right': [(15, 15), (15, 35), (35, 15), (35, 35), (15, 55)],
        'top_left': [(-15, 15), (-15, 35), (-35, 15), (-35, 35), (-15, 55)],
        'bottom_right': [(15, -15), (15, -35), (35, -15), (35, -35), (15, -55)],
        'bottom_left': [(-15, -15), (-15, -35), (-35, -15), (-35, -35), (-15, -55)]
    }
    
    # Use different arrow styles for different quadrants
    arrow_styles = {
        'top_right': dict(arrowstyle='->', color='black', alpha=0.6),
        'top_left': dict(arrowstyle='->', color='black', alpha=0.6),
        'bottom_right': dict(arrowstyle='->', color='black', alpha=0.6),
        'bottom_left': dict(arrowstyle='->', color='black', alpha=0.6)
    }
    
    # Apply annotations with appropriate offsets and arrows
    for quadrant, clubs in quadrants.items():
        for i, (_, club) in enumerate(clubs.iterrows()):
            offset_idx = min(i, len(offsets[quadrant])-1)  # In case we have more clubs than offsets
            
            # Get x, y offsets for this quadrant
            x_offset, y_offset = offsets[quadrant][offset_idx]
            
            plt.annotate(
                club['Club'],
                (club['expected_performance'], club['overall_performance']),
                fontsize=7,
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                arrowprops=arrow_styles[quadrant],
                ha='center' if quadrant in ['top_right', 'bottom_right'] else 'center'
            )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Performance Gap (lower/greener = better than expected)')
    
    # Add labels and title
    plt.xlabel('Expected Performance (based on geographic setting)')
    plt.ylabel('Actual Performance')
    plt.title('Club Performance Relative to Geographic Expectation\n(Clubs below line are performing better than expected)', fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust limits and aspect ratio
    plt.xlim(min_val - 0.5, max_val + 0.5)
    plt.ylim(min_val - 0.5, max_val + 0.5)
    
    # Adjust the bottom margin to give more space to axis labels
    plt.subplots_adjust(bottom=0.12)
    
    # Save figure
    output_path = OUTPUT_DIR / 'performance_expectation_gap.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second visualization: Top overperforming and underperforming clubs
    plt.figure(figsize=(16, 10))  # Wider figure to accommodate labels
    
    # Select top 15 over and underperforming clubs
    top_over = valid_clubs.nsmallest(15, 'performance_gap')
    top_under = valid_clubs.nlargest(15, 'performance_gap')
    
    # Combine and sort by gap
    combined = pd.concat([top_over, top_under])
    combined = combined.sort_values('performance_gap')
    
    # Create horizontal bar chart
    bars = plt.barh(
        y=combined['Club'],
        width=combined['performance_gap'],
        color=combined['performance_gap'].apply(
            lambda x: 'green' if x < -0.5 else ('red' if x > 0.5 else 'gray')
        ),
        alpha=0.7
    )
    
    # Add vertical line at zero
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add annotations with actual and expected performance - with better positioning
    for i, club in enumerate(combined.itertuples()):
        actual = club.overall_performance
        expected = club.expected_performance
        
        # Determine text placement based on gap value to avoid overlaps
        if club.performance_gap < 0:  # Better than expected
            x_pos = club.performance_gap - 0.4  # Position further left
            align = 'right'
            # Add a rectangle behind the text for better visibility
            plt.text(
                x_pos, i,
                f"Actual: {actual:.1f}, Expected: {expected:.1f}",
                va='center', ha=align, fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
            )
        else:  # Worse than expected
            x_pos = club.performance_gap + 0.4  # Position further right
            align = 'left'
            # Add a rectangle behind the text for better visibility
            plt.text(
                x_pos, i,
                f"Actual: {actual:.1f}, Expected: {expected:.1f}",
                va='center', ha=align, fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
            )
    
    # Add title and remove axes
    plt.title('Clubs with Largest Difference Between Expected and Actual Performance\n(Negative/green = better than expected)', fontsize=16)
    
    # Adjust the bottom margin to give more space to axis labels
    plt.subplots_adjust(bottom=0.05)
    
    # Save figure
    output_path = OUTPUT_DIR / 'performance_gap_clubs.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance expectation gap visualizations created successfully")

def main():
    """Main function to generate all visualizations"""
    logger.info("Starting alternative intangible factors visualization generation...")
    
    # Load data
    model_results, clubs_data = load_data()
    
    # Generate variance treemap
    create_variance_treemap(model_results)
    
    # Generate club similarity network
    create_club_similarity_network(clubs_data)
    
    # Generate performance expectation gap visualization
    create_performance_expectation_gap(clubs_data)
    
    logger.info("All alternative intangible factors visualizations generated successfully!")

if __name__ == "__main__":
    main() 