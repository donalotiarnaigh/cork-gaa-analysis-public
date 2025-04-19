#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Club Similarity Network Visualization

This script generates an improved version of the club similarity network
visualization with better clustering and more visible connections.

Author: Daniel Tierney
Date: April 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from community import community_louvain  # For community detection
import adjustText as adjText

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("improved_club_network")

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "report_visualizations" / "intangible_alt"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Data path
CLUBS_DATA_PATH = BASE_DIR / "data" / "processed" / "cork_clubs_complete_graded.csv"

def load_data():
    """Load the required club data"""
    logger.info("Loading club data...")
    
    clubs_data = None
    if CLUBS_DATA_PATH.exists():
        try:
            clubs_data = pd.read_csv(CLUBS_DATA_PATH)
            logger.info(f"Loaded clubs data: {len(clubs_data)} clubs from {CLUBS_DATA_PATH}")
        except Exception as e:
            logger.error(f"Error loading clubs data: {e}")
    else:
        logger.error(f"Clubs data file not found at {CLUBS_DATA_PATH}")
    
    return clubs_data

def create_improved_network(clubs_data):
    """Create an improved network visualization with better clustering"""
    logger.info("Creating improved club similarity network...")
    
    if clubs_data is None:
        logger.error("Cannot create network, no clubs data available")
        return
    
    # Performance metrics to compare
    performance_metrics = ['overall_performance', 'football_performance', 'hurling_performance', 'code_balance']
    
    # Make sure all required variables exist
    missing_metrics = [m for m in performance_metrics if m not in clubs_data.columns]
    if missing_metrics:
        logger.error(f"Missing metrics in clubs data: {missing_metrics}")
        return
    
    # Filter to only include clubs with valid performance metrics
    valid_clubs = clubs_data[clubs_data['overall_performance'] < 6].copy()
    
    # IMPROVEMENT 1: Reduce number of clubs for clearer visualization
    top_n = min(30, len(valid_clubs))  # Reduced from 40 to 30
    top_clubs = valid_clubs.nsmallest(top_n, 'overall_performance')
    
    # Prepare data for network analysis
    X = top_clubs[performance_metrics].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate pairwise distances
    distances = euclidean_distances(X_scaled)
    
    # IMPROVEMENT 2: Adjust similarity calculation for more distinction
    # Use different sigma in Gaussian kernel
    sigma = np.mean(distances) * 0.8  # Adjust sigma for better separation
    similarities = np.exp(-distances**2 / (2 * sigma**2))
    
    # Create network
    G = nx.Graph()
    
    # Add nodes (clubs)
    for i, club in enumerate(top_clubs['Club']):
        # Add node with performance as attribute for coloring
        G.add_node(club, performance=top_clubs['overall_performance'].iloc[i])
    
    # IMPROVEMENT 3: Lower threshold for edge creation
    # Use 75th percentile instead of 90th to include more connections
    threshold = np.percentile(similarities.flatten(), 75)
    
    # Add edges (similarities above threshold)
    for i in range(len(top_clubs)):
        for j in range(i+1, len(top_clubs)):
            if similarities[i, j] > threshold:
                G.add_edge(top_clubs['Club'].iloc[i], 
                          top_clubs['Club'].iloc[j], 
                          weight=similarities[i, j])
    
    # IMPROVEMENT 4: Find communities for coloring clusters
    # First check if we have edges to find communities
    if len(G.edges) == 0:
        logger.warning("No edges in network, lowering threshold further")
        threshold = np.percentile(similarities.flatten(), 60)
        for i in range(len(top_clubs)):
            for j in range(i+1, len(top_clubs)):
                if similarities[i, j] > threshold:
                    G.add_edge(top_clubs['Club'].iloc[i], 
                              top_clubs['Club'].iloc[j], 
                              weight=similarities[i, j])
    
    # Find communities
    try:
        communities = community_louvain.best_partition(G)
        logger.info(f"Found {len(set(communities.values()))} communities")
    except Exception as e:
        logger.warning(f"Could not detect communities: {e}")
        communities = {node: 0 for node in G.nodes}
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # IMPROVEMENT 5: Adjust layout parameters
    pos = nx.spring_layout(G, k=0.4, iterations=100, seed=42)
    
    # Get community colors for nodes
    community_colors = [communities[node] for node in G.nodes]
    
    # Get performance values for node sizes (better performance = larger node)
    node_sizes = [150 + 100 * (6 - G.nodes[node]['performance']) for node in G.nodes]
    
    # IMPROVEMENT 6: Make edges more visible
    edge_weights = [G.edges[edge]['weight'] * 10 for edge in G.edges]  # Thicker edges
    
    # Draw network with community colors
    nodes = nx.draw_networkx_nodes(G, pos, 
                                 node_color=community_colors,
                                 node_size=node_sizes,
                                 alpha=0.8,
                                 cmap=plt.cm.tab10)
    
    # Draw edges with improved visibility
    edges = nx.draw_networkx_edges(G, pos, 
                                  width=edge_weights,
                                  alpha=0.7,  # Increased alpha
                                  edge_color='gray')
    
    # Add labels with adjustText to avoid overlapping
    texts = []
    for node, (x, y) in pos.items():
        # Only label the larger nodes (better performers) or community centers
        if G.nodes[node]['performance'] < 3 or node in ['Nemo Rangers', 'St. Finbarr\'s', 'Ballyhea', 'Sarsfields', 'Castlehaven']:
            texts.append(plt.text(x, y, node, fontsize=8, ha='center', va='center', 
                                 color='black', weight='bold', backgroundcolor='white', alpha=0.8))

    # Adjust text to avoid overlaps
    adjText.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3))
    
    # IMPROVEMENT 7: Add performance indicator using colorbar
    performance_nodes = nx.draw_networkx_nodes(G, pos, 
                                 node_size=0,  # Invisible nodes just for colorbar
                                 node_color=[G.nodes[node]['performance'] for node in G.nodes],
                                 cmap=plt.cm.RdYlGn_r,
                                 alpha=0)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=1, vmax=5))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Performance Grade (lower is better)')
    
    # Add title and remove axes
    plt.title('Club Similarity Network With Community Detection', fontsize=16)
    plt.axis('off')
    
    # Add explanation
    plt.figtext(0.5, 0.02, 
                "Note: Colors represent communities of clubs with similar performance profiles.\n"
                "Node size indicates performance (larger = better performance).\n"
                "Edge thickness represents strength of similarity between clubs.",
                ha='center', fontsize=10, style='italic')
    
    # Save figure
    output_path = OUTPUT_DIR / 'improved_club_similarity_network.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # IMPROVEMENT 8: Create a second visualization focused on top performers only
    logger.info("Creating focused network of top performers...")
    
    # Select even fewer clubs - just the top performers
    top_15 = valid_clubs.nsmallest(15, 'overall_performance')
    
    # Create a complete graph of these clubs
    G_top = nx.Graph()
    
    X_top = top_15[performance_metrics].values
    X_top_scaled = scaler.fit_transform(X_top)
    distances_top = euclidean_distances(X_top_scaled)
    similarities_top = np.exp(-distances_top**2 / (2 * sigma**2))
    
    # Add nodes
    for i, club in enumerate(top_15['Club']):
        G_top.add_node(club, performance=top_15['overall_performance'].iloc[i])
    
    # Add all edges but with weight based on similarity
    for i in range(len(top_15)):
        for j in range(i+1, len(top_15)):
            G_top.add_edge(top_15['Club'].iloc[i], 
                        top_15['Club'].iloc[j], 
                        weight=similarities_top[i, j])
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Position nodes using force-directed layout
    pos_top = nx.spring_layout(G_top, k=0.5, iterations=100, seed=42)
    
    # Color nodes by performance
    node_colors_top = [G_top.nodes[node]['performance'] for node in G_top.nodes]
    
    # Size nodes by inverse of performance (better performance = larger)
    node_sizes_top = [200 + 150 * (6 - G_top.nodes[node]['performance']) for node in G_top.nodes]
    
    # Scale edge widths based on weight but make them more visible
    # Remove weakest connections for clarity
    strong_edges = [(u, v) for u, v in G_top.edges if G_top[u][v]['weight'] > np.percentile([G_top[u][v]['weight'] for u, v in G_top.edges], 25)]
    
    edge_weights_top = [G_top[u][v]['weight'] * 15 for u, v in strong_edges]
    
    # Draw nodes
    nodes_top = nx.draw_networkx_nodes(G_top, pos_top, 
                                   node_color=node_colors_top,
                                   node_size=node_sizes_top,
                                   alpha=0.8,
                                   cmap=plt.cm.RdYlGn_r)
    
    # Draw edges - only the strong ones
    edges_top = nx.draw_networkx_edges(G_top, pos_top, 
                                    edgelist=strong_edges,
                                    width=edge_weights_top,
                                    alpha=0.7,
                                    edge_color='gray')
    
    # Add labels with adjustText to avoid overlapping
    texts_top = []
    for node, (x, y) in pos_top.items():
        texts_top.append(plt.text(x, y, node, fontsize=9, ha='center', va='center', 
                               color='black', weight='bold', backgroundcolor='white', alpha=0.7))

    # Adjust text to avoid overlaps
    adjText.adjust_text(texts_top, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3), 
                       expand_points=(1.5, 1.5), force_points=(0.5, 0.5))
    
    # Add colorbar
    sm_top = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=1, vmax=5))
    sm_top.set_array([])
    cbar_top = plt.colorbar(sm_top, ax=plt.gca())
    cbar_top.set_label('Performance Grade (lower is better)')
    
    # Add title and remove axes
    plt.title('Similarity Network of Top 15 Performing Clubs', fontsize=16)
    plt.axis('off')
    
    # Add explanation
    plt.figtext(0.5, 0.02, 
                "Note: This network shows the top 15 performing clubs. Node size indicates performance (larger = better).\n"
                "Edge thickness represents strength of similarity. Only the strongest connections are shown.",
                ha='center', fontsize=10, style='italic')
    
    # Save figure
    output_path = OUTPUT_DIR / 'top_performers_network.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Improved network visualizations created successfully")

def main():
    """Main function"""
    logger.info("Starting improved club network visualization...")
    
    # Load data
    clubs_data = load_data()
    
    # Create improved network visualization
    create_improved_network(clubs_data)
    
    logger.info("Improved club network visualization completed successfully")

if __name__ == "__main__":
    main() 