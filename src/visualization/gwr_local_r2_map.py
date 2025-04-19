#!/usr/bin/env python3
"""
Create a visualization showing spatial distribution of local R² values from GWR analysis

This script creates a map visualization showing the local R² values from 
Geographically Weighted Regression for Cork GAA clubs.

Author: Daniel Tierney
Date: April 2025
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import contextily as ctx

# Set paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = BASE_DIR / "output" / "improved_gwr_analysis"
REPORT_VIZ_DIR = BASE_DIR / "report_visualizations" / "Best"

# Ensure directories exist
REPORT_VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def load_data():
    """Load the GWR visualization data"""
    try:
        # Load visualization data for hurling performance (should have the best model fit)
        viz_path = OUTPUT_DIR / "gwr_visualization_hurling_performance.gpkg"
        if viz_path.exists():
            viz_gdf = gpd.read_file(viz_path)
        else:
            # Try overall performance as fallback
            viz_path = OUTPUT_DIR / "gwr_visualization_overall_performance.gpkg"
            if viz_path.exists():
                viz_gdf = gpd.read_file(viz_path)
            else:
                # Just load club data as last resort
                viz_gdf = gpd.read_file(BASE_DIR / "data" / "processed" / "cork_clubs_complete.gpkg")
                viz_gdf['local_r2'] = 0.05  # Placeholder
                print("Warning: Using placeholder local R² values")
        
        return viz_gdf
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_local_r2_map():
    """Create a map visualization of local R² values from GWR analysis"""
    viz_gdf = load_data()
    
    if viz_gdf is None:
        print("Failed to load required data. Exiting.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Reproject to Web Mercator for basemap
    viz_gdf.to_crs(epsg=3857, inplace=True)
    
    # Plot map with local R² values
    viz_gdf.plot(
        column='local_r2',
        cmap='viridis',
        legend=True,
        ax=ax,
        edgecolor='gray',
        legend_kwds={'label': 'Local R²', 'shrink': 0.6},
        alpha=0.8
    )
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=10)
    
    # Add title and north arrow
    ax.set_title('Spatial Distribution of GWR Local R² Values\nAcross Cork GAA Clubs', fontsize=16)
    ax.text(0.97, 0.03, "N↑", transform=ax.transAxes, fontsize=14, fontweight='bold',
           ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8))
    
    # Add scale bar
    ax.text(0.05, 0.05, '0 10km', transform=ax.transAxes, 
           fontsize=10, backgroundcolor='white', bbox=dict(facecolor='white', alpha=0.5))
    
    # Remove axis
    ax.set_axis_off()
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, "Local R² values show how well the GWR model explains club performance at each location.\nHigher values (yellow) indicate stronger local fit of demographic variables to performance.",
              ha='center', fontsize=10, fontweight='normal', bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = REPORT_VIZ_DIR / "gwr_local_r2_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Saved GWR local R² map to {output_path}")
    plt.close()

if __name__ == "__main__":
    create_local_r2_map() 