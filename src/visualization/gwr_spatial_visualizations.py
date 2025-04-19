#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate GWR and Spatial Analysis Visualizations

This script generates three key spatial visualizations:
1. GWR local R² map for overall performance
2. Spatial autocorrelation plots for football and overall performance
3. Football performance hotspot map

Author: Daniel Tierney
Date: April 2025
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import contextily as ctx
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import logging
from matplotlib_scalebar.scalebar import ScaleBar
from pysal.explore import esda
from pysal.lib import weights

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("gwr_spatial_visualizations")

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "report_visualizations" / "regression"
SPATIAL_OUTPUT_DIR = BASE_DIR / "report_visualizations" / "spatial"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SPATIAL_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Data paths
GWR_RESULTS_PATH = BASE_DIR / "output" / "gwr_analysis" / "overall_performance_gwr_results.gpkg"
SPATIAL_ANALYSIS_PATH = BASE_DIR / "output" / "spatial_analysis" / "hotspot_analysis.gpkg"
# Also check the visualization output from spatial pattern analysis
SPATIAL_MORANS_PATH = BASE_DIR / "output" / "spatial_analysis" / "club_density_analysis.gpkg"

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def load_data():
    """Load the required data files"""
    logger.info("Loading spatial data files...")
    
    # Check if GWR results file exists
    if not GWR_RESULTS_PATH.exists():
        logger.warning(f"GWR results file not found at {GWR_RESULTS_PATH}")
        gwr_data = None
    else:
        try:
            gwr_data = gpd.read_file(GWR_RESULTS_PATH)
            logger.info(f"Loaded GWR data: {len(gwr_data)} clubs, {gwr_data.shape[1]} columns")
        except Exception as e:
            logger.error(f"Error loading GWR data: {e}")
            gwr_data = None
    
    # Check if spatial analysis file exists
    if not SPATIAL_ANALYSIS_PATH.exists():
        logger.warning(f"Spatial analysis file not found at {SPATIAL_ANALYSIS_PATH}")
        spatial_data = None
    else:
        try:
            spatial_data = gpd.read_file(SPATIAL_ANALYSIS_PATH)
            logger.info(f"Loaded spatial analysis data: {len(spatial_data)} clubs, {spatial_data.shape[1]} columns")
        except Exception as e:
            logger.error(f"Error loading spatial analysis data: {e}")
            spatial_data = None
    
    # If files don't exist, try alternative paths
    if gwr_data is None:
        alt_path = BASE_DIR / "output" / "improved_gwr_analysis" / "club_data_for_gwr.gpkg"
        if alt_path.exists():
            try:
                gwr_data = gpd.read_file(alt_path)
                logger.info(f"Loaded alternative GWR data: {len(gwr_data)} clubs from {alt_path}")
            except Exception as e:
                logger.error(f"Error loading alternative GWR data: {e}")
    
    if spatial_data is None:
        alt_path = BASE_DIR / "data" / "processed" / "cork_clubs_complete.gpkg"
        if alt_path.exists():
            try:
                spatial_data = gpd.read_file(alt_path)
                logger.info(f"Loaded alternative spatial data: {len(spatial_data)} clubs from {alt_path}")
            except Exception as e:
                logger.error(f"Error loading alternative spatial data: {e}")
    
    # Try loading additional Moran's I data
    morans_data = None
    if SPATIAL_MORANS_PATH.exists():
        try:
            morans_data = gpd.read_file(SPATIAL_MORANS_PATH)
            logger.info(f"Loaded Moran's I data: {len(morans_data)} entries from {SPATIAL_MORANS_PATH}")
            
            # If we loaded Moran's data but not spatial data, use this
            if spatial_data is None:
                spatial_data = morans_data
                logger.info("Using Moran's I data as primary spatial data")
        except Exception as e:
            logger.error(f"Error loading Moran's I data: {e}")
    
    return gwr_data, spatial_data

def reproject_to_web_mercator(gdf):
    """Reproject to Web Mercator for basemap compatibility"""
    if gdf is None:
        return None
        
    if gdf.crs != 'EPSG:3857':
        try:
            return gdf.to_crs('EPSG:3857')
        except Exception as e:
            logger.error(f"Error reprojecting data: {e}")
            return gdf
    return gdf

def create_gwr_local_r2_map(gwr_data):
    """Create a visualization of local R² values from GWR analysis"""
    logger.info("Creating GWR local R² map...")
    
    if gwr_data is None:
        logger.error("Cannot create GWR local R² map, no data available")
        return
    
    # Check if local_R2 column exists
    if 'local_R2' not in gwr_data.columns:
        logger.error("local_R2 column not found in GWR data")
        return
    
    # Reproject to Web Mercator for basemap
    gwr_data_web = reproject_to_web_mercator(gwr_data)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot local R² values
    gwr_data_web.plot(
        column='local_R2',
        cmap='viridis',
        legend=True,
        ax=ax,
        edgecolor='gray',
        alpha=0.7,
        markersize=50,
        legend_kwds={'label': 'Local R²'}
    )
    
    # Add title
    ax.set_title('GWR Analysis - Local R² Values for Overall Performance', fontsize=16)
    
    # Add basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=10)
    except Exception as e:
        logger.warning(f"Could not add basemap: {str(e)}")
    
    # Turn off axes
    ax.set_axis_off()
    
    # Add scale bar
    ax.add_artist(ScaleBar(1, location='lower left', box_alpha=0.9))
    
    # Add north arrow
    ax.text(0.97, 0.03, "N↑", transform=ax.transAxes, fontsize=14, fontweight='bold',
           ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
    
    # Save figure
    output_path = SPATIAL_OUTPUT_DIR / 'gwr_analysis_overall_performance_local_r2.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"GWR local R² map created successfully at {output_path}")

def create_spatial_autocorrelation_plots(spatial_data):
    """Create spatial autocorrelation plots (Moran's I scatterplots)"""
    logger.info("Creating spatial autocorrelation plots...")
    
    if spatial_data is None:
        logger.error("Cannot create spatial autocorrelation plots, no data available")
        return
    
    # Define the metrics to create plots for
    metrics = ['football_performance', 'overall_performance']
    
    for metric in metrics:
        if metric not in spatial_data.columns:
            logger.warning(f"{metric} not found in spatial data")
            continue
        
        # Try to find local_moran columns
        local_moran_col = f'{metric}_local_moran'
        local_moran_p_col = f'{metric}_local_moran_p'
        
        # If local Moran columns don't exist, calculate them on the fly
        if local_moran_col not in spatial_data.columns or local_moran_p_col not in spatial_data.columns:
            logger.info(f"Calculating local Moran's I for {metric} on the fly")
            
            # Create spatial weights
            coords = [(p.x, p.y) for p in spatial_data.geometry.centroid]
            kw = weights.distance.KNN.from_array(coords, k=8)
            
            # Get values and standardize
            values = spatial_data[metric].values
            if pd.isna(values).any():
                values = pd.Series(values).fillna(pd.Series(values).mean()).values
            
            values_std = (values - np.mean(values)) / np.std(values)
            
            # Calculate local Moran's I
            local_moran = esda.Moran_Local(values_std, kw)
            
            # Calculate global Moran's I for the title
            moran_global = esda.Moran(values_std, kw)
            
            # Get spatial lag for the plot
            lag_values = weights.lag_spatial(kw, values_std)
            
            # Create the figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create scatter plot
            sc = ax.scatter(values_std, lag_values, alpha=0.6, edgecolor='k', s=50)
            
            # Add vertical and horizontal lines at 0
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            
            # Add labels
            ax.set_xlabel('Standardized Performance')
            ax.set_ylabel('Spatial Lag')
            ax.set_title(f'Moran\'s I Scatterplot for {metric.replace("_", " ").title()}\n'
                        f'I={moran_global.I:.4f}, p={moran_global.p_sim:.4f}')
            
            # Remove top and right spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Label quadrants
            ax.text(-1.5, 1.5, 'LH\n(Low values surrounded\nby high values)', fontsize=9, ha='center')
            ax.text(1.5, 1.5, 'HH\n(High values surrounded\nby high values)', fontsize=9, ha='center')
            ax.text(-1.5, -1.5, 'LL\n(Low values surrounded\nby low values)', fontsize=9, ha='center')
            ax.text(1.5, -1.5, 'HL\n(High values surrounded\nby low values)', fontsize=9, ha='center')
            
            # Add club names if available
            club_col = None
            for col in ['Club', 'club_name', 'Name', 'name']:
                if col in spatial_data.columns:
                    club_col = col
                    break
                    
            if club_col:
                # Only annotate significant points and a subset to avoid overcrowding
                sig_mask = local_moran.p_sim < 0.05
                if sum(sig_mask) > 30:
                    # If too many significant points, select top 30 by absolute value of local Moran's I
                    top_idx = np.argsort(np.abs(local_moran.Is))[-30:]
                    sig_mask = np.zeros_like(sig_mask)
                    sig_mask[top_idx] = True
                
                sig_idx = np.where(sig_mask)[0]
                for idx in sig_idx:
                    try:
                        club = spatial_data.iloc[idx][club_col]
                        x, y = values_std[idx], lag_values[idx]
                        ax.annotate(club, (x, y), fontsize=8, alpha=0.7,
                                  xytext=(5, 5), textcoords='offset points')
                    except Exception as e:
                        # Skip if any issues with annotation
                        logger.warning(f"Error annotating club: {e}")
                        continue
        else:
            # Use existing local Moran's I values
            logger.info(f"Using existing local Moran's I values for {metric}")
            
            # Get values from data
            values = spatial_data[metric].values
            values_std = (values - np.mean(values)) / np.std(values)
            local_moran_values = spatial_data[local_moran_col].values
            
            # Calculate global Moran's I for the title
            try:
                # Use pysal to calculate global Moran's I
                coords = [(p.x, p.y) for p in spatial_data.geometry.centroid]
                kw = weights.distance.KNN.from_array(coords, k=8)
                moran = esda.Moran(values_std, kw)
                moran_i = moran.I
                moran_p = moran.p_sim
            except Exception as e:
                logger.warning(f"Error calculating global Moran's I: {e}")
                moran_i = 0
                moran_p = 1
            
            # Get spatial lag for the plot
            lag_values = weights.lag_spatial(kw, values_std)
            
            # Create the figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create scatter plot
            sc = ax.scatter(values_std, lag_values, alpha=0.6, edgecolor='k', s=50)
            
            # Add vertical and horizontal lines at 0
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            
            # Add labels
            ax.set_xlabel('Standardized Performance')
            ax.set_ylabel('Spatial Lag')
            ax.set_title(f'Moran\'s I Scatterplot for {metric.replace("_", " ").title()}\n'
                        f'I={moran_i:.4f}, p={moran_p:.4f}')
            
            # Remove top and right spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Label quadrants
            ax.text(-1.5, 1.5, 'LH\n(Low values surrounded\nby high values)', fontsize=9, ha='center')
            ax.text(1.5, 1.5, 'HH\n(High values surrounded\nby high values)', fontsize=9, ha='center')
            ax.text(-1.5, -1.5, 'LL\n(Low values surrounded\nby low values)', fontsize=9, ha='center')
            ax.text(1.5, -1.5, 'HL\n(High values surrounded\nby low values)', fontsize=9, ha='center')
            
            # Add club names if available
            club_col = None
            for col in ['Club', 'club_name', 'Name', 'name']:
                if col in spatial_data.columns:
                    club_col = col
                    break
                    
            if club_col:
                # Only annotate significant points and a subset to avoid overcrowding
                sig_mask = spatial_data[local_moran_p_col] < 0.05 if local_moran_p_col in spatial_data.columns else np.ones(len(spatial_data), dtype=bool)
                if sum(sig_mask) > 30:
                    # If too many significant points, select top 30 by absolute value of local Moran's I
                    top_idx = np.argsort(np.abs(local_moran_values))[-30:]
                    sig_mask = np.zeros_like(sig_mask)
                    sig_mask[top_idx] = True
                
                sig_idx = np.where(sig_mask)[0]
                for idx in sig_idx:
                    try:
                        club = spatial_data.iloc[idx][club_col]
                        x, y = values_std[idx], lag_values[idx]
                        ax.annotate(club, (x, y), fontsize=8, alpha=0.7,
                                  xytext=(5, 5), textcoords='offset points')
                    except Exception as e:
                        # Skip if any issues with annotation
                        logger.warning(f"Error annotating club: {e}")
                        continue
        
        # Save the figure
        output_path = SPATIAL_OUTPUT_DIR / f'spatial_{metric}_morans_scatterplot.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Spatial autocorrelation plot for {metric} created successfully at {output_path}")

def create_hotspot_map(spatial_data):
    """Create a hotspot analysis map for football performance"""
    logger.info("Creating football performance hotspot map...")
    
    if spatial_data is None:
        logger.error("Cannot create hotspot map, no data available")
        return
    
    # Check if necessary columns exist
    if 'football_performance_hotspot' not in spatial_data.columns:
        logger.error("football_performance_hotspot column not found in spatial data")
        return
    
    # Reproject to Web Mercator for basemap
    spatial_data_web = reproject_to_web_mercator(spatial_data)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create improved custom colormap for better visual distinction
    # Using a more distinct colormap with clearer separation between categories
    # Red for hot spots, blue for cold spots, light grey for not significant
    colors_list = ['#0000CD', '#6495ED', '#E5E5E5', '#FF7F50', '#FF0000']  # Dark blue, light blue, grey, orange, red
    cmap = LinearSegmentedColormap.from_list('lisa_colors', colors_list, N=5)
    
    # Boundaries for categories
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot the hotspots with improved styling
    spatial_data_web.plot(
        column='football_performance_hotspot',
        categorical=True,
        cmap=cmap,
        norm=norm,
        legend=False,  # We'll create a custom legend
        ax=ax,
        linewidth=0.8,
        edgecolor='#404040',  # Darker edge color for better definition
        alpha=0.85,  # Slightly more opaque
        markersize=60  # Larger markers
    )
    
    # Customize the plot
    ax.set_title('LISA Cluster Map: Football Performance\nSpatial Autocorrelation Patterns', fontsize=18)
    
    # Add a subtitle explaining the map
    plt.figtext(0.5, 0.01, 
               "Hot spots indicate clusters of high-performing clubs; cold spots indicate clusters of low-performing clubs",
               ha='center', fontsize=12, style='italic')
    
    # Add basemap with a cleaner look
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=10, alpha=0.7)
    except Exception as e:
        logger.warning(f"Could not add basemap: {str(e)}")
    
    # Turn off axes
    ax.set_axis_off()
    
    # Add scale bar
    ax.add_artist(ScaleBar(1, location='lower left', box_alpha=0.9, border_pad=1.5))
    
    # Add north arrow with improved styling
    ax.text(0.97, 0.03, "N↑", transform=ax.transAxes, fontsize=16, fontweight='bold',
           ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3'))
    
    # Create a custom legend with better formatting
    legend_labels = {
        -2: "Cold Spot (99% confidence)",
        -1: "Cold Spot (95% confidence)",
        0: "Not Significant",
        1: "Hot Spot (95% confidence)",
        2: "Hot Spot (99% confidence)"
    }
    
    # Create legend patches with improved appearance
    legend_elements = [
        mpatches.Patch(facecolor=cmap(norm(val)), 
                      edgecolor='#404040', 
                      linewidth=1,
                      label=label)
        for val, label in legend_labels.items()
    ]
    
    # Add a better positioned and styled legend
    legend = ax.legend(
        handles=legend_elements,
        title="LISA Cluster Categories",
        title_fontsize=13,
        fontsize=11,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True
    )
    
    # Save figure with improved resolution
    output_path = SPATIAL_OUTPUT_DIR / 'spatial_football_performance_hotspot_map.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    logger.info(f"Football performance hotspot map created successfully at {output_path}")

def main():
    """Main function to generate all visualizations"""
    logger.info("Starting GWR and spatial visualization generation...")
    
    # Load data
    gwr_data, spatial_data = load_data()
    
    # Generate GWR local R² map
    create_gwr_local_r2_map(gwr_data)
    
    # Generate spatial autocorrelation plots
    create_spatial_autocorrelation_plots(spatial_data)
    
    # Generate hotspot map
    create_hotspot_map(spatial_data)
    
    logger.info("All visualizations generated successfully!")

if __name__ == "__main__":
    main() 