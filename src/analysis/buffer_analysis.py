"""
Buffer-based Catchment Analysis Module

This module implements a buffer-based approach to analyzing GAA club catchments,
using tiered buffer zones to represent different levels of club influence.

Author: Daniel Tierney
Date: 2024-04-05
"""

import logging
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def determine_buffer_distances(club_row, population_density):
    """
    Determine appropriate buffer distances based on club characteristics and population density.
    Uses standardized distances that align with typical catchment areas.
    
    Args:
        club_row (Series): Row from clubs GeoDataFrame with club attributes
        population_density (float): Population density in the area (people/km²)
        
    Returns:
        dict: Buffer distances for each tier in meters
    """
    # Standardized distances for each tier (aligned with typical catchment areas)
    base_distances = {
        'core': 3000,      # 3km - Core catchment area
        'extended': 8000,  # 8km - Extended catchment area
        'outer': 15000     # 15km - Outer catchment area
    }
    
    # Adjust for population density
    density_factor = 1.0
    if population_density > 1000:  # Urban
        density_factor = 0.6  # Reduce buffer sizes in urban areas
    elif population_density < 100:  # Rural
        density_factor = 1.4  # Increase buffer sizes in rural areas
    
    # Adjust for club grade
    grade_factor = 1.0
    if 'Premier' in str(club_row['Grade_2024_football']) or 'Premier' in str(club_row['Grade_2024_hurling']):
        grade_factor = 1.3  # Increase buffer sizes for premier grade clubs
    elif 'Junior' in str(club_row['Grade_2024_football']) or 'Junior' in str(club_row['Grade_2024_hurling']):
        grade_factor = 0.7  # Reduce buffer sizes for junior grade clubs
    
    # Calculate final distances
    return {
        tier: int(dist * density_factor * grade_factor)
        for tier, dist in base_distances.items()
    }

def create_club_buffers(clubs_gdf, demographics_gdf, population_stats_df):
    """
    Create tiered buffer zones for each club.
    
    Args:
        clubs_gdf (GeoDataFrame): Club locations and attributes
        demographics_gdf (GeoDataFrame): Demographic data by area
        population_stats_df (DataFrame): Population statistics by club
        
    Returns:
        GeoDataFrame: Buffer zones for each club
    """
    logger.info("Creating club buffer zones...")
    
    # Ensure data is in ITM projection
    if clubs_gdf.crs != "EPSG:2157":
        logger.info("Converting clubs data to EPSG:2157")
        clubs_gdf = clubs_gdf.to_crs("EPSG:2157")
    if demographics_gdf.crs != "EPSG:2157":
        logger.info("Converting demographics data to EPSG:2157")
        demographics_gdf = demographics_gdf.to_crs("EPSG:2157")
    
    # Merge population data into demographics
    logger.info("Merging population data into demographics")
    demographics_gdf = demographics_gdf.merge(
        population_stats_df[['assigned_club', 'total_population']],
        left_on='Club',
        right_on='assigned_club',
        how='left'
    )
    
    # Initialize lists to store buffer data
    buffer_data = []
    
    # Process each club
    for idx, club in clubs_gdf.iterrows():
        # Calculate population density for club's area
        club_area = demographics_gdf[demographics_gdf['Club'] == club['Club']]
        if len(club_area) > 0:
            area_km2 = club_area.geometry.area.iloc[0] / 1e6  # Convert to km²
            # Log available columns for debugging
            logger.info(f"Available columns in demographics_gdf: {demographics_gdf.columns.tolist()}")
            population = club_area['total_population'].iloc[0]  # Updated column name
            density = population / area_km2
            logger.debug(f"Club: {club['Club']}, Area: {area_km2:.2f} km², Pop: {population}, Density: {density:.2f}")
        else:
            density = 500  # Default density if not found
            logger.warning(f"No demographic data found for club {club['Club']}, using default density")
        
        # Get buffer distances for this club
        distances = determine_buffer_distances(club, density)
        
        # Create buffers for each tier
        for tier, distance in distances.items():
            buffer = club.geometry.buffer(distance)
            buffer_data.append({
                'Club': club['Club'],
                'tier': tier,
                'buffer_distance': distance,
                'Grade_2024_football': club['Grade_2024_football'],
                'Grade_2024_hurling': club['Grade_2024_hurling'],
                'geometry': buffer
            })
    
    # Create GeoDataFrame from buffer data
    buffer_gdf = gpd.GeoDataFrame(buffer_data, crs=clubs_gdf.crs)
    
    logger.info(f"Created {len(buffer_gdf)} buffer zones for {len(clubs_gdf)} clubs")
    return buffer_gdf

def calculate_buffer_statistics(buffer_gdf, sa_gdf, population_stats_df):
    """
    Calculate detailed statistics for each buffer zone, including demographic breakdowns.
    
    Args:
        buffer_gdf (GeoDataFrame): Buffer zones for each club
        sa_gdf (GeoDataFrame): Small Areas data
        population_stats_df (DataFrame): Population statistics
        
    Returns:
        DataFrame: Statistics for each buffer zone
    """
    logger.info("Calculating buffer zone statistics...")
    
    # Ensure consistent CRS
    if sa_gdf.crs != "EPSG:2157":
        logger.info("Converting Small Areas data to EPSG:2157")
        sa_gdf = sa_gdf.to_crs("EPSG:2157")
    
    # Initialize list to store statistics
    stats_data = []
    
    # Process each buffer
    for idx, buffer in buffer_gdf.iterrows():
        # Find Small Areas that intersect with buffer
        intersecting_sa = sa_gdf[sa_gdf.geometry.intersects(buffer.geometry)]
        
        # Calculate intersection areas
        intersection_areas = intersecting_sa.geometry.intersection(buffer.geometry).area
        proportions = intersection_areas / intersecting_sa.geometry.area
        
        # Initialize population counter
        total_pop = 0
        
        # Calculate population in buffer (weighted by intersection proportion)
        for sa_idx, prop in proportions.items():
            sa_guid = intersecting_sa.loc[sa_idx, 'SA_GUID_2016_x']
            sa_stats = population_stats_df[population_stats_df['assigned_club'] == buffer['Club']]
            
            if not sa_stats.empty:
                # Total population
                sa_pop = sa_stats['total_population'].iloc[0]
                total_pop += sa_pop * prop
        
        # Calculate area
        buffer_area = buffer.geometry.area / 1e6  # Convert to km²
        
        # Calculate population density
        pop_density = total_pop / buffer_area if buffer_area > 0 else 0
        
        # Store statistics
        stats_data.append({
            'Club': buffer['Club'],
            'tier': buffer['tier'],
            'buffer_distance': buffer['buffer_distance'],
            'area_km2': buffer_area,
            'total_population': int(total_pop),
            'population_density': int(pop_density),
            'num_small_areas': len(intersecting_sa)
        })
    
    # Create DataFrame from statistics
    stats_df = pd.DataFrame(stats_data)
    
    logger.info("Completed buffer statistics calculation")
    return stats_df

def analyze_buffer_overlaps(buffer_gdf):
    """
    Analyze overlaps between buffer zones.
    
    Args:
        buffer_gdf (GeoDataFrame): Buffer zones for each club
        
    Returns:
        DataFrame: Overlap analysis results
    """
    logger.info("Analyzing buffer zone overlaps...")
    
    # Initialize list to store overlap data
    overlap_data = []
    
    # Process each pair of buffers
    for idx1, buffer1 in buffer_gdf.iterrows():
        for idx2, buffer2 in buffer_gdf.iloc[idx1+1:].iterrows():
            # Skip if same club
            if buffer1['Club'] == buffer2['Club']:
                continue
            
            # Calculate intersection
            intersection = buffer1.geometry.intersection(buffer2.geometry)
            if not intersection.is_empty:
                # Calculate overlap area
                overlap_area = intersection.area / 1e6  # Convert to km²
                
                # Calculate overlap percentage
                overlap_pct1 = (intersection.area / buffer1.geometry.area) * 100
                overlap_pct2 = (intersection.area / buffer2.geometry.area) * 100
                
                # Store overlap data
                overlap_data.append({
                    'club1': buffer1['Club'],
                    'club2': buffer2['Club'],
                    'tier1': buffer1['tier'],
                    'tier2': buffer2['tier'],
                    'overlap_area_km2': overlap_area,
                    'overlap_pct1': overlap_pct1,
                    'overlap_pct2': overlap_pct2,
                    'grade1_football': buffer1['Grade_2024_football'],
                    'grade2_football': buffer2['Grade_2024_football'],
                    'grade1_hurling': buffer1['Grade_2024_hurling'],
                    'grade2_hurling': buffer2['Grade_2024_hurling']
                })
    
    # Create DataFrame from overlap data
    overlap_df = pd.DataFrame(overlap_data)
    
    logger.info(f"Found {len(overlap_df)} buffer zone overlaps")
    return overlap_df

def visualize_buffer_analysis(buffer_gdf, clubs_gdf, output_dir):
    """
    Create visualizations of buffer analysis results.
    
    Args:
        buffer_gdf (GeoDataFrame): Buffer zones for each club
        clubs_gdf (GeoDataFrame): Club locations
        output_dir (str): Directory to save visualizations
    """
    logger.info("Creating buffer analysis visualizations...")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot buffer zones by tier
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot each tier with different colors
    tiers = ['core', 'extended', 'outer']
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    for tier, color in zip(tiers, colors):
        tier_buffers = buffer_gdf[buffer_gdf['tier'] == tier]
        tier_buffers.plot(ax=ax, color=color, alpha=0.3, label=f'{tier.capitalize()} Buffer')
    
    # Plot club locations
    clubs_gdf.plot(ax=ax, color='red', markersize=50, label='Club Locations')
    
    # Add legend and title
    ax.legend()
    ax.set_title('GAA Club Buffer Zones by Tier')
    
    # Save the plot
    plt.savefig(output_path / 'buffer_zones.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Completed buffer analysis visualizations")

def main():
    """Main execution function."""
    try:
        logger.info("Starting buffer-based catchment analysis")
        
        # Load input data
        clubs_gdf = gpd.read_file('data/processed/cork_clubs_complete.gpkg')
        sa_gdf = gpd.read_file('data/processed/cork_sa_analysis_full.gpkg')
        demographics_gdf = gpd.read_file('data/interim/catchment_demographics.gpkg')
        population_stats_df = pd.read_csv('data/interim/catchment_population_stats.csv')
        
        # Log data info for debugging
        logger.info(f"\nClubs data info:")
        logger.info(f"CRS: {clubs_gdf.crs}")
        logger.info(f"Number of clubs: {len(clubs_gdf)}")
        
        logger.info(f"\nDemographics data info:")
        logger.info(f"CRS: {demographics_gdf.crs}")
        logger.info(f"Columns: {demographics_gdf.columns.tolist()}")
        logger.info(f"Number of records: {len(demographics_gdf)}")
        
        logger.info(f"\nPopulation stats info:")
        logger.info(f"Columns: {population_stats_df.columns.tolist()}")
        logger.info(f"Number of records: {len(population_stats_df)}")
        
        # Create buffer zones
        buffer_gdf = create_club_buffers(clubs_gdf, demographics_gdf, population_stats_df)
        
        # Calculate buffer statistics
        buffer_stats = calculate_buffer_statistics(buffer_gdf, sa_gdf, population_stats_df)
        
        # Analyze buffer overlaps
        overlap_analysis = analyze_buffer_overlaps(buffer_gdf)
        
        # Create output directory if it doesn't exist
        output_dir = Path('data/interim')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        buffer_gdf.to_file(output_dir / 'club_buffers.gpkg', driver='GPKG')
        buffer_stats.to_csv(output_dir / 'buffer_statistics.csv', index=False)
        overlap_analysis.to_csv(output_dir / 'buffer_overlaps.csv', index=False)
        
        # Generate summary report
        with open(output_dir / 'buffer_analysis_report.md', 'w') as f:
            f.write("# Buffer-based Catchment Analysis Report\n\n")
            
            f.write("## Buffer Zone Statistics\n")
            f.write("\nAverage buffer distances:\n")
            avg_distances = buffer_gdf.groupby('tier')['buffer_distance'].mean()
            for tier, dist in avg_distances.items():
                f.write(f"- {tier.capitalize()}: {dist/1000:.1f}km\n")
            
            f.write("\nPopulation coverage:\n")
            tier_stats = buffer_stats.groupby('tier').agg({
                'total_population': ['sum', 'mean'],
                'area_km2': ['sum', 'mean'],
                'population_density': 'mean'
            })
            for tier in tier_stats.index:
                f.write(f"\n### {tier.capitalize()} Catchment\n")
                f.write(f"- Total population: {int(tier_stats.loc[tier, ('total_population', 'sum')]):,}\n")
                f.write(f"- Average population: {int(tier_stats.loc[tier, ('total_population', 'mean')]):,}\n")
                f.write(f"- Total area: {tier_stats.loc[tier, ('area_km2', 'sum')]:.1f} km²\n")
                f.write(f"- Average area: {tier_stats.loc[tier, ('area_km2', 'mean')]:.1f} km²\n")
                f.write(f"- Average density: {int(tier_stats.loc[tier, ('population_density', 'mean')])} people/km²\n")
            
            f.write("\n## Overlap Analysis\n")
            f.write(f"\nTotal overlaps found: {len(overlap_analysis)}\n")
            
            f.write("\nOverlaps by tier combination:\n")
            tier_overlaps = overlap_analysis.groupby(['tier1', 'tier2']).size()
            for (t1, t2), count in tier_overlaps.items():
                f.write(f"- {t1.capitalize()}-{t2.capitalize()}: {count}\n")
            
            f.write("\nSignificant overlaps (>50% overlap):\n")
            significant = overlap_analysis[
                (overlap_analysis['overlap_pct1'] > 50) |
                (overlap_analysis['overlap_pct2'] > 50)
            ]
            f.write(f"Total significant overlaps: {len(significant)}\n")
            
            # List top 10 most significant overlaps
            f.write("\nTop 10 most significant overlaps:\n")
            top_overlaps = significant.nlargest(10, 'overlap_area_km2')
            for _, overlap in top_overlaps.iterrows():
                f.write(f"- {overlap['club1']} ({overlap['tier1']}) - ")
                f.write(f"{overlap['club2']} ({overlap['tier2']}): ")
                f.write(f"{overlap['overlap_area_km2']:.1f} km² ")
                f.write(f"({overlap['overlap_pct1']:.1f}% / {overlap['overlap_pct2']:.1f}%)\n")
        
        # Visualize buffer analysis
        visualize_buffer_analysis(buffer_gdf, clubs_gdf, output_dir)
        
        logger.info("Buffer-based catchment analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in buffer-based catchment analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main() 