"""
Competition Analysis Module

This module analyzes competition patterns between GAA clubs in Cork, including:
- Distance analysis between clubs
- Catchment area overlap analysis
- Competition pattern identification
- Demographic context of competition

Author: Daniel Tierney
Date: 2024-04-04
"""

import logging
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_validate_data():
    """
    Load and validate input data for competition analysis.
    
    Returns:
        tuple: (assignments_gdf, voronoi_gdf, demographics_gdf, population_stats_df)
    """
    logger.info("Starting data preparation and loading")
    
    # Define input file paths
    input_files = {
        'assignments': 'data/processed/sa_club_assignments.gpkg',
        'voronoi': 'data/processed/voronoi_clipped.gpkg',
        'demographics': 'data/interim/catchment_demographics.gpkg',
        'population': 'data/interim/catchment_population_stats.csv'
    }
    
    # Validate input files exist
    for file_type, file_path in input_files.items():
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Required input file not found: {file_path}")
    
    # Load data
    logger.info("Loading input files")
    assignments_gdf = gpd.read_file(input_files['assignments'], layer='sa_club_assignments')
    voronoi_gdf = gpd.read_file(input_files['voronoi'])
    demographics_gdf = gpd.read_file(input_files['demographics'])
    population_stats_df = pd.read_csv(input_files['population'])
    
    # Set consistent CRS (EPSG:2157 - Irish Transverse Mercator)
    logger.info("Setting consistent coordinate reference system")
    target_crs = "EPSG:2157"
    assignments_gdf = assignments_gdf.to_crs(target_crs)
    voronoi_gdf = voronoi_gdf.to_crs(target_crs)
    demographics_gdf = demographics_gdf.to_crs(target_crs)
    
    # Validate CRS consistency
    logger.info("Validating coordinate reference systems")
    crs_list = [
        assignments_gdf.crs,
        voronoi_gdf.crs,
        demographics_gdf.crs
    ]
    
    if not all(crs == crs_list[0] for crs in crs_list):
        raise ValueError("Inconsistent coordinate reference systems detected")
    
    # Validate data completeness
    logger.info("Validating data completeness")
    required_columns = {
        'assignments': ['Club', 'geometry'],
        'voronoi': ['Club', 'geometry'],
        'demographics': ['Club', 'geometry', 'Grade_2024_football', 'Grade_2024_hurling'],
        'population': ['assigned_club', 'total_population']
    }
    
    for df_name, df in [
        ('assignments', assignments_gdf),
        ('voronoi', voronoi_gdf),
        ('demographics', demographics_gdf),
        ('population', population_stats_df)
    ]:
        missing_cols = [col for col in required_columns[df_name] 
                       if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {df_name}: {missing_cols}")
    
    # Validate club consistency
    logger.info("Validating club consistency across datasets")
    clubs_assignments = set(assignments_gdf['Club'].unique())
    clubs_voronoi = set(voronoi_gdf['Club'].unique())
    clubs_demographics = set(demographics_gdf['Club'].unique())
    clubs_population = set(population_stats_df['assigned_club'].unique())
    
    all_clubs = clubs_assignments | clubs_voronoi | clubs_demographics | clubs_population
    for club in all_clubs:
        if not all(club in s for s in [clubs_assignments, clubs_voronoi, 
                                     clubs_demographics, clubs_population]):
            logger.warning(f"Club {club} missing from one or more datasets")
    
    # Create base analysis dataset
    logger.info("Creating base analysis dataset")
    base_gdf = demographics_gdf[['Club', 'geometry', 'Grade_2024_football', 'Grade_2024_hurling']].copy()
    
    # Log summary statistics
    logger.info("Data loading and validation complete")
    logger.info(f"Number of clubs: {len(base_gdf)}")
    logger.info(f"Number of Small Areas: {len(assignments_gdf)}")
    logger.info(f"Number of Voronoi polygons: {len(voronoi_gdf)}")
    logger.info("\nGrade distribution (Football 2024):")
    logger.info(base_gdf['Grade_2024_football'].value_counts().to_string())
    logger.info("\nGrade distribution (Hurling 2024):")
    logger.info(base_gdf['Grade_2024_hurling'].value_counts().to_string())
    
    return assignments_gdf, voronoi_gdf, demographics_gdf, population_stats_df

def calculate_distances(base_gdf):
    """
    Calculate distances between all club pairs and identify patterns.
    
    Args:
        base_gdf (GeoDataFrame): Base analysis dataset with club locations
        
    Returns:
        tuple: (distance_matrix_df, nearest_neighbors_df, distance_stats_df)
    """
    logger.info("Calculating inter-club distances")
    
    # Calculate pairwise distances (in meters since we're using ITM)
    coords = np.column_stack((
        base_gdf.geometry.centroid.x,
        base_gdf.geometry.centroid.y
    ))
    distances = pdist(coords)
    distance_matrix = squareform(distances)
    
    # Create distance matrix DataFrame
    clubs = base_gdf['Club'].values
    distance_matrix_df = pd.DataFrame(
        distance_matrix,
        index=clubs,
        columns=clubs
    )
    
    # Find nearest neighbors
    logger.info("Identifying nearest neighbors")
    n_neighbors = 5  # Number of nearest neighbors to find
    nearest_neighbors = []
    
    for club in clubs:
        # Get distances to all other clubs
        club_distances = distance_matrix_df.loc[club].sort_values()
        # Skip the first one (distance to self = 0)
        nearest = club_distances[1:n_neighbors+1]
        nearest_neighbors.append({
            'Club': club,
            'Nearest_Neighbors': ', '.join(nearest.index),
            'Distances_m': ', '.join(map(str, nearest.values.round(0)))
        })
    
    nearest_neighbors_df = pd.DataFrame(nearest_neighbors)
    
    # Calculate distance statistics
    logger.info("Calculating distance statistics")
    distance_stats = []
    
    for club in clubs:
        club_distances = distance_matrix_df.loc[club]
        # Skip distance to self (0)
        non_zero_distances = club_distances[club_distances > 0]
        
        stats = {
            'Club': club,
            'Min_Distance_m': non_zero_distances.min(),
            'Max_Distance_m': non_zero_distances.max(),
            'Mean_Distance_m': non_zero_distances.mean(),
            'Median_Distance_m': non_zero_distances.median()
        }
        distance_stats.append(stats)
    
    distance_stats_df = pd.DataFrame(distance_stats)
    
    return distance_matrix_df, nearest_neighbors_df, distance_stats_df

def analyze_distance_patterns(base_gdf, distance_matrix_df):
    """
    Analyze distance patterns to identify clusters and isolated clubs.
    
    Args:
        base_gdf (GeoDataFrame): Base analysis dataset with club locations
        distance_matrix_df (DataFrame): Matrix of inter-club distances
        
    Returns:
        DataFrame: Pattern analysis results
    """
    logger.info("Analyzing distance patterns")
    
    # Use DBSCAN to identify clusters
    # Parameters:
    # eps: maximum distance between points in a cluster (5km)
    # min_samples: minimum points to form a cluster (3 clubs)
    coords = np.column_stack((
        base_gdf.geometry.centroid.x,
        base_gdf.geometry.centroid.y
    ))
    
    clustering = DBSCAN(eps=5000, min_samples=3).fit(coords)
    
    # Add cluster labels to results
    pattern_analysis = pd.DataFrame({
        'Club': base_gdf['Club'],
        'Cluster': clustering.labels_
    })
    
    # Identify isolated clubs (those not in any cluster, label = -1)
    pattern_analysis['Is_Isolated'] = pattern_analysis['Cluster'] == -1
    
    # Calculate average distance to all other clubs
    avg_distances = distance_matrix_df.mean()
    pattern_analysis['Avg_Distance_to_Others_m'] = avg_distances.values
    
    # Classify clubs by location type based on average distance
    distance_quartiles = pattern_analysis['Avg_Distance_to_Others_m'].quantile([0.25, 0.75])
    pattern_analysis['Location_Type'] = 'Suburban'  # Default
    pattern_analysis.loc[pattern_analysis['Avg_Distance_to_Others_m'] <= distance_quartiles[0.25], 'Location_Type'] = 'Urban'
    pattern_analysis.loc[pattern_analysis['Avg_Distance_to_Others_m'] >= distance_quartiles[0.75], 'Location_Type'] = 'Rural'
    
    return pattern_analysis

def perform_distance_analysis(base_gdf):
    """
    Perform complete distance analysis including calculations and pattern analysis.
    
    Args:
        base_gdf (GeoDataFrame): Base analysis dataset
        
    Returns:
        tuple: (distance_matrix_df, nearest_neighbors_df, distance_stats_df, pattern_analysis_df)
    """
    logger.info("Starting distance analysis")
    
    # Calculate distances and basic statistics
    distance_matrix_df, nearest_neighbors_df, distance_stats_df = calculate_distances(base_gdf)
    
    # Analyze patterns
    pattern_analysis_df = analyze_distance_patterns(base_gdf, distance_matrix_df)
    
    # Log summary statistics
    logger.info("\nDistance Analysis Summary:")
    logger.info(f"Average minimum distance between clubs: {distance_stats_df['Min_Distance_m'].mean():.0f}m")
    logger.info(f"Number of urban clubs: {(pattern_analysis_df['Location_Type'] == 'Urban').sum()}")
    logger.info(f"Number of suburban clubs: {(pattern_analysis_df['Location_Type'] == 'Suburban').sum()}")
    logger.info(f"Number of rural clubs: {(pattern_analysis_df['Location_Type'] == 'Rural').sum()}")
    logger.info(f"Number of isolated clubs: {pattern_analysis_df['Is_Isolated'].sum()}")
    logger.info(f"Number of clusters identified: {pattern_analysis_df['Cluster'].nunique() - 1}")  # -1 for noise
    
    # Save results
    logger.info("Saving distance analysis results")
    output_dir = Path('data/interim')
    output_dir.mkdir(exist_ok=True)
    
    distance_matrix_df.to_csv(output_dir / 'club_distances.csv')
    nearest_neighbors_df.to_csv(output_dir / 'nearest_neighbors.csv', index=False)
    distance_stats_df.to_csv(output_dir / 'distance_statistics.csv', index=False)
    pattern_analysis_df.to_csv(output_dir / 'distance_patterns.csv', index=False)
    
    return distance_matrix_df, nearest_neighbors_df, distance_stats_df, pattern_analysis_df

def analyze_catchment_overlaps(voronoi_gdf, demographics_gdf, population_stats_df):
    """
    Analyze overlaps between club catchment areas.
    
    Args:
        voronoi_gdf (GeoDataFrame): Voronoi polygons for each club
        demographics_gdf (GeoDataFrame): Demographic data by catchment
        population_stats_df (DataFrame): Population statistics by catchment
        
    Returns:
        tuple: (overlap_stats_df, significant_overlaps_df)
    """
    logger.info("Analyzing catchment overlaps...")
    
    # Ensure geometries are valid and in the correct CRS
    voronoi_gdf = voronoi_gdf.copy()
    if voronoi_gdf.crs is None or voronoi_gdf.crs != "EPSG:2157":
        logger.warning("Converting Voronoi polygons to EPSG:2157")
        voronoi_gdf = voronoi_gdf.to_crs("EPSG:2157")
    
    # Validate geometries
    logger.info("Validating geometries...")
    invalid_geoms = voronoi_gdf[~voronoi_gdf.geometry.is_valid]
    if len(invalid_geoms) > 0:
        logger.warning(f"Found {len(invalid_geoms)} invalid geometries")
        for idx, row in invalid_geoms.iterrows():
            logger.warning(f"Invalid geometry for club {row['Club']}")
            logger.warning(f"Geometry type: {row.geometry.geom_type}")
            logger.warning(f"Geometry WKT: {row.geometry.wkt[:100]}...")
    
    # Fix any invalid geometries
    voronoi_gdf.geometry = voronoi_gdf.geometry.buffer(0)
    
    # Calculate intersection areas between all pairs of catchments
    overlaps = []
    clubs = voronoi_gdf['Club'].unique()
    logger.info(f"Analyzing overlaps between {len(clubs)} clubs")
    
    # Log geometry types and areas
    geom_types = voronoi_gdf.geometry.type.value_counts()
    logger.info(f"Geometry types in dataset: {geom_types.to_dict()}")
    
    # Calculate total area covered by all polygons
    total_area = voronoi_gdf.geometry.area.sum()
    logger.info(f"Total area covered by all polygons: {total_area/1e6:.2f} km²")
    
    # Log some sample areas and bounds
    sample_areas = voronoi_gdf.head(5)
    logger.info("\nSample catchment areas:")
    for _, row in sample_areas.iterrows():
        bounds = row.geometry.bounds
        logger.info(f"{row['Club']}:")
        logger.info(f"  Area: {row.geometry.area/1e6:.2f} km²")
        logger.info(f"  Bounds: ({bounds[0]:.0f}, {bounds[1]:.0f}) - ({bounds[2]:.0f}, {bounds[3]:.0f})")
        logger.info(f"  Type: {row.geometry.geom_type}")
        logger.info(f"  Valid: {row.geometry.is_valid}")
    
    # Check for potential gaps between polygons
    union = voronoi_gdf.geometry.unary_union
    logger.info(f"\nUnion area: {union.area/1e6:.2f} km²")
    logger.info(f"Sum of individual areas: {total_area/1e6:.2f} km²")
    logger.info(f"Difference: {abs(union.area - total_area)/1e6:.2f} km²")
    
    overlap_count = 0
    total_pairs = 0
    
    for i, club1 in enumerate(clubs):
        for club2 in clubs[i+1:]:
            total_pairs += 1
            try:
                # Get the Voronoi polygons for both clubs
                geom1 = voronoi_gdf[voronoi_gdf['Club'] == club1].geometry.iloc[0]
                geom2 = voronoi_gdf[voronoi_gdf['Club'] == club2].geometry.iloc[0]
                
                # Log every 1000th pair for debugging
                if total_pairs % 1000 == 0:
                    logger.info(f"\nProcessing pair {total_pairs}:")
                    logger.info(f"Club1: {club1}, Type: {geom1.geom_type}, Valid: {geom1.is_valid}")
                    logger.info(f"Club2: {club2}, Type: {geom2.geom_type}, Valid: {geom2.is_valid}")
                
                # Convert MultiPolygons to single Polygons if possible
                if geom1.geom_type == 'MultiPolygon':
                    geom1 = geom1.buffer(0)
                if geom2.geom_type == 'MultiPolygon':
                    geom2 = geom2.buffer(0)
                
                # Calculate intersection
                intersection = geom1.intersection(geom2)
                
                # Skip if no intersection
                if intersection.is_empty:
                    continue
                
                # Calculate areas (in square meters)
                area1 = geom1.area
                area2 = geom2.area
                intersection_area = intersection.area
                
                # Calculate overlap percentages
                overlap_pct1 = (intersection_area / area1) * 100
                overlap_pct2 = (intersection_area / area2) * 100
                
                # Skip if overlap is negligible
                if overlap_pct1 < 0.01 and overlap_pct2 < 0.01:
                    continue
                
                overlap_count += 1
                
                # Log every 100th overlap for debugging
                if overlap_count % 100 == 0:
                    logger.info(f"Processed {overlap_count} overlaps out of {total_pairs} pairs checked")
                
                # Get population data for both clubs
                pop1 = population_stats_df[population_stats_df['assigned_club'] == club1]['total_population'].iloc[0]
                pop2 = population_stats_df[population_stats_df['assigned_club'] == club2]['total_population'].iloc[0]
                
                # Get catchment types
                type1 = demographics_gdf[demographics_gdf['Club'] == club1]['catchment_type'].iloc[0]
                type2 = demographics_gdf[demographics_gdf['Club'] == club2]['catchment_type'].iloc[0]
                
                # Log significant overlaps
                if overlap_pct1 > 5 or overlap_pct2 > 5:
                    logger.info(f"Significant overlap found: {club1} ({type1}) and {club2} ({type2})")
                    logger.info(f"Overlap percentages: {overlap_pct1:.2f}% and {overlap_pct2:.2f}%")
                    logger.info(f"Areas: {area1:.2f}m², {area2:.2f}m², intersection: {intersection_area:.2f}m²")
                
                overlaps.append({
                    'club1': club1,
                    'club2': club2,
                    'intersection_area': intersection_area,
                    'area1': area1,
                    'area2': area2,
                    'overlap_pct1': overlap_pct1,
                    'overlap_pct2': overlap_pct2,
                    'population1': pop1,
                    'population2': pop2,
                    'type1': type1,
                    'type2': type2
                })
            except Exception as e:
                logger.error(f"Error processing overlap between {club1} and {club2}: {str(e)}")
                continue
    
    logger.info(f"\nProcessed {total_pairs} total pairs")
    logger.info(f"Found {overlap_count} pairs with non-zero intersection")
    
    # Handle case when no overlaps are found
    if not overlaps:
        logger.warning("No overlaps found between catchment areas")
        empty_df = pd.DataFrame(columns=[
            'club1', 'club2', 'intersection_area', 'area1', 'area2',
            'overlap_pct1', 'overlap_pct2', 'population1', 'population2',
            'type1', 'type2'
        ])
        return empty_df, empty_df
    
    # Convert to DataFrame
    overlap_stats_df = pd.DataFrame(overlaps)
    
    # Identify significant overlaps (>10%)
    significant_overlaps = overlap_stats_df[
        (overlap_stats_df['overlap_pct1'] > 10) | 
        (overlap_stats_df['overlap_pct2'] > 10)
    ].copy()
    
    # Calculate competition intensity score
    if len(significant_overlaps) > 0:
        significant_overlaps['competition_intensity'] = (
            significant_overlaps['overlap_pct1'] * significant_overlaps['population1'] +
            significant_overlaps['overlap_pct2'] * significant_overlaps['population2']
        ) / (significant_overlaps['population1'] + significant_overlaps['population2'])
        
        # Sort by competition intensity
        significant_overlaps = significant_overlaps.sort_values('competition_intensity', ascending=False)
    
    # Save results
    output_dir = Path('data/interim')
    overlap_stats_df.to_csv(output_dir / 'catchment_overlaps.csv', index=False)
    significant_overlaps.to_csv(output_dir / 'significant_overlaps.csv', index=False)
    
    # Log summary statistics
    logger.info("\nOverlap Analysis Summary:")
    if len(overlap_stats_df) > 0:
        logger.info(f"Found {len(overlap_stats_df)} total overlaps")
        logger.info(f"Found {len(significant_overlaps)} significant overlaps (>10%)")
        logger.info(f"Average overlap percentage: {overlap_stats_df[['overlap_pct1', 'overlap_pct2']].mean().mean():.2f}%")
        logger.info(f"Maximum overlap percentage: {overlap_stats_df[['overlap_pct1', 'overlap_pct2']].max().max():.2f}%")
        logger.info(f"Average intersection area: {overlap_stats_df['intersection_area'].mean():.2f}m²")
        logger.info(f"Maximum intersection area: {overlap_stats_df['intersection_area'].max():.2f}m²")
        
        # Log overlap types
        logger.info("\nNumber of overlaps by type:")
        logger.info(f"- Urban-Urban: {len(overlap_stats_df[(overlap_stats_df['type1'] == 'Urban') & (overlap_stats_df['type2'] == 'Urban')])}")
        logger.info(f"- Urban-Suburban: {len(overlap_stats_df[((overlap_stats_df['type1'] == 'Urban') & (overlap_stats_df['type2'] == 'Suburban')) | ((overlap_stats_df['type1'] == 'Suburban') & (overlap_stats_df['type2'] == 'Urban'))])}")
        logger.info(f"- Urban-Rural: {len(overlap_stats_df[((overlap_stats_df['type1'] == 'Urban') & (overlap_stats_df['type2'] == 'Rural')) | ((overlap_stats_df['type1'] == 'Rural') & (overlap_stats_df['type2'] == 'Urban'))])}")
        logger.info(f"- Suburban-Suburban: {len(overlap_stats_df[(overlap_stats_df['type1'] == 'Suburban') & (overlap_stats_df['type2'] == 'Suburban')])}")
        logger.info(f"- Suburban-Rural: {len(overlap_stats_df[((overlap_stats_df['type1'] == 'Suburban') & (overlap_stats_df['type2'] == 'Rural')) | ((overlap_stats_df['type1'] == 'Rural') & (overlap_stats_df['type2'] == 'Suburban'))])}")
        logger.info(f"- Rural-Rural: {len(overlap_stats_df[(overlap_stats_df['type1'] == 'Rural') & (overlap_stats_df['type2'] == 'Rural')])}")
    else:
        logger.warning("No overlaps found between catchment areas")
    
    return overlap_stats_df, significant_overlaps

if __name__ == "__main__":
    try:
        # Load and validate data
        assignments_gdf, voronoi_gdf, demographics_gdf, population_stats_df = load_and_validate_data()
        logger.info("Data preparation and loading completed successfully")
        
        # Create base analysis dataset
        base_gdf = demographics_gdf[['Club', 'geometry', 'Grade_2024_football', 'Grade_2024_hurling']].copy()
        
        # Perform distance analysis
        distance_results = perform_distance_analysis(base_gdf)
        logger.info("Distance analysis completed successfully")
        
        # Perform catchment overlap analysis
        overlap_results = analyze_catchment_overlaps(voronoi_gdf, demographics_gdf, population_stats_df)
        logger.info("Catchment overlap analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise 