#!/usr/bin/env python3
"""
Assign Small Areas to GAA club catchments based on Voronoi polygons.
"""

import logging
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/small_area_assignment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def validate_geometries(gdf):
    """Ensure all geometries are valid."""
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        logging.warning(f"Found {invalid.sum()} invalid geometries, attempting to fix...")
        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].apply(make_valid)
    return gdf

def check_crs(gdf1, gdf2):
    """Ensure both GeoDataFrames have the same CRS."""
    if gdf1.crs != gdf2.crs:
        logging.warning(f"CRS mismatch: {gdf1.crs} vs {gdf2.crs}")
        gdf2 = gdf2.to_crs(gdf1.crs)
        logging.info(f"Converted CRS to {gdf1.crs}")
    return gdf2

def create_backup(gdf, output_path):
    """Create a backup of the GeoDataFrame."""
    backup_path = output_path.with_suffix('.backup.gpkg')
    gdf.to_file(backup_path, driver='GPKG')
    logging.info(f"Created backup at {backup_path}")

def prepare_data(voronoi_path, sa_path, output_dir):
    """Prepare data for Small Area assignment."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        # Load input data
        logging.info("Loading input data...")
        voronoi_gdf = gpd.read_file(voronoi_path)
        sa_gdf = gpd.read_file(sa_path)
        
        # Create backups
        logging.info("Creating backups of input data...")
        create_backup(voronoi_gdf, output_dir / 'voronoi_clipped.gpkg')
        create_backup(sa_gdf, output_dir / 'sa_prepared.gpkg')
        
        # Ensure consistent CRS
        logging.info("Checking CRS consistency...")
        sa_gdf = check_crs(voronoi_gdf, sa_gdf)
        
        # Validate geometries
        logging.info("Validating geometries...")
        voronoi_gdf = validate_geometries(voronoi_gdf)
        sa_gdf = validate_geometries(sa_gdf)
        
        # Generate preparation report
        logging.info("Generating preparation report...")
        with open(output_dir / 'data_preparation_report.md', 'w') as f:
            f.write("# Data Preparation Report\n\n")
            f.write("## Input Data Summary\n")
            f.write(f"- Voronoi polygons: {len(voronoi_gdf)} features\n")
            f.write(f"- Small Areas: {len(sa_gdf)} features\n")
            f.write(f"- CRS: {voronoi_gdf.crs}\n\n")
            
            f.write("## Geometry Validation\n")
            f.write(f"- Valid Voronoi polygons: {voronoi_gdf.geometry.is_valid.all()}\n")
            f.write(f"- Valid Small Areas: {sa_gdf.geometry.is_valid.all()}\n\n")
            
            f.write("## Data Quality Checks\n")
            f.write(f"- Missing values in Voronoi polygons: {voronoi_gdf.isnull().sum().sum()}\n")
            f.write(f"- Missing values in Small Areas: {sa_gdf.isnull().sum().sum()}\n")
        
        logging.info("Data preparation complete!")
        return voronoi_gdf, sa_gdf
        
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}")
        raise

def perform_spatial_intersection(sa_gdf, voronoi_gdf):
    """Perform spatial intersection between Small Areas and Voronoi polygons."""
    logging.info("Performing spatial intersection...")
    
    # Calculate original areas
    sa_gdf['original_area'] = sa_gdf.geometry.area
    
    # Perform spatial intersection
    intersection_gdf = gpd.overlay(sa_gdf, voronoi_gdf, how='intersection')
    
    # Calculate intersection areas
    intersection_gdf['intersection_area'] = intersection_gdf.geometry.area
    
    # Calculate area proportions
    intersection_gdf['area_proportion'] = intersection_gdf['intersection_area'] / intersection_gdf['original_area']
    
    # Identify split Small Areas
    split_areas = intersection_gdf.groupby('SA_GUID_2022').size()
    split_areas = split_areas[split_areas > 1].index.tolist()
    
    logging.info(f"Found {len(split_areas)} split Small Areas")
    
    return intersection_gdf, split_areas

def generate_intersection_report(intersection_gdf, split_areas, output_dir):
    """Generate a report on the spatial intersection results."""
    logging.info("Generating intersection report...")
    
    # Calculate statistics
    total_sa = intersection_gdf['SA_GUID_2022'].nunique()
    total_splits = len(split_areas)
    avg_splits = intersection_gdf.groupby('SA_GUID_2022').size().mean()
    max_splits = intersection_gdf.groupby('SA_GUID_2022').size().max()
    
    # Generate report
    report_path = output_dir / 'intersection_report.md'
    with open(report_path, 'w') as f:
        f.write("# Spatial Intersection Report\n\n")
        
        f.write("## Summary Statistics\n")
        f.write(f"- Total Small Areas: {total_sa}\n")
        f.write(f"- Split Small Areas: {total_splits}\n")
        f.write(f"- Average splits per Small Area: {avg_splits:.2f}\n")
        f.write(f"- Maximum splits for a Small Area: {max_splits}\n\n")
        
        f.write("## Split Areas Analysis\n")
        if split_areas:
            f.write("### Top 10 Most Split Areas:\n")
            top_splits = intersection_gdf[intersection_gdf['SA_GUID_2022'].isin(split_areas)]
            top_splits = top_splits.groupby(['SA_GUID_2022', 'Club']).agg({
                'area_proportion': 'first',
                'intersection_area': 'sum'
            }).reset_index()
            
            for sa in top_splits.groupby('SA_GUID_2022')['intersection_area'].sum().nlargest(10).index:
                sa_splits = top_splits[top_splits['SA_GUID_2022'] == sa]
                proportions = [f"{row['Club']}: {row['area_proportion']:.2%}" 
                             for _, row in sa_splits.iterrows()]
                f.write(f"- {sa}:\n  - Clubs: {', '.join(proportions)}\n")
        else:
            f.write("No split areas found.\n")
    
    return report_path

def handle_split_areas(intersection_gdf, split_areas, output_dir):
    """Handle Small Areas that are split across multiple Voronoi polygons."""
    logging.info("Processing split areas...")
    
    # Create a copy of the intersection data
    assignments = intersection_gdf.copy()
    
    # Initialize lists to track decisions
    major_splits = []  # Areas with significant splits
    minor_splits = []  # Areas with minor splits
    edge_cases = []    # Areas requiring special handling
    
    # Process each split area
    for sa_guid in split_areas:
        sa_data = assignments[assignments['SA_GUID_2022'] == sa_guid]
        
        # Calculate total area and proportions
        total_area = sa_data['intersection_area'].sum()
        sa_data['proportion'] = sa_data['intersection_area'] / total_area
        
        # Sort by proportion
        sa_data = sa_data.sort_values('proportion', ascending=False)
        
        # Apply assignment rules
        if sa_data['proportion'].iloc[0] >= 0.95:
            # If one club has >95% of the area, assign entire Small Area to that club
            assignments.loc[assignments['SA_GUID_2022'] == sa_guid, 'assigned_club'] = sa_data['Club'].iloc[0]
            minor_splits.append({
                'SA_GUID_2022': sa_guid,
                'assigned_club': sa_data['Club'].iloc[0],
                'proportion': sa_data['proportion'].iloc[0],
                'reason': 'Dominant club (>95% area)'
            })
        elif sa_data['proportion'].iloc[0] >= 0.75:
            # If one club has >75% of the area, assign entire Small Area to that club
            assignments.loc[assignments['SA_GUID_2022'] == sa_guid, 'assigned_club'] = sa_data['Club'].iloc[0]
            major_splits.append({
                'SA_GUID_2022': sa_guid,
                'assigned_club': sa_data['Club'].iloc[0],
                'proportion': sa_data['proportion'].iloc[0],
                'reason': 'Majority club (>75% area)'
            })
        else:
            # For more complex splits, assign to the largest portion
            assignments.loc[assignments['SA_GUID_2022'] == sa_guid, 'assigned_club'] = sa_data['Club'].iloc[0]
            edge_cases.append({
                'SA_GUID_2022': sa_guid,
                'assigned_club': sa_data['Club'].iloc[0],
                'proportions': sa_data[['Club', 'proportion']].to_dict('records'),
                'reason': 'Complex split (assigned to largest portion)'
            })
    
    # Generate split area report
    generate_split_area_report(major_splits, minor_splits, edge_cases, output_dir)
    
    return assignments

def generate_split_area_report(major_splits, minor_splits, edge_cases, output_dir):
    """Generate a detailed report on split area handling."""
    logging.info("Generating split area report...")
    
    report_path = output_dir / 'split_areas_report.md'
    with open(report_path, 'w') as f:
        f.write("# Split Area Handling Report\n\n")
        
        f.write("## Summary\n")
        f.write(f"- Total minor splits (<5% other clubs): {len(minor_splits)}\n")
        f.write(f"- Total major splits (<25% other clubs): {len(major_splits)}\n")
        f.write(f"- Total complex splits: {len(edge_cases)}\n\n")
        
        f.write("## Minor Splits (<5% other clubs)\n")
        for split in minor_splits:
            f.write(f"- {split['SA_GUID_2022']}: Assigned to {split['assigned_club']} ({split['proportion']:.2%})\n")
        
        f.write("\n## Major Splits (<25% other clubs)\n")
        for split in major_splits:
            f.write(f"- {split['SA_GUID_2022']}: Assigned to {split['assigned_club']} ({split['proportion']:.2%})\n")
        
        f.write("\n## Complex Splits\n")
        for case in edge_cases:
            f.write(f"\n### {case['SA_GUID_2022']}\n")
            f.write(f"- Assigned to: {case['assigned_club']}\n")
            f.write("- Area proportions:\n")
            for club in case['proportions']:
                f.write(f"  - {club['Club']}: {club['proportion']:.2%}\n")
            f.write(f"- Reason: {case['reason']}\n")
    
    return report_path

def main():
    """Main function to prepare data and perform spatial intersection."""
    try:
        # Define input and output paths
        voronoi_path = Path('data/processed/voronoi_clipped.gpkg')
        sa_path = Path('data/interim/sa_prepared.gpkg')
        output_dir = Path('data/processed')
        
        # Prepare data
        voronoi_gdf, sa_gdf = prepare_data(voronoi_path, sa_path, output_dir)
        
        # Perform spatial intersection
        intersection_gdf, split_areas = perform_spatial_intersection(sa_gdf, voronoi_gdf)
        
        # Handle split areas
        assignments = handle_split_areas(intersection_gdf, split_areas, output_dir)
        
        # Save final assignments
        logging.info("Saving final assignments...")
        assignments.to_file(output_dir / 'sa_club_assignments.gpkg', driver='GPKG')
        
        logging.info("Small Area assignment complete!")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 