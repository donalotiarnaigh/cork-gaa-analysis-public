import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, box

# Set up logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'voronoi_catchment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def load_data():
    """Load Small Areas, club locations, and QGIS-generated Voronoi polygons."""
    try:
        # Load Small Areas
        sa_gdf = gpd.read_file('data/processed/cork_sa_analysis_full.gpkg')
        logging.info(f"Loaded {len(sa_gdf)} Small Areas")
        
        # Load club locations
        clubs_gdf = gpd.read_file('data/processed/cork_clubs_complete.gpkg')
        logging.info(f"Loaded {len(clubs_gdf)} clubs")
        
        # Load QGIS-generated Voronoi polygons
        voronoi_gdf = gpd.read_file('data/processed/voronoi_clipped.gpkg')
        logging.info(f"Loaded {len(voronoi_gdf)} Voronoi polygons")
        
        return sa_gdf, clubs_gdf, voronoi_gdf
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def clip_voronoi_polygons(voronoi_gdf, sa_gdf):
    """Clip Voronoi polygons to study area boundary."""
    try:
        # Create study area boundary from Small Areas
        study_area = unary_union(sa_gdf.geometry)
        
        # Clip Voronoi polygons to study area
        voronoi_gdf.geometry = voronoi_gdf.geometry.intersection(study_area)
        logging.info("Clipped Voronoi polygons to study area")
        
        return voronoi_gdf
    except Exception as e:
        logging.error(f"Error clipping Voronoi polygons: {e}")
        raise

def assign_small_areas(sa_gdf, voronoi_gdf):
    """Assign Small Areas to clubs based on Voronoi polygons."""
    try:
        # Spatial join between Small Areas and Voronoi polygons
        # Using largest overlap method
        sa_results = gpd.sjoin(sa_gdf, voronoi_gdf, how='left', predicate='intersects')
        
        # Handle split Small Areas (those that intersect multiple Voronoi polygons)
        split_areas = sa_results.index.value_counts()
        split_areas = split_areas[split_areas > 1].index
        
        if len(split_areas) > 0:
            logging.info(f"Found {len(split_areas)} split Small Areas")
            
            # For split areas, calculate overlap area and assign to polygon with largest overlap
            for area_id in split_areas:
                area = sa_gdf.loc[area_id]
                overlaps = voronoi_gdf.intersection(area.geometry)
                areas = overlaps.area
                max_overlap = areas.idxmax()
                
                # Keep only the largest overlap
                sa_results = sa_results.drop(sa_results[
                    (sa_results.index == area_id) & 
                    (sa_results['index_right'] != max_overlap)
                ].index)
        
        # Rename columns for consistency
        sa_results = sa_results.rename(columns={'Club': 'assigned_club'})
        
        logging.info("Completed Small Area assignments")
        return sa_results
    except Exception as e:
        logging.error(f"Error assigning Small Areas: {e}")
        raise

def export_results(sa_results, clubs_gdf, voronoi_gdf):
    """Export analysis results."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path('data/processed')
        output_dir.mkdir(exist_ok=True)
        
        # Export Small Area assignments
        sa_results.to_file(
            output_dir / 'cork_clubs_voronoi_assignment.gpkg',
            driver='GPKG'
        )
        
        # Export as CSV (without geometry)
        sa_results.drop(columns=['geometry']).to_csv(
            output_dir / 'cork_clubs_voronoi_assignment.csv',
            index=False
        )
        
        # Export clipped Voronoi polygons
        voronoi_gdf.to_file(
            output_dir / 'cork_clubs_voronoi_polygons.gpkg',
            driver='GPKG'
        )
        
        logging.info("Exported results to GeoPackage and CSV")
        
        # Generate summary report
        with open('data/analysis/voronoi_assignment_report.md', 'w') as f:
            f.write("# Voronoi Catchment Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Dataset Overview\n")
            f.write(f"- Total Small Areas: {len(sa_results)}\n")
            f.write(f"- Total Clubs: {len(clubs_gdf)}\n")
            f.write(f"- Total Voronoi Polygons: {len(voronoi_gdf)}\n\n")
            
            f.write("## Catchment Statistics\n")
            catchment_sizes = sa_results.groupby('assigned_club').size()
            f.write(f"- Minimum Small Areas per catchment: {catchment_sizes.min()}\n")
            f.write(f"- Maximum Small Areas per catchment: {catchment_sizes.max()}\n")
            f.write(f"- Mean Small Areas per catchment: {catchment_sizes.mean():.2f}\n\n")
            
            f.write("## Area Statistics\n")
            area_stats = voronoi_gdf.geometry.area.describe()
            f.write(f"- Minimum catchment area: {area_stats['min']:.2f} square meters\n")
            f.write(f"- Maximum catchment area: {area_stats['max']:.2f} square meters\n")
            f.write(f"- Mean catchment area: {area_stats['mean']:.2f} square meters\n")
            f.write(f"- Median catchment area: {area_stats['50%']:.2f} square meters\n")
        
        logging.info("Generated analysis report")
        
    except Exception as e:
        logging.error(f"Error exporting results: {e}")
        raise

def main():
    """Main execution function."""
    try:
        logging.info("Starting Voronoi catchment analysis")
        
        # Load data including QGIS-generated Voronoi polygons
        sa_gdf, clubs_gdf, voronoi_gdf = load_data()
        
        # Clip Voronoi polygons to study area
        voronoi_gdf = clip_voronoi_polygons(voronoi_gdf, sa_gdf)
        
        # Assign Small Areas to catchments
        sa_results = assign_small_areas(sa_gdf, voronoi_gdf)
        
        # Export results
        export_results(sa_results, clubs_gdf, voronoi_gdf)
        
        logging.info("Completed Voronoi catchment analysis")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 