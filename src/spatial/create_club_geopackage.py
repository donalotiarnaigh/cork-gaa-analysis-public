"""
Script to convert club data to a spatial dataset (GeoPackage).
This script:
1. Loads the club data from CSV files
2. Creates a GeoDataFrame with point geometries
3. Saves the result as a GeoPackage
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
from shapely.geometry import Point

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Set up paths for input and output files."""
    base_dir = Path(__file__).parent.parent.parent
    return {
        'input': {
            'unique_clubs': base_dir / 'data/raw/clubs/cork_clubs_unique.csv',
            'graded_clubs': base_dir / 'data/processed/cork_clubs_graded.csv'
        },
        'output': {
            'geopackage': base_dir / 'data/processed/cork_clubs.gpkg'
        }
    }

def load_data(paths):
    """Load club data from CSV files."""
    logger.info("Loading club data...")
    
    # Load unique clubs with coordinates
    unique_clubs = pd.read_csv(paths['input']['unique_clubs'])
    logger.info(f"Loaded {len(unique_clubs)} unique clubs with coordinates")
    
    # Load graded clubs data
    graded_clubs = pd.read_csv(paths['input']['graded_clubs'])
    logger.info(f"Loaded {len(graded_clubs)} graded clubs")
    
    return unique_clubs, graded_clubs

def create_geodataframe(unique_clubs, graded_clubs):
    """Create a GeoDataFrame from the club data."""
    logger.info("Creating GeoDataFrame...")
    
    # Merge the datasets
    merged_data = pd.merge(
        unique_clubs,
        graded_clubs,
        on='Club',
        how='left'
    )
    
    # Create point geometries
    geometry = [Point(xy) for xy in zip(merged_data['Longitude'], merged_data['Latitude'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        merged_data,
        geometry=geometry,
        crs="EPSG:4326"
    )
    
    logger.info(f"Created GeoDataFrame with {len(gdf)} features")
    return gdf

def save_geopackage(gdf, output_path):
    """Save the GeoDataFrame as a GeoPackage."""
    logger.info(f"Saving GeoPackage to {output_path}")
    gdf.to_file(output_path, driver="GPKG")
    logger.info("GeoPackage saved successfully")

def main():
    """Main function to run the script."""
    try:
        # Set up paths
        paths = setup_paths()
        
        # Load data
        unique_clubs, graded_clubs = load_data(paths)
        
        # Create GeoDataFrame
        gdf = create_geodataframe(unique_clubs, graded_clubs)
        
        # Save GeoPackage
        save_geopackage(gdf, paths['output']['geopackage'])
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 