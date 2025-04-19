#!/usr/bin/env python3
"""
Clip QGIS-generated Voronoi polygons to Cork county boundary.
"""

import logging
import sys
from pathlib import Path
import geopandas as gpd
from shapely.validation import make_valid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/clip_voronoi.log'),
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

def create_county_mask(sa_boundaries):
    """Create a mask from the Cork county boundary."""
    logging.info("Creating county mask from Small Area boundaries...")
    
    # Ensure geometries are valid
    sa_boundaries = validate_geometries(sa_boundaries)
    
    # Dissolve all Small Areas to create county boundary
    county_mask = sa_boundaries.dissolve()
    
    # Ensure we have a single polygon
    if len(county_mask) != 1:
        logging.warning("Multiple features in dissolved boundary, taking the largest")
        county_mask = county_mask.loc[county_mask.geometry.area.idxmax()]
    
    # Add buffer to ensure complete coverage
    county_mask.geometry = county_mask.geometry.buffer(1000)  # 1km buffer
    
    # Ensure the mask is valid
    county_mask = validate_geometries(county_mask)
    
    return county_mask

def clip_voronoi_polygons(voronoi_gdf, mask):
    """Clip Voronoi polygons to county boundary."""
    logging.info("Clipping Voronoi polygons to county boundary...")
    
    # Ensure geometries are valid
    voronoi_gdf = validate_geometries(voronoi_gdf)
    mask = validate_geometries(mask)
    
    # Clip polygons to mask
    clipped = voronoi_gdf.copy()
    clipped.geometry = clipped.geometry.intersection(mask.geometry.iloc[0])
    
    # Remove any empty geometries
    empty = clipped.geometry.is_empty
    if empty.any():
        logging.warning(f"Removing {empty.sum()} empty geometries after clipping")
        clipped = clipped[~empty]
    
    return clipped

def validate_results(clipped_gdf, original_gdf):
    """Validate the clipped Voronoi polygons."""
    logging.info("Validating results...")
    
    # Check if we lost any clubs during clipping
    missing_clubs = set(original_gdf['Club']) - set(clipped_gdf['Club'])
    if missing_clubs:
        logging.warning(f"Missing clubs after clipping: {missing_clubs}")
    
    # Check if all polygons are valid
    invalid = ~clipped_gdf.geometry.is_valid
    if invalid.any():
        logging.warning(f"Found {invalid.sum()} invalid Voronoi polygons")
    
    return len(missing_clubs) == 0 and not invalid.any()

def main():
    """Main function to clip Voronoi polygons."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        # Load input data
        logging.info("Loading input data...")
        voronoi_gdf = gpd.read_file('data/processed/voronoi_qgis.gpkg')
        sa_boundaries = gpd.read_file('data/interim/sa_prepared.gpkg')
        
        # Create county mask
        county_mask = create_county_mask(sa_boundaries)
        
        # Clip Voronoi polygons
        clipped_voronoi = clip_voronoi_polygons(voronoi_gdf, county_mask)
        
        # Validate results
        is_valid = validate_results(clipped_voronoi, voronoi_gdf)
        if not is_valid:
            logging.error("Validation failed, but proceeding with output")
        
        # Save outputs
        logging.info("Saving outputs...")
        clipped_voronoi.to_file(output_dir / 'voronoi_clipped.gpkg', driver='GPKG')
        
        # Generate validation report
        logging.info("Generating validation report...")
        with open(output_dir / 'voronoi_clipping_report.md', 'w') as f:
            f.write("# Voronoi Clipping Report\n\n")
            f.write(f"Number of original Voronoi polygons: {len(voronoi_gdf)}\n")
            f.write(f"Number of clipped Voronoi polygons: {len(clipped_voronoi)}\n")
            f.write(f"CRS: {clipped_voronoi.crs}\n")
            f.write("\n## Validation Checks\n")
            f.write(f"- All Voronoi polygons are valid: {'Yes' if is_valid else 'No'}\n")
            f.write(f"- All clubs retained after clipping: {'Yes' if len(voronoi_gdf) == len(clipped_voronoi) else 'No'}\n")
            if len(voronoi_gdf) != len(clipped_voronoi):
                missing_clubs = set(voronoi_gdf['Club']) - set(clipped_voronoi['Club'])
                f.write(f"\nMissing clubs after clipping: {missing_clubs}\n")
        
        logging.info("Voronoi clipping complete!")
        
    except Exception as e:
        logging.error(f"Error in Voronoi clipping: {str(e)}")
        raise

if __name__ == "__main__":
    main() 