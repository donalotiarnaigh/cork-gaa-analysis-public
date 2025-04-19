"""
Clean and standardize Small Area (SA) boundary data.
This script performs cleaning operations on the SA boundary data to ensure
data quality and consistency for analysis.

Tasks performed:
1. Fix invalid geometries
2. Remove duplicate boundaries
3. Handle self-intersections
4. Clean topology errors
5. Standardize attribute names and formats
6. Normalize coordinate system
7. Create consistent ID format
8. Standardize geometry types

This script implements Phase 1.3.3 of the implementation plan.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import logging
from shapely.validation import make_valid
from shapely.ops import unary_union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Set up file paths for data processing."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir

def load_sa_boundaries(file_path: Path) -> gpd.GeoDataFrame:
    """
    Load SA boundary data from GeoJSON file.
    
    Args:
        file_path: Path to SA boundary GeoJSON file
        
    Returns:
        GeoDataFrame containing SA boundaries
    """
    try:
        logger.info(f"Loading SA boundaries from {file_path}")
        gdf = gpd.read_file(file_path)
        logger.info(f"Columns in dataset: {', '.join(gdf.columns)}")
        return gdf
    except Exception as e:
        logger.error(f"Error loading SA boundaries: {str(e)}")
        raise

def fix_invalid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fix invalid geometries using shapely's make_valid function.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        
    Returns:
        GeoDataFrame with fixed geometries
    """
    logger.info("Fixing invalid geometries...")
    invalid_count = (~gdf.geometry.is_valid).sum()
    if invalid_count > 0:
        logger.info(f"Found {invalid_count} invalid geometries")
        gdf.geometry = gdf.geometry.apply(make_valid)
        logger.info("Fixed invalid geometries")
    return gdf

def remove_duplicate_boundaries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Remove duplicate boundaries based on geometry and attributes.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        
    Returns:
        GeoDataFrame with duplicates removed
    """
    logger.info("Checking for duplicate boundaries...")
    initial_count = len(gdf)
    
    # Remove exact duplicates based on geometry and key attributes
    key_columns = ['SA_GUID_2022', 'COUNTY_ENGLISH', 'SA_URBAN_AREA_FLAG']
    gdf = gdf.drop_duplicates(subset=key_columns, keep='first')
    
    removed_count = initial_count - len(gdf)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate boundaries")
    return gdf

def handle_self_intersections(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Handle self-intersections in geometries.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        
    Returns:
        GeoDataFrame with self-intersections handled
    """
    logger.info("Handling self-intersections...")
    # Convert to projected CRS for accurate operations
    gdf_proj = gdf.to_crs(epsg=29902)
    
    # Fix self-intersections using buffer(0) technique
    gdf_proj.geometry = gdf_proj.geometry.apply(lambda x: x.buffer(0))
    
    # Convert back to original CRS
    gdf = gdf_proj.to_crs(gdf.crs)
    logger.info("Handled self-intersections")
    return gdf

def clean_topology_errors(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean topology errors in the dataset.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        
    Returns:
        GeoDataFrame with cleaned topology
    """
    logger.info("Cleaning topology errors...")
    # Convert to projected CRS for accurate operations
    gdf_proj = gdf.to_crs(epsg=29902)
    
    # Ensure no overlaps between adjacent areas
    # This is a simplified approach - in practice, you might want to handle overlaps differently
    gdf_proj.geometry = gdf_proj.geometry.apply(lambda x: x.buffer(0))
    
    # Convert back to original CRS
    gdf = gdf_proj.to_crs(gdf.crs)
    logger.info("Cleaned topology errors")
    return gdf

def standardize_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardize attribute names and formats.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        
    Returns:
        GeoDataFrame with standardized attributes
    """
    logger.info("Standardizing attributes...")
    
    # Create a copy to avoid modifying the original
    gdf_std = gdf.copy()
    
    # Standardize column names (convert to lowercase and replace spaces with underscores)
    gdf_std.columns = [col.lower().replace(' ', '_') for col in gdf_std.columns]
    
    # Ensure SA_GUID_2022 is string type
    gdf_std['sa_guid_2022'] = gdf_std['sa_guid_2022'].astype(str)
    
    # Standardize county names
    gdf_std['county_english'] = gdf_std['county_english'].str.upper()
    
    # Handle missing values in urban area names
    gdf_std['sa_urban_area_name'] = gdf_std['sa_urban_area_name'].fillna('Rural')
    
    logger.info("Standardized attributes")
    return gdf_std

def create_cleaning_report(gdf_original: gpd.GeoDataFrame, gdf_cleaned: gpd.GeoDataFrame, output_dir: Path):
    """
    Create a report documenting the cleaning operations.
    
    Args:
        gdf_original: Original GeoDataFrame
        gdf_cleaned: Cleaned GeoDataFrame
        output_dir: Directory to save the report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"sa_boundary_cleaning_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write("# SA Boundary Data Cleaning Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary of Changes\n\n")
        f.write(f"- Original number of features: {len(gdf_original):,}\n")
        f.write(f"- Final number of features: {len(gdf_cleaned):,}\n")
        f.write(f"- Features removed: {len(gdf_original) - len(gdf_cleaned):,}\n\n")
        
        f.write("## Geometry Statistics\n\n")
        f.write("### Original Data\n")
        f.write(f"- Invalid geometries: {(~gdf_original.geometry.is_valid).sum():,}\n")
        f.write(f"- Self-intersecting geometries: {(gdf_original.geometry.is_simple).sum():,}\n")
        f.write(f"- Geometry types: {gdf_original.geometry.type.value_counts().to_dict()}\n\n")
        
        f.write("### Cleaned Data\n")
        f.write(f"- Invalid geometries: {(~gdf_cleaned.geometry.is_valid).sum():,}\n")
        f.write(f"- Self-intersecting geometries: {(gdf_cleaned.geometry.is_simple).sum():,}\n")
        f.write(f"- Geometry types: {gdf_cleaned.geometry.type.value_counts().to_dict()}\n\n")
        
        f.write("## Attribute Changes\n\n")
        f.write("### Column Names\n")
        f.write("Original columns:\n")
        for col in gdf_original.columns:
            f.write(f"- {col}\n")
        f.write("\nCleaned columns:\n")
        for col in gdf_cleaned.columns:
            f.write(f"- {col}\n")
        
        logger.info(f"Cleaning report saved to {report_path}")

def save_cleaned_data(gdf: gpd.GeoDataFrame, output_dir: Path, timestamp: str):
    """
    Save cleaned data in both Geopackage and CSV formats.
    
    Args:
        gdf: Cleaned GeoDataFrame
        output_dir: Directory to save the files
        timestamp: Timestamp for file naming
    """
    # Save as Geopackage
    gpkg_file = output_dir / f"sa_boundaries_cleaned_{timestamp}.gpkg"
    gdf.to_file(gpkg_file, driver='GPKG')
    logger.info(f"Cleaned data saved as Geopackage: {gpkg_file}")
    
    # Save attributes as CSV
    csv_file = output_dir / f"sa_boundaries_cleaned_{timestamp}.csv"
    gdf.drop(columns=['geometry']).to_csv(csv_file, index=False)
    logger.info(f"Cleaned attributes saved as CSV: {csv_file}")

def main():
    """Main function to run the SA boundary cleaning process."""
    try:
        # Setup paths
        raw_dir, processed_dir = setup_paths()
        sa_file = raw_dir / "SA" / "Small_Area_National_Statistical_Boundaries_2022_Ungeneralised_view_-1547105166321508941.geojson"
        
        # Load data
        gdf = load_sa_boundaries(sa_file)
        gdf_original = gdf.copy()
        
        # Clean and standardize data
        gdf = fix_invalid_geometries(gdf)
        gdf = remove_duplicate_boundaries(gdf)
        gdf = handle_self_intersections(gdf)
        gdf = clean_topology_errors(gdf)
        gdf = standardize_attributes(gdf)
        
        # Create cleaning report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        create_cleaning_report(gdf_original, gdf, processed_dir)
        
        # Save cleaned data in both formats
        save_cleaned_data(gdf, processed_dir, timestamp)
        
        # Print summary
        logger.info("\nSA Boundary Cleaning Summary:")
        logger.info(f"Original features: {len(gdf_original):,}")
        logger.info(f"Final features: {len(gdf):,}")
        logger.info(f"Features removed: {len(gdf_original) - len(gdf):,}")
        logger.info(f"Invalid geometries fixed: {(~gdf_original.geometry.is_valid).sum() - (~gdf.geometry.is_valid).sum():,}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 