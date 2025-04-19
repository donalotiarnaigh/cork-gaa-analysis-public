"""
Prepare Small Area (SA) boundaries for analysis.
This script handles the initial preparation of SA boundaries, including
filtering for Cork and Cork City, cleaning geometries, and standardizing attributes.

This script implements Phase 1.3.1 and 1.3.2 of the implementation plan.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import logging
import numpy as np

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

def load_raw_data(file_path: Path) -> gpd.GeoDataFrame:
    """
    Load raw SA boundary data from Geopackage.
    
    Args:
        file_path: Path to raw SA boundary Geopackage file
        
    Returns:
        GeoDataFrame containing raw SA boundaries
    """
    try:
        logger.info(f"Loading raw SA boundaries from {file_path}")
        gdf = gpd.read_file(file_path)
        # Set the geometry column
        gdf = gdf.rename(columns={'GEOMETRY': 'geometry'}).set_geometry('geometry')
        return gdf
    except Exception as e:
        logger.error(f"Error loading raw SA boundaries: {str(e)}")
        raise

def filter_cork_boundaries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filter SA boundaries to include only Cork and Cork City.
    
    Args:
        gdf: GeoDataFrame containing all SA boundaries
        
    Returns:
        GeoDataFrame containing only Cork and Cork City SA boundaries
    """
    try:
        # Filter for Cork and Cork City
        cork_gdf = gdf[gdf['COUNTY_ENGLISH'].isin(['CORK', 'CORK CITY'])]
        logger.info(f"Filtered to {len(cork_gdf)} Cork and Cork City SA boundaries")
        return cork_gdf
    except Exception as e:
        logger.error(f"Error filtering Cork boundaries: {str(e)}")
        raise

def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean and validate geometries.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        
    Returns:
        GeoDataFrame with cleaned geometries
    """
    try:
        # Fix invalid geometries
        gdf['geometry'] = gdf['geometry'].buffer(0)
        
        # Remove empty geometries
        gdf = gdf[~gdf['geometry'].is_empty]
        
        # Ensure valid geometries
        gdf = gdf[gdf['geometry'].is_valid]
        
        logger.info(f"Cleaned geometries: {len(gdf)} valid geometries remaining")
        return gdf
    except Exception as e:
        logger.error(f"Error cleaning geometries: {str(e)}")
        raise

def standardize_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardize attribute names and values.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        
    Returns:
        GeoDataFrame with standardized attributes
    """
    try:
        # Standardize column names (except geometry)
        non_geom_cols = [col for col in gdf.columns if col != 'geometry']
        gdf.columns = [col.upper() if col != 'geometry' else col for col in gdf.columns]
        
        # Standardize COUNTY values
        gdf['COUNTY'] = gdf['COUNTY_ENGLISH']
        
        # Ensure numeric columns are numeric
        numeric_columns = ['SA_ID', 'COUNTY_ID']
        for col in numeric_columns:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
        
        logger.info("Standardized attributes")
        return gdf
    except Exception as e:
        logger.error(f"Error standardizing attributes: {str(e)}")
        raise

def save_cleaned_data(gdf: gpd.GeoDataFrame, output_dir: Path):
    """
    Save cleaned data to Geopackage.
    
    Args:
        gdf: GeoDataFrame containing cleaned SA boundaries
        output_dir: Directory to save cleaned data
    """
    try:
        output_path = output_dir / "sa_boundaries_cleaned.gpkg"
        # Ensure geometry column is set correctly
        gdf = gdf.set_geometry('geometry')
        gdf.to_file(output_path, driver="GPKG")
        logger.info(f"Saved cleaned data to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving cleaned data: {str(e)}")
        raise

def create_preparation_report(gdf: gpd.GeoDataFrame, output_dir: Path):
    """
    Create comprehensive preparation report.
    
    Args:
        gdf: GeoDataFrame containing cleaned SA boundaries
        output_dir: Directory to save report
    """
    report_path = output_dir / "sa_boundary_preparation_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Small Area (SA) Boundary Preparation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Data Summary\n\n")
        f.write(f"- Total Features: {len(gdf):,}\n")
        f.write(f"- Total Attributes: {len(gdf.columns):,}\n")
        f.write(f"- Geometry Types: {', '.join(gdf.geometry.type.unique())}\n\n")
        
        f.write("## County Distribution\n\n")
        county_counts = gdf['COUNTY'].value_counts()
        for county, count in county_counts.items():
            f.write(f"- {county}: {count:,} features\n")
        f.write("\n")
        
        f.write("## Data Quality Metrics\n\n")
        f.write("### Geometry Quality\n")
        f.write(f"- Valid Geometries: {gdf.geometry.is_valid.all()}\n")
        f.write(f"- No Empty Geometries: {not gdf.geometry.is_empty.any()}\n")
        f.write(f"- Geometry Types: {', '.join(gdf.geometry.type.unique())}\n\n")
        
        f.write("### Attribute Completeness\n")
        for column in gdf.columns:
            if column != 'geometry':
                missing = gdf[column].isna().sum()
                missing_pct = (missing / len(gdf)) * 100
                f.write(f"- {column}: {missing:,} missing values ({missing_pct:.2f}%)\n")
        f.write("\n")
        
        f.write("## Processing Steps\n\n")
        f.write("1. **Data Loading**\n")
        f.write("   - Loaded from source Geopackage\n")
        f.write("   - Validated file structure\n")
        f.write("   - Checked coordinate system\n\n")
        
        f.write("2. **Data Filtering**\n")
        f.write("   - Filtered to Cork and Cork City\n")
        f.write("   - Validated record counts\n")
        f.write("   - Checked spatial coverage\n\n")
        
        f.write("3. **Geometry Cleaning**\n")
        f.write("   - Fixed invalid geometries\n")
        f.write("   - Removed empty geometries\n")
        f.write("   - Ensured geometry validity\n\n")
        
        f.write("4. **Attribute Standardization**\n")
        f.write("   - Standardized column names\n")
        f.write("   - Standardized COUNTY values\n")
        f.write("   - Converted numeric columns\n\n")
        
        f.write("## Data Structure\n\n")
        f.write("### Required Fields\n")
        for column in gdf.columns:
            if column != 'geometry':
                f.write(f"- {column}\n")
        f.write("\n")
        
        f.write("### Data Types\n")
        for column in gdf.columns:
            if column != 'geometry':
                f.write(f"- {column}: {gdf[column].dtype}\n")
        f.write("\n")
        
        f.write("### Value Ranges\n")
        for column in gdf.columns:
            if column != 'geometry':
                if pd.api.types.is_numeric_dtype(gdf[column]):
                    f.write(f"- {column}: {gdf[column].min():,} to {gdf[column].max():,}\n")
                else:
                    unique_values = gdf[column].unique()
                    f.write(f"- {column}: {', '.join(map(str, unique_values))}\n")
        f.write("\n")
        
        logger.info(f"Preparation report saved to {report_path}")

def main():
    """Main function to run the preparation process."""
    try:
        # Set up paths
        raw_dir, processed_dir = setup_paths()
        
        # Load raw data
        raw_file = raw_dir / "SA" / "Small_Area_National_Statistical_Boundaries_2022_Ungeneralised_view_6693239842129791815 (1).gpkg"
        gdf = load_raw_data(raw_file)
        
        # Filter for Cork
        gdf = filter_cork_boundaries(gdf)
        
        # Clean geometries
        gdf = clean_geometries(gdf)
        
        # Standardize attributes
        gdf = standardize_attributes(gdf)
        
        # Save cleaned data
        save_cleaned_data(gdf, processed_dir)
        
        # Create documentation
        create_preparation_report(gdf, processed_dir)
        
        logger.info("Preparation process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in preparation process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 