"""
Data preparation script for Voronoi catchment analysis.
This script prepares the input data for Voronoi analysis by:
1. Loading and validating input data
2. Ensuring consistent CRS
3. Cleaning and preparing geometries
4. Validating data completeness
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
from shapely.geometry import Point

# Set up logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'voronoi_data_preparation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def load_and_validate_data():
    """Load and perform initial validation of input data."""
    try:
        # Load club data
        clubs_gdf = gpd.read_file('data/processed/cork_clubs_complete.gpkg')
        logging.info(f"Loaded {len(clubs_gdf)} clubs")
        
        # Load Small Areas data
        sa_gdf = gpd.read_file('data/processed/cork_sa_analysis_key.gpkg')
        logging.info(f"Loaded {len(sa_gdf)} Small Areas")
        
        # Initial validation
        validate_input_data(clubs_gdf, sa_gdf)
        
        return clubs_gdf, sa_gdf
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def validate_input_data(clubs_gdf, sa_gdf):
    """Validate the loaded data meets requirements."""
    # Validate clubs data
    required_club_cols = ['Club', 'geometry']
    missing_cols = [col for col in required_club_cols if col not in clubs_gdf.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in clubs data: {missing_cols}")
    
    # Validate Small Areas data
    required_sa_cols = ['SA_GUID_2022', 'geometry']
    missing_cols = [col for col in required_sa_cols if col not in sa_gdf.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in SA data: {missing_cols}")
    
    # Check for null geometries
    if clubs_gdf.geometry.isna().any():
        raise ValueError("Found null geometries in clubs data")
    if sa_gdf.geometry.isna().any():
        raise ValueError("Found null geometries in SA data")

def ensure_consistent_crs(clubs_gdf, sa_gdf):
    """Ensure both datasets use the same CRS (EPSG:2157 - Irish Grid)."""
    target_crs = "EPSG:2157"
    
    if clubs_gdf.crs is None:
        logging.warning("Clubs data has no CRS. Setting to Irish Grid")
        clubs_gdf.set_crs(target_crs, inplace=True)
    elif clubs_gdf.crs != target_crs:
        logging.info(f"Reprojecting clubs data from {clubs_gdf.crs} to {target_crs}")
        clubs_gdf = clubs_gdf.to_crs(target_crs)
    
    if sa_gdf.crs is None:
        logging.warning("SA data has no CRS. Setting to Irish Grid")
        sa_gdf.set_crs(target_crs, inplace=True)
    elif sa_gdf.crs != target_crs:
        logging.info(f"Reprojecting SA data from {sa_gdf.crs} to {target_crs}")
        sa_gdf = sa_gdf.to_crs(target_crs)
    
    return clubs_gdf, sa_gdf

def clean_and_prepare_geometries(clubs_gdf, sa_gdf):
    """Clean and prepare geometries for Voronoi analysis."""
    # For clubs (points), ensure they are valid points
    if clubs_gdf.geometry.geom_type.unique()[0] == 'Point':
        # No need to buffer points, just ensure they are valid
        invalid_clubs = ~clubs_gdf.geometry.is_valid
        if invalid_clubs.any():
            logging.warning(f"Found {invalid_clubs.sum()} invalid club geometries")
            # For invalid points, we might need to recreate them from coordinates
            if 'Latitude' in clubs_gdf.columns and 'Longitude' in clubs_gdf.columns:
                invalid_idx = clubs_gdf[invalid_clubs].index
                for idx in invalid_idx:
                    clubs_gdf.loc[idx, 'geometry'] = Point(
                        clubs_gdf.loc[idx, 'Longitude'],
                        clubs_gdf.loc[idx, 'Latitude']
                    )
    
    # For SA polygons, ensure they are valid
    sa_gdf['geometry'] = sa_gdf.geometry.buffer(0)
    
    # Check for and fix any remaining invalid geometries
    invalid_sa = ~sa_gdf.geometry.is_valid
    if invalid_sa.any():
        logging.warning(f"Found {invalid_sa.sum()} invalid SA geometries. Attempting to fix...")
        sa_gdf.loc[invalid_sa, 'geometry'] = sa_gdf.loc[invalid_sa, 'geometry'].buffer(0)
    
    return clubs_gdf, sa_gdf

def validate_data_completeness(clubs_gdf, sa_gdf):
    """Perform final validation of prepared data."""
    validation_results = {
        "num_clubs": len(clubs_gdf),
        "num_sa": len(sa_gdf),
        "invalid_club_geoms": (~clubs_gdf.geometry.is_valid).sum(),
        "invalid_sa_geoms": (~sa_gdf.geometry.is_valid).sum(),
        "null_club_geoms": clubs_gdf.geometry.isna().sum(),
        "null_sa_geoms": sa_gdf.geometry.isna().sum()
    }
    
    # Log validation results
    logging.info("Data validation results:")
    for key, value in validation_results.items():
        logging.info(f"{key}: {value}")
    
    return validation_results

def export_prepared_data(clubs_gdf, sa_gdf, validation_results):
    """Export the prepared data and validation report."""
    # Create interim directory if it doesn't exist
    interim_dir = Path('data/interim')
    interim_dir.mkdir(exist_ok=True)
    
    # Export prepared data
    clubs_gdf.to_file(interim_dir / 'clubs_prepared.gpkg', driver='GPKG')
    sa_gdf.to_file(interim_dir / 'sa_prepared.gpkg', driver='GPKG')
    
    # Create validation report
    report_content = "# Data Preparation Report\n\n"
    report_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_content += "## Validation Results\n\n"
    for key, value in validation_results.items():
        report_content += f"- {key}: {value}\n"
    
    with open(interim_dir / 'data_preparation_report.md', 'w') as f:
        f.write(report_content)
    
    logging.info("Exported prepared data and validation report")

def main():
    try:
        # Load and validate input data
        logging.info("Loading and validating input data...")
        clubs_gdf, sa_gdf = load_and_validate_data()
        
        # Ensure consistent CRS
        logging.info("Ensuring consistent CRS...")
        clubs_gdf, sa_gdf = ensure_consistent_crs(clubs_gdf, sa_gdf)
        
        # Clean and prepare geometries
        logging.info("Cleaning and preparing geometries...")
        clubs_gdf, sa_gdf = clean_and_prepare_geometries(clubs_gdf, sa_gdf)
        
        # Validate data completeness
        logging.info("Validating data completeness...")
        validation_results = validate_data_completeness(clubs_gdf, sa_gdf)
        
        # Export prepared data
        logging.info("Exporting prepared data...")
        export_prepared_data(clubs_gdf, sa_gdf, validation_results)
        
        logging.info("Data preparation completed successfully")
        
    except Exception as e:
        logging.error(f"Error in data preparation: {e}")
        raise

if __name__ == "__main__":
    main() 