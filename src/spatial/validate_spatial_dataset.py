import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'spatial_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def validate_spatial_dataset():
    """Validate the spatial dataset for analysis."""
    logging.info("Starting spatial dataset validation")
    
    # Load the spatial dataset
    try:
        gdf = gpd.read_file('data/processed/cork_sa_analysis_full.gpkg')
        logging.info(f"Successfully loaded spatial dataset with {len(gdf)} features")
    except Exception as e:
        logging.error(f"Failed to load spatial dataset: {e}")
        return

    # 1. Validate CRS
    if gdf.crs.to_epsg() != 2157:
        logging.warning(f"CRS is {gdf.crs.to_epsg()}, expected 2157")
    else:
        logging.info("CRS validation passed (EPSG:2157)")

    # 2. Check for complete spatial coverage
    total_area = gdf.geometry.area.sum()
    logging.info(f"Total area covered: {total_area:.2f} square meters")
    
    # Check for gaps or overlaps
    if not gdf.geometry.is_valid.all():
        logging.warning("Some geometries are invalid")
    else:
        logging.info("All geometries are valid")

    # 3. Verify required variables
    required_vars = [
        'SA_GUID_2022', 'SA_PUB2016',
        'basic_education_rate', 'secondary_education_rate', 'third_level_rate',
        'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
        'professional_rate', 'working_class_rate', 'class_verification_rate',
        'youth_proportion', 'school_age_rate', 'youth_gender_ratio'
    ]
    
    missing_vars = [var for var in required_vars if var not in gdf.columns]
    if missing_vars:
        logging.warning(f"Missing required variables: {missing_vars}")
    else:
        logging.info("All required variables are present")

    # 4. Validate identifier consistency
    if gdf['SA_GUID_2022'].is_unique:
        logging.info("SA_GUID_2022 is unique")
    else:
        logging.warning("SA_GUID_2022 contains duplicates")

    # 5. Check for missing values
    missing_data = gdf[required_vars].isnull().sum()
    if missing_data.any():
        logging.warning("Missing values found:")
        for var, count in missing_data[missing_data > 0].items():
            logging.warning(f"  {var}: {count} missing values")
    else:
        logging.info("No missing values in required variables")

    # 6. Validate rate variables
    rate_vars = [
        'basic_education_rate', 'secondary_education_rate', 'third_level_rate',
        'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
        'professional_rate', 'working_class_rate', 'class_verification_rate',
        'youth_proportion', 'school_age_rate'
    ]
    
    for var in rate_vars:
        if var in gdf.columns:
            if (gdf[var] < 0).any() or (gdf[var] > 1).any():
                logging.warning(f"{var} contains values outside [0,1] range")
            else:
                logging.info(f"{var} values are within valid range [0,1]")

    # 7. Generate summary statistics
    summary_stats = gdf[rate_vars].describe()
    logging.info("\nSummary Statistics:")
    logging.info(summary_stats)

    # 8. Create validation report
    report_path = Path('data/analysis/spatial_validation_report.md')
    with open(report_path, 'w') as f:
        f.write("# Spatial Dataset Validation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Overview\n")
        f.write(f"- Total features: {len(gdf)}\n")
        f.write(f"- Total area: {total_area:.2f} square meters\n")
        f.write(f"- CRS: EPSG:{gdf.crs.to_epsg()}\n\n")
        
        f.write("## Data Quality\n")
        f.write(f"- Valid geometries: {gdf.geometry.is_valid.all()}\n")
        f.write(f"- Unique SA_GUID_2022: {gdf['SA_GUID_2022'].is_unique}\n\n")
        
        f.write("## Missing Values\n")
        for var, count in missing_data[missing_data > 0].items():
            f.write(f"- {var}: {count} missing values\n")
        
        f.write("\n## Summary Statistics\n")
        f.write(summary_stats.to_markdown())
        
        f.write("\n## Recommendations\n")
        if missing_vars or not gdf.geometry.is_valid.all() or missing_data.any():
            f.write("1. Address missing variables and invalid geometries\n")
            f.write("2. Review and clean data as needed\n")
        else:
            f.write("Dataset is ready for spatial analysis\n")

    logging.info(f"Validation report saved to {report_path}")

if __name__ == "__main__":
    validate_spatial_dataset() 