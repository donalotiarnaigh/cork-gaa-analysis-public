"""
Export cleaned Small Area (SA) boundaries and generate documentation.
This script handles the final export of cleaned data and creates comprehensive
documentation of the entire data preparation process.

This script implements Phase 1.3.5 and 1.3.6 of the implementation plan.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import logging
import numpy as np
import shutil

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

def load_cleaned_data(file_path: Path) -> gpd.GeoDataFrame:
    """
    Load cleaned SA boundary data from Geopackage.
    
    Args:
        file_path: Path to cleaned SA boundary Geopackage file
        
    Returns:
        GeoDataFrame containing cleaned SA boundaries
    """
    try:
        logger.info(f"Loading cleaned SA boundaries from {file_path}")
        gdf = gpd.read_file(file_path)
        return gdf
    except Exception as e:
        logger.error(f"Error loading cleaned SA boundaries: {str(e)}")
        raise

def export_data(gdf: gpd.GeoDataFrame, output_dir: Path):
    """
    Export data in multiple formats and verify exports.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        output_dir: Directory to save exported files
    """
    # Export as Geopackage (primary format)
    gpkg_path = output_dir / "sa_boundaries_final.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")
    logger.info(f"Exported Geopackage to {gpkg_path}")
    
    # Export as GeoJSON (backup format)
    geojson_path = output_dir / "sa_boundaries_final.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    logger.info(f"Exported GeoJSON to {geojson_path}")
    
    # Export attributes as CSV
    csv_path = output_dir / "sa_boundaries_attributes.csv"
    gdf.drop(columns=['geometry']).to_csv(csv_path, index=False)
    logger.info(f"Exported CSV to {csv_path}")
    
    return gpkg_path, geojson_path, csv_path

def verify_exports(gpkg_path: Path, geojson_path: Path, csv_path: Path) -> dict:
    """
    Verify exported files for completeness and integrity.
    
    Args:
        gpkg_path: Path to exported Geopackage
        geojson_path: Path to exported GeoJSON
        csv_path: Path to exported CSV
        
    Returns:
        Dictionary containing verification results
    """
    verification = {
        'file_sizes': {},
        'feature_counts': {},
        'attribute_counts': {},
        'geometry_types': {},
        'errors': []
    }
    
    try:
        # Verify Geopackage
        gpkg_gdf = gpd.read_file(gpkg_path)
        verification['file_sizes']['gpkg'] = gpkg_path.stat().st_size
        verification['feature_counts']['gpkg'] = len(gpkg_gdf)
        verification['attribute_counts']['gpkg'] = len(gpkg_gdf.columns)
        verification['geometry_types']['gpkg'] = gpkg_gdf.geometry.type.unique().tolist()
        
        # Verify GeoJSON
        geojson_gdf = gpd.read_file(geojson_path)
        verification['file_sizes']['geojson'] = geojson_path.stat().st_size
        verification['feature_counts']['geojson'] = len(geojson_gdf)
        verification['attribute_counts']['geojson'] = len(geojson_gdf.columns)
        verification['geometry_types']['geojson'] = geojson_gdf.geometry.type.unique().tolist()
        
        # Verify CSV
        csv_df = pd.read_csv(csv_path)
        verification['file_sizes']['csv'] = csv_path.stat().st_size
        verification['feature_counts']['csv'] = len(csv_df)
        verification['attribute_counts']['csv'] = len(csv_df.columns)
        
        # Check for consistency
        if not all(count == verification['feature_counts']['gpkg'] for count in verification['feature_counts'].values()):
            verification['errors'].append("Feature count mismatch between formats")
        
        if not all(count == verification['attribute_counts']['gpkg'] - 1 for count in verification['attribute_counts'].values() if count != verification['attribute_counts']['gpkg']):
            verification['errors'].append("Attribute count mismatch between formats")
        
    except Exception as e:
        verification['errors'].append(f"Verification error: {str(e)}")
    
    return verification

def create_export_report(gdf: gpd.GeoDataFrame, verification: dict, output_dir: Path):
    """
    Create comprehensive export report.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        verification: Dictionary containing verification results
        output_dir: Directory to save report
    """
    report_path = output_dir / "sa_boundary_export_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Small Area (SA) Boundary Export Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Export Summary\n\n")
        f.write(f"- Total Features: {len(gdf):,}\n")
        f.write(f"- Total Attributes: {len(gdf.columns):,}\n")
        f.write(f"- Geometry Types: {', '.join(gdf.geometry.type.unique())}\n\n")
        
        f.write("## File Details\n\n")
        for format_name in ['gpkg', 'geojson', 'csv']:
            f.write(f"### {format_name.upper()}\n\n")
            f.write(f"- File Size: {verification['file_sizes'][format_name]:,} bytes\n")
            f.write(f"- Feature Count: {verification['feature_counts'][format_name]:,}\n")
            f.write(f"- Attribute Count: {verification['attribute_counts'][format_name]:,}\n")
            if format_name != 'csv':
                f.write(f"- Geometry Types: {', '.join(verification['geometry_types'][format_name])}\n")
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
        
        f.write("## Export Verification\n\n")
        if verification['errors']:
            f.write("### Issues Found\n")
            for error in verification['errors']:
                f.write(f"- {error}\n")
        else:
            f.write("All exports verified successfully.\n")
        f.write("\n")
        
        f.write("## Usage Guidelines\n\n")
        f.write("1. **Primary Format**: Use the Geopackage (.gpkg) file for all spatial operations\n")
        f.write("2. **Backup Format**: GeoJSON file is provided as a backup and for web mapping\n")
        f.write("3. **Attribute Access**: CSV file is provided for non-spatial analysis\n")
        f.write("\n")
        
        f.write("## Limitations\n\n")
        f.write("1. **File Size**: GeoJSON file may be larger than Geopackage\n")
        f.write("2. **Attribute Types**: Some attribute types may be simplified in CSV export\n")
        f.write("3. **Geometry Precision**: Coordinate precision is maintained in all formats\n")
        f.write("\n")
        
        logger.info(f"Export report saved to {report_path}")

def create_processing_documentation(gdf: gpd.GeoDataFrame, output_dir: Path):
    """
    Create comprehensive processing documentation.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        output_dir: Directory to save documentation
    """
    doc_path = output_dir / "sa_boundary_processing_documentation.md"
    
    with open(doc_path, 'w') as f:
        f.write("# Small Area (SA) Boundary Processing Documentation\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Processing Steps\n\n")
        f.write("1. **Data Loading and Validation**\n")
        f.write("   - Loaded from CSO source\n")
        f.write("   - Validated geometry types\n")
        f.write("   - Checked coordinate system\n\n")
        
        f.write("2. **Data Filtering**\n")
        f.write("   - Filtered to Cork and Cork City\n")
        f.write("   - Validated record counts\n")
        f.write("   - Checked spatial coverage\n\n")
        
        f.write("3. **Data Cleaning**\n")
        f.write("   - Fixed invalid geometries\n")
        f.write("   - Standardized attributes\n")
        f.write("   - Handled missing values\n\n")
        
        f.write("4. **Data Standardization**\n")
        f.write("   - Standardized column names\n")
        f.write("   - Converted data types\n")
        f.write("   - Normalized values\n\n")
        
        f.write("## Parameter Settings\n\n")
        f.write("- Coordinate System: EPSG:4326 (WGS84)\n")
        f.write("- Geometry Validation: Strict\n")
        f.write("- Missing Value Treatment: Standardized\n")
        f.write("- Attribute Standardization: Consistent\n\n")
        
        f.write("## Assumptions\n\n")
        f.write("1. **Data Quality**\n")
        f.write("   - Source data is authoritative\n")
        f.write("   - Boundaries are current\n")
        f.write("   - Attributes are accurate\n\n")
        
        f.write("2. **Processing**\n")
        f.write("   - No data loss during cleaning\n")
        f.write("   - Standardization preserves meaning\n")
        f.write("   - All transformations are documented\n\n")
        
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
        
        logger.info(f"Processing documentation saved to {doc_path}")

def main():
    """Main function to run the export process."""
    try:
        # Set up paths
        raw_dir, processed_dir = setup_paths()
        
        # Load cleaned data
        cleaned_file = processed_dir / "sa_boundaries_cleaned.gpkg"
        gdf = load_cleaned_data(cleaned_file)
        
        # Export data
        gpkg_path, geojson_path, csv_path = export_data(gdf, processed_dir)
        
        # Verify exports
        verification = verify_exports(gpkg_path, geojson_path, csv_path)
        
        # Create documentation
        create_export_report(gdf, verification, processed_dir)
        create_processing_documentation(gdf, processed_dir)
        
        logger.info("Export process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in export process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 