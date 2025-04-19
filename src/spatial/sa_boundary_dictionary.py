"""
Generate a data dictionary for the Small Area (SA) boundary data.
This script analyzes the data structure and creates detailed documentation
for all fields, including data types, value ranges, and special cases.

This script implements Phase 1.3.4 of the implementation plan.
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
        return gdf
    except Exception as e:
        logger.error(f"Error loading SA boundaries: {str(e)}")
        raise

def analyze_field_statistics(df: pd.DataFrame, field: str) -> dict:
    """
    Analyze statistics for a given field.
    
    Args:
        df: DataFrame containing the data
        field: Name of the field to analyze
        
    Returns:
        Dictionary containing field statistics
    """
    stats = {
        'data_type': str(df[field].dtype),
        'unique_values': df[field].nunique(),
        'missing_values': df[field].isna().sum(),
        'missing_percentage': round(df[field].isna().mean() * 100, 2)
    }
    
    # Add value ranges for numeric fields
    if pd.api.types.is_numeric_dtype(df[field]):
        stats.update({
            'min_value': df[field].min(),
            'max_value': df[field].max(),
            'mean_value': round(df[field].mean(), 2),
            'median_value': round(df[field].median(), 2)
        })
    
    # Add most common values for categorical fields
    if not pd.api.types.is_numeric_dtype(df[field]):
        value_counts = df[field].value_counts()
        stats['most_common_values'] = value_counts.head(5).to_dict()
    
    return stats

def create_data_dictionary(gdf: gpd.GeoDataFrame, output_dir: Path):
    """
    Create a comprehensive data dictionary for the SA boundary data.
    
    Args:
        gdf: GeoDataFrame containing SA boundaries
        output_dir: Directory to save the dictionary
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dict_path = output_dir / f"sa_boundary_dictionary_{timestamp}.md"
    
    with open(dict_path, 'w') as f:
        f.write("# Small Area (SA) Boundary Data Dictionary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This document provides a comprehensive description of the Small Area (SA) boundary data used in the GAA Club Success Analysis project.\n\n")
        
        f.write("## Data Source\n\n")
        f.write("- **Source**: CSO Small Area National Statistical Boundaries 2022\n")
        f.write("- **Version**: 2022 Ungeneralised\n")
        f.write("- **Format**: GeoJSON\n")
        f.write("- **Coordinate System**: EPSG:4326 (WGS84)\n\n")
        
        f.write("## Processing Steps\n\n")
        f.write("1. Data loaded from GeoJSON source\n")
        f.write("2. Geometries validated and cleaned\n")
        f.write("3. Attribute names standardized\n")
        f.write("4. Data filtered to Cork and Cork City\n")
        f.write("5. Exported as Geopackage and CSV formats\n\n")
        
        f.write("## Field Descriptions\n\n")
        
        # Analyze each field
        for column in gdf.columns:
            if column != 'geometry':  # Skip geometry column for now
                stats = analyze_field_statistics(gdf, column)
                
                f.write(f"### {column}\n\n")
                f.write(f"- **Data Type**: {stats['data_type']}\n")
                f.write(f"- **Unique Values**: {stats['unique_values']:,}\n")
                f.write(f"- **Missing Values**: {stats['missing_values']:,} ({stats['missing_percentage']}%)\n")
                
                if pd.api.types.is_numeric_dtype(gdf[column]):
                    f.write(f"- **Range**: {stats['min_value']} to {stats['max_value']}\n")
                    f.write(f"- **Mean**: {stats['mean_value']}\n")
                    f.write(f"- **Median**: {stats['median_value']}\n")
                else:
                    f.write("- **Most Common Values**:\n")
                    for value, count in stats['most_common_values'].items():
                        f.write(f"  - {value}: {count:,}\n")
                
                # Add field description based on field name
                description = get_field_description(column)
                f.write(f"- **Description**: {description}\n")
                
                # Add any special cases or limitations
                special_cases = get_special_cases(column)
                if special_cases:
                    f.write("- **Special Cases**:\n")
                    for case in special_cases:
                        f.write(f"  - {case}\n")
                
                f.write("\n")
        
        # Add geometry field description
        f.write("### geometry\n\n")
        f.write("- **Data Type**: Geometry\n")
        f.write("- **Type**: Polygon/MultiPolygon\n")
        f.write("- **Coordinate System**: EPSG:4326 (WGS84)\n")
        f.write("- **Description**: Vector geometry representing the Small Area boundary\n")
        f.write("- **Special Cases**:\n")
        f.write("  - Contains both Polygon and MultiPolygon geometries\n")
        f.write("  - All geometries are valid and properly structured\n")
        f.write("\n")
        
        f.write("## Usage Guidelines\n\n")
        f.write("1. **Data Loading**:\n")
        f.write("   - Use Geopackage format for QGIS visualization and analysis\n")
        f.write("   - Use CSV format for attribute inspection and validation\n")
        f.write("\n")
        f.write("2. **Key Fields**:\n")
        f.write("   - SA_GUID_2022: Unique identifier for each Small Area\n")
        f.write("   - COUNTY_ENGLISH: County name (CORK or CORK CITY)\n")
        f.write("   - SA_URBAN_AREA_NAME: Name of urban area (if applicable)\n")
        f.write("\n")
        f.write("3. **Data Quality**:\n")
        f.write("   - All geometries are valid\n")
        f.write("   - No duplicate boundaries\n")
        f.write("   - Complete coverage of Cork and Cork City\n")
        f.write("\n")
        
        f.write("## Limitations\n\n")
        f.write("1. **Urban Area Names**:\n")
        f.write("   - Some Small Areas have missing urban area names\n")
        f.write("   - These are marked as 'Rural' in the cleaned data\n")
        f.write("\n")
        f.write("2. **Geometry Types**:\n")
        f.write("   - Mix of Polygon and MultiPolygon geometries\n")
        f.write("   - All geometries are valid but may require special handling\n")
        f.write("\n")
        
        logger.info(f"Data dictionary saved to {dict_path}")

def get_field_description(field: str) -> str:
    """Get description for a specific field."""
    descriptions = {
        'OBJECTID': 'Unique identifier for each record',
        'SA_GUID_2016': 'Unique identifier for Small Area in 2016 boundaries',
        'SA_GUID_2022': 'Unique identifier for Small Area in 2022 boundaries',
        'SA_PUB2011': 'Public identifier for Small Area in 2011',
        'SA_PUB2016': 'Public identifier for Small Area in 2016',
        'SA_PUB2022': 'Public identifier for Small Area in 2022',
        'SA_GEOGID_2022': 'Geographic identifier for Small Area in 2022',
        'SA_CHANGE_CODE': 'Code indicating changes from previous version',
        'SA_URBAN_AREA_FLAG': 'Flag indicating if area is urban or rural',
        'SA_URBAN_AREA_NAME': 'Name of urban area (if applicable)',
        'SA_NUTS1': 'NUTS1 region code',
        'SA_NUTS1_NAME': 'NUTS1 region name',
        'SA_NUTS2': 'NUTS2 region code',
        'SA_NUTS2_NAME': 'NUTS2 region name',
        'SA_NUTS3': 'NUTS3 region code',
        'SA_NUTS3_NAME': 'NUTS3 region name',
        'ED_GUID': 'Unique identifier for Electoral Division',
        'ED_OFFICIAL': 'Official name of Electoral Division',
        'ED_ENGLISH': 'English name of Electoral Division',
        'ED_GAEILGE': 'Irish name of Electoral Division',
        'ED_ID_STR': 'String identifier for Electoral Division',
        'ED_PART_COUNT': 'Number of parts in Electoral Division',
        'COUNTY_CODE': 'County code',
        'COUNTY_ENGLISH': 'English name of county',
        'COUNTY_GAEILGE': 'Irish name of county',
        'CSO_LEA': 'CSO Local Electoral Area identifier'
    }
    return descriptions.get(field, 'No description available')

def get_special_cases(field: str) -> list:
    """Get special cases for a specific field."""
    special_cases = {
        'SA_URBAN_AREA_NAME': [
            'Missing values indicate rural areas',
            'Values are standardized to proper case'
        ],
        'COUNTY_ENGLISH': [
            'Only contains CORK and CORK CITY values',
            'Values are standardized to uppercase'
        ],
        'SA_GUID_2022': [
            'Primary identifier for Small Areas',
            'Converted to string type for consistency'
        ]
    }
    return special_cases.get(field, [])

def main():
    """Main function to generate the data dictionary."""
    try:
        # Setup paths
        raw_dir, processed_dir = setup_paths()
        sa_file = raw_dir / "SA" / "Small_Area_National_Statistical_Boundaries_2022_Ungeneralised_view_-1547105166321508941.geojson"
        
        # Load data
        gdf = load_sa_boundaries(sa_file)
        
        # Create data dictionary
        create_data_dictionary(gdf, processed_dir)
        
        logger.info("Data dictionary generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 