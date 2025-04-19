#!/usr/bin/env python3
"""
Demographic Data Integration Script

This script integrates demographic data with club catchment areas using three methods:
1. Nearest Club Method
2. Voronoi Catchment Method
3. Buffer Overlap Method

The script follows a structured approach with validation, integration, and output generation.
"""

import os
import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/demographic_integration.log'),
        logging.StreamHandler()
    ]
)

# Constants
CRS = "EPSG:2157"
JOIN_KEY = "SA_GUID_2022"

# Input file paths
INPUT_FILES = {
    'nearest': 'data/processed/cork_clubs_nearest_full.gpkg',
    'voronoi': 'data/processed/cork_clubs_voronoi_assignment.gpkg',
    'buffer': 'data/analysis/buffer_overlaps.gpkg',
    'demographics': 'data/processed/cork_sa_analysis_full.gpkg',
    'competition': 'data/analysis/competition_metrics.gpkg',
    'scores': 'data/analysis/competition_scores.csv'
}

# Output file paths
OUTPUT_FILES = {
    'nearest': 'data/processed/nearest_demographics.gpkg',
    'voronoi': 'data/processed/voronoi_demographics.gpkg',
    'buffer': 'data/processed/buffer_demographics.csv'
}

def validate_input_data():
    """Validate input data consistency and prepare for integration."""
    logging.info("Starting input data validation")
    
    # Check file existence
    for file_type, path in INPUT_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input file not found: {path}")
    
    # Load and validate data
    data = {}
    for file_type, path in INPUT_FILES.items():
        try:
            if path.endswith('.gpkg'):
                data[file_type] = gpd.read_file(path)
                # Validate CRS only for spatial data
                if file_type not in ['buffer', 'competition']:  # Skip CRS check for non-spatial data
                    if data[file_type].crs != CRS:
                        logging.warning(f"CRS mismatch in {file_type}. Converting to {CRS}")
                        data[file_type] = data[file_type].to_crs(CRS)
            else:
                data[file_type] = pd.read_csv(path)
        except Exception as e:
            logging.error(f"Error loading {file_type} data: {str(e)}")
            raise
    
    # Validate join keys for required datasets only
    required_join_keys = ['nearest', 'voronoi', 'demographics']
    for file_type in required_join_keys:
        df = data[file_type]
        if JOIN_KEY in df.columns:
            logging.info(f"Join key {JOIN_KEY} found in {file_type}")
        else:
            raise ValueError(f"Join key {JOIN_KEY} not found in {file_type}")
    
    return data

def integrate_nearest_method(data):
    """Integrate demographic data using nearest club method."""
    logging.info("Starting nearest method integration")
    
    # Merge demographic data with nearest club assignments
    integrated = data['nearest'].merge(
        data['demographics'], 
        on='SA_GUID_2022',
        suffixes=('_nearest', '_demog')
    )
    
    logging.info(f"Columns after merge: {integrated.columns.tolist()}")
    
    # Calculate validation metrics
    metrics = {
        'total_population': integrated['T1_1AGETT_x_demog'].sum(),  # Total population from demographics
        'average_area': integrated.geometry_nearest.area.mean(),
        'null_count': integrated.isnull().sum().sum()
    }
    
    logging.info(f"Nearest method validation metrics: {metrics}")
    
    return integrated

def integrate_voronoi_method(data):
    """Integrate demographic data using Voronoi catchment areas."""
    logging.info("Starting Voronoi method integration")
    
    # Merge demographic data with Voronoi assignments
    integrated = data['voronoi'].merge(
        data['demographics'],
        on='SA_GUID_2022',
        suffixes=('_voronoi', '_demog')
    )
    
    # Calculate validation metrics
    metrics = {
        'total_population': integrated['T1_1AGETT_x_demog'].sum(),
        'average_area': integrated.geometry_voronoi.area.mean(),
        'null_count': integrated.isnull().sum().sum()
    }
    
    logging.info(f"Voronoi method validation metrics: {metrics}")
    
    return integrated

def integrate_buffer_method(buffer_data, demographics_path):
    """
    Process buffer overlaps data. Since this data is already aggregated with demographics,
    we only need to validate and return it.
    """
    logging.info("Processing buffer overlaps data...")
    
    # Validate metrics
    total_population = buffer_data['total_population'].sum()
    avg_overlap_area = buffer_data['overlap_area_km2'].mean()
    null_count = buffer_data.isnull().sum().sum()
    
    logging.info(f"Buffer Method Validation Metrics:")
    logging.info(f"Total Population: {total_population}")
    logging.info(f"Average Overlap Area (km2): {avg_overlap_area}")
    logging.info(f"Null Count: {null_count}")
    
    return buffer_data

def save_integrated_data(nearest_data, voronoi_data, buffer_data):
    """Save integrated data to output files."""
    try:
        # Convert additional geometry columns to WKT for nearest data
        if 'geometry_demog' in nearest_data.columns:
            nearest_data['geometry_demog_wkt'] = nearest_data['geometry_demog'].to_wkt()
            nearest_data = nearest_data.drop(columns=['geometry_demog'])
        nearest_data.set_geometry('geometry_nearest', inplace=True)
        nearest_data.to_file(OUTPUT_FILES['nearest'], driver='GPKG')
        logging.info(f"Saved nearest method data to {OUTPUT_FILES['nearest']}")

        # Convert additional geometry columns to WKT for voronoi data
        if 'geometry_demog' in voronoi_data.columns:
            voronoi_data['geometry_demog_wkt'] = voronoi_data['geometry_demog'].to_wkt()
            voronoi_data = voronoi_data.drop(columns=['geometry_demog'])
        voronoi_data.to_file(OUTPUT_FILES['voronoi'], driver='GPKG')
        logging.info(f"Saved Voronoi method data to {OUTPUT_FILES['voronoi']}")

        # Save buffer data as CSV since it's a regular DataFrame
        buffer_data.to_csv(OUTPUT_FILES['buffer'], index=False)
        logging.info(f"Saved buffer method data to {OUTPUT_FILES['buffer']}")

    except Exception as e:
        logging.error(f"Error saving integrated data: {str(e)}")
        raise

def generate_integration_report(nearest_data: gpd.GeoDataFrame, voronoi_data: gpd.GeoDataFrame, buffer_data: pd.DataFrame) -> None:
    """Generate a comprehensive report comparing the three integration methods.
    
    Args:
        nearest_data: GeoDataFrame containing nearest method results
        voronoi_data: GeoDataFrame containing Voronoi method results
        buffer_data: DataFrame containing buffer method results
    """
    # Set active geometry columns
    nearest_data = nearest_data.set_geometry('geometry_nearest')
    voronoi_data = voronoi_data.set_geometry('geometry_demog')
    
    # Calculate metrics for each method
    nearest_metrics = {
        'total_population': nearest_data['T1_1AGETT_y_demog'].sum(),
        'average_area': nearest_data.geometry.area.mean() / 1_000_000,  # Convert to km²
        'null_count': nearest_data['nearest_club'].isnull().sum(),
        'unique_clubs': nearest_data['nearest_club'].nunique()
    }
    
    voronoi_metrics = {
        'total_population': voronoi_data['T1_1AGETT_y_demog'].sum(),
        'average_area': voronoi_data.geometry.area.mean() / 1_000_000,  # Convert to km²
        'null_count': voronoi_data['assigned_club_name'].isnull().sum(),
        'unique_clubs': voronoi_data['assigned_club_name'].nunique()
    }
    
    buffer_metrics = {
        'total_population': buffer_data['total_population'].sum(),
        'average_area': buffer_data['overlap_area_km2'].mean(),
        'null_count': buffer_data['club1'].isnull().sum(),
        'unique_clubs': buffer_data['club1'].nunique()
    }
    
    # Generate report content
    report_content = f"""# Demographic Data Integration Report

## Overview
This report compares three methods of integrating demographic data with GAA club catchment areas:
1. Nearest Club Method
2. Voronoi Catchment Method
3. Buffer Overlap Method

## Method Comparison

### Population Distribution
- **Nearest Method**: {nearest_metrics['total_population']:,} total population
- **Voronoi Method**: {voronoi_metrics['total_population']:,} total population
- **Buffer Method**: {buffer_metrics['total_population']:,} total population

### Area Coverage
- **Nearest Method**: {nearest_metrics['average_area']:.2f} km² average area
- **Voronoi Method**: {voronoi_metrics['average_area']:.2f} km² average area
- **Buffer Method**: {buffer_metrics['average_area']:.2f} km² average area

### Data Quality
- **Nearest Method**: {nearest_metrics['null_count']} null values
- **Voronoi Method**: {voronoi_metrics['null_count']} null values
- **Buffer Method**: {buffer_metrics['null_count']} null values

### Club Coverage
- **Nearest Method**: {nearest_metrics['unique_clubs']} unique clubs
- **Voronoi Method**: {voronoi_metrics['unique_clubs']} unique clubs
- **Buffer Method**: {buffer_metrics['unique_clubs']} unique clubs

## Method Analysis

### Nearest Club Method
**Strengths**:
- Simple and intuitive assignment
- Clear distance-based relationships
- Easy to understand and explain

**Limitations**:
- May not account for natural boundaries
- Can create artificial catchment boundaries
- May not reflect actual club influence areas

### Voronoi Catchment Method
**Strengths**:
- Creates natural catchment boundaries
- Ensures complete coverage
- No overlapping areas

**Limitations**:
- May not reflect actual travel patterns
- Can create unrealistic boundaries in urban areas
- Doesn't account for physical barriers

### Buffer Overlap Method
**Strengths**:
- Accounts for overlapping influence areas
- More realistic representation of club reach
- Can handle complex urban environments

**Limitations**:
- More complex to implement
- Requires careful buffer distance selection
- May overestimate total population

## Recommendations

### Primary Analysis
- Use the Voronoi method for initial catchment area definition
- Use the nearest method for distance-based analysis
- Use the buffer method for detailed urban area analysis

### Secondary Analysis
- Combine Voronoi and nearest methods for validation
- Use buffer method for sensitivity analysis
- Consider hybrid approaches for complex areas

### Supplementary Analysis
- Use nearest method for accessibility studies
- Use Voronoi method for service area planning
- Use buffer method for competition analysis

## Output Files
- Nearest Method Data: `data/processed/nearest_demographics.gpkg`
- Voronoi Method Data: `data/processed/voronoi_demographics.gpkg`
- Buffer Method Data: `data/processed/buffer_demographics.csv`
"""
    
    # Save report
    report_path = 'data/analysis/demographic_integration_report.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logging.info(f"Integration report saved to {report_path}")

def main():
    """Main function to orchestrate the integration process."""
    try:
        # Validate input data
        data = validate_input_data()
        
        # Integrate data using different methods
        nearest_data = integrate_nearest_method(data)
        voronoi_data = integrate_voronoi_method(data)
        buffer_data = integrate_buffer_method(data['buffer'], data['demographics'])
        
        # Save integrated data
        save_integrated_data(nearest_data, voronoi_data, buffer_data)
        
        # Generate integration report
        generate_integration_report(nearest_data, voronoi_data, buffer_data)
        
        logging.info("Demographic data integration completed successfully")
        
    except Exception as e:
        logging.error(f"Error in demographic data integration: {str(e)}")
        raise

if __name__ == "__main__":
    main() 