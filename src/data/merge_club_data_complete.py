"""
Script to merge all club data while preserving spatial and environmental attributes.
This script:
1. Loads core club data with spatial and environmental attributes
2. Loads grade data from 2022 and 2024
3. Merges all data while preserving all unique clubs
4. Adds numerical grade values and performance metrics
5. Creates a GeoPackage with all attributes
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
from shapely.geometry import Point
import numpy as np

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
            'core': base_dir / 'data/raw/clubs/cork_clubs_unique.csv',
            'grades_2022': base_dir / 'data/raw/clubs/cork_clubs_grades_2022.csv',
            'grades_2024': base_dir / 'data/raw/clubs/cork_clubs_grades_2024.csv'
        },
        'output': {
            'merged': base_dir / 'data/processed/cork_clubs_complete.csv',
            'graded': base_dir / 'data/processed/cork_clubs_complete_graded.csv',
            'gpkg': base_dir / 'data/processed/cork_clubs_complete.gpkg'
        }
    }

def load_core_data(file_path):
    """Load core club data with spatial and environmental attributes."""
    logger.info("Loading core club data...")
    df = pd.read_csv(file_path)
    # Strip whitespace from club names
    df['Club'] = df['Club'].str.strip()
    logger.info(f"Loaded {len(df)} clubs with spatial and environmental data")
    return df

def load_grade_data(paths):
    """Load grade data from 2022 and 2024."""
    logger.info("Loading grade data...")
    
    # Load 2022 grades
    grades_2022 = pd.read_csv(paths['input']['grades_2022'])
    grades_2022.rename(columns={
        'club_name': 'Club',
        'football_grade': 'Grade_2022_football',
        'hurling_grade': 'Grade_2022_hurling'
    }, inplace=True)
    # Strip whitespace from club names
    grades_2022['Club'] = grades_2022['Club'].str.strip()
    
    # Load 2024 grades
    grades_2024 = pd.read_csv(paths['input']['grades_2024'])
    grades_2024.rename(columns={
        'club_name': 'Club',
        'football_grade': 'Grade_2024_football',
        'hurling_grade': 'Grade_2024_hurling'
    }, inplace=True)
    # Strip whitespace from club names
    grades_2024['Club'] = grades_2024['Club'].str.strip()
    
    logger.info(f"Loaded grades for {len(grades_2022)} clubs (2022) and {len(grades_2024)} clubs (2024)")
    return grades_2022, grades_2024

def merge_data(core_df, grades_2022, grades_2024):
    """Merge all data while preserving all unique clubs."""
    logger.info("Merging data...")
    
    # First merge 2022 grades
    merged_df = pd.merge(
        core_df,
        grades_2022[['Club', 'Grade_2022_football', 'Grade_2022_hurling']],
        on='Club',
        how='left'
    )
    
    # Then merge 2024 grades
    merged_df = pd.merge(
        merged_df,
        grades_2024[['Club', 'Grade_2024_football', 'Grade_2024_hurling']],
        on='Club',
        how='left'
    )
    
    # Log merge results
    logger.info(f"Total clubs after merge: {len(merged_df)}")
    logger.info(f"Clubs with 2022 football grades: {merged_df['Grade_2022_football'].notna().sum()}")
    logger.info(f"Clubs with 2022 hurling grades: {merged_df['Grade_2022_hurling'].notna().sum()}")
    logger.info(f"Clubs with 2024 football grades: {merged_df['Grade_2024_football'].notna().sum()}")
    logger.info(f"Clubs with 2024 hurling grades: {merged_df['Grade_2024_hurling'].notna().sum()}")
    
    return merged_df

def add_grade_values(df):
    """Add numerical values to grades and calculate performance metrics."""
    logger.info("Adding grade values and calculating metrics...")
    
    # Define grade value mappings (1 is best, 6 is NA)
    FOOTBALL_GRADE_VALUES = {
        'Premier Senior': 1,
        'Senior A': 2,
        'Premier Intermediate': 3,
        'Intermediate A': 4,
        'Premier Junior': 5,
        'NA': 6
    }
    
    HURLING_GRADE_VALUES = {
        'Premier Senior': 1,
        'Senior A': 2,
        'Premier Intermediate': 3,
        'Intermediate A': 4,
        'Premier Junior': 5,
        'Junior A': 5,  # Same level as Premier Junior
        'NA': 6
    }
    
    # Fill NA values in grade columns
    grade_columns = [
        'Grade_2022_football', 'Grade_2022_hurling',
        'Grade_2024_football', 'Grade_2024_hurling'
    ]
    df[grade_columns] = df[grade_columns].fillna('NA')
    
    # Map grades to numerical values
    for year in ['2022', '2024']:
        for code in ['football', 'hurling']:
            col = f'Grade_{year}_{code}'
            value_col = f'Grade_{year}_{code}_value'
            values = FOOTBALL_GRADE_VALUES if code == 'football' else HURLING_GRADE_VALUES
            df[value_col] = df[col].map(values)
    
    # Calculate combined values for dual clubs
    df['combined_2022'] = df[['Grade_2022_football_value', 'Grade_2022_hurling_value']].mean(axis=1)
    df['combined_2024'] = df[['Grade_2024_football_value', 'Grade_2024_hurling_value']].mean(axis=1)
    
    # Calculate grade changes (negative means improvement)
    df['football_improvement'] = df['Grade_2024_football_value'] - df['Grade_2022_football_value']
    df['hurling_improvement'] = df['Grade_2024_hurling_value'] - df['Grade_2022_hurling_value']
    
    # Add new composite performance metrics
    df['overall_performance'] = df[['Grade_2024_football_value', 'Grade_2024_hurling_value']].mean(axis=1)
    df['football_performance'] = df['Grade_2024_football_value']
    df['hurling_performance'] = df['Grade_2024_hurling_value']
    df['code_balance'] = abs(df['Grade_2024_football_value'] - df['Grade_2024_hurling_value'])
    df['is_dual_2022'] = (df['Grade_2022_football_value'] != 6) & (df['Grade_2022_hurling_value'] != 6)
    df['is_dual_2024'] = (df['Grade_2024_football_value'] != 6) & (df['Grade_2024_hurling_value'] != 6)
    
    logger.info("Added grade values and calculated metrics")
    return df

def create_geopackage(df, output_path):
    """Create a GeoPackage with point geometries and all attributes."""
    logger.info("Creating GeoPackage...")
    
    # Create point geometries
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Save to GeoPackage
    gdf.to_file(output_path, driver='GPKG')
    logger.info(f"Created {len(gdf)} records")
    logger.info(f"Saved GeoPackage to {output_path}")
    
    return gdf

def save_data(df, output_path):
    """Save dataset to CSV."""
    logger.info(f"Saving dataset to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved successfully")

def validate_data(df):
    """Validate the processed dataset."""
    logger.info("Validating dataset...")
    
    # Check grade value distributions
    for year in ['2022', '2024']:
        for code in ['football', 'hurling']:
            col = f'Grade_{year}_{code}_value'
            logger.info(f"\n{year} {code} grade value distribution:")
            value_counts = df[col].value_counts().sort_index()
            for value, count in value_counts.items():
                logger.info(f"Value {int(value)}: {count} clubs")
    
    # Performance metrics summary
    logger.info("\nPerformance metrics summary:")
    logger.info(f"Average overall performance: {df['overall_performance'].mean():.2f}")
    logger.info(f"Average football performance: {df['football_performance'].mean():.2f}")
    logger.info(f"Average hurling performance: {df['hurling_performance'].mean():.2f}")
    logger.info(f"Number of dual clubs (2024): {df['is_dual_2024'].sum()}")
    
    # Spatial data summary
    logger.info("\nSpatial data summary:")
    logger.info(f"Number of clubs with coordinates: {df[['Latitude', 'Longitude']].notna().all(axis=1).sum()}")
    logger.info(f"Number of clubs with elevation data: {df['Elevation'].notna().sum()}")
    logger.info(f"Number of clubs with rainfall data: {df['annual_rainfall'].notna().sum()}")

def main():
    """Main function to run the script."""
    try:
        # Set up paths
        paths = setup_paths()
        
        # Load data
        core_df = load_core_data(paths['input']['core'])
        grades_2022, grades_2024 = load_grade_data(paths)
        
        # Merge data
        merged_df = merge_data(core_df, grades_2022, grades_2024)
        
        # Save merged data
        save_data(merged_df, paths['output']['merged'])
        
        # Add grade values and calculate metrics
        graded_df = add_grade_values(merged_df)
        
        # Save graded data
        save_data(graded_df, paths['output']['graded'])
        
        # Create GeoPackage
        gdf = create_geopackage(graded_df, paths['output']['gpkg'])
        
        # Validate data
        validate_data(graded_df)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 