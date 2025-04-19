"""
Script to merge club data from multiple sources.
This script:
1. Loads core club data with coordinates
2. Loads 2022 and 2024 grade data
3. Merges all data while preserving all unique clubs
4. Validates the merged dataset
"""

import pandas as pd
from pathlib import Path
import logging

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
            'core_data': base_dir / 'data/raw/clubs/cork_clubs_unique.csv',
            'grades_2022': base_dir / 'data/raw/clubs/cork_clubs_grades_2022.csv',
            'grades_2024': base_dir / 'data/raw/clubs/cork_clubs_grades_2024.csv'
        },
        'output': {
            'merged': base_dir / 'data/processed/cork_clubs_merged.csv'
        }
    }

def load_data(paths):
    """Load data from CSV files."""
    logger.info("Loading data files...")
    
    # Load core club data
    core_data = pd.read_csv(paths['input']['core_data'])
    logger.info(f"Loaded {len(core_data)} clubs from core data")
    
    # Load grade data
    grades_2022 = pd.read_csv(paths['input']['grades_2022'])
    grades_2024 = pd.read_csv(paths['input']['grades_2024'])
    logger.info(f"Loaded {len(grades_2022)} records from 2022 grades")
    logger.info(f"Loaded {len(grades_2024)} records from 2024 grades")
    
    return core_data, grades_2022, grades_2024

def merge_data(core_data, grades_2022, grades_2024):
    """Merge all club data while preserving all unique clubs."""
    logger.info("Merging club data...")
    
    # First merge with 2022 grades
    merged_data = pd.merge(
        core_data,
        grades_2022,
        on='Club',
        how='left',
        suffixes=('', '_2022')
    )
    
    # Then merge with 2024 grades
    merged_data = pd.merge(
        merged_data,
        grades_2024,
        on='Club',
        how='left',
        suffixes=('', '_2024')
    )
    
    # Rename columns to be more explicit
    column_mapping = {
        'Grade_football': 'Grade_2022_football',
        'Grade_hurling': 'Grade_2022_hurling',
        'Grade_football_2024': 'Grade_2024_football',
        'Grade_hurling_2024': 'Grade_2024_hurling'
    }
    merged_data = merged_data.rename(columns=column_mapping)
    
    logger.info(f"Merged dataset contains {len(merged_data)} clubs")
    return merged_data

def validate_data(df):
    """Validate the merged dataset."""
    logger.info("Validating merged dataset...")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    logger.info("\nMissing values by column:")
    for col, count in missing_counts[missing_counts > 0].items():
        logger.info(f"{col}: {count} missing values")
    
    # Check for duplicate clubs
    duplicates = df[df.duplicated(subset=['Club'], keep=False)]
    if not duplicates.empty:
        logger.warning(f"Found {len(duplicates)} duplicate clubs")
        logger.warning("Duplicate clubs:")
        for club in duplicates['Club'].unique():
            logger.warning(f"- {club}")
    
    # Check grade distributions
    for year in ['2022', '2024']:
        for code in ['football', 'hurling']:
            col = f'Grade_{year}_{code}'
            if col in df.columns:
                distribution = df[col].value_counts()
                logger.info(f"\n{year} {code} grade distribution:")
                for grade, count in distribution.items():
                    logger.info(f"{grade}: {count} clubs")

def save_data(df, output_path):
    """Save the merged dataset."""
    logger.info(f"Saving merged dataset to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved successfully")

def main():
    """Main function to run the script."""
    try:
        # Set up paths
        paths = setup_paths()
        
        # Load data
        core_data, grades_2022, grades_2024 = load_data(paths)
        
        # Merge data
        merged_data = merge_data(core_data, grades_2022, grades_2024)
        
        # Validate data
        validate_data(merged_data)
        
        # Save data
        save_data(merged_data, paths['output']['merged'])
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 