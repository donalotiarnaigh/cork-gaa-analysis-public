"""
Script to add numerical values to club grades and calculate composite performance metrics.
This script:
1. Loads the merged club dataset
2. Assigns numerical values to grades (1-6, where 1 is best)
3. Calculates composite performance metrics
4. Saves the enhanced dataset
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
            'merged': base_dir / 'data/processed/cork_clubs_merged.csv'
        },
        'output': {
            'graded': base_dir / 'data/processed/cork_clubs_graded.csv'
        }
    }

def load_data(paths):
    """Load the merged club dataset."""
    logger.info("Loading merged club data...")
    df = pd.read_csv(paths['input']['merged'])
    logger.info(f"Loaded {len(df)} clubs")
    return df

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

def validate_data(df):
    """Validate the enhanced dataset."""
    logger.info("Validating enhanced dataset...")
    
    # Check grade value distributions
    for year in ['2022', '2024']:
        for code in ['football', 'hurling']:
            col = f'Grade_{year}_{code}_value'
            distribution = df[col].value_counts().sort_index()
            logger.info(f"\n{year} {code} grade value distribution:")
            for value, count in distribution.items():
                logger.info(f"Value {value}: {count} clubs")
    
    # Check performance metrics
    logger.info("\nPerformance metrics summary:")
    logger.info(f"Average overall performance: {df['overall_performance'].mean():.2f}")
    logger.info(f"Average football performance: {df['football_performance'].mean():.2f}")
    logger.info(f"Average hurling performance: {df['hurling_performance'].mean():.2f}")
    logger.info(f"Number of dual clubs (2024): {df['is_dual_2024'].sum()}")

def save_data(df, output_path):
    """Save the enhanced dataset."""
    logger.info(f"Saving enhanced dataset to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved successfully")

def main():
    """Main function to run the script."""
    try:
        # Set up paths
        paths = setup_paths()
        
        # Load data
        df = load_data(paths)
        
        # Add grade values and calculate metrics
        df = add_grade_values(df)
        
        # Validate data
        validate_data(df)
        
        # Save data
        save_data(df, paths['output']['graded'])
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 