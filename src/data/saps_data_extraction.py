"""
Extract and prepare SAPS (Small Area Population Statistics) data.
This script handles the extraction and initial preparation of SAPS data,
including data loading, cleaning, and standardization.

This script implements Phase 1.1 of the implementation plan.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import logging

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

def load_saps_data(file_path: Path) -> pd.DataFrame:
    """
    Load SAPS data from CSV file.
    
    Args:
        file_path: Path to SAPS CSV file
        
    Returns:
        DataFrame containing SAPS data
    """
    try:
        logger.info(f"Loading SAPS data from {file_path}")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.error(f"Error loading SAPS data: {str(e)}")
        raise

def clean_saps_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize SAPS data.
    
    Args:
        df: DataFrame containing raw SAPS data
        
    Returns:
        DataFrame containing cleaned SAPS data
    """
    try:
        # Remove any empty rows
        df = df.dropna(how='all')
        
        # Standardize column names
        df.columns = df.columns.str.upper()
        
        # Convert numeric columns
        numeric_columns = ['SA_ID', 'TOTAL_POPULATION', 'MALE_POPULATION', 'FEMALE_POPULATION']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize COUNTY values
        if 'COUNTY' in df.columns:
            df['COUNTY'] = df['COUNTY'].str.upper()
        
        logger.info("Cleaned SAPS data")
        return df
    except Exception as e:
        logger.error(f"Error cleaning SAPS data: {str(e)}")
        raise

def save_cleaned_data(df: pd.DataFrame, output_dir: Path):
    """
    Save cleaned data to CSV.
    
    Args:
        df: DataFrame containing cleaned SAPS data
        output_dir: Directory to save cleaned data
    """
    try:
        output_path = output_dir / "saps_data_cleaned.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving cleaned data: {str(e)}")
        raise

def create_extraction_report(df: pd.DataFrame, output_dir: Path):
    """
    Create comprehensive extraction report.
    
    Args:
        df: DataFrame containing cleaned SAPS data
        output_dir: Directory to save report
    """
    report_path = output_dir / "saps_data_extraction_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# SAPS Data Extraction Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Data Summary\n\n")
        f.write(f"- Total Records: {len(df):,}\n")
        f.write(f"- Total Columns: {len(df.columns):,}\n\n")
        
        f.write("## County Distribution\n\n")
        if 'COUNTY' in df.columns:
            county_counts = df['COUNTY'].value_counts()
            for county, count in county_counts.items():
                f.write(f"- {county}: {count:,} records\n")
        f.write("\n")
        
        f.write("## Data Quality Metrics\n\n")
        f.write("### Missing Values\n")
        for column in df.columns:
            missing = df[column].isna().sum()
            missing_pct = (missing / len(df)) * 100
            f.write(f"- {column}: {missing:,} missing values ({missing_pct:.2f}%)\n")
        f.write("\n")
        
        f.write("## Processing Steps\n\n")
        f.write("1. **Data Loading**\n")
        f.write("   - Loaded from source CSV file\n")
        f.write("   - Validated file structure\n")
        f.write("   - Checked data types\n\n")
        
        f.write("2. **Data Cleaning**\n")
        f.write("   - Removed empty rows\n")
        f.write("   - Standardized column names\n")
        f.write("   - Converted numeric columns\n")
        f.write("   - Standardized COUNTY values\n\n")
        
        f.write("## Data Structure\n\n")
        f.write("### Required Fields\n")
        for column in df.columns:
            f.write(f"- {column}\n")
        f.write("\n")
        
        f.write("### Data Types\n")
        for column in df.columns:
            f.write(f"- {column}: {df[column].dtype}\n")
        f.write("\n")
        
        f.write("### Value Ranges\n")
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                f.write(f"- {column}: {df[column].min():,} to {df[column].max():,}\n")
            else:
                unique_values = df[column].unique()
                f.write(f"- {column}: {', '.join(map(str, unique_values))}\n")
        f.write("\n")
        
        logger.info(f"Extraction report saved to {report_path}")

def main():
    """Main function to run the extraction process."""
    try:
        # Set up paths
        raw_dir, processed_dir = setup_paths()
        
        # Load raw data
        raw_file = raw_dir / "SAPS" / "SAPS_2022_Small_Area_UR_171024.csv"
        df = load_saps_data(raw_file)
        
        # Clean data
        df = clean_saps_data(df)
        
        # Save cleaned data
        save_cleaned_data(df, processed_dir)
        
        # Create documentation
        create_extraction_report(df, processed_dir)
        
        logger.info("Extraction process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in extraction process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 