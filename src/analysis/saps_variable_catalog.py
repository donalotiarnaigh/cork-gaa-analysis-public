#!/usr/bin/env python3
"""
Script to analyze and catalog SAPS variables for the GAA Club Success Analysis project.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
WORKSPACE_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = WORKSPACE_ROOT / 'data/processed/cork_sa_saps_joined_guid.csv'
GLOSSARY_PATH = WORKSPACE_ROOT / 'data/raw/SAPS/SAPS_Glossary - Sheet1.csv'
OUTPUT_DIR = WORKSPACE_ROOT / 'data/analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging():
    """Configure logging for the script."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("../../logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"saps_variable_catalog_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_data():
    """Load the integrated SAPS dataset and glossary."""
    logger.info(f'Loading data from {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)
    
    logger.info(f'Loading SAPS glossary from {GLOSSARY_PATH}')
    try:
        glossary = pd.read_csv(GLOSSARY_PATH)
        logger.info(f'Loaded glossary with {len(glossary)} definitions')
    except Exception as e:
        logger.warning(f'Could not load glossary: {str(e)}')
        glossary = None
    
    return df, glossary

def analyze_variables(df, glossary=None):
    """Analyze SAPS variables and create a catalog."""
    # Get all columns
    columns = df.columns.tolist()
    
    # Create a catalog DataFrame
    catalog = pd.DataFrame({
        'variable_name': columns,
        'data_type': df[columns].dtypes.astype(str),
        'missing_count': df[columns].isnull().sum(),
        'missing_percentage': (df[columns].isnull().sum() / len(df) * 100).round(2),
        'unique_values': df[columns].nunique(),
        'zero_count': (df[columns] == 0).sum(),
        'zero_percentage': ((df[columns] == 0).sum() / len(df) * 100).round(2)
    })
    
    # Add glossary definitions if available
    if glossary is not None:
        # Create a dictionary of variable definitions
        definitions = {}
        themes = {}
        tables = {}
        
        for _, row in glossary.iterrows():
            if 'Column Names' in row and 'Description of Field' in row:
                var_name = row['Column Names']
                definitions[var_name] = row['Description of Field']
                themes[var_name] = row['Themes']
                tables[var_name] = row['Tables Within Themes']
        
        # Add definitions and metadata to catalog
        catalog['definition'] = catalog['variable_name'].map(definitions)
        catalog['theme'] = catalog['variable_name'].map(themes)
        catalog['table'] = catalog['variable_name'].map(tables)
        
        # Log variables without definitions
        undefined = catalog[catalog['definition'].isnull()]['variable_name'].tolist()
        if undefined:
            logger.warning(f'Found {len(undefined)} variables without definitions')
            logger.debug(f'Variables without definitions: {undefined}')
    
    # Add variable categories based on naming patterns
    def categorize_variable(name):
        if name.startswith('T1_1AGE'):
            return 'Age Distribution'
        elif name.startswith('T1_2'):
            return 'Marital Status'
        elif name.startswith('T2_'):
            return 'Nationality and Ethnicity'
        elif name.startswith('T3_'):
            return 'Irish Language'
        elif name.startswith('T4_'):
            return 'Education'
        elif name.startswith('T5_'):
            return 'Occupation'
        elif name.startswith('T6_'):
            return 'Housing'
        elif name.startswith('T7_'):
            return 'Voluntary Work'
        elif name.startswith('T8_'):
            return 'Travel to Work'
        elif name.startswith('T9_'):
            return 'Social Class'
        elif name.startswith('T10_'):
            return 'Employment'
        elif name.startswith('T11_'):
            return 'Work Pattern'
        elif name.startswith('T12_'):
            return 'Health'
        elif name.startswith('T13_'):
            return 'Religion'
        elif name.startswith('T14_'):
            return 'Ethnicity'
        elif name.startswith('T15_'):
            return 'Car Ownership'
        else:
            return 'Other'
    
    catalog['category'] = catalog['variable_name'].apply(categorize_variable)
    
    # Add basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        catalog.loc[catalog['variable_name'] == col, 'min_value'] = df[col].min()
        catalog.loc[catalog['variable_name'] == col, 'max_value'] = df[col].max()
        catalog.loc[catalog['variable_name'] == col, 'mean_value'] = df[col].mean()
        catalog.loc[catalog['variable_name'] == col, 'median_value'] = df[col].median()
    
    return catalog

def generate_reports(catalog):
    """Generate various reports from the catalog."""
    # Create output directory
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    
    # Save full catalog
    catalog.to_csv(output_dir / "saps_variable_catalog.csv", index=False)
    logger.info(f"Saved full catalog to {output_dir}/saps_variable_catalog.csv")
    
    # Generate category summary
    category_summary = catalog.groupby('category').agg({
        'variable_name': 'count',
        'missing_percentage': 'mean',
        'zero_percentage': 'mean'
    }).round(2)
    category_summary.to_csv(output_dir / "saps_variable_categories.csv")
    logger.info(f"Saved category summary to {output_dir}/saps_variable_categories.csv")
    
    # Generate quality report
    quality_report = catalog[catalog['missing_percentage'] > 0].sort_values('missing_percentage', ascending=False)
    quality_report.to_csv(output_dir / "saps_variable_quality_report.csv")
    logger.info(f"Saved quality report to {output_dir}/saps_variable_quality_report.csv")
    
    # Generate markdown documentation
    with open(output_dir / "saps_variable_documentation.md", 'w') as f:
        f.write("# SAPS Variable Documentation\n\n")
        
        # Write overview
        f.write("## Overview\n")
        f.write(f"- Total variables: {len(catalog)}\n")
        f.write(f"- Variables with missing values: {len(catalog[catalog['missing_percentage'] > 0])}\n")
        f.write(f"- Variables with high zero counts: {len(catalog[catalog['zero_percentage'] > 50])}\n\n")
        
        # Write category summaries
        f.write("## Category Summaries\n")
        for category in catalog['category'].unique():
            cat_vars = catalog[catalog['category'] == category]
            f.write(f"\n### {category}\n")
            f.write(f"- Number of variables: {len(cat_vars)}\n")
            f.write(f"- Average missing percentage: {cat_vars['missing_percentage'].mean():.2f}%\n")
            f.write(f"- Average zero percentage: {cat_vars['zero_percentage'].mean():.2f}%\n")
            
            # Add theme information if available
            if 'theme' in cat_vars.columns:
                themes = cat_vars['theme'].unique()
                themes = [t for t in themes if pd.notna(t)]
                if themes:
                    f.write("\nThemes:\n")
                    for theme in themes:
                        f.write(f"- {theme}\n")
            
            f.write("\nVariables:\n")
            for _, row in cat_vars.iterrows():
                f.write(f"- {row['variable_name']}: {row['data_type']}\n")
                if 'definition' in row and pd.notna(row['definition']):
                    f.write(f"  - Definition: {row['definition']}\n")
                if 'table' in row and pd.notna(row['table']):
                    f.write(f"  - Table: {row['table']}\n")
                if row['missing_percentage'] > 0:
                    f.write(f"  - Missing: {row['missing_percentage']}%\n")
                if row['zero_percentage'] > 50:
                    f.write(f"  - High zero count: {row['zero_percentage']}%\n")
    
    logger.info(f"Saved markdown documentation to {output_dir}/saps_variable_documentation.md")

def main():
    """Main function to run the variable catalog analysis."""
    # Setup logging
    log_file = setup_logging()
    logger.info("Starting SAPS variable catalog analysis")
    
    try:
        # Load data and glossary
        df, glossary = load_data()
        
        # Analyze variables
        catalog = analyze_variables(df, glossary)
        
        # Generate reports
        generate_reports(catalog)
        
        logger.info("SAPS variable catalog analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in SAPS variable catalog analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 