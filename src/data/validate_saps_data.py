import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

def get_project_root():
    """Get the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def load_data():
    """
    Load the cleaned SAPS data.
    """
    print("Loading data...")
    project_root = get_project_root()
    input_file = os.path.join(project_root, "data/processed/cork_saps_2022_cleaned.csv")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} records with {len(df.columns):,} columns")
    return df

def validate_record_count(df):
    """
    Validate record count and geographic coverage.
    """
    # Debug information for geographic IDs
    print("\nGeographic ID Analysis:")
    print("Sample of GEOGIDs:", df['geogid'].astype(str).head().tolist())
    print("\nUnique GEOGID prefixes:", sorted(set(df['geogid'].astype(str).str[:3].unique())))
    
    validation_results = {
        'total_records': int(len(df)),
        'unique_areas': int(df['geogid'].nunique()),
        'expected_record_count': 2028,  # From previous processing
        'validation_passed': bool(len(df) == 2028)
    }
    
    # Check if all records are for Cork (using '47' and '48' as prefixes)
    cork_geogids = df['geogid'].astype(str).str.startswith(('47', '48'))
    non_cork = df[~cork_geogids]
    if len(non_cork) > 0:
        print("\nFound non-Cork records:")
        print("Number of non-Cork records:", len(non_cork))
        print("Sample of non-Cork GEOGIDs:", non_cork['geogid'].astype(str).head().tolist())
    
    validation_results['all_cork_areas'] = bool(cork_geogids.all())
    validation_results['non_cork_count'] = int(len(non_cork))
    
    return validation_results

def validate_value_ranges(df):
    """
    Validate value ranges for numeric columns.
    """
    validation_results = {}
    
    # Define expected ranges for key numeric columns
    expected_ranges = {
        'ur_category': {'min': 1, 'max': 6},
        't1_1agetm': {'min': 0, 'max': 1000},  # Example range for total male population
        't1_1agetf': {'min': 0, 'max': 1000},  # Example range for total female population
        't1_1agett': {'min': 0, 'max': 2000}   # Example range for total population
    }
    
    for col, ranges in expected_ranges.items():
        if col in df.columns:
            validation_results[col] = {
                'min_value': int(df[col].min()),
                'max_value': int(df[col].max()),
                'expected_min': ranges['min'],
                'expected_max': ranges['max'],
                'validation_passed': bool((df[col].min() >= ranges['min']) and (df[col].max() <= ranges['max']))
            }
    
    return validation_results

def validate_data_types(df):
    """
    Validate data types of columns.
    """
    validation_results = {}
    
    # Define expected data types
    expected_types = {
        'guid': 'object',
        'geogid': 'int64',
        'geogdesc': 'object',
        'ur_category': 'int64',
        'ur_category_desc': 'object'
    }
    
    # Add expected types for numeric columns
    numeric_patterns = ['T1_', 'T2_', 'T3_', 'T4_', 'T5_', 'T6_', 'T7_', 'T8_', 'T9_', 
                       'T10_', 'T11_', 'T12_', 'T13_', 'T14_', 'T15_']
    for pattern in numeric_patterns:
        numeric_cols = [col for col in df.columns if col.startswith(pattern)]
        for col in numeric_cols:
            expected_types[col] = 'int64'
    
    for col, expected_type in expected_types.items():
        if col in df.columns:
            validation_results[col] = {
                'actual_type': str(df[col].dtype),
                'expected_type': expected_type,
                'validation_passed': bool(str(df[col].dtype) == expected_type)
            }
    
    return validation_results

def check_missing_values(df):
    """
    Check for missing values and patterns.
    """
    validation_results = {
        'missing_counts': df.isnull().sum().to_dict(),
        'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'columns_with_missing': df.columns[df.isnull().any()].tolist(),
        'total_missing': int(df.isnull().sum().sum()),
        'validation_passed': bool(not df.isnull().any().any())
    }
    
    return validation_results

def calculate_data_quality_metrics(df):
    """
    Calculate data quality metrics.
    """
    # Get numeric columns only for numeric operations
    numeric_df = df.select_dtypes(include=[np.number])
    
    metrics = {
        'completeness': {
            'total_cells': int(df.size),
            'missing_cells': int(df.isnull().sum().sum()),
            'completeness_rate': float((1 - df.isnull().sum().sum() / df.size) * 100)
        },
        'consistency': {
            'numeric_columns': int(len(numeric_df.columns)),
            'categorical_columns': int(len(df.select_dtypes(include=['category', 'object']).columns)),
            'unique_values_ratio': float(df.nunique().mean() / len(df) * 100)
        },
        'validity': {
            'zero_values': int((numeric_df == 0).sum().sum()),
            'negative_values': int((numeric_df < 0).sum().sum()),
            'outlier_threshold': float(numeric_df.quantile(0.99).mean())
        }
    }
    
    return metrics

def create_validation_report(validation_results):
    """
    Create a comprehensive validation report in markdown format.
    """
    print("Creating validation report...")
    
    # Create markdown content
    md_content = [
        "# SAPS Cork Dataset Validation Report",
        f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Overview",
        "\nThis report documents the validation checks performed on the Small Area Population Statistics (SAPS) dataset for Cork.",
        "\n## Validation Results",
        "\n### Record Count Validation",
        f"- Total records: {validation_results['record_count']['total_records']:,}",
        f"- Unique areas: {validation_results['record_count']['unique_areas']:,}",
        f"- Expected record count: {validation_results['record_count']['expected_record_count']:,}",
        f"- All records for Cork: {'Yes' if validation_results['record_count']['all_cork_areas'] else 'No'}",
        "\n### Value Range Validation",
        "\n| Column | Min Value | Max Value | Expected Min | Expected Max | Validation Passed |",
        "|--------|-----------|-----------|--------------|--------------|-------------------|"
    ]
    
    # Add value range validation results
    for col, results in validation_results['value_ranges'].items():
        md_content.append(
            f"| {col} | {results['min_value']} | {results['max_value']} | "
            f"{results['expected_min']} | {results['expected_max']} | "
            f"{'Yes' if results['validation_passed'] else 'No'} |"
        )
    
    # Add data type validation results
    md_content.extend([
        "\n### Data Type Validation",
        "\n| Column | Actual Type | Expected Type | Validation Passed |",
        "|--------|-------------|---------------|-------------------|"
    ])
    
    for col, results in validation_results['data_types'].items():
        md_content.append(
            f"| {col} | {results['actual_type']} | {results['expected_type']} | "
            f"{'Yes' if results['validation_passed'] else 'No'} |"
        )
    
    # Add missing value results
    md_content.extend([
        "\n### Missing Value Analysis",
        f"- Total missing values: {validation_results['missing_values']['total_missing']:,}",
        f"- Columns with missing values: {len(validation_results['missing_values']['columns_with_missing'])}",
        "\nColumns with missing values:",
        "| Column | Missing Count | Missing Percentage |",
        "|--------|---------------|-------------------|"
    ])
    
    for col in validation_results['missing_values']['columns_with_missing']:
        count = validation_results['missing_values']['missing_counts'][col]
        percentage = validation_results['missing_values']['missing_percentages'][col]
        md_content.append(f"| {col} | {count:,} | {percentage:.2f}% |")
    
    # Add data quality metrics
    metrics = validation_results['data_quality_metrics']
    md_content.extend([
        "\n## Data Quality Metrics",
        "\n### Completeness",
        f"- Total cells: {metrics['completeness']['total_cells']:,}",
        f"- Missing cells: {metrics['completeness']['missing_cells']:,}",
        f"- Completeness rate: {metrics['completeness']['completeness_rate']:.2f}%",
        "\n### Consistency",
        f"- Numeric columns: {metrics['consistency']['numeric_columns']}",
        f"- Categorical columns: {metrics['consistency']['categorical_columns']}",
        f"- Unique values ratio: {metrics['consistency']['unique_values_ratio']:.2f}%",
        "\n### Validity",
        f"- Zero values: {metrics['validity']['zero_values']:,}",
        f"- Negative values: {metrics['validity']['negative_values']:,}",
        f"- Outlier threshold: {metrics['validity']['outlier_threshold']:.2f}"
    ])
    
    # Add summary
    md_content.extend([
        "\n## Summary",
        "\n### Validation Status",
        "- Record count validation: " + ("Passed" if validation_results['record_count']['validation_passed'] else "Failed"),
        "- Value range validation: " + ("Passed" if all(r['validation_passed'] for r in validation_results['value_ranges'].values()) else "Failed"),
        "- Data type validation: " + ("Passed" if all(r['validation_passed'] for r in validation_results['data_types'].values()) else "Failed"),
        "- Missing value validation: " + ("Passed" if validation_results['missing_values']['validation_passed'] else "Failed"),
        "\n### Data Quality Assessment",
        "The dataset has been validated for completeness, consistency, and validity. Any issues found have been documented above."
    ])
    
    # Write to file
    project_root = get_project_root()
    output_file = os.path.join(project_root, "data/processed/validation_report.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    print(f"\nValidation report saved to: {output_file}")
    
    # Also save validation results as JSON for programmatic access
    json_file = os.path.join(project_root, "data/processed/validation_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"Validation results saved to: {json_file}")

def main():
    # Load data
    df = load_data()
    
    # Perform validation checks
    validation_results = {
        'record_count': validate_record_count(df),
        'value_ranges': validate_value_ranges(df),
        'data_types': validate_data_types(df),
        'missing_values': check_missing_values(df),
        'data_quality_metrics': calculate_data_quality_metrics(df)
    }
    
    # Create validation report
    create_validation_report(validation_results)

if __name__ == "__main__":
    main() 