#!/usr/bin/env python3
"""
Document Transformed Performance Metrics Methodology

This script:
1. Loads both original and transformed club data
2. Validates the transformation calculation
3. Compares original and transformed metrics
4. Generates visualizations showing the relationship
5. Updates the data dictionary with transformed metrics
6. Saves a comprehensive documentation report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.ticker as mtick

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'output' / 'statistics'
REPORTS_DIR = BASE_DIR / 'reports'

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load both original and transformed club data."""
    logger.info("Loading club data...")
    
    try:
        original_filepath = DATA_DIR / 'cork_clubs_complete_graded.csv'
        transformed_filepath = DATA_DIR / 'cork_clubs_transformed.csv'
        
        df_original = pd.read_csv(original_filepath)
        df_transformed = pd.read_csv(transformed_filepath)
        
        logger.info(f"Loaded original data for {len(df_original)} clubs")
        logger.info(f"Loaded transformed data for {len(df_transformed)} clubs")
        
        return df_original, df_transformed
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def validate_transformations(df_original, df_transformed):
    """Validate that transformations were applied correctly."""
    logger.info("Validating transformations...")
    
    validation_results = {}
    
    # Define pairs of original and transformed metrics to compare
    metric_pairs = [
        ('overall_performance', 'transformed_performance'),
        ('football_performance', 'transformed_football'),
        ('hurling_performance', 'transformed_hurling'),
        ('code_balance', 'transformed_code_balance'),
        ('football_improvement', 'transformed_football_improvement'),
        ('hurling_improvement', 'transformed_hurling_improvement')
    ]
    
    for original, transformed in metric_pairs:
        if original == 'code_balance' and transformed == 'transformed_code_balance':
            # Code balance should be the same (not transformed)
            is_valid = df_transformed[transformed].equals(df_transformed[original])
            formula = "No transformation (same as original)"
        elif original.endswith('_improvement') and transformed.endswith('_improvement'):
            # Improvement metrics are reversed (positive = improvement)
            is_valid = df_transformed[transformed].equals(-df_original[original])
            formula = "Reversed sign (negative = improvement → positive = improvement)"
        else:
            # Performance metrics are transformed as 7 - original_value
            calculated = 7 - df_original[original]
            is_valid = df_transformed[transformed].equals(calculated)
            formula = "7 - original_value"
        
        validation_results[original] = {
            'transformed_metric': transformed,
            'transformation_formula': formula,
            'is_valid': is_valid,
            'original_range': f"{df_original[original].min()} to {df_original[original].max()}",
            'transformed_range': f"{df_transformed[transformed].min()} to {df_transformed[transformed].max()}"
        }
    
    return validation_results

def generate_comparison_visualizations(df_original, df_transformed):
    """Generate visualizations comparing original and transformed metrics."""
    logger.info("Generating comparison visualizations...")
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Define pairs of original and transformed metrics to visualize
    visualization_pairs = [
        ('overall_performance', 'transformed_performance', 'Overall Performance'),
        ('football_performance', 'transformed_football', 'Football Performance'),
        ('hurling_performance', 'transformed_hurling', 'Hurling Performance'),
        ('football_improvement', 'transformed_football_improvement', 'Football Improvement'),
        ('hurling_improvement', 'transformed_hurling_improvement', 'Hurling Improvement')
    ]
    
    visualizations = []
    
    # Create a visualization for each pair
    for original, transformed, title in visualization_pairs:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Filter out NA values (6) for cleaner visualization
        if not original.endswith('_improvement'):
            original_data = df_original[df_original[original] < 6][original]
            transformed_data = df_transformed[df_transformed[original] < 6][transformed]
        else:
            original_data = df_original[original]
            transformed_data = df_transformed[transformed]
        
        # Plot original metric on the left
        sns.histplot(original_data, kde=True, bins=10, ax=axes[0])
        axes[0].set_title(f'Original {title}\n(Lower = Better)', fontsize=14)
        axes[0].set_xlabel(original, fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        
        # Add mean and median lines
        axes[0].axvline(original_data.mean(), color='r', linestyle='--', label=f'Mean: {original_data.mean():.2f}')
        axes[0].axvline(original_data.median(), color='g', linestyle='-.', label=f'Median: {original_data.median():.2f}')
        axes[0].legend()
        
        # Plot transformed metric on the right
        sns.histplot(transformed_data, kde=True, bins=10, ax=axes[1])
        axes[1].set_title(f'Transformed {title}\n(Higher = Better)', fontsize=14)
        axes[1].set_xlabel(transformed, fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        
        # Add mean and median lines
        axes[1].axvline(transformed_data.mean(), color='r', linestyle='--', label=f'Mean: {transformed_data.mean():.2f}')
        axes[1].axvline(transformed_data.median(), color='g', linestyle='-.', label=f'Median: {transformed_data.median():.2f}')
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save the figure
        output_path = OUTPUT_DIR / f'{original}_vs_{transformed}.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        visualizations.append({
            'title': f'{title} Comparison',
            'description': f'Comparison of original ({original}) and transformed ({transformed}) metrics',
            'file_path': output_path
        })
    
    # Create a correlation matrix visualization for transformed metrics
    plt.figure(figsize=(12, 10))
    
    transformed_metrics = [
        'transformed_performance', 
        'transformed_football', 
        'transformed_hurling', 
        'transformed_code_balance',
        'transformed_football_improvement',
        'transformed_hurling_improvement'
    ]
    
    corr_matrix = df_transformed[transformed_metrics].corr()
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0,
        square=True,
        linewidths=.5,
        fmt='.2f'
    )
    
    plt.title('Correlation Matrix for Transformed Metrics', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    correlation_path = OUTPUT_DIR / 'transformed_metrics_correlation.png'
    plt.savefig(correlation_path, dpi=300)
    plt.close()
    
    visualizations.append({
        'title': 'Transformed Metrics Correlation',
        'description': 'Correlation matrix showing relationships between transformed metrics',
        'file_path': correlation_path
    })
    
    return visualizations

def update_data_dictionary(validation_results):
    """Create a data dictionary entry for transformed metrics."""
    logger.info("Updating data dictionary with transformed metrics...")
    
    dictionary_md = """# Transformed Performance Metrics Data Dictionary

## Overview
This document provides a detailed description of the transformed performance metrics used in the Cork GAA club analysis. Transformed metrics convert the original scoring system (where lower values represent better performance) to an intuitive system (where higher values represent better performance).

## Transformation Methodology
The primary transformation formula applied to performance metrics is:
```
transformed_value = 7 - original_value
```

This transformation preserves the relative differences between values while inverting the scale so that higher values now represent better performance.

For improvement metrics, the transformation simply reverses the sign:
```
transformed_improvement = -original_improvement
```

This makes positive values represent improvement (rather than negative values in the original system).

## Metric Definitions

"""
    
    for original, info in validation_results.items():
        transformed = info['transformed_metric']
        formula = info['transformation_formula']
        orig_range = info['original_range']
        trans_range = info['transformed_range']
        
        if original == 'overall_performance':
            description = "Combined performance across football and hurling"
        elif original == 'football_performance':
            description = "Performance level in football competitions"
        elif original == 'hurling_performance':
            description = "Performance level in hurling competitions"
        elif original == 'code_balance':
            description = "Similarity of performance across football and hurling"
        elif original == 'football_improvement':
            description = "Change in football performance from 2022 to 2024"
        elif original == 'hurling_improvement':
            description = "Change in hurling performance from 2022 to 2024"
        else:
            description = "Performance metric"
        
        dictionary_md += f"### {transformed}\n"
        dictionary_md += f"- **Description**: {description}\n"
        dictionary_md += f"- **Original Metric**: {original}\n"
        dictionary_md += f"- **Transformation Formula**: {formula}\n"
        dictionary_md += f"- **Original Range**: {orig_range}\n"
        dictionary_md += f"- **Transformed Range**: {trans_range}\n"
        
        if original == 'overall_performance' or original == 'football_performance' or original == 'hurling_performance':
            dictionary_md += f"- **Interpretation**: Higher values (closer to 6) represent better performance. A value of 6 represents Premier Senior level, while a value of 1 represents not competing.\n"
        elif original == 'code_balance':
            dictionary_md += f"- **Interpretation**: Lower values still represent more balanced performance across codes. No transformation was applied to this metric.\n"
        elif original.endswith('_improvement'):
            dictionary_md += f"- **Interpretation**: Positive values now represent improvement (moving up in grade), while negative values represent decline.\n"
        
        dictionary_md += "\n"
    
    # Add usage examples
    dictionary_md += """## Usage Examples

### Filtering
When filtering or selecting clubs based on performance using transformed metrics:
- To select the best-performing clubs: `df[df['transformed_performance'] >= 5]`
- To select the worst-performing clubs: `df[df['transformed_performance'] <= 2]`

### Sorting
When sorting clubs by performance using transformed metrics:
- Best first: `df.sort_values('transformed_performance', ascending=False)`
- Worst first: `df.sort_values('transformed_performance', ascending=True)`

### Visualization
Transformed metrics are particularly useful for visualizations where users expect higher values to represent better performance:
- Axis labels should clarify the direction of performance (Higher = Better)
- Color scales can be aligned with expectations (darker/more intense = better)

### Statistical Analysis
For statistical analyses like correlation and regression, transformed metrics provide more intuitive interpretation:
- Positive correlation with a demographic variable means that increasing the demographic variable is associated with better performance
- Negative correlation means increasing the demographic variable is associated with worse performance
"""
    
    # Save the data dictionary
    dictionary_path = REPORTS_DIR / 'transformed_metrics_data_dictionary.md'
    with open(dictionary_path, 'w') as f:
        f.write(dictionary_md)
    
    return dictionary_path

def generate_documentation_report(validation_results, visualizations, dictionary_path):
    """Generate a comprehensive documentation report for transformed metrics."""
    logger.info("Generating documentation report...")
    
    report_md = """# Transformed Performance Metrics Documentation

## 1. Overview

This report documents the transformation of performance metrics for Cork GAA club analysis. The original performance scoring system uses a convention where lower values represent better performance (e.g., 1 = Premier Senior, 6 = Not Competing). While this aligns with typical ranking systems, it can be counterintuitive in statistical analysis contexts.

To facilitate more intuitive interpretation, particularly in visualizations and correlation analyses, we have created transformed versions of these metrics where higher values represent better performance.

## 2. Transformation Methodology

"""
    
    # Add transformation details for each metric
    for original, info in validation_results.items():
        transformed = info['transformed_metric']
        formula = info['transformation_formula']
        is_valid = info['is_valid']
        
        report_md += f"### {original} → {transformed}\n"
        report_md += f"- **Transformation Formula**: {formula}\n"
        report_md += f"- **Validation Status**: {'Valid' if is_valid else 'Invalid - requires review'}\n"
        
        if original == 'overall_performance' or original == 'football_performance' or original == 'hurling_performance':
            report_md += f"- **Interpretation Change**: Original (1 = best, 6 = worst) → Transformed (6 = best, 1 = worst)\n"
        elif original == 'code_balance':
            report_md += f"- **Interpretation Change**: None (lower values still represent more balanced performance)\n"
        elif original.endswith('_improvement'):
            report_md += f"- **Interpretation Change**: Original (negative = improvement) → Transformed (positive = improvement)\n"
        
        report_md += "\n"
    
    # Add visualizations section
    report_md += """## 3. Visualizations

The following visualizations compare original and transformed metrics:

"""
    
    for viz in visualizations:
        report_md += f"### {viz['title']}\n"
        report_md += f"{viz['description']}\n"
        report_md += f"[View visualization]({viz['file_path'].relative_to(BASE_DIR)})\n\n"
    
    # Add data dictionary reference
    report_md += """## 4. Data Dictionary

A comprehensive data dictionary for the transformed metrics has been created:

"""
    report_md += f"[View data dictionary]({dictionary_path.relative_to(BASE_DIR)})\n\n"
    
    # Add benefits section
    report_md += """## 5. Benefits of Transformed Metrics

### 5.1 Intuitive Interpretation
- Alignment with common expectation that higher values = better
- More natural interpretation of correlations and relationships
- Clearer visualizations with expected orientation

### 5.2 Statistical Analysis Advantages
- Direct interpretation of correlation coefficients (positive correlation = positive effect)
- Clearer model coefficients in regression analyses
- Easier communication of findings to non-technical audiences

### 5.3 Visualization Benefits
- Intuitive color schemes (darker/more intense = better)
- Natural axis orientation (higher = better)
- Reduced need for explanatory notes

## 6. Implementation Notes

The transformed metrics have been added to the main dataset and are available for all analyses:
- File path: `data/processed/cork_clubs_transformed.csv`
- All original metrics are preserved
- All transformed metrics use a consistent naming pattern: `transformed_*`
- Both sets of metrics can be used depending on context and needs

## 7. Recommendation

For most visualization and statistical analysis purposes, we recommend using the transformed metrics to provide more intuitive interpretation. However, for contexts where the original convention is more appropriate (such as direct grade comparison), the original metrics remain available.
"""
    
    # Save the report
    report_path = REPORTS_DIR / 'transformed_metrics_documentation.md'
    with open(report_path, 'w') as f:
        f.write(report_md)
    
    return report_path

def main():
    """Main function to execute the documentation process."""
    try:
        logger.info("Starting transformed metrics documentation process...")
        
        # Load data
        df_original, df_transformed = load_data()
        
        # Validate transformations
        validation_results = validate_transformations(df_original, df_transformed)
        
        # Generate comparison visualizations
        visualizations = generate_comparison_visualizations(df_original, df_transformed)
        
        # Update data dictionary
        dictionary_path = update_data_dictionary(validation_results)
        
        # Generate documentation report
        report_path = generate_documentation_report(validation_results, visualizations, dictionary_path)
        
        logger.info(f"Transformed metrics documentation complete.")
        logger.info(f"Data dictionary saved to: {dictionary_path}")
        logger.info(f"Documentation report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error in documentation process: {e}")
        raise

if __name__ == "__main__":
    main() 