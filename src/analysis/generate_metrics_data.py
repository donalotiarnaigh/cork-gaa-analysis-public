import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import matplotlib.ticker as mtick

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = ROOT_DIR / "data"
ORIGINAL_INPUT_FILE = DATA_DIR / "processed" / "cork_clubs_complete_graded.csv"
TRANSFORMED_INPUT_FILE = DATA_DIR / "processed" / "cork_clubs_transformed.csv"
OUTPUT_DIR = ROOT_DIR / "output" / "data"
REPORTS_DIR = ROOT_DIR / "reports"
STATS_DIR = ROOT_DIR / "output" / "statistics"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
STATS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load and prepare the club data with all metrics."""
    try:
        # Load original data
        df_original = pd.read_csv(ORIGINAL_INPUT_FILE)
        logger.info(f"Loaded original data for {len(df_original)} clubs from {ORIGINAL_INPUT_FILE}")
        
        # Check if transformed file exists
        if TRANSFORMED_INPUT_FILE.exists():
            df_transformed = pd.read_csv(TRANSFORMED_INPUT_FILE)
            logger.info(f"Loaded transformed data for {len(df_transformed)} clubs from {TRANSFORMED_INPUT_FILE}")
            
            # Merge original and transformed data
            df = df_original.merge(
                df_transformed[['Club', 'transformed_performance', 'transformed_football', 
                                'transformed_hurling', 'transformed_code_balance', 
                                'transformed_football_improvement', 'transformed_hurling_improvement']],
                on='Club',
                how='left'
            )
            logger.info(f"Merged data for {len(df)} clubs")
        else:
            # If transformed file doesn't exist, calculate transformations
            logger.info("Transformed data file not found. Calculating transformations...")
            df = df_original.copy()
            
            # Apply transformations
            df['transformed_performance'] = 7 - df['overall_performance']
            df['transformed_football'] = 7 - df['football_performance']
            df['transformed_hurling'] = 7 - df['hurling_performance']
            df['transformed_code_balance'] = df['code_balance']  # No transformation
            df['transformed_football_improvement'] = -df['football_improvement']
            df['transformed_hurling_improvement'] = -df['hurling_improvement']
            
            logger.info("Calculated transformed metrics")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_metrics_data_file(df):
    """Create a clean metrics data file with all performance metrics."""
    try:
        # Select relevant columns for the metrics data file
        metrics_cols = [
            'Club', 
            'Latitude', 'Longitude',
            'Grade_2022_football', 'Grade_2022_hurling',
            'Grade_2024_football', 'Grade_2024_hurling',
            'Grade_2022_football_value', 'Grade_2022_hurling_value',
            'Grade_2024_football_value', 'Grade_2024_hurling_value',
            'overall_performance', 'football_performance', 'hurling_performance',
            'code_balance', 'football_improvement', 'hurling_improvement',
            'transformed_performance', 'transformed_football', 'transformed_hurling',
            'transformed_code_balance', 'transformed_football_improvement', 'transformed_hurling_improvement',
            'is_dual_2022', 'is_dual_2024'
        ]
        
        # Create a clean dataframe with only the metrics
        metrics_df = df[metrics_cols].copy()
        
        # Make sure club name is clean
        metrics_df['Club'] = metrics_df['Club'].str.strip()
        
        # Calculate is_dual_club if not already present
        if 'is_dual_club' not in metrics_df.columns:
            metrics_df['is_dual_club'] = metrics_df['is_dual_2024']
        
        # Save to CSV
        output_file = OUTPUT_DIR / "cork_clubs_metrics.csv"
        metrics_df.to_csv(output_file, index=False)
        logger.info(f"Created metrics data file with {len(metrics_df)} clubs at {output_file}")
        
        return metrics_df, output_file
    except Exception as e:
        logger.error(f"Error creating metrics data file: {e}")
        raise

def generate_comparison_table(df):
    """Generate a comparison table of original vs transformed metrics."""
    # Create a comparison table
    stats_orig = df[['overall_performance', 'football_performance', 'hurling_performance', 
                     'code_balance', 'football_improvement', 'hurling_improvement']].describe()
    
    stats_trans = df[['transformed_performance', 'transformed_football', 'transformed_hurling',
                      'transformed_code_balance', 'transformed_football_improvement', 'transformed_hurling_improvement']].describe()
    
    # Rename columns for clarity in the report
    stats_orig.columns = ['Overall Perf.', 'Football Perf.', 'Hurling Perf.', 
                          'Code Balance', 'Football Improv.', 'Hurling Improv.']
    
    stats_trans.columns = ['Trans. Overall', 'Trans. Football', 'Trans. Hurling',
                           'Trans. Code Balance', 'Trans. Football Improv.', 'Trans. Hurling Improv.']
    
    return stats_orig, stats_trans

def generate_dual_vs_single_stats(df):
    """Generate statistics comparing dual vs single-code clubs."""
    # Use is_dual_2024 column to identify dual clubs
    dual_clubs = df[df['is_dual_2024'] == 1]
    single_clubs = df[df['is_dual_2024'] == 0]
    
    metrics = ['overall_performance', 'football_performance', 'hurling_performance', 
               'transformed_performance', 'transformed_football', 'transformed_hurling']
    
    dual_stats = dual_clubs[metrics].mean()
    single_stats = single_clubs[metrics].mean()
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Dual Clubs': dual_stats,
        'Single-Code Clubs': single_stats,
        'Difference': dual_stats - single_stats
    })
    
    return comparison, len(dual_clubs), len(single_clubs)

def create_success_metrics_report(df, metrics_file, stats_orig, stats_trans, dual_comparison, dual_count, single_count):
    """Create a comprehensive success metrics report."""
    try:
        # Calculate additional statistics for the report
        total_clubs = len(df)
        football_clubs = df[df['football_performance'] < 6].shape[0]
        hurling_clubs = df[df['hurling_performance'] < 6].shape[0]
        
        # Create markdown report
        report_content = f"""# Comprehensive Success Metrics Report

## Overview
This report provides a comprehensive overview of the performance metrics used to evaluate Cork GAA clubs. 
The analysis includes both original and transformed metrics, with a focus on dual vs. single-code clubs.

### Dataset Summary
- **Total Clubs Analyzed**: {total_clubs}
- **Football Clubs**: {football_clubs} ({football_clubs/total_clubs*100:.1f}%)
- **Hurling Clubs**: {hurling_clubs} ({hurling_clubs/total_clubs*100:.1f}%)
- **Dual Clubs**: {dual_count} ({dual_count/total_clubs*100:.1f}%)
- **Single-Code Clubs**: {single_count} ({single_count/total_clubs*100:.1f}%)

### Metrics Data File
A clean metrics data file has been created at:
- `{metrics_file}`

This file contains all performance metrics for each club, including both original and transformed versions.

## Performance Metrics Analysis

### Original Metrics Statistics
The following table shows key statistics for the original performance metrics, where **lower scores indicate better performance**:

```
{stats_orig.round(2).to_markdown()}
```

### Transformed Metrics Statistics
The following table shows key statistics for the transformed performance metrics, where **higher scores indicate better performance**:

```
{stats_trans.round(2).to_markdown()}
```

### Dual vs. Single-Code Club Comparison
The table below compares the average performance of dual clubs vs. single-code clubs:

```
{dual_comparison.round(2).to_markdown()}
```

#### Interpretation:
- For original metrics (lower is better): Negative difference means dual clubs perform better
- For transformed metrics (higher is better): Positive difference means dual clubs perform better

## Performance Distribution
![Performance Metrics Distribution](../output/statistics/performance_metrics_distribution.png)

## Dual vs. Single-Code Club Performance
![Dual vs. Single-Code Club Performance](../output/statistics/dual_vs_single_club_performance.png)

## Football vs. Hurling Performance Comparison
![Football vs. Hurling Performance](../output/statistics/football_vs_hurling_performance.png)

## Performance Metrics Correlation
![Performance Metrics Correlation](../output/statistics/performance_metrics_correlation.png)

## Key Findings

1. **Performance Distribution**: The distribution of club performance shows significant variability across Cork GAA clubs.

2. **Dual Club Advantage**: Dual clubs show {("better" if dual_comparison.loc["overall_performance", "Difference"] < 0 else "worse")} overall performance compared to single-code clubs in original metrics.

3. **Transformation Impact**: The transformed metrics provide a more intuitive interpretation of club performance, with higher values indicating better performance.

4. **Performance Balance**: There is a correlation of {df['football_performance'].corr(df['hurling_performance']):.2f} between football and hurling performance, indicating that clubs with strong football teams tend to also have strong hurling teams.

5. **Performance Improvement**: Between 2022 and 2024, {(df['football_improvement'] < 0).sum()} clubs improved in football and {(df['hurling_improvement'] < 0).sum()} clubs improved in hurling.

## Data Dictionary
For a detailed explanation of each metric, please refer to the data dictionary:
- `reports/transformed_metrics_data_dictionary.md`

## References
- Original metrics documentation: `reports/performance_score_documentation.md`
- Transformation methodology: `reports/transformed_metrics_documentation.md`
- Detailed performance statistics: `reports/performance_metrics_statistics.md`
"""
        
        # Write report to file
        report_path = REPORTS_DIR / "comprehensive_success_metrics_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Created comprehensive success metrics report at {report_path}")
        
        return report_path
    except Exception as e:
        logger.error(f"Error creating success metrics report: {e}")
        raise

def main():
    """Main function to orchestrate the analysis process."""
    try:
        # Load data
        df = load_data()
        
        # Create metrics data file
        metrics_df, metrics_file = create_metrics_data_file(df)
        
        # Generate statistics for report
        stats_orig, stats_trans = generate_comparison_table(df)
        dual_comparison, dual_count, single_count = generate_dual_vs_single_stats(df)
        
        # Create comprehensive success metrics report
        report_path = create_success_metrics_report(
            df, metrics_file, stats_orig, stats_trans, 
            dual_comparison, dual_count, single_count
        )
        
        logger.info(f"Success metrics analysis completed. Report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main() 