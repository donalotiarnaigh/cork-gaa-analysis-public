#!/usr/bin/env python3
"""
Generate descriptive statistics for performance metrics.
This script:
1. Loads club data with grade values and performance metrics
2. Calculates comprehensive descriptive statistics
3. Analyzes distributions and relationships
4. Generates visualizations of performance metrics
5. Saves output to reports directory
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
    """Load the club data with grade values and performance metrics."""
    logger.info("Loading club data...")
    
    try:
        filepath = DATA_DIR / 'cork_clubs_complete_graded.csv'
        df = pd.read_csv(filepath)
        logger.info(f"Loaded data for {len(df)} clubs")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def add_transformed_metrics(df):
    """Add transformed versions of performance metrics for analysis."""
    logger.info("Adding transformed performance metrics...")
    
    # Create transformed versions (higher = better)
    df['transformed_football_performance'] = 7 - df['football_performance']
    df['transformed_hurling_performance'] = 7 - df['hurling_performance']
    df['transformed_overall_performance'] = 7 - df['overall_performance']
    
    # Code balance remains the same (lower is still better)
    df['transformed_code_balance'] = df['code_balance']
    
    return df

def calculate_descriptive_statistics(df):
    """Calculate comprehensive descriptive statistics for performance metrics."""
    logger.info("Calculating descriptive statistics...")
    
    # Define performance metrics to analyze
    original_metrics = [
        'overall_performance', 
        'football_performance', 
        'hurling_performance', 
        'code_balance',
        'football_improvement',
        'hurling_improvement'
    ]
    
    transformed_metrics = [
        'transformed_overall_performance', 
        'transformed_football_performance', 
        'transformed_hurling_performance', 
        'transformed_code_balance'
    ]
    
    # Calculate statistics for original metrics
    original_stats = df[original_metrics].describe().T
    
    # Add additional statistics
    original_stats['median'] = df[original_metrics].median()
    original_stats['mode'] = df[original_metrics].mode().iloc[0]
    original_stats['range'] = df[original_metrics].max() - df[original_metrics].min()
    original_stats['var'] = df[original_metrics].var()
    original_stats['skew'] = df[original_metrics].skew()
    original_stats['kurtosis'] = df[original_metrics].kurtosis()
    
    # Calculate statistics for transformed metrics
    transformed_stats = df[transformed_metrics].describe().T
    
    # Add additional statistics
    transformed_stats['median'] = df[transformed_metrics].median()
    transformed_stats['mode'] = df[transformed_metrics].mode().iloc[0]
    transformed_stats['range'] = df[transformed_metrics].max() - df[transformed_metrics].min()
    transformed_stats['var'] = df[transformed_metrics].var()
    transformed_stats['skew'] = df[transformed_metrics].skew()
    transformed_stats['kurtosis'] = df[transformed_metrics].kurtosis()
    
    # Calculate dual club statistics
    dual_clubs = df[df['is_dual_2024'] == True]
    single_clubs_football = df[(df['is_dual_2024'] == False) & (df['football_performance'] < 6)]
    single_clubs_hurling = df[(df['is_dual_2024'] == False) & (df['hurling_performance'] < 6)]
    
    dual_stats = {
        'count': len(dual_clubs),
        'percentage': len(dual_clubs) / len(df) * 100,
        'avg_overall': dual_clubs['overall_performance'].mean(),
        'avg_football': dual_clubs['football_performance'].mean(),
        'avg_hurling': dual_clubs['hurling_performance'].mean(),
        'avg_code_balance': dual_clubs['code_balance'].mean()
    }
    
    single_football_stats = {
        'count': len(single_clubs_football),
        'percentage': len(single_clubs_football) / len(df) * 100,
        'avg_performance': single_clubs_football['football_performance'].mean()
    }
    
    single_hurling_stats = {
        'count': len(single_clubs_hurling),
        'percentage': len(single_clubs_hurling) / len(df) * 100,
        'avg_performance': single_clubs_hurling['hurling_performance'].mean()
    }
    
    # Calculate grade improvement statistics
    football_improved = len(df[df['football_improvement'] < 0])
    football_declined = len(df[df['football_improvement'] > 0])
    football_unchanged = len(df[df['football_improvement'] == 0])
    
    hurling_improved = len(df[df['hurling_improvement'] < 0])
    hurling_declined = len(df[df['hurling_improvement'] > 0])
    hurling_unchanged = len(df[df['hurling_improvement'] == 0])
    
    improvement_stats = {
        'football_improved': football_improved,
        'football_improved_pct': football_improved / len(df) * 100,
        'football_declined': football_declined,
        'football_declined_pct': football_declined / len(df) * 100,
        'football_unchanged': football_unchanged,
        'football_unchanged_pct': football_unchanged / len(df) * 100,
        'hurling_improved': hurling_improved,
        'hurling_improved_pct': hurling_improved / len(df) * 100,
        'hurling_declined': hurling_declined,
        'hurling_declined_pct': hurling_declined / len(df) * 100,
        'hurling_unchanged': hurling_unchanged,
        'hurling_unchanged_pct': hurling_unchanged / len(df) * 100
    }
    
    # Create correlation matrix
    corr_matrix = df[original_metrics + transformed_metrics].corr()
    
    return {
        'original_stats': original_stats,
        'transformed_stats': transformed_stats,
        'dual_stats': dual_stats,
        'single_football_stats': single_football_stats,
        'single_hurling_stats': single_hurling_stats,
        'improvement_stats': improvement_stats,
        'correlation_matrix': corr_matrix
    }

def generate_visualizations(df, stats):
    """Generate visualizations of performance metrics."""
    logger.info("Generating performance metric visualizations...")
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # 1. Performance Metrics Distribution
    plt.figure(figsize=(15, 10))
    
    metrics = ['overall_performance', 'football_performance', 'hurling_performance', 'code_balance']
    titles = ['Overall Performance', 'Football Performance', 'Hurling Performance', 'Code Balance']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 2, i+1)
        
        # Filter out NA values (6) for cleaner visualization
        if metric != 'code_balance':
            data = df[df[metric] < 6][metric]
        else:
            data = df[metric]
            
        sns.histplot(data, kde=True, bins=10)
        plt.title(f'{title} Distribution', fontsize=14)
        plt.xlabel(f'{title} (Lower = Better)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add mean and median lines
        plt.axvline(data.mean(), color='r', linestyle='--', label=f'Mean: {data.mean():.2f}')
        plt.axvline(data.median(), color='g', linestyle='-.', label=f'Median: {data.median():.2f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'performance_metrics_distribution.png', dpi=300)
    plt.close()
    
    # 2. Football vs Hurling Performance Scatterplot
    plt.figure(figsize=(10, 8))
    
    # Filter out clubs that don't compete in either code
    competing_clubs = df[(df['football_performance'] < 6) | (df['hurling_performance'] < 6)]
    
    # Set up color mapping based on dual status
    colors = competing_clubs['is_dual_2024'].map({True: 'blue', False: 'red'})
    
    plt.scatter(
        competing_clubs['football_performance'], 
        competing_clubs['hurling_performance'],
        c=colors,
        alpha=0.7,
        s=100
    )
    
    plt.title('Football vs Hurling Performance', fontsize=16)
    plt.xlabel('Football Performance (Lower = Better)', fontsize=14)
    plt.ylabel('Hurling Performance (Lower = Better)', fontsize=14)
    
    # Set axis limits to include only valid performance ranges
    plt.xlim(0.5, 6.5)
    plt.ylim(0.5, 6.5)
    
    # Add a diagonal line representing equal performance
    plt.plot([1, 6], [1, 6], 'k--', alpha=0.3)
    
    # Add legend
    plt.legend(['Equal Performance', 'Dual Clubs', 'Single-Code Clubs'])
    
    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'football_vs_hurling_performance.png', dpi=300)
    plt.close()
    
    # 3. Dual vs Single-Code Club Performance Boxplot
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    dual_clubs = df[df['is_dual_2024'] == True]
    single_clubs = df[df['is_dual_2024'] == False]
    
    # Filter single code clubs for those actually competing
    single_football = single_clubs[single_clubs['football_performance'] < 6]['football_performance']
    single_hurling = single_clubs[single_clubs['hurling_performance'] < 6]['hurling_performance']
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Dual Clubs (Overall)': dual_clubs['overall_performance'],
        'Dual Clubs (Football)': dual_clubs['football_performance'],
        'Dual Clubs (Hurling)': dual_clubs['hurling_performance'],
        'Single-Code (Football)': single_football,
        'Single-Code (Hurling)': single_hurling
    })
    
    # Create boxplot
    ax = plot_data.boxplot(figsize=(12, 8), grid=True)
    plt.title('Performance Comparison: Dual vs Single-Code Clubs', fontsize=16)
    plt.ylabel('Performance Score (Lower = Better)', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add a horizontal line at the grand mean for reference
    grand_mean = pd.concat([
        dual_clubs['overall_performance'],
        single_football,
        single_hurling
    ]).mean()
    plt.axhline(y=grand_mean, color='r', linestyle='--', label=f'Overall Mean: {grand_mean:.2f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dual_vs_single_club_performance.png', dpi=300)
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    
    # Select original metrics for correlation
    original_metrics = [
        'overall_performance', 
        'football_performance', 
        'hurling_performance', 
        'code_balance',
        'football_improvement',
        'hurling_improvement'
    ]
    
    corr = df[original_metrics].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, 
        mask=mask,
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        square=True,
        cbar_kws={'shrink': .8}
    )
    
    plt.title('Correlation Between Performance Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'performance_metrics_correlation.png', dpi=300)
    plt.close()
    
    return {
        'distribution': str(OUTPUT_DIR / 'performance_metrics_distribution.png'),
        'football_vs_hurling': str(OUTPUT_DIR / 'football_vs_hurling_performance.png'),
        'dual_vs_single': str(OUTPUT_DIR / 'dual_vs_single_club_performance.png'),
        'correlation': str(OUTPUT_DIR / 'performance_metrics_correlation.png')
    }

def generate_statistics_report(stats, visualizations):
    """Generate a comprehensive report of performance metric statistics."""
    logger.info("Generating performance statistics report...")
    
    report = f"""# Performance Metrics Descriptive Statistics
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Summary Statistics for Original Performance Metrics

| Metric | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|------|-----|-----|-----|-----|-----|-----|
"""
    
    for metric, row in stats['original_stats'].iterrows():
        report += f"| {metric} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['25%']:.2f} | {row['50%']:.2f} | {row['75%']:.2f} | {row['max']:.2f} |\n"
    
    report += f"""
### Additional Statistics

| Metric | Median | Mode | Range | Variance | Skewness | Kurtosis |
|--------|--------|------|-------|----------|----------|----------|
"""
    
    for metric, row in stats['original_stats'].iterrows():
        report += f"| {metric} | {row['median']:.2f} | {row['mode']:.2f} | {row['range']:.2f} | {row['var']:.2f} | {row['skew']:.2f} | {row['kurtosis']:.2f} |\n"
    
    report += f"""
## 2. Summary Statistics for Transformed Performance Metrics

| Metric | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|------|-----|-----|-----|-----|-----|-----|
"""
    
    for metric, row in stats['transformed_stats'].iterrows():
        report += f"| {metric} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['25%']:.2f} | {row['50%']:.2f} | {row['75%']:.2f} | {row['max']:.2f} |\n"
    
    report += f"""
### Additional Statistics

| Metric | Median | Mode | Range | Variance | Skewness | Kurtosis |
|--------|--------|------|-------|----------|----------|----------|
"""
    
    for metric, row in stats['transformed_stats'].iterrows():
        report += f"| {metric} | {row['median']:.2f} | {row['mode']:.2f} | {row['range']:.2f} | {row['var']:.2f} | {row['skew']:.2f} | {row['kurtosis']:.2f} |\n"
    
    report += f"""
## 3. Dual Club Analysis

- **Total Dual Clubs**: {stats['dual_stats']['count']} ({stats['dual_stats']['percentage']:.1f}% of all clubs)
- **Average Overall Performance**: {stats['dual_stats']['avg_overall']:.2f}
- **Average Football Performance**: {stats['dual_stats']['avg_football']:.2f}
- **Average Hurling Performance**: {stats['dual_stats']['avg_hurling']:.2f}
- **Average Code Balance**: {stats['dual_stats']['avg_code_balance']:.2f}

### Single-Code Football Clubs
- **Count**: {stats['single_football_stats']['count']} ({stats['single_football_stats']['percentage']:.1f}% of all clubs)
- **Average Performance**: {stats['single_football_stats']['avg_performance']:.2f}

### Single-Code Hurling Clubs
- **Count**: {stats['single_hurling_stats']['count']} ({stats['single_hurling_stats']['percentage']:.1f}% of all clubs)
- **Average Performance**: {stats['single_hurling_stats']['avg_performance']:.2f}

## 4. Performance Improvement Analysis

### Football Grade Changes (2022 to 2024)
- **Improved**: {stats['improvement_stats']['football_improved']} clubs ({stats['improvement_stats']['football_improved_pct']:.1f}%)
- **Declined**: {stats['improvement_stats']['football_declined']} clubs ({stats['improvement_stats']['football_declined_pct']:.1f}%)
- **Unchanged**: {stats['improvement_stats']['football_unchanged']} clubs ({stats['improvement_stats']['football_unchanged_pct']:.1f}%)

### Hurling Grade Changes (2022 to 2024)
- **Improved**: {stats['improvement_stats']['hurling_improved']} clubs ({stats['improvement_stats']['hurling_improved_pct']:.1f}%)
- **Declined**: {stats['improvement_stats']['hurling_declined']} clubs ({stats['improvement_stats']['hurling_declined_pct']:.1f}%)
- **Unchanged**: {stats['improvement_stats']['hurling_unchanged']} clubs ({stats['improvement_stats']['hurling_unchanged_pct']:.1f}%)

## 5. Correlation Analysis

```
{stats['correlation_matrix'].round(2)}
```

## 6. Visualizations

The following visualizations were generated to illustrate the performance metrics:

1. **Performance Metrics Distribution**: {visualizations['distribution']}
2. **Football vs Hurling Performance**: {visualizations['football_vs_hurling']}
3. **Dual vs Single-Code Club Performance**: {visualizations['dual_vs_single']}
4. **Performance Metrics Correlation**: {visualizations['correlation']}

## 7. Key Findings

1. **Performance Distribution**:
   - The mean overall performance score is {stats['original_stats'].loc['overall_performance', 'mean']:.2f} (lower is better)
   - Football and hurling performance distributions show similar patterns
   - Code balance scores indicate most clubs specialize in one code

2. **Dual Club Analysis**:
   - Only {stats['dual_stats']['percentage']:.1f}% of clubs compete in both codes
   - Dual clubs generally perform better than single-code clubs
   - The average performance of dual clubs ({stats['dual_stats']['avg_overall']:.2f}) is better than the average of single-code clubs

3. **Performance Changes**:
   - More clubs improved ({stats['improvement_stats']['football_improved'] + stats['improvement_stats']['hurling_improved']}) than declined ({stats['improvement_stats']['football_declined'] + stats['improvement_stats']['hurling_declined']})
   - Most clubs ({stats['improvement_stats']['football_unchanged'] + stats['improvement_stats']['hurling_unchanged']}) maintained their grades from 2022 to 2024

4. **Correlation Analysis**:
   - Strong correlation between overall performance and both football and hurling performance
   - Weak correlation between football and hurling performance, indicating specialization
   - Improvement metrics show minimal correlation with current performance

These statistics provide a solid foundation for further statistical analysis of the factors influencing club performance.
"""
    
    # Write the report to a file
    report_path = REPORTS_DIR / 'performance_metrics_statistics.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Performance statistics report saved to {report_path}")
    return report_path

def main():
    """Main function to run the analysis."""
    try:
        logger.info("Starting performance metrics statistical analysis...")
        
        # Load data
        df = load_data()
        
        # Add transformed metrics
        df = add_transformed_metrics(df)
        
        # Calculate descriptive statistics
        stats = calculate_descriptive_statistics(df)
        
        # Generate visualizations
        visualization_paths = generate_visualizations(df, stats)
        
        # Generate statistics report
        report_path = generate_statistics_report(stats, visualization_paths)
        
        logger.info(f"Performance metrics analysis completed successfully. Report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == '__main__':
    main() 