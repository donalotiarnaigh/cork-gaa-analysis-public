import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define data directories
DATA_DIR = Path("data/processed")
ANALYSIS_DIR = Path("data/analysis")

# Define theme prefixes for variable grouping
THEME_PREFIXES = {
    'education': ['T10_4'],
    'employment': ['T8_1', 'T11_4'],
    'social_class': ['T9_2'],
    'youth': ['T1_1'],
    'cultural': ['T2_4', 'T3_1', 'T3_2'],
    'economic': ['T9_1']
}

def load_data() -> pd.DataFrame:
    """Load the SAPS data."""
    logger.info("Loading SAPS data...")
    df = pd.read_csv(DATA_DIR / "cork_sa_saps_joined_guid.csv", low_memory=False)
    logger.info(f"Loaded {len(df)} records from SAPS data file")
    return df

def analyze_zero_values(df: pd.DataFrame) -> Dict:
    """Analyze zero value patterns across variables."""
    logger.info("Analyzing zero value patterns...")
    
    zero_analysis = {}
    
    # Analyze by theme
    for theme, prefixes in THEME_PREFIXES.items():
        theme_vars = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
        zero_counts = (df[theme_vars] == 0).sum()
        zero_percentages = (zero_counts / len(df)) * 100
        
        zero_analysis[theme] = {
            'total_variables': len(theme_vars),
            'zero_percentage_mean': zero_percentages.mean(),
            'zero_percentage_std': zero_percentages.std(),
            'zero_percentage_min': zero_percentages.min(),
            'zero_percentage_max': zero_percentages.max(),
            'high_zero_vars': zero_percentages[zero_percentages > 20].to_dict()
        }
    
    # Analyze rural vs urban
    rural_mask = df['SA_URBAN_AREA_NAME'] == 'Rural'
    urban_mask = ~rural_mask
    
    for theme, prefixes in THEME_PREFIXES.items():
        theme_vars = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
        rural_zeros = (df.loc[rural_mask, theme_vars] == 0).mean() * 100
        urban_zeros = (df.loc[urban_mask, theme_vars] == 0).mean() * 100
        
        zero_analysis[theme]['rural_urban_comparison'] = {
            'rural_zero_mean': rural_zeros.mean(),
            'urban_zero_mean': urban_zeros.mean(),
            'difference': rural_zeros.mean() - urban_zeros.mean()
        }
    
    return zero_analysis

def analyze_missing_values(df: pd.DataFrame) -> Dict:
    """Analyze missing value patterns across variables."""
    logger.info("Analyzing missing value patterns...")
    
    missing_analysis = {}
    
    # Analyze by theme
    for theme, prefixes in THEME_PREFIXES.items():
        theme_vars = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
        missing_counts = df[theme_vars].isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_analysis[theme] = {
            'total_variables': len(theme_vars),
            'missing_percentage_mean': missing_percentages.mean(),
            'missing_percentage_std': missing_percentages.std(),
            'missing_percentage_min': missing_percentages.min(),
            'missing_percentage_max': missing_percentages.max(),
            'variables_with_missing': missing_percentages[missing_percentages > 0].to_dict()
        }
    
    # Calculate data quality score by area
    quality_scores = {}
    for theme, prefixes in THEME_PREFIXES.items():
        theme_vars = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
        completeness = 1 - df[theme_vars].isnull().mean(axis=1)
        quality_scores[theme] = completeness
    
    missing_analysis['area_quality_scores'] = {
        'mean_scores': {theme: scores.mean() for theme, scores in quality_scores.items()},
        'std_scores': {theme: scores.std() for theme, scores in quality_scores.items()},
        'min_scores': {theme: scores.min() for theme, scores in quality_scores.items()},
        'max_scores': {theme: scores.max() for theme, scores in quality_scores.items()}
    }
    
    return missing_analysis

def validate_data_types(df: pd.DataFrame) -> Dict:
    """Validate data types across variables."""
    logger.info("Validating data types...")
    
    type_analysis = {}
    
    # Analyze by theme
    for theme, prefixes in THEME_PREFIXES.items():
        theme_vars = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
        type_info = df[theme_vars].dtypes
        
        type_analysis[theme] = {
            'total_variables': len(theme_vars),
            'type_distribution': type_info.value_counts().to_dict(),
            'numeric_vars': type_info[type_info.isin(['int64', 'float64'])].index.tolist(),
            'categorical_vars': type_info[~type_info.isin(['int64', 'float64'])].index.tolist()
        }
    
    # Check consistency across themes
    consistency_issues = []
    for theme, prefixes in THEME_PREFIXES.items():
        theme_vars = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
        for var in theme_vars:
            if df[var].dtype not in ['int64', 'float64']:
                try:
                    pd.to_numeric(df[var])
                    consistency_issues.append(f"Variable {var} could be numeric but is {df[var].dtype}")
                except:
                    pass
    
    type_analysis['consistency_issues'] = consistency_issues
    return type_analysis

def analyze_distributions(df: pd.DataFrame) -> Dict:
    """Analyze distributions of variables."""
    logger.info("Analyzing variable distributions...")
    
    distribution_analysis = {}
    
    # Analyze by theme
    for theme, prefixes in THEME_PREFIXES.items():
        theme_vars = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
        numeric_vars = df[theme_vars].select_dtypes(include=['int64', 'float64']).columns
        
        theme_stats = {}
        for var in numeric_vars:
            stats = df[var].describe()
            skewness = df[var].skew()
            kurtosis = df[var].kurtosis()
            
            # Identify outliers using IQR method
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[var][(df[var] < (Q1 - 1.5 * IQR)) | (df[var] > (Q3 + 1.5 * IQR))]
            
            theme_stats[var] = {
                'basic_stats': stats.to_dict(),
                'skewness': skewness,
                'kurtosis': kurtosis,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100
            }
        
        distribution_analysis[theme] = theme_stats
    
    return distribution_analysis

def generate_report(zero_analysis: Dict, missing_analysis: Dict, 
                   type_analysis: Dict, distribution_analysis: Dict) -> str:
    """Generate a comprehensive data quality report."""
    logger.info("Generating data quality report...")
    
    report = "# Data Quality Assessment Report\n\n"
    
    # Zero Value Analysis
    report += "## Zero Value Analysis\n\n"
    for theme, analysis in zero_analysis.items():
        report += f"### {theme.title()}\n"
        report += f"- Total Variables: {analysis['total_variables']}\n"
        report += f"- Average Zero Percentage: {analysis['zero_percentage_mean']:.2f}%\n"
        report += f"- Standard Deviation: {analysis['zero_percentage_std']:.2f}%\n"
        report += f"- Range: {analysis['zero_percentage_min']:.2f}% to {analysis['zero_percentage_max']:.2f}%\n"
        
        if 'rural_urban_comparison' in analysis:
            comp = analysis['rural_urban_comparison']
            report += f"- Rural vs Urban Zero Rate Difference: {comp['difference']:.2f}%\n"
        
        if analysis['high_zero_vars']:
            report += "\nHigh Zero Variables (>20%):\n"
            for var, pct in analysis['high_zero_vars'].items():
                report += f"- {var}: {pct:.2f}%\n"
        report += "\n"
    
    # Missing Value Analysis
    report += "## Missing Value Analysis\n\n"
    for theme, analysis in missing_analysis.items():
        if theme != 'area_quality_scores':
            report += f"### {theme.title()}\n"
            report += f"- Total Variables: {analysis['total_variables']}\n"
            report += f"- Average Missing Percentage: {analysis['missing_percentage_mean']:.2f}%\n"
            report += f"- Standard Deviation: {analysis['missing_percentage_std']:.2f}%\n"
            report += f"- Range: {analysis['missing_percentage_min']:.2f}% to {analysis['missing_percentage_max']:.2f}%\n"
            
            if analysis['variables_with_missing']:
                report += "\nVariables with Missing Values:\n"
                for var, pct in analysis['variables_with_missing'].items():
                    report += f"- {var}: {pct:.2f}%\n"
            report += "\n"
    
    # Data Quality Scores
    report += "### Area Quality Scores\n\n"
    scores = missing_analysis['area_quality_scores']
    for metric, theme_scores in scores.items():
        report += f"#### {metric.replace('_', ' ').title()}\n"
        for theme, score in theme_scores.items():
            report += f"- {theme}: {score:.3f}\n"
        report += "\n"
    
    # Data Type Validation
    report += "## Data Type Validation\n\n"
    for theme, analysis in type_analysis.items():
        if theme != 'consistency_issues':
            report += f"### {theme.title()}\n"
            report += f"- Total Variables: {analysis['total_variables']}\n"
            report += "\nType Distribution:\n"
            for dtype, count in analysis['type_distribution'].items():
                report += f"- {dtype}: {count}\n"
            report += "\n"
    
    if type_analysis['consistency_issues']:
        report += "### Consistency Issues\n"
        for issue in type_analysis['consistency_issues']:
            report += f"- {issue}\n"
        report += "\n"
    
    # Distribution Analysis
    report += "## Distribution Analysis\n\n"
    for theme, analysis in distribution_analysis.items():
        report += f"### {theme.title()}\n"
        for var, stats in analysis.items():
            report += f"\n#### {var}\n"
            report += "Basic Statistics:\n"
            for stat, value in stats['basic_stats'].items():
                report += f"- {stat}: {value:.3f}\n"
            report += f"- Skewness: {stats['skewness']:.3f}\n"
            report += f"- Kurtosis: {stats['kurtosis']:.3f}\n"
            report += f"- Outlier Percentage: {stats['outlier_percentage']:.2f}%\n"
        report += "\n"
    
    return report

def create_visualizations(df: pd.DataFrame, zero_analysis: Dict, 
                         missing_analysis: Dict, distribution_analysis: Dict):
    """Create visualizations for data quality assessment."""
    logger.info("Creating data quality visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Zero Value Distribution by Theme
    zero_means = [analysis['zero_percentage_mean'] for analysis in zero_analysis.values()]
    themes = list(zero_analysis.keys())
    bars1 = ax1.bar(themes, zero_means)
    ax1.set_title('Average Zero Value Percentage by Theme')
    ax1.set_ylabel('Percentage')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Data Completeness by Theme
    completeness_means = [100 - analysis['missing_percentage_mean'] 
                         for analysis in missing_analysis.values() 
                         if isinstance(analysis, dict) and 'missing_percentage_mean' in analysis]
    themes = [theme for theme, analysis in missing_analysis.items() 
             if isinstance(analysis, dict) and 'missing_percentage_mean' in analysis]
    bars2 = ax2.bar(themes, completeness_means)
    ax2.set_title('Data Completeness by Theme')
    ax2.set_ylabel('Completeness Percentage')
    ax2.set_ylim([95, 100.5])  # Set y-axis range to highlight small variations
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Quality Score Summary
    scores = missing_analysis['area_quality_scores']['mean_scores']
    score_std = missing_analysis['area_quality_scores']['std_scores']
    themes = list(scores.keys())
    means = list(scores.values())
    stds = list(score_std.values())
    
    bars3 = ax3.bar(themes, means, yerr=stds, capsize=5)
    ax3.set_title('Quality Scores by Theme (Mean ± Std)')
    ax3.set_ylabel('Score')
    ax3.set_ylim([0.95, 1.005])  # Set y-axis range to highlight small variations
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 4: Outlier Distribution
    outlier_data = {}
    for theme, analysis in distribution_analysis.items():
        outlier_data[theme] = [stats['outlier_percentage'] for stats in analysis.values()]
    
    bp = ax4.boxplot(list(outlier_data.values()), labels=list(outlier_data.keys()))
    ax4.set_title('Outlier Percentage Distribution by Theme')
    ax4.set_ylabel('Percentage')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add median values on top of boxplots
    for i, median in enumerate(bp['medians']):
        median_val = median.get_ydata()[0]
        ax4.text(i+1, median_val, f'{median_val:.1f}%',
                ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "data_quality_analysis.png", dpi=300, bbox_inches='tight')
    logger.info("Visualization saved to data/analysis/data_quality_analysis.png")

def main():
    """Main function to run the data quality assessment."""
    logger.info("Starting data quality assessment...")
    
    # Load data
    df = load_data()
    
    # Run analyses
    zero_analysis = analyze_zero_values(df)
    missing_analysis = analyze_missing_values(df)
    type_analysis = validate_data_types(df)
    distribution_analysis = analyze_distributions(df)
    
    # Generate report
    report = generate_report(zero_analysis, missing_analysis, type_analysis, distribution_analysis)
    
    # Save report
    report_path = ANALYSIS_DIR / "data_quality_assessment_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    # Create visualizations
    create_visualizations(df, zero_analysis, missing_analysis, distribution_analysis)
    
    logger.info("Data quality assessment completed successfully")

if __name__ == "__main__":
    main() 