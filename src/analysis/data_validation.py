import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define data directories
DATA_DIR = Path("data/processed")
ANALYSIS_DIR = Path("data/analysis")

def load_data() -> pd.DataFrame:
    """Load the transformed data."""
    logger.info("Loading transformed data...")
    df = pd.read_csv(DATA_DIR / "cork_sa_transformed.csv", low_memory=False)
    logger.info(f"Loaded {len(df)} records")
    return df

def validate_rate_ranges(df: pd.DataFrame) -> dict:
    """Validate that all rates are between 0 and 1."""
    logger.info("Validating rate ranges...")
    
    rate_columns = [
        'basic_education_rate', 'secondary_education_rate', 'third_level_rate',
        'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
        'professional_rate', 'working_class_rate', 'class_verification_rate',
        'youth_proportion', 'school_age_rate'
    ]
    
    results = {}
    for col in rate_columns:
        invalid_mask = df[col].notna() & ((df[col] < 0) | (df[col] > 1))
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            invalid_values = df[invalid_mask][col]
            results[col] = {
                'invalid_count': invalid_count,
                'min': invalid_values.min(),
                'max': invalid_values.max()
            }
            logger.warning(f"Found {invalid_count} invalid values in {col}")
            logger.warning(f"Range: {invalid_values.min():.3f} to {invalid_values.max():.3f}")
    
    return results

def validate_denominators(df: pd.DataFrame) -> dict:
    """Validate denominators used in rate calculations."""
    logger.info("Validating denominators...")
    
    results = {}
    
    # Education denominators
    education_total = (
        df['T10_4_NFT'] + df['T10_4_PT'] + df['T10_4_LST'] +  # Basic
        df['T10_4_UST'] + df['T10_4_TVT'] +                    # Secondary
        df['T10_4_ACCAT'] + df['T10_4_HCT'] + df['T10_4_ODNDT'] + 
        df['T10_4_HDPQT'] + df['T10_4_PDT'] + df['T10_4_DT']  # Third level
    )
    results['education'] = {
        'total': education_total.mean(),
        'min': education_total.min(),
        'max': education_total.max(),
        'zero_count': (education_total == 0).sum()
    }
    
    # Employment denominators
    labor_force = df['T8_1_WT'] + df['T8_1_LFFJT'] + df['T8_1_STUT']
    results['labor_force'] = {
        'total': labor_force.mean(),
        'min': labor_force.min(),
        'max': labor_force.max(),
        'zero_count': (labor_force == 0).sum()
    }
    
    # Social class denominators
    total_classified = df['T9_2_PT'] - df['T9_2_PZ']
    results['social_class'] = {
        'total': total_classified.mean(),
        'min': total_classified.min(),
        'max': total_classified.max(),
        'zero_count': (total_classified == 0).sum()
    }
    
    # Youth denominators
    total_pop = df['T1_1AGETT']
    results['population'] = {
        'total': total_pop.mean(),
        'min': total_pop.min(),
        'max': total_pop.max(),
        'zero_count': (total_pop == 0).sum()
    }
    
    return results

def check_anomalies(df: pd.DataFrame) -> dict:
    """Check for anomalous values using statistical methods."""
    logger.info("Checking for anomalies...")
    
    rate_columns = [
        'basic_education_rate', 'secondary_education_rate', 'third_level_rate',
        'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
        'professional_rate', 'working_class_rate', 'class_verification_rate',
        'youth_proportion', 'school_age_rate', 'youth_gender_ratio'
    ]
    
    results = {}
    for col in rate_columns:
        # Calculate z-scores
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = df[z_scores > 3][[col]]
        
        if not outliers.empty:
            results[col] = {
                'outlier_count': len(outliers),
                'min': outliers[col].min(),
                'max': outliers[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
            logger.warning(f"Found {len(outliers)} outliers in {col}")
            logger.warning(f"Outlier range: {outliers[col].min():.3f} to {outliers[col].max():.3f}")
    
    return results

def create_validation_plots(df: pd.DataFrame) -> None:
    """Create validation plots for transformed variables."""
    logger.info("Creating validation plots...")
    
    rate_columns = [
        'basic_education_rate', 'secondary_education_rate', 'third_level_rate',
        'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
        'professional_rate', 'working_class_rate', 'class_verification_rate',
        'youth_proportion', 'school_age_rate', 'youth_gender_ratio'
    ]
    
    # Create a figure with multiple subplots
    n_cols = 3
    n_rows = (len(rate_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    # Create box plots for each variable
    for i, col in enumerate(rate_columns):
        sns.boxplot(data=df, y=col, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_ylabel('Rate')
    
    # Remove empty subplots if any
    for i in range(len(rate_columns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'rate_validation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_validation_report(range_results: dict, denominator_results: dict, anomaly_results: dict) -> None:
    """Generate a validation report in markdown format."""
    logger.info("Generating validation report...")
    
    report_path = ANALYSIS_DIR / "rate_validation_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Rate Validation Report\n\n")
        
        # Range validation results
        f.write("## Rate Range Validation\n")
        if range_results:
            f.write("The following variables had values outside the valid range [0,1]:\n\n")
            for col, results in range_results.items():
                f.write(f"- **{col}**:\n")
                f.write(f"  - Invalid values: {results['invalid_count']}\n")
                f.write(f"  - Range: {results['min']:.3f} to {results['max']:.3f}\n\n")
        else:
            f.write("All rates are within valid range [0,1].\n\n")
        
        # Denominator validation results
        f.write("## Denominator Validation\n")
        for category, results in denominator_results.items():
            f.write(f"\n### {category.title()}\n")
            f.write(f"- Mean: {results['total']:.1f}\n")
            f.write(f"- Range: {results['min']:.1f} to {results['max']:.1f}\n")
            f.write(f"- Zero counts: {results['zero_count']}\n")
        
        # Anomaly detection results
        f.write("\n## Anomaly Detection\n")
        if anomaly_results:
            f.write("The following variables had statistical outliers (|z-score| > 3):\n\n")
            for col, results in anomaly_results.items():
                f.write(f"### {col}\n")
                f.write(f"- Outlier count: {results['outlier_count']}\n")
                f.write(f"- Outlier range: {results['min']:.3f} to {results['max']:.3f}\n")
                f.write(f"- Variable mean: {results['mean']:.3f}\n")
                f.write(f"- Variable std: {results['std']:.3f}\n\n")
        else:
            f.write("No statistical outliers detected.\n")
        
        f.write("\n## Visualization\n")
        f.write("Box plots for all rate variables are available in 'rate_validation_plots.png'.\n")
    
    logger.info(f"Validation report saved to {report_path}")

def main():
    """Main function to run the data validation."""
    logger.info("Starting data validation...")
    
    # Load data
    df = load_data()
    
    # Run validations
    range_results = validate_rate_ranges(df)
    denominator_results = validate_denominators(df)
    anomaly_results = check_anomalies(df)
    
    # Create visualizations
    create_validation_plots(df)
    
    # Generate report
    generate_validation_report(range_results, denominator_results, anomaly_results)
    
    logger.info("Data validation completed successfully")

if __name__ == "__main__":
    main() 