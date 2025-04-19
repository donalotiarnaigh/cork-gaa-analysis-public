import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define data directories
DATA_DIR = Path("data/processed")
ANALYSIS_DIR = Path("data/analysis")

def load_data():
    """Load the SAPS data."""
    logger.info("Loading SAPS data...")
    df = pd.read_csv(DATA_DIR / "cork_sa_saps_joined_guid.csv")
    logger.info(f"Loaded {len(df)} records from SAPS data file")
    return df

def calculate_education_metrics(df):
    """Calculate education level metrics."""
    logger.info("Calculating education metrics...")
    
    # Calculate total population for education theme
    df['total_education'] = df['T10_4_TT']
    
    # Calculate secondary education (lower and upper secondary)
    df['secondary_education'] = df['T10_4_LST'] + df['T10_4_UST']
    df['secondary_education_rate'] = df['secondary_education'] / df['total_education']
    
    # Calculate college education (ordinary degree and higher)
    df['college_education'] = df['T10_4_ODNDT'] + df['T10_4_HDPQT']
    df['college_education_rate'] = df['college_education'] / df['total_education']
    
    # Calculate postgraduate education
    df['postgrad_education'] = df['T10_4_PDT'] + df['T10_4_DT']
    df['postgrad_education_rate'] = df['postgrad_education'] / df['total_education']
    
    # Calculate gender-specific rates
    df['male_total'] = df['T10_4_TM']
    df['female_total'] = df['T10_4_TF']
    
    # Male rates
    df['male_secondary_rate'] = (df['T10_4_LSM'] + df['T10_4_USM']) / df['male_total']
    df['male_college_rate'] = (df['T10_4_ODNDM'] + df['T10_4_HDPQM']) / df['male_total']
    df['male_postgrad_rate'] = (df['T10_4_PDM'] + df['T10_4_DM']) / df['male_total']
    
    # Female rates
    df['female_secondary_rate'] = (df['T10_4_LSF'] + df['T10_4_USF']) / df['female_total']
    df['female_college_rate'] = (df['T10_4_ODNDF'] + df['T10_4_HDPQF']) / df['female_total']
    df['female_postgrad_rate'] = (df['T10_4_PDF'] + df['T10_4_DF']) / df['female_total']
    
    # Add verification totals
    df['sum_all_education'] = df['T10_4_NFT'] + df['T10_4_PT'] + df['T10_4_LST'] + \
                            df['T10_4_UST'] + df['T10_4_TVT'] + df['T10_4_ACCAT'] + \
                            df['T10_4_HCT'] + df['T10_4_ODNDT'] + df['T10_4_HDPQT'] + \
                            df['T10_4_PDT'] + df['T10_4_DT'] + df['T10_4_NST']
    df['verification_rate'] = df['sum_all_education'] / df['total_education']
    
    # Log verification statistics
    logger.info(f"Average verification rate: {df['verification_rate'].mean():.3f}")
    logger.info(f"Min verification rate: {df['verification_rate'].min():.3f}")
    logger.info(f"Max verification rate: {df['verification_rate'].max():.3f}")
    
    return df

def generate_report(df):
    """Generate a report on education patterns."""
    logger.info("Generating education analysis report...")
    
    report = "# Education Level Analysis\n\n"
    
    # Data Quality Verification
    report += "## Data Quality Verification\n\n"
    report += f"- Average verification rate: {df['verification_rate'].mean():.3f}\n"
    report += f"- Minimum verification rate: {df['verification_rate'].min():.3f}\n"
    report += f"- Maximum verification rate: {df['verification_rate'].max():.3f}\n\n"
    
    # Summary Statistics
    report += "## Summary Statistics\n\n"
    
    # Total Population Context
    report += "### Population Context\n"
    report += f"- Total Population in Analysis: {df['total_education'].sum():,}\n"
    report += f"- Average Small Area Population: {df['total_education'].mean():.1f}\n"
    report += f"- Population Range: {df['total_education'].min():.0f} to {df['total_education'].max():.0f}\n\n"
    
    # Secondary Education
    report += "### Secondary Education (Lower and Upper Secondary)\n"
    report += f"- Average Rate: {df['secondary_education_rate'].mean():.3f}\n"
    report += f"- Median Rate: {df['secondary_education_rate'].median():.3f}\n"
    report += f"- Standard Deviation: {df['secondary_education_rate'].std():.3f}\n"
    report += f"- Range: {df['secondary_education_rate'].min():.3f} to {df['secondary_education_rate'].max():.3f}\n"
    report += f"- 25th Percentile: {df['secondary_education_rate'].quantile(0.25):.3f}\n"
    report += f"- 75th Percentile: {df['secondary_education_rate'].quantile(0.75):.3f}\n\n"
    
    # College Education
    report += "### College Education (Ordinary Degree and Higher)\n"
    report += f"- Average Rate: {df['college_education_rate'].mean():.3f}\n"
    report += f"- Median Rate: {df['college_education_rate'].median():.3f}\n"
    report += f"- Standard Deviation: {df['college_education_rate'].std():.3f}\n"
    report += f"- Range: {df['college_education_rate'].min():.3f} to {df['college_education_rate'].max():.3f}\n"
    report += f"- 25th Percentile: {df['college_education_rate'].quantile(0.25):.3f}\n"
    report += f"- 75th Percentile: {df['college_education_rate'].quantile(0.75):.3f}\n\n"
    
    # Postgraduate Education
    report += "### Postgraduate Education\n"
    report += f"- Average Rate: {df['postgrad_education_rate'].mean():.3f}\n"
    report += f"- Median Rate: {df['postgrad_education_rate'].median():.3f}\n"
    report += f"- Standard Deviation: {df['postgrad_education_rate'].std():.3f}\n"
    report += f"- Range: {df['postgrad_education_rate'].min():.3f} to {df['postgrad_education_rate'].max():.3f}\n"
    report += f"- 25th Percentile: {df['postgrad_education_rate'].quantile(0.25):.3f}\n"
    report += f"- 75th Percentile: {df['postgrad_education_rate'].quantile(0.75):.3f}\n\n"
    
    # Gender Analysis
    report += "## Gender Analysis\n\n"
    
    # Secondary Education by Gender
    report += "### Secondary Education by Gender\n"
    report += f"- Male Rate: {df['male_secondary_rate'].mean():.3f}\n"
    report += f"- Female Rate: {df['female_secondary_rate'].mean():.3f}\n"
    report += f"- Gender Gap: {(df['male_secondary_rate'].mean() - df['female_secondary_rate'].mean()):.3f}\n\n"
    
    # College Education by Gender
    report += "### College Education by Gender\n"
    report += f"- Male Rate: {df['male_college_rate'].mean():.3f}\n"
    report += f"- Female Rate: {df['female_college_rate'].mean():.3f}\n"
    report += f"- Gender Gap: {(df['male_college_rate'].mean() - df['female_college_rate'].mean()):.3f}\n\n"
    
    # Postgraduate Education by Gender
    report += "### Postgraduate Education by Gender\n"
    report += f"- Male Rate: {df['male_postgrad_rate'].mean():.3f}\n"
    report += f"- Female Rate: {df['female_postgrad_rate'].mean():.3f}\n"
    report += f"- Gender Gap: {(df['male_postgrad_rate'].mean() - df['female_postgrad_rate'].mean()):.3f}\n\n"
    
    # Implications for GAA
    report += "## Implications for GAA\n\n"
    report += "1. **Education Level Distribution**\n"
    report += f"   - Areas with higher education rates (>{(df['college_education_rate'].mean() + df['college_education_rate'].std()):.3f}) may have stronger GAA presence\n"
    report += f"   - Areas with lower education rates (<{(df['college_education_rate'].mean() - df['college_education_rate'].std()):.3f}) may need additional support\n\n"
    
    report += "2. **Gender Patterns**\n"
    report += "   - Gender gaps in education levels may influence club leadership and participation\n"
    report += "   - Need to consider gender balance in different education levels\n\n"
    
    report += "3. **Educational Support**\n"
    report += "   - Areas with lower education rates may benefit from additional educational support\n"
    report += "   - Consider partnerships with educational institutions\n"
    
    # Save report
    report_path = ANALYSIS_DIR / "education_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

def create_visualization(df):
    """Create visualizations for education analysis."""
    logger.info("Creating education analysis visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Education Level Distribution
    education_rates = ['secondary_education_rate', 'college_education_rate', 'postgrad_education_rate']
    education_labels = ['Secondary', 'College', 'Postgraduate']
    
    data = [df[rate] for rate in education_rates]
    ax1.boxplot(data, labels=education_labels)
    ax1.set_title('Distribution of Education Level Rates')
    ax1.set_ylabel('Rate')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Gender Comparison in College Education
    gender_data = [df['male_college_rate'], df['female_college_rate']]
    gender_labels = ['Male', 'Female']
    ax2.boxplot(gender_data, labels=gender_labels)
    ax2.set_title('College Education by Gender')
    ax2.set_ylabel('Rate')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "education_analysis.png")
    logger.info("Visualization saved to data/analysis/education_analysis.png")

def main():
    """Main function to run the education analysis."""
    logger.info("Starting education analysis...")
    
    # Load data
    df = load_data()
    
    # Calculate metrics
    df = calculate_education_metrics(df)
    
    # Generate report
    generate_report(df)
    
    # Create visualization
    create_visualization(df)
    
    logger.info("Education analysis completed successfully")

if __name__ == "__main__":
    main() 