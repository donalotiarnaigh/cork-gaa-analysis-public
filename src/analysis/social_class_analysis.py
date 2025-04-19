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

def calculate_social_class_metrics(df):
    """Calculate social class and occupation metrics."""
    logger.info("Calculating social class metrics...")
    
    # Calculate total population by social class (using T9_1_TT which is specific to this theme)
    df['total_social_class'] = df['T9_1_TT']  # Total for social class theme
    
    # Calculate professional/managerial class (middle class)
    df['professional_managerial'] = df['T9_1_PWM'] + df['T9_1_MTM'] + \
                                  df['T9_1_PWF'] + df['T9_1_MTF']
    df['professional_managerial_rate'] = df['professional_managerial'] / df['total_social_class']
    
    # Calculate non-manual class
    df['non_manual'] = df['T9_1_NMM'] + df['T9_1_NMF']
    df['non_manual_rate'] = df['non_manual'] / df['total_social_class']
    
    # Calculate skilled manual class
    df['skilled_manual'] = df['T9_1_SM'] + df['T9_1_SF']
    df['skilled_manual_rate'] = df['skilled_manual'] / df['total_social_class']
    
    # Calculate semi-skilled class
    df['semi_skilled'] = df['T9_1_SSM'] + df['T9_1_SSF']
    df['semi_skilled_rate'] = df['semi_skilled'] / df['total_social_class']
    
    # Calculate unskilled class
    df['unskilled'] = df['T9_1_USM'] + df['T9_1_USF']
    df['unskilled_rate'] = df['unskilled'] / df['total_social_class']
    
    # Calculate gender-specific rates using gender totals
    df['male_total'] = df['T9_1_TM']  # Total males for social class theme
    df['female_total'] = df['T9_1_TF']  # Total females for social class theme
    
    df['male_professional_rate'] = (df['T9_1_PWM'] + df['T9_1_MTM']) / df['male_total']
    df['female_professional_rate'] = (df['T9_1_PWF'] + df['T9_1_MTF']) / df['female_total']
    
    # Add verification totals
    df['sum_all_classes'] = df['professional_managerial'] + df['non_manual'] + \
                           df['skilled_manual'] + df['semi_skilled'] + df['unskilled']
    df['verification_rate'] = df['sum_all_classes'] / df['total_social_class']
    
    # Log verification statistics
    logger.info(f"Average verification rate: {df['verification_rate'].mean():.3f}")
    logger.info(f"Min verification rate: {df['verification_rate'].min():.3f}")
    logger.info(f"Max verification rate: {df['verification_rate'].max():.3f}")
    
    return df

def generate_report(df):
    """Generate a report on social class patterns."""
    logger.info("Generating social class analysis report...")
    
    report = "# Social Class and Occupation Analysis\n\n"
    
    # Data Quality Verification
    report += "## Data Quality Verification\n\n"
    report += f"- Average verification rate: {df['verification_rate'].mean():.3f}\n"
    report += f"- Minimum verification rate: {df['verification_rate'].min():.3f}\n"
    report += f"- Maximum verification rate: {df['verification_rate'].max():.3f}\n\n"
    
    # Summary Statistics
    report += "## Summary Statistics\n\n"
    
    # Total Population Context
    report += "### Population Context\n"
    report += f"- Total Population in Analysis: {df['total_social_class'].sum():,}\n"
    report += f"- Average Small Area Population: {df['total_social_class'].mean():.1f}\n"
    report += f"- Population Range: {df['total_social_class'].min():.0f} to {df['total_social_class'].max():.0f}\n\n"
    
    # Professional/Managerial Class
    report += "### Professional/Managerial Class (Middle Class)\n"
    report += f"- Average Rate: {df['professional_managerial_rate'].mean():.3f}\n"
    report += f"- Median Rate: {df['professional_managerial_rate'].median():.3f}\n"
    report += f"- Standard Deviation: {df['professional_managerial_rate'].std():.3f}\n"
    report += f"- Range: {df['professional_managerial_rate'].min():.3f} to {df['professional_managerial_rate'].max():.3f}\n"
    report += f"- 25th Percentile: {df['professional_managerial_rate'].quantile(0.25):.3f}\n"
    report += f"- 75th Percentile: {df['professional_managerial_rate'].quantile(0.75):.3f}\n\n"
    
    # Non-Manual Class
    report += "### Non-Manual Class\n"
    report += f"- Average Rate: {df['non_manual_rate'].mean():.3f}\n"
    report += f"- Median Rate: {df['non_manual_rate'].median():.3f}\n"
    report += f"- Standard Deviation: {df['non_manual_rate'].std():.3f}\n"
    report += f"- Range: {df['non_manual_rate'].min():.3f} to {df['non_manual_rate'].max():.3f}\n\n"
    
    # Skilled Manual Class
    report += "### Skilled Manual Class\n"
    report += f"- Average Rate: {df['skilled_manual_rate'].mean():.3f}\n"
    report += f"- Median Rate: {df['skilled_manual_rate'].median():.3f}\n"
    report += f"- Standard Deviation: {df['skilled_manual_rate'].std():.3f}\n"
    report += f"- Range: {df['skilled_manual_rate'].min():.3f} to {df['skilled_manual_rate'].max():.3f}\n\n"
    
    # Semi-Skilled Class
    report += "### Semi-Skilled Class\n"
    report += f"- Average Rate: {df['semi_skilled_rate'].mean():.3f}\n"
    report += f"- Median Rate: {df['semi_skilled_rate'].median():.3f}\n"
    report += f"- Standard Deviation: {df['semi_skilled_rate'].std():.3f}\n"
    report += f"- Range: {df['semi_skilled_rate'].min():.3f} to {df['semi_skilled_rate'].max():.3f}\n\n"
    
    # Unskilled Class
    report += "### Unskilled Class\n"
    report += f"- Average Rate: {df['unskilled_rate'].mean():.3f}\n"
    report += f"- Median Rate: {df['unskilled_rate'].median():.3f}\n"
    report += f"- Standard Deviation: {df['unskilled_rate'].std():.3f}\n"
    report += f"- Range: {df['unskilled_rate'].min():.3f} to {df['unskilled_rate'].max():.3f}\n\n"
    
    # Gender Analysis
    report += "## Gender Analysis\n\n"
    report += "### Professional/Managerial Class by Gender\n"
    report += f"- Male Rate: {df['male_professional_rate'].mean():.3f}\n"
    report += f"- Female Rate: {df['female_professional_rate'].mean():.3f}\n"
    report += f"- Gender Gap: {(df['male_professional_rate'].mean() - df['female_professional_rate'].mean()):.3f}\n\n"
    
    # Implications for GAA
    report += "## Implications for GAA\n\n"
    report += "1. **Middle Class Concentration**\n"
    report += f"   - Areas with higher professional/managerial rates (>{(df['professional_managerial_rate'].mean() + df['professional_managerial_rate'].std()):.3f}) may have stronger GAA presence\n"
    report += f"   - Areas with lower professional/managerial rates (<{(df['professional_managerial_rate'].mean() - df['professional_managerial_rate'].std()):.3f}) may need additional support\n\n"
    
    report += "2. **Class Distribution**\n"
    report += "   - Balance between professional and manual classes may affect club resources\n"
    report += "   - Areas with diverse class representation may have broader community engagement\n\n"
    
    report += "3. **Gender Patterns**\n"
    report += "   - Gender gaps in professional class may influence club leadership and participation\n"
    report += "   - Need to consider gender balance in different social classes\n"
    
    # Save report
    report_path = ANALYSIS_DIR / "social_class_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

def create_visualization(df):
    """Create visualizations for social class analysis."""
    logger.info("Creating social class analysis visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')  # Using a built-in style that mimics seaborn
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Social Class Distribution
    class_rates = ['professional_managerial_rate', 'non_manual_rate', 
                   'skilled_manual_rate', 'semi_skilled_rate', 'unskilled_rate']
    class_labels = ['Professional/Managerial', 'Non-Manual', 'Skilled Manual', 
                   'Semi-Skilled', 'Unskilled']
    
    data = [df[rate] for rate in class_rates]
    ax1.boxplot(data, labels=class_labels)
    ax1.set_title('Distribution of Social Class Rates')
    ax1.set_ylabel('Rate')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Gender Comparison in Professional Class
    gender_data = [df['male_professional_rate'], df['female_professional_rate']]
    gender_labels = ['Male', 'Female']
    ax2.boxplot(gender_data, labels=gender_labels)
    ax2.set_title('Professional/Managerial Class by Gender')
    ax2.set_ylabel('Rate')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "social_class_analysis.png")
    logger.info("Visualization saved to data/analysis/social_class_analysis.png")

def main():
    """Main function to run the social class analysis."""
    logger.info("Starting social class analysis...")
    
    # Load data
    df = load_data()
    
    # Calculate metrics
    df = calculate_social_class_metrics(df)
    
    # Generate report
    generate_report(df)
    
    # Create visualization
    create_visualization(df)
    
    logger.info("Social class analysis completed successfully")

if __name__ == "__main__":
    main() 