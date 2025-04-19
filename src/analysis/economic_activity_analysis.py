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

# Define directories
DATA_DIR = Path("data/processed")
ANALYSIS_DIR = Path("data/analysis")

def load_data():
    """Load processed SAPS data."""
    logger.info("Loading SAPS data...")
    df = pd.read_csv(DATA_DIR / "cork_sa_saps_joined_guid.csv")
    logger.info(f"Loaded {len(df)} records")
    return df

def calculate_economic_metrics(df):
    """Calculate economic activity metrics."""
    logger.info("Calculating economic activity metrics...")
    
    # First calculate total sums for each category to understand the complete breakdown
    total_working = df['T8_1_WT'].sum()
    total_looking_first_job = df['T8_1_LFFJT'].sum()
    total_short_term_unemployed = df['T8_1_STUT'].sum()
    total_long_term_unemployed = df['T8_1_LTUT'].sum()
    total_students = df['T8_1_ST'].sum()
    total_home_duties = df['T8_1_LAHFT'].sum()
    total_retired = df['T8_1_RT'].sum()
    total_unable_to_work = df['T8_1_UTWSDT'].sum()
    total_other = df['T8_1_OTHT'].sum()
    total_population = df['T8_1_TT'].sum()
    
    # Log the breakdown
    logger.info("\nEconomic Status Breakdown:")
    logger.info(f"Total Population (15+): {total_population}")
    logger.info(f"Working: {total_working} ({total_working/total_population*100:.1f}%)")
    logger.info(f"Looking for First Job: {total_looking_first_job} ({total_looking_first_job/total_population*100:.1f}%)")
    logger.info(f"Short Term Unemployed: {total_short_term_unemployed} ({total_short_term_unemployed/total_population*100:.1f}%)")
    logger.info(f"Long Term Unemployed: {total_long_term_unemployed} ({total_long_term_unemployed/total_population*100:.1f}%)")
    logger.info(f"Students: {total_students} ({total_students/total_population*100:.1f}%)")
    logger.info(f"Home Duties: {total_home_duties} ({total_home_duties/total_population*100:.1f}%)")
    logger.info(f"Retired: {total_retired} ({total_retired/total_population*100:.1f}%)")
    logger.info(f"Unable to Work: {total_unable_to_work} ({total_unable_to_work/total_population*100:.1f}%)")
    logger.info(f"Other: {total_other} ({total_other/total_population*100:.1f}%)")
    
    # Calculate employment rate (15+ years) using total respondents as denominator
    df['employment_rate'] = df['T8_1_WT'] / df['T8_1_TT']
    
    # Calculate unemployment rate (15+ years) using labor force as denominator
    # Labor force = Working + Looking for First Job + Short Term Unemployed + Long Term Unemployed
    df['labor_force'] = df['T8_1_WT'] + df['T8_1_LFFJT'] + df['T8_1_STUT'] + df['T8_1_LTUT']
    df['unemployment_rate'] = (df['T8_1_STUT'] + df['T8_1_LTUT'] + df['T8_1_LFFJT']) / df['labor_force']
    
    # Calculate student rate (15+ years) using total respondents as denominator
    df['student_rate'] = df['T8_1_ST'] / df['T8_1_TT']
    
    # Calculate retired rate (15+ years) using total respondents as denominator
    df['retired_rate'] = df['T8_1_RT'] / df['T8_1_TT']
    
    # Calculate looking after home/family rate (15+ years) using total respondents as denominator
    df['home_family_rate'] = df['T8_1_LAHFT'] / df['T8_1_TT']
    
    # Calculate unable to work rate (15+ years) using total respondents as denominator
    df['unable_to_work_rate'] = df['T8_1_UTWSDT'] / df['T8_1_TT']
    
    # Calculate economic activity rate (employed + unemployed + students) using total respondents
    df['economic_activity_rate'] = (df['T8_1_WT'] + df['T8_1_STUT'] + df['T8_1_LTUT'] + df['T8_1_LFFJT'] + df['T8_1_ST']) / df['T8_1_TT']
    
    # Calculate gender-specific rates using gender-specific labor force
    df['male_labor_force'] = df['T8_1_WM'] + df['T8_1_LFFJM'] + df['T8_1_STUM'] + df['T8_1_LTUM']
    df['female_labor_force'] = df['T8_1_WF'] + df['T8_1_LFFJF'] + df['T8_1_STUF'] + df['T8_1_LTUF']
    
    df['male_employment_rate'] = df['T8_1_WM'] / df['T8_1_TM']
    df['female_employment_rate'] = df['T8_1_WF'] / df['T8_1_TF']
    
    df['male_unemployment_rate'] = (df['T8_1_STUM'] + df['T8_1_LTUM'] + df['T8_1_LFFJM']) / df['male_labor_force']
    df['female_unemployment_rate'] = (df['T8_1_STUF'] + df['T8_1_LTUF'] + df['T8_1_LFFJF']) / df['female_labor_force']
    
    return df

def generate_economic_report(df):
    """Generate a comprehensive report on economic activity patterns."""
    logger.info("Generating economic activity report...")
    
    # Calculate summary statistics
    summary_stats = {
        'Average Employment Rate': df['employment_rate'].mean(),
        'Median Employment Rate': df['employment_rate'].median(),
        'Average Unemployment Rate': df['unemployment_rate'].mean(),
        'Median Unemployment Rate': df['unemployment_rate'].median(),
        'Average Student Rate': df['student_rate'].mean(),
        'Median Student Rate': df['student_rate'].median(),
        'Average Retired Rate': df['retired_rate'].mean(),
        'Median Retired Rate': df['retired_rate'].median(),
        'Average Economic Activity Rate': df['economic_activity_rate'].mean(),
        'Median Economic Activity Rate': df['economic_activity_rate'].median()
    }
    
    # Calculate gender-specific statistics
    gender_stats = {
        'Average Male Employment Rate': df['male_employment_rate'].mean(),
        'Average Female Employment Rate': df['female_employment_rate'].mean(),
        'Average Male Unemployment Rate': df['male_unemployment_rate'].mean(),
        'Average Female Unemployment Rate': df['female_unemployment_rate'].mean()
    }
    
    # Generate report
    report = "# Economic Activity Analysis Report\n\n"
    
    # Add summary statistics
    report += "## Summary Statistics\n\n"
    for stat, value in summary_stats.items():
        report += f"- {stat}: {value:.3f}\n"
    
    # Add gender-specific statistics
    report += "\n## Gender-Specific Statistics\n\n"
    for stat, value in gender_stats.items():
        report += f"- {stat}: {value:.3f}\n"
    
    # Add distribution statistics
    report += "\n## Distribution Statistics\n\n"
    report += "### Employment Rate Distribution\n"
    report += f"- 25th percentile: {df['employment_rate'].quantile(0.25):.3f}\n"
    report += f"- 75th percentile: {df['employment_rate'].quantile(0.75):.3f}\n"
    report += f"- Standard deviation: {df['employment_rate'].std():.3f}\n"
    
    report += "\n### Unemployment Rate Distribution\n"
    report += f"- 25th percentile: {df['unemployment_rate'].quantile(0.25):.3f}\n"
    report += f"- 75th percentile: {df['unemployment_rate'].quantile(0.75):.3f}\n"
    report += f"- Standard deviation: {df['unemployment_rate'].std():.3f}\n"
    
    report += "\n### Economic Activity Rate Distribution\n"
    report += f"- 25th percentile: {df['economic_activity_rate'].quantile(0.25):.3f}\n"
    report += f"- 75th percentile: {df['economic_activity_rate'].quantile(0.75):.3f}\n"
    report += f"- Standard deviation: {df['economic_activity_rate'].std():.3f}\n"
    
    # Save report
    report_path = ANALYSIS_DIR / "economic_activity_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    return df

def create_visualizations(df):
    """Create visualizations for economic activity analysis."""
    logger.info("Creating visualizations...")
    
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Employment Rate Distribution
        sns.histplot(data=df, x='employment_rate', bins=50, ax=ax1)
        ax1.set_title('Distribution of Employment Rate')
        ax1.set_xlabel('Employment Rate')
        ax1.set_ylabel('Count')
        
        # 2. Unemployment Rate Distribution
        sns.histplot(data=df, x='unemployment_rate', bins=50, ax=ax2)
        ax2.set_title('Distribution of Unemployment Rate')
        ax2.set_xlabel('Unemployment Rate')
        ax2.set_ylabel('Count')
        
        # 3. Economic Activity Rate vs Employment Rate
        sns.scatterplot(data=df, x='economic_activity_rate', y='employment_rate', alpha=0.5, ax=ax3)
        ax3.set_title('Economic Activity Rate vs Employment Rate')
        ax3.set_xlabel('Economic Activity Rate')
        ax3.set_ylabel('Employment Rate')
        
        # 4. Gender Employment Rate Comparison
        gender_data = pd.DataFrame({
            'Rate': pd.concat([df['male_employment_rate'], df['female_employment_rate']]),
            'Gender': ['Male'] * len(df) + ['Female'] * len(df)
        })
        sns.boxplot(data=gender_data, x='Gender', y='Rate', ax=ax4)
        ax4.set_title('Employment Rate by Gender')
        ax4.set_ylabel('Employment Rate')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "economic_activity_analysis.png")
        logger.info("Visualizations saved successfully")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        logger.info("Continuing with report generation only")

def main():
    """Main function to run the economic activity analysis."""
    logger.info("Starting economic activity analysis...")
    
    # Load data
    df = load_data()
    
    # Calculate metrics
    df = calculate_economic_metrics(df)
    
    # Generate report
    df = generate_economic_report(df)
    
    # Create visualizations
    create_visualizations(df)
    
    logger.info("Economic activity analysis completed")

if __name__ == "__main__":
    main() 