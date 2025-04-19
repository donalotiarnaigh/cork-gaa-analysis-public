import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = DATA_DIR / "analysis"

# Create analysis directory if it doesn't exist
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the processed SAPS data."""
    logger.info("Loading SAPS data...")
    df = pd.read_csv(PROCESSED_DIR / "cork_sa_saps_joined_guid.csv")
    logger.info(f"Loaded {len(df)} records")
    return df

def calculate_youth_metrics(df):
    """Calculate youth population metrics."""
    logger.info("Calculating youth population metrics...")
    
    # Define age groups
    youth_vars = {
        'under_5': ['T1_1AGE0T', 'T1_1AGE1T', 'T1_1AGE2T', 'T1_1AGE3T', 'T1_1AGE4T'],
        'primary_school': ['T1_1AGE5T', 'T1_1AGE6T', 'T1_1AGE7T', 'T1_1AGE8T', 'T1_1AGE9T', 'T1_1AGE10T', 'T1_1AGE11T', 'T1_1AGE12T'],
        'secondary_school': ['T1_1AGE13T', 'T1_1AGE14T', 'T1_1AGE15T', 'T1_1AGE16T', 'T1_1AGE17T', 'T1_1AGE18T', 'T1_1AGE19T']
    }
    
    # Calculate totals for each age group
    for group, vars in youth_vars.items():
        df[f'{group}_total'] = df[vars].sum(axis=1)
    
    # Calculate youth ratios
    df['youth_ratio'] = (df['under_5_total'] + df['primary_school_total'] + df['secondary_school_total']) / df['T1_1AGETT']
    
    # Calculate gender ratios for youth
    male_vars = ['T1_1AGE0M', 'T1_1AGE1M', 'T1_1AGE2M', 'T1_1AGE3M', 'T1_1AGE4M', 
                 'T1_1AGE5M', 'T1_1AGE6M', 'T1_1AGE7M', 'T1_1AGE8M', 'T1_1AGE9M',
                 'T1_1AGE10M', 'T1_1AGE11M', 'T1_1AGE12M', 'T1_1AGE13M', 'T1_1AGE14M',
                 'T1_1AGE15M', 'T1_1AGE16M', 'T1_1AGE17M', 'T1_1AGE18M', 'T1_1AGE19M']
    
    female_vars = ['T1_1AGE0F', 'T1_1AGE1F', 'T1_1AGE2F', 'T1_1AGE3F', 'T1_1AGE4F',
                   'T1_1AGE5F', 'T1_1AGE6F', 'T1_1AGE7F', 'T1_1AGE8F', 'T1_1AGE9F',
                   'T1_1AGE10F', 'T1_1AGE11F', 'T1_1AGE12F', 'T1_1AGE13F', 'T1_1AGE14F',
                   'T1_1AGE15F', 'T1_1AGE16F', 'T1_1AGE17F', 'T1_1AGE18F', 'T1_1AGE19F']
    
    df['youth_male_total'] = df[male_vars].sum(axis=1)
    df['youth_female_total'] = df[female_vars].sum(axis=1)
    df['youth_gender_ratio'] = df['youth_male_total'] / df['youth_female_total']
    
    return df

def generate_youth_report(df):
    """Generate a comprehensive report on youth population patterns."""
    logger.info("Generating youth population report...")
    
    # Calculate summary statistics
    summary_stats = {
        'Total Youth Population': df['youth_male_total'].sum() + df['youth_female_total'].sum(),
        'Average Youth Ratio': df['youth_ratio'].mean(),
        'Median Youth Ratio': df['youth_ratio'].median(),
        'Average Gender Ratio': df['youth_gender_ratio'].mean(),
        'Median Gender Ratio': df['youth_gender_ratio'].median()
    }
    
    # Create age group breakdown
    age_groups = {
        'Under 5': df['under_5_total'].sum(),
        'Primary School (5-12)': df['primary_school_total'].sum(),
        'Secondary School (13-19)': df['secondary_school_total'].sum()
    }
    
    # Generate report
    report = "# Youth Population Analysis Report\n\n"
    
    # Add summary statistics
    report += "## Summary Statistics\n\n"
    for stat, value in summary_stats.items():
        report += f"- {stat}: {value:.2f}\n"
    
    # Add age group breakdown
    report += "\n## Age Group Breakdown\n\n"
    for group, count in age_groups.items():
        report += f"- {group}: {count:,} people\n"
    
    # Add distribution statistics
    report += "\n## Distribution Statistics\n\n"
    report += "### Youth Ratio Distribution\n"
    report += f"- 25th percentile: {df['youth_ratio'].quantile(0.25):.3f}\n"
    report += f"- 75th percentile: {df['youth_ratio'].quantile(0.75):.3f}\n"
    report += f"- Standard deviation: {df['youth_ratio'].std():.3f}\n"
    
    report += "\n### Gender Ratio Distribution\n"
    report += f"- 25th percentile: {df['youth_gender_ratio'].quantile(0.25):.3f}\n"
    report += f"- 75th percentile: {df['youth_gender_ratio'].quantile(0.75):.3f}\n"
    report += f"- Standard deviation: {df['youth_gender_ratio'].std():.3f}\n"
    
    # Save report
    report_path = ANALYSIS_DIR / "youth_population_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

def create_visualizations(df):
    """Create visualizations for youth population analysis."""
    logger.info("Creating visualizations...")
    
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Youth Ratio Distribution
        sns.histplot(data=df, x='youth_ratio', bins=50, ax=ax1)
        ax1.set_title('Distribution of Youth Ratio')
        ax1.set_xlabel('Youth Ratio')
        ax1.set_ylabel('Count')
        
        # 2. Gender Ratio Distribution
        sns.histplot(data=df, x='youth_gender_ratio', bins=50, ax=ax2)
        ax2.set_title('Distribution of Youth Gender Ratio')
        ax2.set_xlabel('Male/Female Ratio')
        ax2.set_ylabel('Count')
        
        # 3. Age Group Distribution
        age_groups = ['under_5_total', 'primary_school_total', 'secondary_school_total']
        age_data = df[age_groups].sum()
        age_data.plot(kind='bar', ax=ax3)
        ax3.set_title('Total Population by Age Group')
        ax3.set_xlabel('Age Group')
        ax3.set_ylabel('Population')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Youth Ratio vs Total Population
        sns.scatterplot(data=df, x='T1_1AGETT', y='youth_ratio', alpha=0.5, ax=ax4)
        ax4.set_title('Youth Ratio vs Total Population')
        ax4.set_xlabel('Total Population')
        ax4.set_ylabel('Youth Ratio')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "youth_population_analysis.png")
        logger.info("Visualizations saved successfully")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        logger.info("Continuing with report generation only")

def main():
    """Main function to run the youth population analysis."""
    logger.info("Starting youth population analysis...")
    
    # Load data
    df = load_data()
    
    # Calculate metrics
    df = calculate_youth_metrics(df)
    
    # Generate report
    generate_youth_report(df)
    
    # Create visualizations
    create_visualizations(df)
    
    logger.info("Youth population analysis completed")

if __name__ == "__main__":
    main() 