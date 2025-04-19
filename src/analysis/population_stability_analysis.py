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

def calculate_stability_metrics(df):
    """Calculate population stability metrics."""
    logger.info("Calculating population stability metrics...")
    
    # Migration Indicators
    df['internal_migration_ratio'] = (df['T2_3EC'] + df['T2_3EI']) / df['T2_3T']  # Movement within Ireland
    df['external_migration_ratio'] = df['T2_3OI'] / df['T2_3T']  # Movement from outside Ireland
    
    # Household Composition Metrics
    df['family_household_ratio'] = df['T6_1_FA_H'] / df['T6_1_TH']
    df['single_person_ratio'] = df['T6_1_BS_H'] / df['T6_1_TH']
    df['multi_generation_ratio'] = df['T6_1_CM_H'] / df['T6_1_TH']
    
    # Mobility Scores
    df['residential_mobility'] = (df['T2_3EC'] + df['T2_3EI'] + df['T2_3OI']) / df['T2_3T']  # All movement
    df['tenure_stability'] = df['T6_3_OMLH'] / df['T6_3_TH']
    
    # Population Stability Index (weighted combination)
    df['population_stability_index'] = (
        0.3 * (1 - df['internal_migration_ratio']) +  # Lower migration = higher stability
        0.2 * (1 - df['external_migration_ratio']) +  # Lower external migration = higher stability
        0.2 * df['family_household_ratio'] +          # Higher family households = higher stability
        0.15 * df['tenure_stability'] +               # Higher owner occupancy = higher stability
        0.15 * (1 - df['single_person_ratio'])       # Lower single person households = higher stability
    )
    
    return df

def generate_stability_report(df):
    """Generate a comprehensive report on population stability patterns."""
    logger.info("Generating population stability report...")
    
    # Calculate summary statistics
    summary_stats = {
        'Average Internal Migration': df['internal_migration_ratio'].mean(),
        'Median Internal Migration': df['internal_migration_ratio'].median(),
        'Average External Migration': df['external_migration_ratio'].mean(),
        'Median External Migration': df['external_migration_ratio'].median(),
        'Average Family Household Ratio': df['family_household_ratio'].mean(),
        'Median Family Household Ratio': df['family_household_ratio'].median(),
        'Average Tenure Stability': df['tenure_stability'].mean(),
        'Median Tenure Stability': df['tenure_stability'].median(),
        'Average Population Stability Index': df['population_stability_index'].mean(),
        'Median Population Stability Index': df['population_stability_index'].median()
    }
    
    # Generate report
    report = "# Population Stability Analysis Report\n\n"
    
    # Add summary statistics
    report += "## Summary Statistics\n\n"
    for stat, value in summary_stats.items():
        report += f"- {stat}: {value:.3f}\n"
    
    # Add distribution statistics
    report += "\n## Distribution Statistics\n\n"
    
    report += "### Migration Distribution\n"
    report += f"- Internal Migration (25th-75th): {df['internal_migration_ratio'].quantile(0.25):.3f} - {df['internal_migration_ratio'].quantile(0.75):.3f}\n"
    report += f"- External Migration (25th-75th): {df['external_migration_ratio'].quantile(0.25):.3f} - {df['external_migration_ratio'].quantile(0.75):.3f}\n"
    
    report += "\n### Household Composition Distribution\n"
    report += f"- Family Households (25th-75th): {df['family_household_ratio'].quantile(0.25):.3f} - {df['family_household_ratio'].quantile(0.75):.3f}\n"
    report += f"- Single Person Households (25th-75th): {df['single_person_ratio'].quantile(0.25):.3f} - {df['single_person_ratio'].quantile(0.75):.3f}\n"
    report += f"- Multi-Generation Households (25th-75th): {df['multi_generation_ratio'].quantile(0.25):.3f} - {df['multi_generation_ratio'].quantile(0.75):.3f}\n"
    
    report += "\n### Stability Metrics Distribution\n"
    report += f"- Residential Mobility (25th-75th): {df['residential_mobility'].quantile(0.25):.3f} - {df['residential_mobility'].quantile(0.75):.3f}\n"
    report += f"- Tenure Stability (25th-75th): {df['tenure_stability'].quantile(0.25):.3f} - {df['tenure_stability'].quantile(0.75):.3f}\n"
    
    # Save report
    report_path = ANALYSIS_DIR / "population_stability_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

def create_visualizations(df):
    """Create visualizations for population stability analysis."""
    logger.info("Creating visualizations...")
    
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Migration Distribution
        sns.histplot(data=df, x='internal_migration_ratio', bins=50, ax=ax1)
        ax1.set_title('Distribution of Internal Migration Ratio')
        ax1.set_xlabel('Internal Migration Ratio')
        ax1.set_ylabel('Count')
        
        # 2. Household Composition Distribution
        household_data = df[['family_household_ratio', 'single_person_ratio', 'multi_generation_ratio']].mean()
        household_data.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Household Composition Distribution')
        ax2.set_xlabel('Household Type')
        ax2.set_ylabel('Average Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Stability Index Distribution
        sns.histplot(data=df, x='population_stability_index', bins=50, ax=ax3)
        ax3.set_title('Distribution of Population Stability Index')
        ax3.set_xlabel('Population Stability Index')
        ax3.set_ylabel('Count')
        
        # 4. Stability Index vs Total Population
        sns.scatterplot(data=df, x='T1_1AGETT', y='population_stability_index', alpha=0.5, ax=ax4)
        ax4.set_title('Population Stability Index vs Total Population')
        ax4.set_xlabel('Total Population')
        ax4.set_ylabel('Population Stability Index')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "population_stability_analysis.png")
        logger.info("Visualizations saved successfully")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        logger.info("Continuing with report generation only")

def main():
    """Main function to run the population stability analysis."""
    logger.info("Starting population stability analysis...")
    
    # Load data
    df = load_data()
    
    # Calculate metrics
    df = calculate_stability_metrics(df)
    
    # Generate report
    generate_stability_report(df)
    
    # Create visualizations
    create_visualizations(df)
    
    logger.info("Population stability analysis completed")

if __name__ == "__main__":
    main() 