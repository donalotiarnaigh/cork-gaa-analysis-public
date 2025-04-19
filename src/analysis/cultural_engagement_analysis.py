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

def calculate_cultural_metrics(df):
    """Calculate cultural engagement metrics."""
    logger.info("Calculating cultural engagement metrics...")
    
    # Irish Language Metrics
    df['irish_speaker_ratio'] = df['T3_1YES'] / df['T3_1T']
    df['daily_irish_ratio'] = (df['T3_2DIT'] + df['T3_2DIDOT'] + df['T3_2DIWOT'] + 
                              df['T3_2DILOOT'] + df['T3_2DINOT'] + df['T3_2DOEST']) / df['T3_2ALLT']
    
    # Religious Affiliation Metrics
    df['catholic_ratio'] = df['T2_4CA'] / df['T2_4T']
    df['other_religion_ratio'] = df['T2_4OR'] / df['T2_4T']
    df['no_religion_ratio'] = df['T2_4NR'] / df['T2_4T']
    
    # Cultural Diversity Metrics
    df['foreign_birth_ratio'] = (df['T2_1UKBP'] + df['T2_1PLBP'] + df['T2_1INBP'] + 
                                df['T2_1EUBP'] + df['T2_1RWBP']) / df['T2_1TBP']
    
    # Update ethnic diversity calculation to include Irish Travellers
    df['ethnic_diversity'] = (df['T2_2WIT'] + df['T2_2OW'] + df['T2_2BBI'] + 
                            df['T2_2AAI'] + df['T2_2OTH']) / (df['T2_2T'] - df['T2_2NS'])
    
    # Cultural Engagement Index (weighted combination)
    df['cultural_engagement_index'] = (
        0.3 * df['irish_speaker_ratio'] +
        0.2 * df['daily_irish_ratio'] +
        0.2 * df['catholic_ratio'] +
        0.15 * df['foreign_birth_ratio'] +
        0.15 * df['ethnic_diversity']
    )
    
    return df

def generate_cultural_report(df):
    """Generate a comprehensive report on cultural engagement patterns."""
    logger.info("Generating cultural engagement report...")
    
    # Calculate summary statistics
    summary_stats = {
        'Average Irish Speaker Ratio': df['irish_speaker_ratio'].mean(),
        'Median Irish Speaker Ratio': df['irish_speaker_ratio'].median(),
        'Average Daily Irish Usage': df['daily_irish_ratio'].mean(),
        'Median Daily Irish Usage': df['daily_irish_ratio'].median(),
        'Average Catholic Ratio': df['catholic_ratio'].mean(),
        'Median Catholic Ratio': df['catholic_ratio'].median(),
        'Average Cultural Diversity': df['ethnic_diversity'].mean(),
        'Median Cultural Diversity': df['ethnic_diversity'].median(),
        'Average Cultural Engagement Index': df['cultural_engagement_index'].mean(),
        'Median Cultural Engagement Index': df['cultural_engagement_index'].median()
    }
    
    # Generate report
    report = "# Cultural Engagement Analysis Report\n\n"
    
    # Add summary statistics
    report += "## Summary Statistics\n\n"
    for stat, value in summary_stats.items():
        report += f"- {stat}: {value:.3f}\n"
    
    # Add distribution statistics
    report += "\n## Distribution Statistics\n\n"
    
    report += "### Irish Language Distribution\n"
    report += f"- 25th percentile: {df['irish_speaker_ratio'].quantile(0.25):.3f}\n"
    report += f"- 75th percentile: {df['irish_speaker_ratio'].quantile(0.75):.3f}\n"
    report += f"- Standard deviation: {df['irish_speaker_ratio'].std():.3f}\n"
    
    report += "\n### Religious Affiliation Distribution\n"
    report += f"- Catholic (25th-75th): {df['catholic_ratio'].quantile(0.25):.3f} - {df['catholic_ratio'].quantile(0.75):.3f}\n"
    report += f"- Other Religion (25th-75th): {df['other_religion_ratio'].quantile(0.25):.3f} - {df['other_religion_ratio'].quantile(0.75):.3f}\n"
    report += f"- No Religion (25th-75th): {df['no_religion_ratio'].quantile(0.25):.3f} - {df['no_religion_ratio'].quantile(0.75):.3f}\n"
    
    report += "\n### Cultural Diversity Distribution\n"
    report += f"- Foreign Birth (25th-75th): {df['foreign_birth_ratio'].quantile(0.25):.3f} - {df['foreign_birth_ratio'].quantile(0.75):.3f}\n"
    report += f"- Ethnic Diversity (25th-75th): {df['ethnic_diversity'].quantile(0.25):.3f} - {df['ethnic_diversity'].quantile(0.75):.3f}\n"
    
    # Save report
    report_path = ANALYSIS_DIR / "cultural_engagement_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

def create_visualizations(df):
    """Create visualizations for cultural engagement analysis."""
    logger.info("Creating visualizations...")
    
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Irish Speaker Ratio Distribution
        sns.histplot(data=df, x='irish_speaker_ratio', bins=50, ax=ax1)
        ax1.set_title('Distribution of Irish Speaker Ratio')
        ax1.set_xlabel('Irish Speaker Ratio')
        ax1.set_ylabel('Count')
        
        # 2. Religious Affiliation Distribution
        religion_data = df[['catholic_ratio', 'other_religion_ratio', 'no_religion_ratio']].mean()
        religion_data.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Religious Affiliation Distribution')
        ax2.set_xlabel('Religion Type')
        ax2.set_ylabel('Average Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Cultural Diversity Distribution
        sns.histplot(data=df, x='ethnic_diversity', bins=50, ax=ax3)
        ax3.set_title('Distribution of Ethnic Diversity')
        ax3.set_xlabel('Ethnic Diversity Ratio')
        ax3.set_ylabel('Count')
        
        # 4. Cultural Engagement Index vs Total Population
        sns.scatterplot(data=df, x='T1_1AGETT', y='cultural_engagement_index', alpha=0.5, ax=ax4)
        ax4.set_title('Cultural Engagement Index vs Total Population')
        ax4.set_xlabel('Total Population')
        ax4.set_ylabel('Cultural Engagement Index')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "cultural_engagement_analysis.png")
        logger.info("Visualizations saved successfully")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        logger.info("Continuing with report generation only")

def main():
    """Main function to run the cultural engagement analysis."""
    logger.info("Starting cultural engagement analysis...")
    
    # Load data
    df = load_data()
    
    # Calculate metrics
    df = calculate_cultural_metrics(df)
    
    # Generate report
    generate_cultural_report(df)
    
    # Create visualizations
    create_visualizations(df)
    
    logger.info("Cultural engagement analysis completed")

if __name__ == "__main__":
    main() 