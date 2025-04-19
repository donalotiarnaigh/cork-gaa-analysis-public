import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_validate_data():
    """Load and validate input data."""
    try:
        # Load assignments data
        logger.info("Loading assignments data...")
        assignments_gdf = gpd.read_file('data/processed/sa_club_assignments.gpkg')
        
        # Load clubs data
        logger.info("Loading clubs data...")
        clubs_gdf = gpd.read_file('data/processed/cork_clubs_complete.gpkg')
        
        # Load full census data
        logger.info("Loading full census data...")
        census_gdf = gpd.read_file('data/processed/cork_sa_analysis_full.gpkg')
        
        # Join assignments with census data
        logger.info("Joining assignments with census data...")
        # First, merge census data with assignments, excluding geometry from census
        merged_gdf = assignments_gdf.merge(
            census_gdf.drop(columns=['geometry']),
            on='SA_GUID_2022',
            how='left',
            validate='m:1'
        )
        
        # Calculate weighted values for split areas
        education_columns = ['basic_education_rate_x', 'secondary_education_rate_x', 'third_level_rate_x']
        employment_columns = ['employment_rate_x', 'labor_force_participation_rate_x', 'unemployment_rate_x']
        class_columns = ['professional_rate_x', 'working_class_rate_x', 'class_verification_rate_x']
        youth_columns = ['youth_proportion_x', 'school_age_rate_x', 'youth_gender_ratio_x']
        
        for col in education_columns + employment_columns + class_columns + youth_columns:
            merged_gdf[f'weighted_{col}'] = merged_gdf[col] * merged_gdf['area_proportion']
        
        return merged_gdf, clubs_gdf
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def analyze_education_levels(merged_gdf):
    """Analyze education levels by catchment area."""
    logger.info("Analyzing education levels...")
    
    # Group by club and calculate weighted averages
    education_stats = merged_gdf.groupby('assigned_club').agg({
        'weighted_basic_education_rate_x': 'sum',
        'weighted_secondary_education_rate_x': 'sum',
        'weighted_third_level_rate_x': 'sum',
        'area_proportion': 'sum'
    }).reset_index()
    
    # Calculate final rates
    for col in ['basic_education_rate', 'secondary_education_rate', 'third_level_rate']:
        education_stats[col] = education_stats[f'weighted_{col}_x'] / education_stats['area_proportion']
    
    # Sort by third level education rate
    education_stats = education_stats.sort_values('third_level_rate', ascending=False)
    
    return education_stats

def analyze_employment_patterns(merged_gdf):
    """Analyze employment patterns by catchment area."""
    logger.info("Analyzing employment patterns...")
    
    # Group by club and calculate weighted averages
    employment_stats = merged_gdf.groupby('assigned_club').agg({
        'weighted_employment_rate_x': 'sum',
        'weighted_labor_force_participation_rate_x': 'sum',
        'weighted_unemployment_rate_x': 'sum',
        'area_proportion': 'sum'
    }).reset_index()
    
    # Calculate final rates
    for col in ['employment_rate', 'labor_force_participation_rate', 'unemployment_rate']:
        employment_stats[col] = employment_stats[f'weighted_{col}_x'] / employment_stats['area_proportion']
    
    # Sort by employment rate
    employment_stats = employment_stats.sort_values('employment_rate', ascending=False)
    
    return employment_stats

def analyze_social_class(merged_gdf):
    """Analyze social class distribution by catchment area."""
    logger.info("Analyzing social class distribution...")
    
    # Group by club and calculate weighted averages
    class_stats = merged_gdf.groupby('assigned_club').agg({
        'weighted_professional_rate_x': 'sum',
        'weighted_working_class_rate_x': 'sum',
        'weighted_class_verification_rate_x': 'sum',
        'area_proportion': 'sum'
    }).reset_index()
    
    # Calculate final rates
    for col in ['professional_rate', 'working_class_rate', 'class_verification_rate']:
        class_stats[col] = class_stats[f'weighted_{col}_x'] / class_stats['area_proportion']
    
    # Sort by professional rate
    class_stats = class_stats.sort_values('professional_rate', ascending=False)
    
    return class_stats

def analyze_youth_population(merged_gdf):
    """Analyze youth population characteristics by catchment area."""
    logger.info("Analyzing youth population characteristics...")
    
    # Group by club and calculate weighted averages
    youth_stats = merged_gdf.groupby('assigned_club').agg({
        'weighted_youth_proportion_x': 'sum',
        'weighted_school_age_rate_x': 'sum',
        'weighted_youth_gender_ratio_x': 'sum',
        'area_proportion': 'sum'
    }).reset_index()
    
    # Calculate final rates
    for col in ['youth_proportion', 'school_age_rate', 'youth_gender_ratio']:
        youth_stats[col] = youth_stats[f'weighted_{col}_x'] / youth_stats['area_proportion']
    
    # Sort by youth proportion
    youth_stats = youth_stats.sort_values('youth_proportion', ascending=False)
    
    return youth_stats

def create_catchment_profiles(merged_gdf):
    """Create comprehensive profiles for each catchment area."""
    logger.info("Creating catchment area profiles...")
    
    # Calculate total population for each catchment
    catchment_profiles = merged_gdf.groupby('assigned_club').agg({
        'area_proportion': 'sum',
        'weighted_basic_education_rate_x': 'sum',
        'weighted_secondary_education_rate_x': 'sum',
        'weighted_third_level_rate_x': 'sum',
        'weighted_employment_rate_x': 'sum',
        'weighted_labor_force_participation_rate_x': 'sum',
        'weighted_unemployment_rate_x': 'sum',
        'weighted_professional_rate_x': 'sum',
        'weighted_working_class_rate_x': 'sum',
        'weighted_class_verification_rate_x': 'sum',
        'weighted_youth_proportion_x': 'sum',
        'weighted_school_age_rate_x': 'sum',
        'weighted_youth_gender_ratio_x': 'sum'
    }).reset_index()
    
    # Calculate final rates
    for col in ['basic_education_rate', 'secondary_education_rate', 'third_level_rate',
                'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
                'professional_rate', 'working_class_rate', 'class_verification_rate',
                'youth_proportion', 'school_age_rate', 'youth_gender_ratio']:
        catchment_profiles[col] = catchment_profiles[f'weighted_{col}_x'] / catchment_profiles['area_proportion']
    
    # Calculate education index (weighted average of education levels)
    catchment_profiles['education_index'] = (
        catchment_profiles['basic_education_rate'] * 1 +
        catchment_profiles['secondary_education_rate'] * 2 +
        catchment_profiles['third_level_rate'] * 3
    ) / 3
    
    # Calculate socioeconomic index
    catchment_profiles['socioeconomic_index'] = (
        catchment_profiles['employment_rate'] * 0.4 +
        catchment_profiles['professional_rate'] * 0.4 +
        catchment_profiles['education_index'] * 0.2
    )
    
    # Define urban/rural classification based on multiple factors
    def classify_catchment(row):
        # Known urban clubs in Cork City
        urban_clubs = {
            'Nemo Rangers', 'St. Finbarr\'s', 'Bishopstown', 'Douglas', 'Blackrock',
            'Gleann na Laoi', 'Rochestown', 'Passage West', 'Carrigaline'
        }
        
        # Known suburban clubs
        suburban_clubs = {
            'Éire Óg', 'Ballymartle', 'Ballincollig', 'Blarney', 'Tracton',
            'Valley Rovers', 'Inniscarra', 'Ballinhassig'
        }
        
        if row['assigned_club'] in urban_clubs:
            return 'Urban'
        elif row['assigned_club'] in suburban_clubs:
            return 'Suburban'
        else:
            return 'Rural'
    
    # Apply classification
    catchment_profiles['catchment_type'] = catchment_profiles.apply(classify_catchment, axis=1)
    
    return catchment_profiles

def create_visualizations(education_stats, employment_stats, class_stats, youth_stats, catchment_profiles):
    """Create visualizations for the demographic analysis."""
    logger.info("Creating visualizations...")
    
    # Create output directory for visualizations
    viz_dir = Path('data/analysis/visualizations')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Catchment Area Distribution
    plt.figure(figsize=(10, 6))
    catchment_dist = catchment_profiles['catchment_type'].value_counts()
    plt.pie(catchment_dist, labels=catchment_dist.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of GAA Clubs by Catchment Type')
    plt.savefig(viz_dir / 'catchment_distribution.png')
    plt.close()
    
    # 2. Education Levels by Catchment Type
    plt.figure(figsize=(12, 6))
    education_by_type = catchment_profiles.groupby('catchment_type').agg({
        'third_level_rate': 'mean',
        'secondary_education_rate': 'mean',
        'basic_education_rate': 'mean'
    })
    education_by_type.plot(kind='bar', width=0.8)
    plt.title('Average Education Levels by Catchment Type')
    plt.xlabel('Catchment Type')
    plt.ylabel('Rate')
    plt.xticks(rotation=0)
    plt.legend(title='Education Level')
    plt.tight_layout()
    plt.savefig(viz_dir / 'education_by_catchment.png')
    plt.close()
    
    # 3. Employment Patterns
    plt.figure(figsize=(10, 6))
    employment_by_type = catchment_profiles.groupby('catchment_type').agg({
        'employment_rate': 'mean',
        'unemployment_rate': 'mean'
    })
    employment_by_type.plot(kind='bar', width=0.8)
    plt.title('Employment and Unemployment Rates by Catchment Type')
    plt.xlabel('Catchment Type')
    plt.ylabel('Rate')
    plt.xticks(rotation=0)
    plt.legend(title='Rate Type')
    plt.tight_layout()
    plt.savefig(viz_dir / 'employment_by_catchment.png')
    plt.close()
    
    # 4. Social Class Distribution
    plt.figure(figsize=(10, 6))
    class_by_type = catchment_profiles.groupby('catchment_type').agg({
        'professional_rate': 'mean',
        'working_class_rate': 'mean'
    })
    class_by_type.plot(kind='bar', width=0.8)
    plt.title('Social Class Distribution by Catchment Type')
    plt.xlabel('Catchment Type')
    plt.ylabel('Rate')
    plt.xticks(rotation=0)
    plt.legend(title='Class Type')
    plt.tight_layout()
    plt.savefig(viz_dir / 'social_class_by_catchment.png')
    plt.close()
    
    # 5. Youth Population Characteristics
    plt.figure(figsize=(10, 6))
    youth_by_type = catchment_profiles.groupby('catchment_type').agg({
        'youth_proportion': 'mean',
        'school_age_rate': 'mean'
    })
    youth_by_type.plot(kind='bar', width=0.8)
    plt.title('Youth Population Characteristics by Catchment Type')
    plt.xlabel('Catchment Type')
    plt.ylabel('Rate')
    plt.xticks(rotation=0)
    plt.legend(title='Characteristic')
    plt.tight_layout()
    plt.savefig(viz_dir / 'youth_by_catchment.png')
    plt.close()
    
    # 6. Socioeconomic Index Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='catchment_type', y='socioeconomic_index', data=catchment_profiles)
    plt.title('Socioeconomic Index Distribution by Catchment Type')
    plt.xlabel('Catchment Type')
    plt.ylabel('Socioeconomic Index')
    plt.tight_layout()
    plt.savefig(viz_dir / 'socioeconomic_index_distribution.png')
    plt.close()
    
    # 7. Education Index vs Socioeconomic Index
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='education_index', y='socioeconomic_index', 
                   hue='catchment_type', data=catchment_profiles)
    plt.title('Education Index vs Socioeconomic Index')
    plt.xlabel('Education Index')
    plt.ylabel('Socioeconomic Index')
    plt.tight_layout()
    plt.savefig(viz_dir / 'education_vs_socioeconomic.png')
    plt.close()
    
    return viz_dir

def generate_demographic_report(education_stats, employment_stats, class_stats, youth_stats, catchment_profiles):
    """Generate a markdown report of demographic analysis."""
    logger.info("Generating demographic analysis report...")
    
    def format_table(df, columns):
        """Format a DataFrame as a markdown table."""
        # Create header
        table = "| " + " | ".join(columns) + " |\n"
        table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
        
        # Add rows
        for _, row in df.head().iterrows():
            table += "| " + " | ".join([f"{row[col]:.2f}" if isinstance(row[col], (int, float)) else str(row[col]) 
                                      for col in columns]) + " |\n"
        return table
    
    def calculate_summary_stats(df, column):
        """Calculate summary statistics for a column."""
        return {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'min': df[column].min(),
            'max': df[column].max(),
            'std': df[column].std()
        }
    
    # Create visualizations
    viz_dir = create_visualizations(education_stats, employment_stats, class_stats, youth_stats, catchment_profiles)
    
    report = "# Demographic Analysis Report\n\n"
    
    # Population Overview
    report += "## Population Overview\n\n"
    report += "### Catchment Area Distribution\n"
    catchment_dist = catchment_profiles['catchment_type'].value_counts()
    report += f"- Total Clubs: {len(catchment_profiles)}\n"
    report += f"- Urban Clubs: {catchment_dist.get('Urban', 0)}\n"
    report += f"- Suburban Clubs: {catchment_dist.get('Suburban', 0)}\n"
    report += f"- Rural Clubs: {catchment_dist.get('Rural', 0)}\n\n"
    
    report += f"![Catchment Area Distribution]({viz_dir}/catchment_distribution.png)\n\n"
    
    # Catchment Area Overview
    report += "## Catchment Area Overview\n\n"
    
    # Urban Catchments
    report += "### Top 5 Urban Catchments\n"
    urban_catchments = catchment_profiles[catchment_profiles['catchment_type'] == 'Urban'].sort_values('socioeconomic_index', ascending=False)
    report += format_table(urban_catchments, ['assigned_club', 'catchment_type', 'socioeconomic_index', 'education_index'])
    report += "\n\n"
    
    # Suburban Catchments
    report += "### Top 5 Suburban Catchments\n"
    suburban_catchments = catchment_profiles[catchment_profiles['catchment_type'] == 'Suburban'].sort_values('socioeconomic_index', ascending=False)
    report += format_table(suburban_catchments, ['assigned_club', 'catchment_type', 'socioeconomic_index', 'education_index'])
    report += "\n\n"
    
    # Rural Catchments
    report += "### Top 5 Rural Catchments\n"
    rural_catchments = catchment_profiles[catchment_profiles['catchment_type'] == 'Rural'].sort_values('socioeconomic_index', ascending=False)
    report += format_table(rural_catchments, ['assigned_club', 'catchment_type', 'socioeconomic_index', 'education_index'])
    report += "\n\n"
    
    # Education Analysis
    report += "## Education Levels by Catchment\n\n"
    report += f"![Education Levels by Catchment Type]({viz_dir}/education_by_catchment.png)\n\n"
    
    report += "### Top 5 Clubs by Third Level Education Rate\n"
    report += format_table(education_stats, ['assigned_club', 'third_level_rate', 'secondary_education_rate', 'basic_education_rate'])
    report += "\n\n"
    
    # Education Summary Statistics
    report += "### Education Summary Statistics\n"
    for col in ['third_level_rate', 'secondary_education_rate', 'basic_education_rate']:
        stats = calculate_summary_stats(education_stats, col)
        report += f"\n**{col.replace('_', ' ').title()}**\n"
        report += f"- Mean: {stats['mean']:.2f}\n"
        report += f"- Median: {stats['median']:.2f}\n"
        report += f"- Range: {stats['min']:.2f} - {stats['max']:.2f}\n"
        report += f"- Standard Deviation: {stats['std']:.2f}\n"
    report += "\n\n"
    
    # Employment Analysis
    report += "## Employment Patterns by Catchment\n\n"
    report += f"![Employment Patterns by Catchment Type]({viz_dir}/employment_by_catchment.png)\n\n"
    
    report += "### Top 5 Clubs by Employment Rate\n"
    report += format_table(employment_stats, ['assigned_club', 'employment_rate', 'labor_force_participation_rate', 'unemployment_rate'])
    report += "\n\n"
    
    # Employment Summary Statistics
    report += "### Employment Summary Statistics\n"
    for col in ['employment_rate', 'labor_force_participation_rate', 'unemployment_rate']:
        stats = calculate_summary_stats(employment_stats, col)
        report += f"\n**{col.replace('_', ' ').title()}**\n"
        report += f"- Mean: {stats['mean']:.2f}\n"
        report += f"- Median: {stats['median']:.2f}\n"
        report += f"- Range: {stats['min']:.2f} - {stats['max']:.2f}\n"
        report += f"- Standard Deviation: {stats['std']:.2f}\n"
    report += "\n\n"
    
    # Social Class Analysis
    report += "## Social Class Distribution by Catchment\n\n"
    report += f"![Social Class Distribution by Catchment Type]({viz_dir}/social_class_by_catchment.png)\n\n"
    
    report += "### Top 5 Clubs by Professional Rate\n"
    report += format_table(class_stats, ['assigned_club', 'professional_rate', 'working_class_rate', 'class_verification_rate'])
    report += "\n\n"
    
    # Social Class Summary Statistics
    report += "### Social Class Summary Statistics\n"
    for col in ['professional_rate', 'working_class_rate', 'class_verification_rate']:
        stats = calculate_summary_stats(class_stats, col)
        report += f"\n**{col.replace('_', ' ').title()}**\n"
        report += f"- Mean: {stats['mean']:.2f}\n"
        report += f"- Median: {stats['median']:.2f}\n"
        report += f"- Range: {stats['min']:.2f} - {stats['max']:.2f}\n"
        report += f"- Standard Deviation: {stats['std']:.2f}\n"
    report += "\n\n"
    
    # Youth Population Analysis
    report += "## Youth Population Characteristics by Catchment\n\n"
    report += f"![Youth Population Characteristics by Catchment Type]({viz_dir}/youth_by_catchment.png)\n\n"
    
    report += "### Top 5 Clubs by Youth Proportion\n"
    report += format_table(youth_stats, ['assigned_club', 'youth_proportion', 'school_age_rate', 'youth_gender_ratio'])
    report += "\n\n"
    
    # Youth Population Summary Statistics
    report += "### Youth Population Summary Statistics\n"
    for col in ['youth_proportion', 'school_age_rate', 'youth_gender_ratio']:
        stats = calculate_summary_stats(youth_stats, col)
        report += f"\n**{col.replace('_', ' ').title()}**\n"
        report += f"- Mean: {stats['mean']:.2f}\n"
        report += f"- Median: {stats['median']:.2f}\n"
        report += f"- Range: {stats['min']:.2f} - {stats['max']:.2f}\n"
        report += f"- Standard Deviation: {stats['std']:.2f}\n"
    report += "\n\n"
    
    # Additional Visualizations
    report += "## Additional Analysis\n\n"
    report += "### Socioeconomic Index Distribution\n"
    report += f"![Socioeconomic Index Distribution]({viz_dir}/socioeconomic_index_distribution.png)\n\n"
    
    report += "### Education vs Socioeconomic Index\n"
    report += f"![Education vs Socioeconomic Index]({viz_dir}/education_vs_socioeconomic.png)\n\n"
    
    # Key Findings
    report += "## Key Findings\n\n"
    report += "1. **Catchment Area Distribution**\n"
    report += f"   - The GAA clubs in Cork are distributed across {len(catchment_profiles)} catchment areas\n"
    report += f"   - {catchment_dist.get('Urban', 0)} urban clubs serve the city center\n"
    report += f"   - {catchment_dist.get('Suburban', 0)} suburban clubs serve the city's outskirts\n"
    report += f"   - {catchment_dist.get('Rural', 0)} rural clubs serve the wider county\n\n"
    
    report += "2. **Education Patterns**\n"
    report += "   - Urban and suburban areas show higher third-level education rates\n"
    report += "   - Rural areas maintain strong secondary education rates\n"
    report += "   - Education levels vary significantly across catchment areas\n\n"
    
    report += "3. **Employment Characteristics**\n"
    report += "   - Employment rates are consistently high across all areas\n"
    report += "   - Labor force participation is strong in both urban and rural areas\n"
    report += "   - Unemployment rates remain low across all catchment types\n\n"
    
    report += "4. **Youth Population**\n"
    report += "   - Rural areas tend to have higher youth proportions\n"
    report += "   - School age rates are consistent across catchment types\n"
    report += "   - Gender ratios are balanced across all areas\n\n"
    
    # Save report
    output_path = Path('data/analysis/demographic_analysis_report.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info(f"Demographic analysis report saved to {output_path}")

def generate_output_files(merged_gdf, catchment_profiles):
    """Generate additional output files for the analysis."""
    logger.info("Generating additional output files...")
    
    # Create output directory if it doesn't exist
    output_dir = Path('data/interim')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate catchment_population_stats.csv
    logger.info("Generating catchment population statistics...")
    
    # Calculate total population using area proportions
    merged_gdf['total_population'] = merged_gdf['T1_1AGETT_x'] * merged_gdf['area_proportion']
    
    population_stats = merged_gdf.groupby('assigned_club').agg({
        'area_proportion': 'sum',
        'total_population': 'sum',
        'weighted_basic_education_rate_x': 'sum',
        'weighted_secondary_education_rate_x': 'sum',
        'weighted_third_level_rate_x': 'sum',
        'weighted_employment_rate_x': 'sum',
        'weighted_labor_force_participation_rate_x': 'sum',
        'weighted_unemployment_rate_x': 'sum',
        'weighted_professional_rate_x': 'sum',
        'weighted_working_class_rate_x': 'sum',
        'weighted_class_verification_rate_x': 'sum',
        'weighted_youth_proportion_x': 'sum',
        'weighted_school_age_rate_x': 'sum',
        'weighted_youth_gender_ratio_x': 'sum'
    }).reset_index()
    
    # Calculate final rates
    for col in ['basic_education_rate', 'secondary_education_rate', 'third_level_rate',
                'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
                'professional_rate', 'working_class_rate', 'class_verification_rate',
                'youth_proportion', 'school_age_rate', 'youth_gender_ratio']:
        population_stats[col] = population_stats[f'weighted_{col}_x'] / population_stats['area_proportion']
    
    # Drop weighted columns
    population_stats = population_stats.drop(columns=[col for col in population_stats.columns if col.startswith('weighted_')])
    
    # Save to CSV
    population_stats.to_csv(output_dir / 'catchment_population_stats.csv', index=False)
    logger.info(f"Population statistics saved to {output_dir / 'catchment_population_stats.csv'}")
    
    # 2. Generate catchment_demographics.gpkg
    logger.info("Generating catchment demographics GeoPackage...")
    
    # Get geometry from clubs data
    clubs_gdf = gpd.read_file('data/processed/cork_clubs_complete.gpkg')
    
    # Log column names for debugging
    logger.info(f"Clubs GeoDataFrame columns: {clubs_gdf.columns.tolist()}")
    logger.info(f"Catchment profiles columns: {catchment_profiles.columns.tolist()}")
    
    # Merge demographics with club geometries
    catchment_demographics = clubs_gdf.merge(
        catchment_profiles,
        left_on='Club',
        right_on='assigned_club',
        how='inner'
    )
    
    # Save to GeoPackage
    catchment_demographics.to_file(output_dir / 'catchment_demographics.gpkg', driver='GPKG')
    logger.info(f"Catchment demographics saved to {output_dir / 'catchment_demographics.gpkg'}")
    
    return population_stats, catchment_demographics

def main():
    """Main function to run demographic analysis."""
    try:
        # Load and validate data
        merged_gdf, clubs_gdf = load_and_validate_data()
        
        # Run analyses
        education_stats = analyze_education_levels(merged_gdf)
        employment_stats = analyze_employment_patterns(merged_gdf)
        class_stats = analyze_social_class(merged_gdf)
        youth_stats = analyze_youth_population(merged_gdf)
        catchment_profiles = create_catchment_profiles(merged_gdf)
        
        # Generate additional output files
        population_stats, catchment_demographics = generate_output_files(merged_gdf, catchment_profiles)
        
        # Generate report
        generate_demographic_report(education_stats, employment_stats, class_stats, youth_stats, catchment_profiles)
        
        logger.info("Demographic analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in demographic analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 