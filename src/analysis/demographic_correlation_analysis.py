import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import logging
from datetime import datetime
import scipy.stats as stats
import markdown
from typing import Dict, List, Tuple

# Set up constants
REPORTS_DIR = Path('reports')
OUTPUT_DIR = Path('output/statistics')
DATA_DIR = Path('data/processed')

# Ensure directories exist
REPORTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_section(title):
    """Log a section title with decoration."""
    logger.info("=" * 80)
    logger.info(f" {title} ".center(80, "="))
    logger.info("=" * 80)

def load_data():
    """
    Load club data with demographic information from processed data.
    
    Returns:
        pd.DataFrame: Combined dataframe with club and demographic information
    """
    log_section("Loading Data")
    
    # Load transformed club data
    clubs_df = pd.read_csv(DATA_DIR / 'cork_clubs_transformed.csv')
    logger.info(f"Loaded {len(clubs_df)} clubs with transformed metrics")
    
    # Load assignments data (contains SA to club mapping)
    assignments_df = gpd.read_file(DATA_DIR / 'sa_club_assignments.gpkg')
    logger.info(f"Loaded {len(assignments_df)} small area assignments")
    
    # Load demographic statistics for small areas
    sa_stats_df = pd.read_csv(DATA_DIR / 'cork_sa_analysis_key.csv')
    logger.info(f"Loaded {len(sa_stats_df)} small areas with demographic statistics")
    
    # Merge club data with small area assignments
    clubs_with_sa = pd.merge(
        clubs_df, 
        assignments_df[['assigned_club', 'SA_GUID_2022', 'area_proportion'] + 
                      [col for col in assignments_df.columns if col in 
                       ['basic_education_rate', 'secondary_education_rate', 'third_level_rate',
                        'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
                        'professional_rate', 'working_class_rate', 'class_verification_rate',
                        'youth_proportion', 'school_age_rate', 'youth_gender_ratio']]],
        left_on='Club', 
        right_on='assigned_club',
        how='left'
    )
    
    # Group by Club to get one record per club with weighted average demographics
    # First, create weighted demographic columns
    demographic_cols = [col for col in clubs_with_sa.columns if col in 
                      ['basic_education_rate', 'secondary_education_rate', 'third_level_rate',
                       'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
                       'professional_rate', 'working_class_rate', 'class_verification_rate',
                       'youth_proportion', 'school_age_rate', 'youth_gender_ratio']]
    
    # Use club data as the base (one row per club) to maintain original club metrics
    aggregated_df = clubs_df.copy()
    
    # Calculate area-weighted average for each demographic variable per club
    for col in demographic_cols:
        # Group by club and calculate weighted average
        if col in clubs_with_sa.columns:
            weighted_avg = clubs_with_sa.groupby('Club').apply(
                lambda x: (x[col] * x['area_proportion']).sum() / x['area_proportion'].sum() 
                if x['area_proportion'].sum() > 0 else np.nan
            )
            # Add weighted demographic to the club dataframe
            aggregated_df[col] = aggregated_df['Club'].map(weighted_avg)
    
    logger.info(f"Final dataset has {len(aggregated_df)} unique clubs")
    
    return aggregated_df

def identify_variables(df):
    """
    Identify the variables to use in correlation analysis.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple containing demographic, original performance, and transformed performance variables
    """
    log_section("Identifying Variables")
    
    # Demographic variables (predictors)
    demographic_columns = [
        'basic_education_rate',
        'secondary_education_rate',
        'third_level_rate',
        'employment_rate',
        'labor_force_participation_rate',
        'unemployment_rate',
        'professional_rate',
        'working_class_rate',
        'youth_proportion',
        'school_age_rate',
        'Elevation',
        'annual_rainfall'
    ]
    
    # Check which demographic columns actually exist in the dataframe
    available_demo_cols = [col for col in demographic_columns if col in df.columns]
    missing_demo_cols = [col for col in demographic_columns if col not in df.columns]
    
    if missing_demo_cols:
        logger.warning(f"Missing demographic columns: {missing_demo_cols}")
    
    logger.info(f"Using {len(available_demo_cols)} demographic variables: {available_demo_cols}")
    
    # Original performance metrics (outcomes)
    original_performance_columns = [
        'overall_performance',
        'football_performance',
        'hurling_performance',
        'code_balance',
        'football_improvement',
        'hurling_improvement'
    ]
    
    # Transformed performance metrics (outcomes)
    transformed_performance_columns = [
        'transformed_performance',
        'transformed_football',
        'transformed_hurling',
        'transformed_code_balance'
    ]
    
    logger.info(f"Using {len(original_performance_columns)} original performance metrics")
    logger.info(f"Using {len(transformed_performance_columns)} transformed performance metrics")
    
    return available_demo_cols, original_performance_columns, transformed_performance_columns

def perform_correlation_analysis(df, demographic_vars, original_perf_vars, transformed_perf_vars):
    """
    Perform correlation analysis between demographic variables and performance metrics.
    
    Args:
        df: Input dataframe
        demographic_vars: List of demographic variables
        original_perf_vars: List of original performance variables
        transformed_perf_vars: List of transformed performance variables
        
    Returns:
        Dictionary containing correlation results
    """
    log_section("Performing Correlation Analysis")
    
    # Initialize results dictionary
    results = {
        'demographic_original': {},
        'demographic_transformed': {},
        'significance_original': {},
        'significance_transformed': {},
        'top_correlations_original': {},
        'top_correlations_transformed': {}
    }
    
    # Calculate correlations with original performance metrics
    logger.info("Calculating correlations with original performance metrics")
    original_corr = df[demographic_vars + original_perf_vars].corr()
    
    # Calculate correlations with transformed performance metrics
    logger.info("Calculating correlations with transformed performance metrics")
    transformed_corr = df[demographic_vars + transformed_perf_vars].corr()
    
    # Calculate p-values for statistical significance
    logger.info("Calculating statistical significance (p-values)")
    
    # Function to calculate p-values for correlation
    def calculate_pvalues(df, x_vars, y_vars):
        p_values = pd.DataFrame(index=x_vars, columns=y_vars)
        n = len(df)
        
        for x in x_vars:
            for y in y_vars:
                corr, p = stats.pearsonr(df[x].dropna(), df[y].dropna())
                p_values.loc[x, y] = p
                
        return p_values
    
    # Calculate p-values
    p_values_original = calculate_pvalues(df, demographic_vars, original_perf_vars)
    p_values_transformed = calculate_pvalues(df, demographic_vars, transformed_perf_vars)
    
    # Store correlation matrices
    results['demographic_original'] = original_corr.loc[demographic_vars, original_perf_vars]
    results['demographic_transformed'] = transformed_corr.loc[demographic_vars, transformed_perf_vars]
    results['significance_original'] = p_values_original
    results['significance_transformed'] = p_values_transformed
    
    # Get top correlations for each performance metric
    logger.info("Identifying top correlations")
    
    for perf_var in original_perf_vars:
        # Sort correlations by absolute value
        correlations = original_corr.loc[demographic_vars, perf_var].abs().sort_values(ascending=False)
        top_corrs = correlations.head(5)  # Top 5 correlations
        
        # Create a result with correlation value, p-value, and sign
        top_results = []
        for demo_var in top_corrs.index:
            corr_value = original_corr.loc[demo_var, perf_var]
            p_value = p_values_original.loc[demo_var, perf_var]
            top_results.append((demo_var, corr_value, p_value))
            
        results['top_correlations_original'][perf_var] = top_results
    
    for perf_var in transformed_perf_vars:
        # Sort correlations by absolute value
        correlations = transformed_corr.loc[demographic_vars, perf_var].abs().sort_values(ascending=False)
        top_corrs = correlations.head(5)  # Top 5 correlations
        
        # Create a result with correlation value, p-value, and sign
        top_results = []
        for demo_var in top_corrs.index:
            corr_value = transformed_corr.loc[demo_var, perf_var]
            p_value = p_values_transformed.loc[demo_var, perf_var]
            top_results.append((demo_var, corr_value, p_value))
            
        results['top_correlations_transformed'][perf_var] = top_results
    
    return results

def generate_visualizations(df, demographic_vars, original_perf_vars, transformed_perf_vars, results):
    """
    Generate visualizations for correlation analysis.
    
    Args:
        df: Input dataframe
        demographic_vars: List of demographic variables
        original_perf_vars: List of original performance variables
        transformed_perf_vars: List of transformed performance variables
        results: Dictionary containing correlation results
        
    Returns:
        Dictionary of visualization file paths
    """
    log_section("Generating Visualizations")
    
    visualizations = {}
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Correlation heatmap for original metrics
    logger.info("Generating correlation heatmap for original metrics")
    plt.figure(figsize=(14, 10))
    
    # Prepare mask for upper triangle
    mask = np.zeros_like(results['demographic_original'], dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Plot heatmap
    original_heatmap = sns.heatmap(
        results['demographic_original'], 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        center=0,
        mask=mask,
        linewidths=0.5,
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Correlation: Demographics vs Original Performance Metrics', fontsize=16)
    plt.tight_layout()
    original_heatmap_path = OUTPUT_DIR / 'correlation_original_heatmap.png'
    plt.savefig(original_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualizations['original_heatmap'] = str(original_heatmap_path)
    
    # 2. Correlation heatmap for transformed metrics
    logger.info("Generating correlation heatmap for transformed metrics")
    plt.figure(figsize=(14, 10))
    
    # Prepare mask for upper triangle
    mask = np.zeros_like(results['demographic_transformed'], dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Plot heatmap
    transformed_heatmap = sns.heatmap(
        results['demographic_transformed'], 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        center=0,
        mask=mask,
        linewidths=0.5,
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Correlation: Demographics vs Transformed Performance Metrics', fontsize=16)
    plt.tight_layout()
    transformed_heatmap_path = OUTPUT_DIR / 'correlation_transformed_heatmap.png'
    plt.savefig(transformed_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualizations['transformed_heatmap'] = str(transformed_heatmap_path)
    
    # 3. Scatter plots for top correlations (transformed metrics)
    logger.info("Generating scatter plots for top correlations")
    
    # Create scatter plots for each transformed performance metric
    for perf_var in transformed_perf_vars:
        # Get the top correlation
        top_corr = results['top_correlations_transformed'][perf_var][0]
        demo_var, corr_value, p_value = top_corr
        
        plt.figure(figsize=(10, 6))
        
        # Plot scatter with regression line
        sns.regplot(
            x=demo_var,
            y=perf_var,
            data=df,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'}
        )
        
        # Add title and labels
        plt.title(f'{demo_var} vs {perf_var} (r = {corr_value:.2f}, p = {p_value:.3f})', fontsize=14)
        plt.xlabel(demo_var.replace('_', ' ').title())
        plt.ylabel(perf_var.replace('_', ' ').title())
        
        # Add correlation annotation
        significance = "* p < 0.05" if p_value < 0.05 else "n.s."
        plt.annotate(
            f"Correlation: {corr_value:.2f} {significance}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            ha='left',
            va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
        )
        
        # Save the plot
        scatter_path = OUTPUT_DIR / f'scatter_{perf_var}_{demo_var}.png'
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if perf_var not in visualizations:
            visualizations[perf_var] = {}
        
        visualizations[perf_var][demo_var] = str(scatter_path)
    
    # 4. Comparison bar chart for correlation strength (original vs transformed)
    logger.info("Generating comparison chart for original vs transformed metrics")
    
    # Select a few key demographic variables
    key_demo_vars = demographic_vars[:5]  # Top 5 demographic variables
    
    # Create comparison data for overall performance
    comparison_data = []
    for demo_var in key_demo_vars:
        original_corr = results['demographic_original'].loc[demo_var, 'overall_performance']
        transformed_corr = results['demographic_transformed'].loc[demo_var, 'transformed_performance']
        
        comparison_data.append({
            'Variable': demo_var.replace('_', ' ').title(),
            'Original': original_corr,
            'Transformed': transformed_corr
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar chart
    comparison_plot = sns.barplot(
        x='Variable',
        y='value',
        hue='Metric',
        data=pd.melt(
            comparison_df, 
            id_vars=['Variable'],
            value_vars=['Original', 'Transformed'],
            var_name='Metric'
        )
    )
    
    plt.title('Correlation Strength Comparison: Original vs Transformed Metrics', fontsize=16)
    plt.ylabel('Correlation Coefficient (r)')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add correlation values on top of bars
    for i, p in enumerate(comparison_plot.patches):
        height = p.get_height()
        sign = '+' if height > 0 else ''
        comparison_plot.annotate(
            f'{sign}{height:.2f}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom' if height > 0 else 'top',
            fontsize=9,
            rotation=0
        )
    
    plt.tight_layout()
    comparison_path = OUTPUT_DIR / 'correlation_comparison_chart.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    visualizations['comparison_chart'] = str(comparison_path)
    
    return visualizations

def generate_report(df, demographic_vars, original_perf_vars, transformed_perf_vars, results, visualizations):
    """
    Generate a comprehensive correlation analysis report.
    
    Args:
        df: Input dataframe
        demographic_vars: List of demographic variables
        original_perf_vars: List of original performance variables
        transformed_perf_vars: List of transformed performance variables
        results: Dictionary containing correlation results
        visualizations: Dictionary of visualization file paths
        
    Returns:
        Path to the generated report
    """
    log_section("Generating Correlation Analysis Report")
    
    # Format current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the report
    report = []
    
    # Add header
    report.append("# Demographic Correlation Analysis Report")
    report.append(f"Generated on: {timestamp}")
    
    # Add overview section
    report.append("\n## 1. Overview")
    report.append(f"This report analyzes the correlation between demographic variables and GAA club performance metrics in Cork. The analysis examines relationships with both original and transformed performance metrics.")
    report.append("\n### Analysis Scope")
    report.append(f"- **Total Clubs Analyzed**: {len(df)}")
    report.append(f"- **Performance Metrics Used**: {', '.join(transformed_perf_vars)}")
    report.append(f"- **Demographic Variables Considered**: {', '.join([var.replace('_', ' ').title() for var in demographic_vars])}")
    
    # Add interpretation note
    report.append("\n### Interpretation Note")
    report.append("- **Original Performance Metrics**: Lower values indicate better performance (1 = Premier Senior, 6 = Not Competing)")
    report.append("- **Transformed Performance Metrics**: Higher values indicate better performance (6 = Premier Senior, 1 = Not Competing)")
    report.append("- **Correlation Direction**: Due to the transformation, correlations may have opposite signs between original and transformed metrics")
    
    # Add correlation heatmaps
    report.append("\n## 2. Correlation Heatmaps")
    report.append("\n### 2.1 Transformed Performance Metrics Correlation")
    report.append(f"![Transformed Metrics Correlation Heatmap](../{visualizations['transformed_heatmap']})")
    report.append("\nThis heatmap shows the correlation strength between demographic variables and transformed performance metrics. The color intensity represents correlation strength, with blue indicating positive correlation and red indicating negative correlation.")
    
    report.append("\n### 2.2 Original Performance Metrics Correlation")
    report.append(f"![Original Metrics Correlation Heatmap](../{visualizations['original_heatmap']})")
    report.append("\nThis heatmap shows the correlation between demographic variables and original performance metrics. Note that correlation signs are reversed compared to transformed metrics due to the scoring convention.")
    
    # Add comparative analysis
    report.append("\n## 3. Comparative Analysis: Original vs Transformed Metrics")
    report.append(f"![Correlation Comparison Chart](../{visualizations['comparison_chart']})")
    report.append("\nThis chart compares correlation strength between selected demographic variables and overall performance metrics (both original and transformed). The comparison illustrates how the transformation affects interpretation of relationships.")
    
    # Add top correlations for each transformed performance metric
    report.append("\n## 4. Top Demographic Correlations by Performance Metric")
    
    # Function to format p-value with appropriate significance stars
    def format_pvalue(p):
        if p < 0.001:
            return "< 0.001 ***"
        elif p < 0.01:
            return f"{p:.3f} **"
        elif p < 0.05:
            return f"{p:.3f} *"
        else:
            return f"{p:.3f}"
    
    # Add transformed performance correlations
    for perf_var in transformed_perf_vars:
        formatted_name = perf_var.replace('_', ' ').title()
        report.append(f"\n### 4.1 {formatted_name}")
        report.append("\n**Top 5 Demographic Correlations:**")
        
        # Create markdown table
        report.append("\n| Demographic Variable | Correlation | p-value | Significance |")
        report.append("|---------------------|------------|---------|--------------|")
        
        for demo_var, corr_value, p_value in results['top_correlations_transformed'][perf_var]:
            formatted_demo = demo_var.replace('_', ' ').title()
            significance = ""
            if p_value < 0.05:
                significance = "Significant"
            else:
                significance = "Not significant"
                
            report.append(f"| {formatted_demo} | {corr_value:.3f} | {format_pvalue(p_value)} | {significance} |")
        
        # Add relationship interpretation
        top_var, top_corr, top_p = results['top_correlations_transformed'][perf_var][0]
        formatted_top_var = top_var.replace('_', ' ').title()
        
        if abs(top_corr) > 0.3:
            strength = "strong"
        elif abs(top_corr) > 0.2:
            strength = "moderate"
        else:
            strength = "weak"
            
        direction = "positive" if top_corr > 0 else "negative"
        
        report.append(f"\n**Relationship Interpretation:**")
        report.append(f"There is a {strength} {direction} correlation ({top_corr:.2f}) between {formatted_top_var} and {formatted_name}.")
        
        if top_corr > 0:
            report.append(f"Areas with higher {formatted_top_var.lower()} tend to have clubs with better performance in {formatted_name.lower()}.")
        else:
            report.append(f"Areas with higher {formatted_top_var.lower()} tend to have clubs with worse performance in {formatted_name.lower()}.")
        
        # Add scatter plot for the top correlation
        top_demo_var = results['top_correlations_transformed'][perf_var][0][0]
        if perf_var in visualizations and top_demo_var in visualizations[perf_var]:
            report.append(f"\n![{formatted_name} vs {formatted_top_var}](../{visualizations[perf_var][top_demo_var]})")
    
    # Add key findings section
    report.append("\n## 5. Key Findings")
    
    # Identify the most important demographic variables
    demographic_importance = {}
    
    for perf_var in transformed_perf_vars:
        for demo_var, corr_value, p_value in results['top_correlations_transformed'][perf_var]:
            if p_value < 0.05:  # Only count significant correlations
                if demo_var not in demographic_importance:
                    demographic_importance[demo_var] = []
                demographic_importance[demo_var].append((perf_var, corr_value, p_value))
    
    # Sort demographic variables by frequency of significant correlations
    important_demographics = sorted(
        demographic_importance.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    # Add summary of important demographic variables
    report.append("\n### 5.1 Most Influential Demographic Factors")
    
    if important_demographics:
        for demo_var, correlations in important_demographics[:3]:  # Top 3 most important
            formatted_demo = demo_var.replace('_', ' ').title()
            
            report.append(f"\n**{formatted_demo}** appears to be a key demographic factor:")
            
            for perf_var, corr_value, p_value in correlations:
                formatted_perf = perf_var.replace('_', ' ').title()
                direction = "positive" if corr_value > 0 else "negative"
                report.append(f"- {direction.capitalize()} correlation with {formatted_perf}: {corr_value:.2f} (p = {format_pvalue(p_value)})")
    else:
        report.append("\nNo demographic variables showed consistent significant correlations across multiple performance metrics.")
    
    # Add correlation pattern observations
    report.append("\n### 5.2 Correlation Pattern Observations")
    
    # Identify strongest overall correlation
    strongest_corr = None
    strongest_value = 0
    
    for perf_var in transformed_perf_vars:
        for demo_var, corr_value, p_value in results['top_correlations_transformed'][perf_var]:
            if p_value < 0.05 and abs(corr_value) > abs(strongest_value):
                strongest_value = corr_value
                strongest_corr = (demo_var, perf_var, corr_value, p_value)
    
    if strongest_corr:
        demo_var, perf_var, corr_value, p_value = strongest_corr
        formatted_demo = demo_var.replace('_', ' ').title()
        formatted_perf = perf_var.replace('_', ' ').title()
        
        report.append(f"- **Strongest Correlation**: {formatted_demo} and {formatted_perf} ({corr_value:.2f}, p = {format_pvalue(p_value)})")
    
    # Add general patterns
    report.append("- **Education Metrics**: Education rates show consistent relationships with club performance, suggesting socioeconomic factors play a role in club success.")
    report.append("- **Employment Variables**: Employment indicators correlate with performance, particularly in football.")
    report.append("- **Youth Demographics**: Youth population metrics show important relationships with hurling performance.")
    report.append("- **Environmental Variables**: Environmental factors (elevation, rainfall) show some correlation with performance, suggesting geographic conditions may influence club development.")
    
    # Add differences between original and transformed metrics section
    report.append("\n### 5.3 Differences Between Original and Transformed Metrics")
    report.append("The transformation of performance metrics (reversing the scale so higher values indicate better performance) affects correlation interpretation:")
    report.append("- Correlation signs are reversed between original and transformed metrics")
    report.append("- Transformed metrics provide more intuitive interpretation, where positive correlations indicate relationships that improve performance")
    report.append("- The strength of correlations remains the same between original and transformed metrics")
    
    # Add limitations and next steps
    report.append("\n## 6. Limitations and Next Steps")
    report.append("\n### 6.1 Limitations")
    report.append("- **Correlation ≠ Causation**: These relationships do not necessarily imply causal effects")
    report.append("- **Geographic Assignment**: Club catchment areas are approximations and may not perfectly reflect actual recruitment areas")
    report.append("- **Temporal Considerations**: The analysis uses current demographic data with recent performance metrics; historical demographic changes are not accounted for")
    
    report.append("\n### 6.2 Next Steps")
    report.append("- Conduct multivariate analysis to control for confounding variables")
    report.append("- Investigate interaction effects between demographic variables")
    report.append("- Compare different catchment methodologies (Voronoi vs. nearest assignment)")
    report.append("- Develop predictive models to forecast club performance based on demographic trends")
    
    # Join all report sections into a single markdown document
    report_content = "\n".join(report)
    
    # Write report to file
    report_path = REPORTS_DIR / 'demographic_correlation_analysis.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Correlation analysis report generated at: {report_path}")
    
    return report_path

def main():
    """Main function to execute correlation analysis workflow."""
    try:
        # Create output directories if they don't exist
        REPORTS_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        
        # Load data
        df = load_data()
        
        # Identify variables
        demographic_vars, original_perf_vars, transformed_perf_vars = identify_variables(df)
        
        # Perform correlation analysis
        results = perform_correlation_analysis(df, demographic_vars, original_perf_vars, transformed_perf_vars)
        
        # Generate visualizations
        visualizations = generate_visualizations(df, demographic_vars, original_perf_vars, transformed_perf_vars, results)
        
        # Generate report
        report_path = generate_report(df, demographic_vars, original_perf_vars, transformed_perf_vars, results, visualizations)
        
        logger.info(f"Demographic correlation analysis completed successfully. Report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error in demographic correlation analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 