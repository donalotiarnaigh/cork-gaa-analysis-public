import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import logging
from scipy import stats

# Set up constants
OUTPUT_DIR = Path('report_visualizations/correlation')
DATA_DIR = Path('data/processed')

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Load club data with demographic information from processed data.
    
    Returns:
        pd.DataFrame: Combined dataframe with club and demographic information
    """
    logger.info("Loading data for correlation visualizations...")
    
    # Load transformed club data
    clubs_df = pd.read_csv(DATA_DIR / 'cork_clubs_transformed.csv')
    logger.info(f"Loaded {len(clubs_df)} clubs with transformed metrics")
    
    # Load demographic data (assuming it has already been processed and merged with clubs)
    try:
        demographic_df = pd.read_csv(DATA_DIR / 'club_demographics_processed.csv')
        logger.info(f"Loaded demographic data for {len(demographic_df)} clubs")
        
        # Merge club data with demographic data
        merged_df = pd.merge(
            clubs_df,
            demographic_df,
            on='Club',
            how='inner'
        )
        logger.info(f"Merged data contains {len(merged_df)} clubs")
        return merged_df
    except FileNotFoundError:
        # If the demographic file doesn't exist, fallback to using the club data directly
        # (assuming demographic data is already merged into the clubs data)
        logger.warning("Dedicated demographic file not found, using club data directly")
        return clubs_df

def create_education_correlation_scatterplots(df):
    """
    Create scatterplots showing the correlation between education metrics and performance metrics.
    
    Args:
        df (pd.DataFrame): Dataframe containing club data with education and performance metrics
        
    Returns:
        int: Number of plots created
    """
    logging.info("Creating education correlation scatterplots...")
    logging.info(f"All columns in dataframe: {df.columns.tolist()}")
    
    # Find education columns
    education_cols = [col for col in df.columns if 'education' in col.lower() or 'third_level' in col.lower()]
    if not education_cols:
        logging.warning("No education columns found in data. Education correlation scatterplots will not be created.")
        return 0
    
    logging.info(f"Found education columns: {education_cols}")
    
    # Clean up duplicate columns by selecting the ones without suffix if available
    performance_metrics = []
    for metric in ['transformed_performance', 'transformed_football', 'transformed_hurling', 'transformed_code_balance']:
        if metric in df.columns:
            performance_metrics.append(metric)
        elif f"{metric}_x" in df.columns:
            # Rename the column to remove the suffix
            df[metric] = df[f"{metric}_x"]
            performance_metrics.append(metric)
    
    logging.info(f"Using performance metrics: {performance_metrics}")
    
    if not performance_metrics:
        logging.warning("No performance metrics found in data. Education correlation scatterplots will not be created.")
        return 0
    
    plots_created = 0
    
    for education_col in education_cols:
        for metric in performance_metrics:
            # Check if we have enough valid data points
            valid_data = df[[education_col, metric]].dropna()
            if len(valid_data) < 5:
                logging.warning(f"Not enough valid data points for {education_col} vs {metric}. Skipping plot.")
                continue
                
            logging.info(f"Creating scatterplot for {education_col} vs {metric} with {len(valid_data)} valid data points")
            
            plt.figure(figsize=(10, 8))
            sns.set_style("whitegrid")
            
            # Calculate correlation
            correlation = valid_data[education_col].corr(valid_data[metric])
            p_value = stats.pearsonr(valid_data[education_col], valid_data[metric])[1]
            
            logging.info(f"Correlation between {education_col} and {metric}: {correlation:.3f} (p-value: {p_value:.3f})")
            
            # Create the plot
            ax = sns.regplot(x=education_col, y=metric, data=valid_data, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
            
            # Add correlation info to plot
            plt.title(f"Correlation between {education_col.replace('_', ' ').title()} and {metric.replace('_', ' ').title()}")
            plt.figtext(0.5, 0.01, f"Correlation: {correlation:.3f} (p-value: {p_value:.3f})", 
                       ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            
            plt.xlabel(education_col.replace('_', ' ').title())
            plt.ylabel(metric.replace('_', ' ').title())
            
            # Save the figure
            output_dir = 'report_visualizations/correlation'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"education_{education_col}_vs_{metric}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logging.info(f"Saved scatterplot to {output_path}")
            plots_created += 1
    
    return plots_created

def create_youth_proportion_boxplots(df):
    """
    Create boxplots showing the relationship between youth proportion and performance metrics.
    
    Args:
        df (pd.DataFrame): Dataframe containing club data with youth proportion and performance metrics
        
    Returns:
        int: Number of plots created
    """
    logging.info("Creating youth proportion boxplots...")
    
    # Find youth columns
    youth_cols = [col for col in df.columns if 'youth' in col.lower()]
    if not youth_cols:
        logging.warning("No youth proportion columns found in data. Youth proportion boxplots will not be created.")
        return 0
        
    logging.info(f"Found youth columns: {youth_cols}")
    
    # Use the first youth column found
    youth_col = youth_cols[0]
    sample_values = df[youth_col].head()
    logging.info(f"Using youth column: {youth_col}, sample values: {sample_values}")
    
    # Clean up duplicate columns by selecting the ones without suffix if available
    performance_metrics = []
    for metric in ['transformed_performance', 'transformed_football', 'transformed_hurling']:
        if metric in df.columns:
            performance_metrics.append(metric)
        elif f"{metric}_x" in df.columns:
            # Rename the column to remove the suffix
            df[metric] = df[f"{metric}_x"]
            performance_metrics.append(metric)
    
    logging.info(f"Using performance metrics: {performance_metrics}")
    
    if not performance_metrics:
        logging.warning("No performance metrics found in data. Youth proportion boxplots will not be created.")
        return 0
    
    # Create youth proportion categories
    df['youth_category'] = pd.qcut(df[youth_col], 3, labels=['Low', 'Medium', 'High'])
    category_counts = df['youth_category'].value_counts()
    logging.info(f"Created youth categories with counts: {category_counts}")
    
    plots_created = 0
    
    for metric in performance_metrics:
        try:
            plt.figure(figsize=(12, 8))
            sns.set_style("whitegrid")
            
            # Check for valid data
            valid_data = df.dropna(subset=[youth_col, metric, 'youth_category'])
            if len(valid_data) < 10:
                logging.warning(f"Not enough valid data points for {youth_col} vs {metric}. Skipping plot.")
                continue
                
            # Check if we have enough data in each group for ANOVA
            group_sizes = valid_data.groupby('youth_category').size()
            logging.info(f"Group sizes for ANOVA: {group_sizes}")
            
            if (group_sizes < 5).any():
                logging.warning(f"Some groups have fewer than 5 samples for {youth_col} vs {metric}. ANOVA results may be unreliable.")
            
            # Perform ANOVA
            try:
                groups = [group[metric].values for name, group in valid_data.groupby('youth_category')]
                f_val, p_val = stats.f_oneway(*groups)
                anova_result = f"ANOVA: F={f_val:.3f}, p={p_val:.3f}"
                logging.info(f"ANOVA result for {youth_col} categories vs {metric}: {anova_result}")
            except Exception as e:
                logging.error(f"Error performing ANOVA: {str(e)}")
                anova_result = "ANOVA: Error"
            
            # Create boxplot
            ax = sns.boxplot(x='youth_category', y=metric, data=valid_data, palette='viridis',
                            order=['Low', 'Medium', 'High'])
            
            # Add stripplot for individual data points
            sns.stripplot(x='youth_category', y=metric, data=valid_data, size=4, color=".3",
                        alpha=0.5, order=['Low', 'Medium', 'High'])
            
            plt.title(f"Relationship between Youth Proportion and {metric.replace('_', ' ').title()}")
            plt.figtext(0.5, 0.01, anova_result, ha="center", fontsize=12, 
                      bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            
            plt.xlabel("Youth Proportion Category")
            plt.ylabel(metric.replace('_', ' ').title())
            
            # Save figure
            output_dir = 'report_visualizations/correlation'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"youth_proportion_vs_{metric}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logging.info(f"Saved boxplot to {output_path}")
            plots_created += 1
            
        except Exception as e:
            logging.error(f"Error creating boxplot for {youth_col} vs {metric}: {str(e)}")
    
    return plots_created

def create_demographic_correlation_scatterplots(df):
    """
    Create a single figure with correlation scatterplots for all demographic variables against performance.
    
    Args:
        df (pd.DataFrame): Dataframe containing club data with demographic and performance variables
        
    Returns:
        int: Number of plots created (0 or 1)
    """
    logging.info("Creating demographic correlation scatterplots...")
    
    # Clean up duplicate columns by selecting the ones without suffix if available
    performance_col = None
    for metric in ['transformed_performance']:
        if metric in df.columns:
            performance_col = metric
            break
        elif f"{metric}_x" in df.columns:
            # Rename the column to remove the suffix
            df[metric] = df[f"{metric}_x"]
            performance_col = metric
            break
    
    if not performance_col:
        logging.warning("Performance column not found in data. Demographic correlation scatterplots will not be created.")
        return 0
    
    # Select demographic columns
    demographic_cols = []
    for col in ['employment_rate', 'professional_rate', 'education_rate', 'youth_proportion', 'population_density']:
        if col in df.columns:
            demographic_cols.append(col)
    
    if not demographic_cols:
        logging.warning("No demographic columns found in data. Demographic correlation scatterplots will not be created.")
        return 0
    
    # Create a figure with subplots
    n_cols = min(2, len(demographic_cols))
    n_rows = (len(demographic_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    plt.suptitle(f"Demographic Correlations with Overall Performance", fontsize=16, y=1.02)
    
    for i, demo_col in enumerate(demographic_cols):
        ax = axes[i]
        
        # Filter valid data
        valid_data = df[[demo_col, performance_col]].dropna()
        
        # Calculate correlation
        correlation = valid_data[demo_col].corr(valid_data[performance_col])
        p_value = stats.pearsonr(valid_data[demo_col], valid_data[performance_col])[1]
        
        # Create the plot
        sns.regplot(x=demo_col, y=performance_col, data=valid_data, 
                   scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax)
        
        # Format the plot
        ax.set_title(f"{demo_col.replace('_', ' ').title()}")
        ax.set_xlabel(demo_col.replace('_', ' ').title())
        ax.set_ylabel(performance_col.replace('_', ' ').title())
        
        # Add correlation info
        ax.text(0.05, 0.95, f"r = {correlation:.3f}\np = {p_value:.3f}", 
               transform=ax.transAxes, verticalalignment='top',
               bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    
    # Save the figure
    output_dir = 'report_visualizations/correlation'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"demographic_correlations_{performance_col}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Created demographic correlations plot: {output_path}")
    
    return 1

def main():
    """Main function to run all correlation visualizations."""
    try:
        # Load the data
        df = load_data()
        
        # Create all visualizations
        education_plots = create_education_correlation_scatterplots(df)
        youth_plots = create_youth_proportion_boxplots(df)
        demographic_plots = create_demographic_correlation_scatterplots(df)
        
        # Combine all output files
        all_plots = education_plots + youth_plots + demographic_plots
        
        logger.info(f"Generated {len(all_plots)} correlation visualizations")
        logger.info(f"Visualizations saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Error in correlation visualizations: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 