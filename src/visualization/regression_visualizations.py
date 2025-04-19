import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
OUTPUT_DIR = Path('report_visualizations/regression')
MODEL_DATA_DIR = Path('output/models')
MULTICOLLINEARITY_DIR = Path('output/analysis/multicollinearity')

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def create_hurling_coefficient_plot():
    """
    Create a coefficient plot for hurling performance showing the strength 
    and significance of predictors.
    """
    logger.info("Creating coefficient plot for hurling performance")
    
    # Load regression model results
    try:
        with open(MODEL_DATA_DIR / 'regression_model_results_20250411.json', 'r') as f:
            model_data = json.load(f)
            
        # Extract hurling performance coefficients
        if 'coefficients' in model_data and 'hurling_performance' in model_data['coefficients']:
            coefficients = model_data['coefficients']['hurling_performance']
            
            # Convert to DataFrame for plotting
            coef_data = []
            for var, stats in coefficients.items():
                if var == 'const':
                    continue  # Skip intercept
                
                # Determine significance level
                p_value = stats['p_value']
                if p_value < 0.001:
                    significance = 'p < 0.001'
                elif p_value < 0.01:
                    significance = 'p < 0.01'
                elif p_value < 0.05:
                    significance = 'p < 0.05'
                elif p_value < 0.1:
                    significance = 'p < 0.1'
                else:
                    significance = 'Not significant'
                
                coef_data.append({
                    'Variable': var,
                    'Coefficient': stats['value'],
                    'Lower CI': stats['conf_int_lower'],
                    'Upper CI': stats['conf_int_upper'],
                    'P-value': p_value,
                    'Significance': significance
                })
            
            coef_df = pd.DataFrame(coef_data)
            
            # Sort by coefficient value
            coef_df = coef_df.sort_values('Coefficient')
            
            # Define colors by significance level
            colors = {
                'Not significant': 'lightgray',
                'p < 0.1': 'lightblue',
                'p < 0.05': 'blue',
                'p < 0.01': 'darkblue',
                'p < 0.001': 'purple'
            }
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            
            # Plot the coefficients and their confidence intervals
            for i, (_, row) in enumerate(coef_df.iterrows()):
                plt.plot(
                    [row['Lower CI'], row['Upper CI']], 
                    [i, i], 
                    'o-', 
                    color=colors[row['Significance']], 
                    linewidth=2, 
                    markersize=8
                )
                plt.plot(
                    [row['Coefficient']], 
                    [i], 
                    'o', 
                    color='red', 
                    markersize=8
                )
            
            # Add zero reference line
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Customize the plot
            plt.yticks(range(len(coef_df)), coef_df['Variable'])
            plt.xlabel('Coefficient Value (with 95% Confidence Interval)')
            plt.title('Regression Coefficients for Hurling Performance\n'
                     'Note: Negative coefficients indicate better performance (lower grade)')
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=sig)
                for sig, color in colors.items()
            ]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Coefficient'))
            
            plt.legend(handles=legend_elements, loc='best')
            plt.grid(alpha=0.3)
            
            # Save the figure
            output_path = OUTPUT_DIR / "model_coefficient_plot_hurling_performance.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved coefficient plot to {output_path}")
            return output_path
            
        else:
            logger.error("Hurling performance data not found in model results")
            return None
            
    except Exception as e:
        logger.error(f"Error creating hurling coefficient plot: {str(e)}")
        return None

def create_pca_biplot():
    """
    Create a PCA biplot showing variable loadings on principal components.
    """
    logger.info("Creating PCA biplot")
    
    try:
        # Load PCA data
        loadings_path = MULTICOLLINEARITY_DIR / 'pca_loadings.csv'
        variance_path = MULTICOLLINEARITY_DIR / 'pca_explained_variance.csv'
        
        if not loadings_path.exists() or not variance_path.exists():
            logger.error("PCA data files not found")
            return None
        
        # Load PCA loadings and explained variance
        loadings = pd.read_csv(loadings_path, index_col=0)
        explained_variance = pd.read_csv(variance_path)
        
        # Extract first two components
        pc1 = 'PC1'
        pc2 = 'PC2'
        
        # Get explained variance percentages
        pc1_var = explained_variance.loc[0, 'Explained_Variance']
        pc2_var = explained_variance.loc[1, 'Explained_Variance']
        
        # Create the biplot
        plt.figure(figsize=(12, 10))
        
        # Plot the loadings
        for var in loadings.index:
            plt.arrow(0, 0, loadings.loc[var, pc1] * 5, loadings.loc[var, pc2] * 5, 
                     head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
            plt.text(loadings.loc[var, pc1] * 5.2, loadings.loc[var, pc2] * 5.2, 
                    var, fontsize=12, ha='center', va='center')
        
        # Add a circle to represent the unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        plt.gca().add_patch(circle)
        
        # Add axis lines
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Set axis labels and title
        plt.xlabel(f'PC1 ({pc1_var:.1%} explained variance)', fontsize=14)
        plt.ylabel(f'PC2 ({pc2_var:.1%} explained variance)', fontsize=14)
        plt.title('PCA Biplot: Variable Loadings on Principal Components', fontsize=16)
        
        # Set equal aspect ratio to avoid distortion
        plt.axis('equal')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        output_path = OUTPUT_DIR / "multicollinearity_pca_biplot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved PCA biplot to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating PCA biplot: {str(e)}")
        return None

def create_r2_comparison_plot():
    """
    Create a visualization comparing R² values across different performance metrics.
    """
    logger.info("Creating R² comparison visualization")
    
    try:
        # Load model statistics
        with open(MODEL_DATA_DIR / 'regression_model_results_20250411.json', 'r') as f:
            model_data = json.load(f)
        
        if 'model_stats' not in model_data:
            logger.error("Model statistics not found in model results")
            return None
        
        # Extract R² values
        r2_data = {
            metric: {
                'R²': stats['r_squared'],
                'Adjusted R²': stats['adj_r_squared']
            }
            for metric, stats in model_data['model_stats'].items()
        }
        
        # Convert to DataFrame
        r2_df = pd.DataFrame(r2_data).T
        r2_df.index = [idx.replace('_', ' ').title() for idx in r2_df.index]
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Plot R² and Adjusted R²
        bar_width = 0.35
        x = np.arange(len(r2_df.index))
        
        plt.bar(x - bar_width/2, r2_df['R²'], bar_width, label='R²', color='steelblue')
        plt.bar(x + bar_width/2, r2_df['Adjusted R²'], bar_width, label='Adjusted R²', color='lightsteelblue')
        
        # Add value labels on top of bars
        for i, v in enumerate(r2_df['R²']):
            plt.text(i - bar_width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
            
        for i, v in enumerate(r2_df['Adjusted R²']):
            plt.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Customize the plot
        plt.xlabel('Performance Metric', fontsize=14)
        plt.ylabel('R² Value', fontsize=14)
        plt.title('Model Explanatory Power Across Performance Metrics', fontsize=16)
        plt.xticks(x, r2_df.index, rotation=45, ha='right')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Add interpretive note
        note = "Note: Higher values indicate greater explanatory power of demographic variables."
        plt.figtext(0.5, 0.01, note, ha='center', fontsize=12)
        
        # Save the figure
        output_path = OUTPUT_DIR / "model_r2_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved R² comparison plot to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating R² comparison visualization: {str(e)}")
        return None

def main():
    """Main function to generate all regression visualizations."""
    try:
        logger.info("Starting regression visualizations generation")
        
        # Create coefficient plot for hurling performance
        hurling_coef_path = create_hurling_coefficient_plot()
        
        # Create PCA biplot
        pca_biplot_path = create_pca_biplot()
        
        # Create R² comparison visualization
        r2_plot_path = create_r2_comparison_plot()
        
        # Report results
        logger.info("Regression visualizations generated successfully:")
        if hurling_coef_path:
            logger.info(f"- Hurling coefficient plot: {hurling_coef_path}")
        if pca_biplot_path:
            logger.info(f"- PCA biplot: {pca_biplot_path}")
        if r2_plot_path:
            logger.info(f"- R² comparison plot: {r2_plot_path}")
            
    except Exception as e:
        logger.error(f"Error in regression visualizations generation: {str(e)}")

if __name__ == "__main__":
    main() 