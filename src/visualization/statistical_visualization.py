import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalVisualization:
    """Class for generating statistical visualizations of analysis results."""
    
    def __init__(self, output_dir: str = "data/analysis/visualizations"):
        """
        Initialize the StatisticalVisualization class.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for visualizations
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid")
        
    def create_method_comparison_charts(self, 
                                      buffer_data: pd.DataFrame,
                                      voronoi_data: pd.DataFrame,
                                      nearest_data: pd.DataFrame) -> Dict[str, str]:
        """
        Create comparison charts for different analysis methods.
        
        Args:
            buffer_data: Buffer analysis results
            voronoi_data: Voronoi analysis results
            nearest_data: Nearest analysis results
            
        Returns:
            Dictionary of saved visualization paths
        """
        saved_paths = {}
        
        # Competition score comparison
        plt.figure(figsize=(12, 6))
        methods = ['Buffer', 'Voronoi', 'Nearest']
        scores = [
            buffer_data['competition_score'].mean(),
            voronoi_data['competition_score'].mean() if 'competition_score' in voronoi_data.columns else 0,
            nearest_data['competition_score'].mean() if 'competition_score' in nearest_data.columns else 0
        ]
        
        plt.bar(methods, scores)
        plt.title('Average Competition Score by Method')
        plt.ylabel('Competition Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / 'competition_score_comparison.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths['competition_score'] = str(path)
        
        # Youth population comparison
        plt.figure(figsize=(12, 6))
        youth = [
            buffer_data['youth_population'].sum(),
            voronoi_data['youth_population'].sum() if 'youth_population' in voronoi_data.columns else 0,
            nearest_data['youth_population'].sum() if 'youth_population' in nearest_data.columns else 0
        ]
        
        plt.bar(methods, youth)
        plt.title('Total Youth Population by Method')
        plt.ylabel('Youth Population')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / 'youth_population_comparison.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths['youth_population'] = str(path)
        
        # Population density comparison
        plt.figure(figsize=(12, 6))
        density = [
            buffer_data['population_density'].mean(),
            voronoi_data['population_density'].mean() if 'population_density' in voronoi_data.columns else 0,
            nearest_data['population_density'].mean() if 'population_density' in nearest_data.columns else 0
        ]
        
        plt.bar(methods, density)
        plt.title('Average Population Density by Method')
        plt.ylabel('Population Density')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / 'population_density_comparison.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths['population_density'] = str(path)
        
        return saved_paths
    
    def create_validation_metric_plots(self, validation_metrics: pd.DataFrame) -> Dict[str, str]:
        """
        Create plots for validation metrics.
        
        Args:
            validation_metrics: DataFrame containing validation metrics
            
        Returns:
            Dictionary of saved visualization paths
        """
        saved_paths = {}
        
        # Create bar plot of metrics
        plt.figure(figsize=(12, 6))
        sns.barplot(data=validation_metrics, x='metric', y='value')
        plt.title('Validation Metrics Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        path = self.output_dir / 'validation_metrics_distribution.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths['distribution'] = str(path)
        
        return saved_paths
    
    def create_quality_check_visualizations(self, quality_metrics: pd.DataFrame) -> Dict[str, str]:
        """
        Create visualizations for quality check metrics.
        
        Args:
            quality_metrics: DataFrame containing quality check metrics
            
        Returns:
            Dictionary of saved visualization paths
        """
        saved_paths = {}
        
        # Get the mean values for each metric
        mean_values = quality_metrics.loc['mean']
        
        # Create bar plot of mean values
        plt.figure(figsize=(15, 6))
        plt.bar(mean_values.index, mean_values.values)
        plt.title('Mean Values of Quality Metrics')
        plt.ylabel('Mean Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / 'quality_metrics_means.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths['means'] = str(path)
        
        # Create error bar plot using mean and std
        plt.figure(figsize=(15, 6))
        mean_values = quality_metrics.loc['mean']
        std_values = quality_metrics.loc['std']
        
        plt.errorbar(range(len(mean_values)), mean_values.values, yerr=std_values.values, fmt='o')
        plt.xticks(range(len(mean_values)), mean_values.index, rotation=45, ha='right')
        plt.title('Quality Metrics with Standard Deviation')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / 'quality_metrics_error_bars.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths['error_bars'] = str(path)
        
        return saved_paths
    
    def create_pattern_identification_charts(self, pattern_data: pd.DataFrame) -> Dict[str, str]:
        """
        Create charts for pattern identification results.
        
        Args:
            pattern_data: DataFrame containing pattern identification results
            
        Returns:
            Dictionary of saved visualization paths
        """
        saved_paths = {}
        
        # Create method comparison bar plot
        plt.figure(figsize=(12, 6))
        method_values = pattern_data[['Nearest', 'Voronoi', 'Buffer']].mean()
        plt.bar(method_values.index, method_values.values)
        plt.title('Average Pattern Values by Method')
        plt.ylabel('Pattern Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / 'pattern_method_comparison.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths['method_comparison'] = str(path)
        
        # Create difference comparison box plot
        plt.figure(figsize=(12, 6))
        diff_data = pattern_data[['Nearest_Voronoi_Diff', 'Nearest_Buffer_Diff', 'Voronoi_Buffer_Diff']]
        diff_data.boxplot()
        plt.title('Pattern Differences Between Methods')
        plt.ylabel('Difference')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / 'pattern_differences.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths['differences'] = str(path)
        
        return saved_paths

def main():
    """Main function to demonstrate usage of the StatisticalVisualization class."""
    try:
        # Initialize the visualization class
        visualizer = StatisticalVisualization()
        
        # Example usage (replace with actual data loading)
        logger.info("Loading analysis data...")
        # buffer_data = pd.read_csv('path_to_buffer_data.csv')
        # voronoi_data = pd.read_csv('path_to_voronoi_data.csv')
        # nearest_data = pd.read_csv('path_to_nearest_data.csv')
        # validation_metrics = pd.read_csv('path_to_validation_metrics.csv')
        # quality_metrics = pd.read_csv('path_to_quality_metrics.csv')
        # pattern_data = pd.read_csv('path_to_pattern_data.csv')
        
        # Generate visualizations
        logger.info("Generating method comparison charts...")
        # method_charts = visualizer.create_method_comparison_charts(
        #     buffer_data, voronoi_data, nearest_data
        # )
        
        logger.info("Generating validation metric plots...")
        # validation_plots = visualizer.create_validation_metric_plots(validation_metrics)
        
        logger.info("Generating quality check visualizations...")
        # quality_plots = visualizer.create_quality_check_visualizations(quality_metrics)
        
        logger.info("Generating pattern identification charts...")
        # pattern_charts = visualizer.create_pattern_identification_charts(pattern_data)
        
        logger.info("Statistical visualizations completed successfully")
        
    except Exception as e:
        logger.error(f"Error in statistical visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 