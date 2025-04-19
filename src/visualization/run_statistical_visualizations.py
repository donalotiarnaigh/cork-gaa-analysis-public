import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
from statistical_visualization import StatisticalVisualization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load the necessary data files for visualization."""
    try:
        # Load buffer analysis data
        logger.info("Loading buffer analysis data...")
        buffer_data = pd.read_csv('data/processed/buffer_demographics.csv')
        
        # Load Voronoi analysis data
        logger.info("Loading Voronoi analysis data...")
        voronoi_data = gpd.read_file('data/processed/voronoi_demographics.gpkg')
        
        # Load nearest analysis data
        logger.info("Loading nearest analysis data...")
        nearest_data = gpd.read_file('data/processed/nearest_demographics.gpkg')
        
        # Load validation metrics
        logger.info("Loading validation metrics...")
        validation_metrics = pd.read_csv('data/analysis/overlap_statistics.csv')
        
        # Load quality metrics
        logger.info("Loading quality metrics...")
        quality_metrics = pd.read_csv('data/analysis/transformed_variables_summary.csv', index_col=0)
        
        # Load pattern data
        logger.info("Loading pattern data...")
        pattern_data = pd.read_csv('data/analysis/demographic_pattern_comparison.csv')
        
        return {
            'buffer_data': buffer_data,
            'voronoi_data': voronoi_data,
            'nearest_data': nearest_data,
            'validation_metrics': validation_metrics,
            'quality_metrics': quality_metrics,
            'pattern_data': pattern_data
        }
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def main():
    """Main function to run statistical visualizations."""
    try:
        # Initialize the visualization class
        visualizer = StatisticalVisualization()
        
        # Load the data
        data = load_data()
        
        # Generate method comparison charts
        logger.info("Generating method comparison charts...")
        method_charts = visualizer.create_method_comparison_charts(
            data['buffer_data'],
            data['voronoi_data'],
            data['nearest_data']
        )
        logger.info(f"Method comparison charts saved to: {method_charts}")
        
        # Generate validation metric plots
        logger.info("Generating validation metric plots...")
        validation_plots = visualizer.create_validation_metric_plots(
            data['validation_metrics']
        )
        logger.info(f"Validation metric plots saved to: {validation_plots}")
        
        # Generate quality check visualizations
        logger.info("Generating quality check visualizations...")
        quality_plots = visualizer.create_quality_check_visualizations(
            data['quality_metrics']
        )
        logger.info(f"Quality check visualizations saved to: {quality_plots}")
        
        # Generate pattern identification charts
        logger.info("Generating pattern identification charts...")
        pattern_charts = visualizer.create_pattern_identification_charts(
            data['pattern_data']
        )
        logger.info(f"Pattern identification charts saved to: {pattern_charts}")
        
        logger.info("Statistical visualizations completed successfully")
        
    except Exception as e:
        logger.error(f"Error in statistical visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 