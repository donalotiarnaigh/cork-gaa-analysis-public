import logging
from validation_metrics import ValidationMetrics
import pandas as pd
import geopandas as gpd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the validation metrics analysis."""
    try:
        # Initialize the validation metrics class
        validator = ValidationMetrics()
        
        # Load the data
        logger.info("Loading analysis data...")
        buffer_data = pd.read_csv('data/processed/buffer_demographics.csv')
        voronoi_data = gpd.read_file('data/processed/voronoi_demographics.gpkg')
        nearest_data = gpd.read_file('data/processed/nearest_demographics.gpkg')
        
        # Calculate validation metrics
        logger.info("Calculating population validation metrics...")
        validator.calculate_population_validation(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating area coverage metrics...")
        validator.calculate_area_coverage(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating method bias metrics...")
        validator.calculate_method_bias(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating quality control metrics...")
        validator.calculate_quality_control(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating urban-rural metrics...")
        validator.calculate_urban_rural_metrics(buffer_data, voronoi_data, nearest_data)
        
        # Generate validation report
        logger.info("Generating validation report...")
        report_path = validator.generate_validation_report()
        logger.info(f"Validation report generated at: {report_path}")
        
    except Exception as e:
        logger.error(f"Error in validation metrics calculation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 