import logging
from pathlib import Path
from generate_report import ReportGenerator

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Set up paths
        analysis_dir = Path("data/analysis")
        output_dir = Path("data/analysis/reports")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report (markdown only for now)
        logger.info("Starting report generation")
        generator = ReportGenerator(analysis_dir, output_dir)
        generator.generate_report(generate_html=False)
        logger.info("Report generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during report generation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 