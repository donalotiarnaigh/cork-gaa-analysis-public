#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.quality_assurance import QualityAssurance
import logging
import yaml

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Set up base directory
        base_dir = project_root / 'data' / 'analysis'
        
        # Initialize quality assurance
        logger.info(f"Initializing quality assurance with base directory: {base_dir}")
        qa = QualityAssurance(base_dir=base_dir)
        
        # Run quality checks
        logger.info("Running quality checks...")
        results = qa.perform_final_validation()
        
        # Log results
        logger.info("\nQuality Assurance Results:")
        logger.info(f"Total checks: {results['summary']['total_checks']}")
        logger.info(f"Passed checks: {results['summary']['passed_checks']}")
        logger.info(f"Pass rate: {results['summary']['pass_rate']:.2%}")
        
        # Save results to YAML file
        report_path = base_dir / 'quality_assurance_report.yaml'
        with open(report_path, 'w') as f:
            yaml.dump(results, f, sort_keys=True, indent=2)
        
        logger.info(f"\nQuality assurance report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error during quality assurance: {str(e)}")
        raise

if __name__ == "__main__":
    main() 