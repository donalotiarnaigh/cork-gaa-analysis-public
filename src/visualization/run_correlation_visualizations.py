#!/usr/bin/env python3
"""
Runner script for generating correlation visualizations for the research report.
This script executes the correlation_visualizations module to create various
plots showing relationships between demographic factors and club performance.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project directory to the Python path if not already there
project_dir = Path(__file__).resolve().parents[2]
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Import the correlation visualizations module
from src.visualization.correlation_visualizations import main as run_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run correlation visualizations for the report."""
    logger.info("Starting correlation visualizations for research report...")
    
    try:
        # Create the output directory if it doesn't exist
        output_dir = Path('report_visualizations/correlation')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Run the visualization module
        run_visualizations()
        
        logger.info("Correlation visualizations completed successfully.")
    except Exception as e:
        logger.error(f"Error generating correlation visualizations: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 