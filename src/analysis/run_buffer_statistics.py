#!/usr/bin/env python3
"""
Run Buffer Statistics Analysis

This script executes the buffer statistics analysis to calculate demographic
profiles for GAA club buffer zones, including age group distributions.
"""

import logging
from pathlib import Path
from buffer_statistics import BufferAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    try:
        logger.info("Starting buffer statistics analysis")
        
        # Initialize analyzer
        analyzer = BufferAnalyzer(
            buffers_path='data/processed/club_buffers.gpkg',
            sa_boundaries_path='data/processed/cork_sa_analysis_full.gpkg',
            sa_analysis_path='data/processed/cork_sa_analysis_full.gpkg'
        )
        
        # Run analysis
        stats_df = analyzer.analyze_all_buffers()
        
        logger.info("Buffer statistics analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in buffer statistics analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 