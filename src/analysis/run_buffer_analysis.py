#!/usr/bin/env python3
"""
Run Buffer Analysis Pipeline

This script executes the complete buffer analysis pipeline:
1. Generate buffer zones around GAA clubs
2. Calculate demographic statistics for buffer zones
"""

import logging
from pathlib import Path
from buffer_generation import BufferGenerator
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
        logger.info("Starting buffer analysis pipeline")
        
        # Step 1: Generate buffer zones
        logger.info("Generating buffer zones...")
        buffer_generator = BufferGenerator(
            clubs_path='data/processed/cork_clubs_complete.gpkg',
            sa_boundaries_path='data/processed/cork_sa_analysis_full.gpkg',
            sa_analysis_path='data/processed/cork_sa_analysis_full.gpkg'
        )
        buffers_gdf = buffer_generator.generate_buffers()
        
        # Step 2: Calculate buffer statistics
        logger.info("Calculating buffer statistics...")
        buffer_analyzer = BufferAnalyzer(
            buffers_path='data/interim/club_buffers.gpkg',
            sa_boundaries_path='data/processed/cork_sa_analysis_full.gpkg',
            sa_analysis_path='data/processed/cork_sa_analysis_full.gpkg'
        )
        stats_df = buffer_analyzer.analyze_all_buffers()
        
        logger.info("Buffer analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in buffer analysis pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 