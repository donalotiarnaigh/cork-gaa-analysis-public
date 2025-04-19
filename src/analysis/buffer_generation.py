#!/usr/bin/env python3
"""
Buffer Generation Module for GAA Club Competition Analysis

This script generates tiered buffer zones around GAA clubs in Cork, adjusting
buffer sizes based on club grades and population density.
"""

import os
import logging
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define buffer tiers (in meters)
BUFFER_TIERS = {
    'primary': 2000,    # 2km
    'secondary': 5000,  # 5km
    'tertiary': 10000   # 10km
}

# Define target CRS (Irish Grid)
TARGET_CRS = 'EPSG:2157'

class BufferGenerator:
    def __init__(self, clubs_path, sa_boundaries_path, sa_analysis_path):
        """
        Initialize the BufferGenerator with input file paths.
        
        Args:
            clubs_path (str): Path to club locations GeoPackage
            sa_boundaries_path (str): Path to Small Area boundaries GeoPackage
            sa_analysis_path (str): Path to Small Area analysis data GeoPackage
        """
        self.clubs_path = clubs_path
        self.sa_boundaries_path = sa_boundaries_path
        self.sa_analysis_path = sa_analysis_path
        
        # Load data
        self.load_data()
        
        # Set up output paths
        self.output_dir = Path('data/interim')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load and validate input data."""
        logger.info("Loading input data...")
        
        # Load club data
        self.clubs = gpd.read_file(self.clubs_path)
        logger.info(f"Loaded {len(self.clubs)} clubs")
        
        # Load Small Area boundaries
        self.sa_boundaries = gpd.read_file(self.sa_boundaries_path)
        logger.info(f"Loaded {len(self.sa_boundaries)} Small Areas")
        
        # Load Small Area analysis data
        self.sa_analysis = gpd.read_file(self.sa_analysis_path)
        logger.info(f"Loaded Small Area analysis data")
        
        # Convert all data to target CRS
        logger.info("Converting data to target CRS...")
        self.clubs = self.clubs.to_crs(TARGET_CRS)
        self.sa_boundaries = self.sa_boundaries.to_crs(TARGET_CRS)
        self.sa_analysis = self.sa_analysis.to_crs(TARGET_CRS)
        
        # Validate data
        self.validate_data()
        
    def validate_data(self):
        """Validate input data for required fields and consistency."""
        logger.info("Validating input data...")
        
        # Check required columns in clubs data
        required_club_columns = ['Club', 'geometry', 'Grade_2024_football', 'Grade_2024_hurling']
        missing_columns = [col for col in required_club_columns if col not in self.clubs.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in clubs data: {missing_columns}")
        
        # Check required columns in Small Area data
        required_sa_columns = ['SA_GUID_2022', 'geometry']
        missing_columns = [col for col in required_sa_columns if col not in self.sa_boundaries.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in Small Area boundaries: {missing_columns}")
        
        # Check CRS consistency
        if not (self.clubs.crs == self.sa_boundaries.crs == self.sa_analysis.crs == TARGET_CRS):
            raise ValueError("CRS mismatch between input datasets")
            
        logger.info("Data validation complete")
        
    def adjust_buffer_size(self, club):
        """
        Adjust buffer size based on club characteristics.
        
        Args:
            club (Series): Club data row
            
        Returns:
            dict: Adjusted buffer sizes for each tier
        """
        # Base buffer sizes
        buffers = BUFFER_TIERS.copy()
        
        # Adjust based on club grade
        # Higher grades get larger buffers
        grade_adjustment = 1.0
        if pd.notna(club['Grade_2024_football']) or pd.notna(club['Grade_2024_hurling']):
            # Check if club has a high grade in either code
            high_grades = ['Premier Senior', 'Senior A']
            if (club['Grade_2024_football'] in high_grades or 
                club['Grade_2024_hurling'] in high_grades):
                grade_adjustment = 1.3
            elif pd.isna(club['Grade_2024_football']) and pd.isna(club['Grade_2024_hurling']):
                grade_adjustment = 0.8  # Smaller buffers for clubs with no grades
        
        buffers = {k: v * grade_adjustment for k, v in buffers.items()}
        
        return buffers
        
    def generate_buffers(self):
        """Generate buffer zones for all clubs."""
        logger.info("Generating buffer zones...")
        
        # Create empty GeoDataFrame for buffers
        buffers_list = []
        
        for _, club in self.clubs.iterrows():
            # Get adjusted buffer sizes
            buffer_sizes = self.adjust_buffer_size(club)
            
            # Create buffers for each tier
            for tier, size in buffer_sizes.items():
                buffer = club.geometry.buffer(size)
                buffers_list.append({
                    'club_name': club['Club'],
                    'tier': tier,
                    'buffer_size': size,
                    'geometry': buffer
                })
        
        # Create GeoDataFrame
        buffers_gdf = gpd.GeoDataFrame(
            buffers_list,
            crs=TARGET_CRS,
            geometry='geometry'
        )
        
        # Save results
        self.save_results(buffers_gdf)
        
        return buffers_gdf
        
    def save_results(self, buffers_gdf):
        """Save buffer generation results."""
        logger.info("Saving results...")
        
        # Save buffer zones
        buffers_path = self.output_dir / 'club_buffers.gpkg'
        buffers_gdf.to_file(buffers_path, driver='GPKG')
        logger.info(f"Saved buffer zones to {buffers_path}")
        
        # Save buffer parameters
        params_df = buffers_gdf[['club_name', 'tier', 'buffer_size']].copy()
        params_path = self.output_dir / 'buffer_parameters.csv'
        params_df.to_csv(params_path, index=False)
        logger.info(f"Saved buffer parameters to {params_path}")
        
        # Generate report
        self.generate_report(buffers_gdf)
        
    def generate_report(self, buffers_gdf):
        """Generate buffer generation report."""
        logger.info("Generating report...")
        
        report_path = self.output_dir / 'buffer_generation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Buffer Generation Report\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- Total clubs processed: {len(self.clubs)}\n")
            f.write(f"- Total buffer zones created: {len(buffers_gdf)}\n")
            
            # Buffer size statistics
            f.write("\n## Buffer Size Statistics\n\n")
            for tier in ['primary', 'secondary', 'tertiary']:
                tier_buffers = buffers_gdf[buffers_gdf['tier'] == tier]
                f.write(f"### {tier.title()} Buffer\n")
                f.write(f"- Average size: {tier_buffers['buffer_size'].mean():.2f}m\n")
                f.write(f"- Minimum size: {tier_buffers['buffer_size'].min():.2f}m\n")
                f.write(f"- Maximum size: {tier_buffers['buffer_size'].max():.2f}m\n\n")
            
            # Clubs with missing grades
            missing_grades = self.clubs[
                (self.clubs['Grade_2024_football'].isna()) & 
                (self.clubs['Grade_2024_hurling'].isna())
            ]
            if not missing_grades.empty:
                f.write("## Clubs with Missing Grades\n\n")
                f.write("The following clubs have no grade information and received default buffer sizes:\n\n")
                for club in missing_grades['Club']:
                    f.write(f"- {club}\n")
            
        logger.info(f"Generated report at {report_path}")

def main():
    """Main function to run buffer generation."""
    # Define input paths
    clubs_path = 'data/processed/cork_clubs_complete.gpkg'
    sa_boundaries_path = 'data/processed/sa_boundaries_final.gpkg'
    sa_analysis_path = 'data/processed/cork_sa_analysis_full.gpkg'
    
    # Initialize and run buffer generator
    generator = BufferGenerator(clubs_path, sa_boundaries_path, sa_analysis_path)
    generator.generate_buffers()

if __name__ == '__main__':
    main() 