#!/usr/bin/env python3
"""
Buffer Statistics Analysis Module

This script calculates demographic profiles and metrics for GAA club buffer zones,
including population statistics, education levels, employment patterns, and
social class distribution.
"""

import os
import logging
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BufferAnalyzer:
    def __init__(self, buffers_path, sa_boundaries_path, sa_analysis_path):
        """
        Initialize the BufferAnalyzer with input file paths.
        
        Args:
            buffers_path (str): Path to buffer zones GeoPackage
            sa_boundaries_path (str): Path to Small Area boundaries GeoPackage
            sa_analysis_path (str): Path to Small Area analysis data GeoPackage
        """
        self.buffers_path = buffers_path
        self.sa_boundaries_path = sa_boundaries_path
        self.sa_analysis_path = sa_analysis_path
        
        # Set up output paths
        self.output_dir = Path('data/interim')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load and validate input data."""
        logger.info("Loading input data...")
        
        # Load buffer zones
        self.buffers = gpd.read_file(self.buffers_path)
        logger.info(f"Loaded {len(self.buffers)} buffer zones")
        
        # Load Small Area boundaries with analysis data
        self.sa_boundaries = gpd.read_file(self.sa_boundaries_path)
        self.sa_analysis = gpd.read_file(self.sa_analysis_path)
        
        # Drop geometry from analysis data before merge
        if 'geometry' in self.sa_analysis.columns:
            self.sa_analysis = self.sa_analysis.drop(columns=['geometry'])
        
        # Merge SA boundaries with analysis data
        self.sa_data = self.sa_boundaries.merge(
            self.sa_analysis,
            on='SA_GUID_2022',
            how='left'
        )
        logger.info(f"Loaded and merged {len(self.sa_data)} Small Areas")
        logger.info(f"Available columns after merge: {self.sa_data.columns.tolist()}")
        
    def calculate_intersection_weights(self, buffer_geometry, intersecting_areas):
        """Calculate area-based weights for intersecting Small Areas."""
        # Calculate intersection areas
        intersection_areas = intersecting_areas.geometry.intersection(buffer_geometry)
        intersection_areas = gpd.GeoSeries(intersection_areas)
        
        # Calculate original areas
        original_areas = intersecting_areas.geometry.area
        
        # Calculate weights (proportion of each SA that intersects with buffer)
        weights = intersection_areas.area / original_areas
        
        return weights
        
    def calculate_weighted_stats(self, data, weights):
        """Calculate weighted statistics for demographic variables."""
        # List of variables to calculate weighted stats for
        rate_variables = [
            'basic_education_rate',
            'secondary_education_rate',
            'third_level_rate',
            'employment_rate',
            'labor_force_participation_rate',
            'professional_rate',
            'working_class_rate',
            'youth_proportion'
        ]
        
        # Calculate weighted means for each variable
        weighted_stats = {}
        for var in rate_variables:
            if var in data.columns:
                weighted_stats[var] = np.average(
                    data[var],
                    weights=weights
                )
            else:
                weighted_stats[var] = None
                logger.warning(f"Variable {var} not found in data")
        
        return weighted_stats
        
    def analyze_buffer(self, buffer_row):
        """Analyze a single buffer zone."""
        # Log available columns
        logger.info("Available columns in sa_data:")
        logger.info(self.sa_data.columns.tolist())
        
        # Find intersecting Small Areas
        intersecting_areas = self.sa_data[
            self.sa_data.geometry.intersects(buffer_row.geometry)
        ]
        
        if intersecting_areas.empty:
            logger.warning(f"No intersecting areas found for buffer {buffer_row['club_name']}")
            return None
        
        # Log columns after join
        logger.info("Available columns after spatial join:")
        logger.info(intersecting_areas.columns.tolist())
        
        # Calculate intersection weights
        weights = self.calculate_intersection_weights(
            buffer_row.geometry,
            intersecting_areas
        )
        
        # Calculate weighted demographic statistics
        demographics = self.calculate_weighted_stats(
            intersecting_areas,
            weights
        )
        
        # Calculate total population (sum of weighted populations)
        total_pop = (intersecting_areas['T1_1AGETT_y_x'] * weights).sum()
        
        # Calculate youth population (ages 0-19)
        youth_pop = 0
        for i in range(20):
            col = f'T1_1AGE{i}T_y_x'
            if col in intersecting_areas.columns:
                youth_pop += (intersecting_areas[col] * weights).sum()
        
        # Calculate working age population (ages 20-64)
        working_age_pop = 0
        for i in range(20, 65):
            col = f'T1_1AGE{i}T_y_x' if i < 20 else f'T1_1AGE{i}_T_y'
            if col in intersecting_areas.columns:
                working_age_pop += (intersecting_areas[col] * weights).sum()
        
        # Calculate elderly population (ages 65+)
        elderly_pop = 0
        for i in range(65, 100):  # Assuming max age is 99
            col = f'T1_1AGE{i}_T_y'
            if col in intersecting_areas.columns:
                elderly_pop += (intersecting_areas[col] * weights).sum()
        
        # Calculate area and density
        buffer_area = buffer_row.geometry.area / 1_000_000  # Convert to km²
        population_density = total_pop / buffer_area
        youth_density = youth_pop / buffer_area
        
        # Combine all metrics
        metrics = {
            'club_name': buffer_row['club_name'],
            'tier': buffer_row['tier'],
            'buffer_size': buffer_row['buffer_size'],
            'area_km2': buffer_area,
            'total_population': total_pop,
            'youth_population': youth_pop,
            'working_age_population': working_age_pop,
            'elderly_population': elderly_pop,
            'population_density': population_density,
            'youth_density': youth_density,
            'num_small_areas': len(intersecting_areas),
            **demographics
        }
        
        return metrics
        
    def analyze_all_buffers(self):
        """Analyze all buffer zones."""
        logger.info("Analyzing buffer zones...")
        
        # Process each buffer
        buffer_stats = []
        for idx, buffer_row in self.buffers.iterrows():
            metrics = self.analyze_buffer(buffer_row)
            if metrics:
                buffer_stats.append(metrics)
        
        # Create DataFrame
        stats_df = pd.DataFrame(buffer_stats)
        
        # Save results
        self.save_results(stats_df)
        
        return stats_df
        
    def save_results(self, stats_df):
        """Save analysis results."""
        logger.info("Saving results...")
        
        # Save detailed statistics
        stats_path = self.output_dir / 'buffer_statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Saved buffer statistics to {stats_path}")
        
        # Create GeoDataFrame with demographics
        demographics_gdf = self.buffers.merge(
            stats_df,
            on=['club_name', 'tier', 'buffer_size']
        )
        
        # Save demographic profiles
        demographics_path = self.output_dir / 'buffer_demographics.gpkg'
        demographics_gdf.to_file(demographics_path, driver='GPKG')
        logger.info(f"Saved demographic profiles to {demographics_path}")
        
        # Save demographics as CSV for compatibility
        demographics_csv_path = self.output_dir / 'buffer_demographics.csv'
        stats_df[['club_name', 'tier', 'buffer_size', 'area_km2', 
                 'total_population', 'youth_population', 'working_age_population', 
                 'elderly_population', 'population_density', 'youth_density']].to_csv(
            demographics_csv_path, index=False)
        logger.info(f"Saved demographic profiles to {demographics_csv_path}")
        
        # Generate report
        self.generate_report(stats_df)
        
    def generate_report(self, stats_df):
        """Generate analysis report."""
        logger.info("Generating report...")
        
        report_path = Path('data/analysis/buffer_analysis_report.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Buffer Analysis Report\n\n")
            
            # Overall statistics
            f.write("## Overall Statistics\n\n")
            f.write(f"Total buffer zones analyzed: {len(stats_df)}\n\n")
            
            # Statistics by tier
            f.write("## Statistics by Buffer Tier\n\n")
            for tier in ['primary', 'secondary', 'tertiary']:
                tier_stats = stats_df[stats_df['tier'] == tier]
                f.write(f"### {tier.title()} Buffer Statistics\n")
                f.write("#### Population Metrics\n")
                
                # Safely calculate and format statistics
                try:
                    avg_total_pop = tier_stats['total_population'].astype(float).mean()
                    f.write(f"- Average total population: {avg_total_pop:.0f}\n")
                except (TypeError, ValueError):
                    f.write("- Average total population: N/A\n")
                
                try:
                    avg_youth_pop = tier_stats['youth_population'].astype(float).mean()
                    f.write(f"- Average youth population: {avg_youth_pop:.0f}\n")
                except (TypeError, ValueError):
                    f.write("- Average youth population: N/A\n")
                
                try:
                    avg_pop_density = tier_stats['population_density'].astype(float).mean()
                    f.write(f"- Average population density: {avg_pop_density:.1f}/km²\n")
                except (TypeError, ValueError):
                    f.write("- Average population density: N/A\n")
                
                try:
                    avg_youth_density = tier_stats['youth_density'].astype(float).mean()
                    f.write(f"- Average youth density: {avg_youth_density:.1f}/km²\n\n")
                except (TypeError, ValueError):
                    f.write("- Average youth density: N/A\n\n")
                
                f.write("#### Demographic Rates\n")
                f.write("Education:\n")
                
                try:
                    basic_rate = tier_stats['basic_education_rate'].astype(float).mean() * 100
                    f.write(f"- Basic: {basic_rate:.1f}%\n")
                except (TypeError, ValueError):
                    f.write("- Basic: N/A\n")
                
                try:
                    secondary_rate = tier_stats['secondary_education_rate'].astype(float).mean() * 100
                    f.write(f"- Secondary: {secondary_rate:.1f}%\n")
                except (TypeError, ValueError):
                    f.write("- Secondary: N/A\n")
                
                try:
                    third_level_rate = tier_stats['third_level_rate'].astype(float).mean() * 100
                    f.write(f"- Third Level: {third_level_rate:.1f}%\n\n")
                except (TypeError, ValueError):
                    f.write("- Third Level: N/A\n\n")
                
                f.write("Employment:\n")
                try:
                    employment_rate = tier_stats['employment_rate'].astype(float).mean() * 100
                    f.write(f"- Employment rate: {employment_rate:.1f}%\n")
                except (TypeError, ValueError):
                    f.write("- Employment rate: N/A\n")
                
                try:
                    labor_rate = tier_stats['labor_force_participation_rate'].astype(float).mean() * 100
                    f.write(f"- Labor force participation: {labor_rate:.1f}%\n\n")
                except (TypeError, ValueError):
                    f.write("- Labor force participation: N/A\n\n")
                
                f.write("Social Class:\n")
                try:
                    professional_rate = tier_stats['professional_rate'].astype(float).mean() * 100
                    f.write(f"- Professional rate: {professional_rate:.1f}%\n")
                except (TypeError, ValueError):
                    f.write("- Professional rate: N/A\n")
                
                try:
                    working_class_rate = tier_stats['working_class_rate'].astype(float).mean() * 100
                    f.write(f"- Working class rate: {working_class_rate:.1f}%\n\n")
                except (TypeError, ValueError):
                    f.write("- Working class rate: N/A\n\n")
                
                f.write("Youth:\n")
                try:
                    youth_proportion = tier_stats['youth_proportion'].astype(float).mean() * 100
                    f.write(f"- Youth proportion: {youth_proportion:.1f}%\n\n")
                except (TypeError, ValueError):
                    f.write("- Youth proportion: N/A\n\n")
            
            # Notable patterns
            f.write("## Notable Patterns\n\n")
            
            # Population density patterns
            try:
                high_density = stats_df.nlargest(5, 'population_density')
                f.write("### Highest Population Density Areas\n")
                for _, row in high_density.iterrows():
                    f.write(f"- {row['club_name']} ({row['tier']}): {row['population_density']:.1f}/km²\n")
                f.write("\n")
            except (TypeError, ValueError):
                f.write("### Population density patterns not available\n\n")
            
            # Youth proportion patterns
            try:
                high_youth = stats_df.nlargest(5, 'youth_proportion')
                f.write("### Highest Youth Proportion Areas\n")
                for _, row in high_youth.iterrows():
                    f.write(f"- {row['club_name']} ({row['tier']}): {row['youth_proportion']*100:.1f}%\n")
            except (TypeError, ValueError):
                f.write("### Youth proportion patterns not available\n")
            
        logger.info(f"Generated report at {report_path}")

def main():
    """Main function to run buffer analysis."""
    # Define input paths
    buffers_path = 'data/interim/club_buffers.gpkg'
    sa_boundaries_path = 'data/processed/sa_boundaries_final.gpkg'
    sa_analysis_path = 'data/processed/cork_sa_analysis_full.gpkg'
    
    # Initialize and run analyzer
    analyzer = BufferAnalyzer(buffers_path, sa_boundaries_path, sa_analysis_path)
    analyzer.analyze_all_buffers()

if __name__ == '__main__':
    main() 