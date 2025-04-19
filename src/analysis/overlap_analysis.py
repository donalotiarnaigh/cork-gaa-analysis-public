import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
from shapely.ops import unary_union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OverlapAnalyzer:
    def __init__(self, buffer_path, sa_path, output_dir):
        self.buffer_path = Path(buffer_path)
        self.sa_path = Path(sa_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffer tier weights based on density ratios
        self.tier_weights = {
            'primary': 1.0,    # 330.3/km²
            'secondary': 0.75, # 245.9/km²
            'tertiary': 0.5    # 165.1/km²
        }
        
        # Load data
        self.buffers = None
        self.sa_data = None
        self.overlaps = None
        
    def load_data(self):
        """Load and prepare spatial data."""
        logger.info("Loading buffer and Small Area data")
        self.buffers = gpd.read_file(self.buffer_path)
        self.sa_data = gpd.read_file(self.sa_path)
        
        # Ensure consistent CRS
        if self.buffers.crs != self.sa_data.crs:
            self.sa_data = self.sa_data.to_crs(self.buffers.crs)
            
        logger.info(f"Loaded {len(self.buffers)} buffer zones and {len(self.sa_data)} Small Areas")
        
    def calculate_overlaps(self):
        """Calculate overlaps between buffer zones."""
        logger.info("Calculating buffer overlaps")
        
        # Create a list to store overlap results
        overlap_results = []
        
        # Group buffers by tier
        for tier in ['primary', 'secondary', 'tertiary']:
            tier_buffers = self.buffers[self.buffers['tier'] == tier]
            
            # Calculate pairwise overlaps
            for i, buffer1 in tier_buffers.iterrows():
                for j, buffer2 in tier_buffers.iterrows():
                    if i < j:  # Avoid duplicate calculations
                        # Calculate intersection
                        intersection = buffer1.geometry.intersection(buffer2.geometry)
                        if not intersection.is_empty:
                            # Calculate overlap area
                            overlap_area = intersection.area / 1000000  # Convert to km²
                            
                            # Get intersecting Small Areas
                            intersecting_sa = self.sa_data[self.sa_data.geometry.intersects(intersection)]
                            
                            if len(intersecting_sa) > 0:
                                # Calculate total population
                                total_pop = intersecting_sa['T1_1AGETT_x'].sum()
                                
                                # Calculate youth population (ages 0-19)
                                youth_pop = 0
                                for age in range(20):  # 0 to 19
                                    col_name = f'T1_1AGE{age}T_x'
                                    if col_name in intersecting_sa.columns:
                                        youth_pop += intersecting_sa[col_name].sum()
                                
                                density = total_pop / overlap_area
                                
                                # Calculate competition score
                                competition_score = (
                                    self.tier_weights[tier] * 
                                    (density / 330.3) *  # Normalize by primary buffer density
                                    (1 + (youth_pop / total_pop))  # Weight by youth proportion
                                )
                                
                                overlap_results.append({
                                    'club1': buffer1['club_name'],
                                    'club2': buffer2['club_name'],
                                    'tier': tier,
                                    'overlap_area_km2': overlap_area,
                                    'total_population': total_pop,
                                    'youth_population': youth_pop,
                                    'population_density': density,
                                    'competition_score': competition_score
                                })
        
        # Create GeoDataFrame from results
        self.overlaps = gpd.GeoDataFrame(overlap_results)
        logger.info(f"Found {len(self.overlaps)} buffer overlaps")
        
    def analyze_high_density_areas(self):
        """Analyze overlaps in high-density areas."""
        logger.info("Analyzing high-density area overlaps")
        
        # Focus areas from buffer analysis
        focus_areas = [
            'Ballyphephane',
            'St. Vincent\'s',
            'St. Nicholas',
            'Glen Rovers',
            'St. Finbarr\'s'
        ]
        
        # Filter overlaps involving focus areas
        focus_overlaps = self.overlaps[
            (self.overlaps['club1'].isin(focus_areas)) |
            (self.overlaps['club2'].isin(focus_areas))
        ]
        
        # Calculate summary statistics
        summary_stats = {
            'total_overlaps': len(focus_overlaps),
            'average_population': focus_overlaps['total_population'].mean(),
            'average_density': focus_overlaps['population_density'].mean(),
            'average_competition': focus_overlaps['competition_score'].mean()
        }
        
        logger.info(f"Found {summary_stats['total_overlaps']} overlaps in focus areas")
        return focus_overlaps, summary_stats
        
    def analyze_youth_rich_areas(self):
        """Analyze overlaps in youth-rich areas."""
        logger.info("Analyzing youth-rich area overlaps")
        
        # Youth-rich areas from buffer analysis
        youth_areas = [
            'Ilen Rovers',
            'Béal Athan Ghaorthaidh',
            'Goleen',
            'Cill na Martra'
        ]
        
        # Filter overlaps involving youth-rich areas
        youth_overlaps = self.overlaps[
            (self.overlaps['club1'].isin(youth_areas)) |
            (self.overlaps['club2'].isin(youth_areas))
        ]
        
        # Calculate summary statistics
        summary_stats = {
            'total_overlaps': len(youth_overlaps),
            'average_youth_population': youth_overlaps['youth_population'].mean(),
            'average_youth_proportion': (youth_overlaps['youth_population'] / 
                                       youth_overlaps['total_population']).mean(),
            'average_competition': youth_overlaps['competition_score'].mean()
        }
        
        logger.info(f"Found {summary_stats['total_overlaps']} overlaps in youth-rich areas")
        return youth_overlaps, summary_stats
        
    def generate_reports(self, focus_overlaps, focus_stats, youth_overlaps, youth_stats):
        """Generate analysis reports."""
        logger.info("Generating analysis reports")
        
        # Save spatial data
        self.overlaps.to_file(
            self.output_dir / 'buffer_overlaps.gpkg',
            driver='GPKG'
        )
        
        # Save statistics
        stats_df = pd.DataFrame({
            'metric': [
                'Total overlaps',
                'Average population in focus areas',
                'Average density in focus areas',
                'Average competition score in focus areas',
                'Total youth area overlaps',
                'Average youth population',
                'Average youth proportion',
                'Average competition score in youth areas'
            ],
            'value': [
                focus_stats['total_overlaps'],
                focus_stats['average_population'],
                focus_stats['average_density'],
                focus_stats['average_competition'],
                youth_stats['total_overlaps'],
                youth_stats['average_youth_population'],
                youth_stats['average_youth_proportion'],
                youth_stats['average_competition']
            ]
        })
        stats_df.to_csv(
            self.output_dir / 'overlap_statistics.csv',
            index=False
        )
        
        # Generate markdown report
        report_path = self.output_dir / 'overlap_analysis_report.md'
        with open(report_path, 'w') as f:
            f.write("# Buffer Overlap Analysis Report\n\n")
            
            f.write("## High-Density Area Analysis\n\n")
            f.write(f"- Total overlaps in focus areas: {focus_stats['total_overlaps']}\n")
            f.write(f"- Average population in overlap areas: {focus_stats['average_population']:.0f}\n")
            f.write(f"- Average population density: {focus_stats['average_density']:.1f}/km²\n")
            f.write(f"- Average competition score: {focus_stats['average_competition']:.3f}\n\n")
            
            f.write("## Youth-Rich Area Analysis\n\n")
            f.write(f"- Total overlaps in youth-rich areas: {youth_stats['total_overlaps']}\n")
            f.write(f"- Average youth population: {youth_stats['average_youth_population']:.0f}\n")
            f.write(f"- Average youth proportion: {youth_stats['average_youth_proportion']:.1%}\n")
            f.write(f"- Average competition score: {youth_stats['average_competition']:.3f}\n\n")
            
            f.write("## Notable Overlaps\n\n")
            f.write("### High-Density Areas\n")
            for _, row in focus_overlaps.iterrows():
                f.write(f"- {row['club1']} and {row['club2']}: ")
                f.write(f"{row['population_density']:.1f}/km², ")
                f.write(f"Competition Score: {row['competition_score']:.3f}\n")
            
            f.write("\n### Youth-Rich Areas\n")
            for _, row in youth_overlaps.iterrows():
                f.write(f"- {row['club1']} and {row['club2']}: ")
                f.write(f"Youth Proportion: {(row['youth_population']/row['total_population']):.1%}, ")
                f.write(f"Competition Score: {row['competition_score']:.3f}\n")
        
        logger.info(f"Reports generated in {self.output_dir}")
        
    def run_analysis(self):
        """Run the complete overlap analysis."""
        self.load_data()
        self.calculate_overlaps()
        
        focus_overlaps, focus_stats = self.analyze_high_density_areas()
        youth_overlaps, youth_stats = self.analyze_youth_rich_areas()
        
        self.generate_reports(focus_overlaps, focus_stats, youth_overlaps, youth_stats)
        
if __name__ == "__main__":
    analyzer = OverlapAnalyzer(
        buffer_path="data/interim/buffer_demographics.gpkg",
        sa_path="data/processed/cork_sa_analysis_full.gpkg",
        output_dir="data/analysis"
    )
    analyzer.run_analysis() 