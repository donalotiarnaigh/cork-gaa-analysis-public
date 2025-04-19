"""
Competition Metrics Analysis Module

This module implements the competition intensity analysis for GAA clubs,
calculating competition metrics and identifying competition hotspots.

Author: Daniel Tierney
Date: 2024-04-05
"""

import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompetitionMetricsCalculator:
    def __init__(self, buffer_overlaps_path, clubs_path, sa_path, output_dir):
        self.buffer_overlaps_path = Path(buffer_overlaps_path)
        self.clubs_path = Path(clubs_path)
        self.sa_path = Path(sa_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Constants from overlap analysis
        self.URBAN_DENSITY_THRESHOLD = 2400  # km²
        self.PRIMARY_BUFFER_DENSITY = 330.3  # km²
        self.BASELINE_YOUTH_PROPORTION = 0.263
        
        # Load data
        self.overlaps = None
        self.clubs = None
        self.sa_data = None
        self.metrics = None
        
    def load_data(self):
        """Load required datasets."""
        logger.info("Loading input data")
        self.overlaps = gpd.read_file(self.buffer_overlaps_path)
        self.clubs = gpd.read_file(self.clubs_path)
        self.sa_data = gpd.read_file(self.sa_path)
        
        # Ensure consistent CRS
        if self.clubs.crs != self.sa_data.crs:
            self.clubs = self.clubs.to_crs(self.sa_data.crs)
            
        logger.info(f"Loaded {len(self.clubs)} clubs and {len(self.overlaps)} overlaps")
        
    def calculate_base_competition_index(self):
        """Calculate base competition index using validated weights."""
        logger.info("Calculating base competition indices")
        
        # Initialize results dictionary
        club_metrics = {}
        
        for club_name in self.clubs['Club'].unique():
            # Get overlaps involving this club
            club_overlaps = self.overlaps[
                (self.overlaps['club1'] == club_name) |
                (self.overlaps['club2'] == club_name)
            ]
            
            if len(club_overlaps) > 0:
                # Calculate weighted metrics
                tier_weights = {
                    'primary': 1.0,
                    'secondary': 0.75,
                    'tertiary': 0.5
                }
                
                weighted_scores = []
                for _, overlap in club_overlaps.iterrows():
                    # Calculate normalized density score
                    density_score = overlap['population_density'] / self.PRIMARY_BUFFER_DENSITY
                    
                    # Calculate youth proportion score
                    youth_prop = overlap['youth_population'] / overlap['total_population']
                    youth_score = youth_prop / self.BASELINE_YOUTH_PROPORTION
                    
                    # Combined weighted score
                    weighted_score = (
                        tier_weights[overlap['tier']] *
                        density_score *
                        youth_score
                    )
                    weighted_scores.append(weighted_score)
                
                if weighted_scores:  # If there are any weighted scores
                    # Store metrics
                    club_metrics[club_name] = {
                        'base_competition_index': np.mean(weighted_scores),
                        'max_competition_score': np.max(weighted_scores),
                        'total_overlaps': len(club_overlaps)
                    }
                else:
                    # No overlaps - isolated club
                    club_metrics[club_name] = {
                        'base_competition_index': 0,
                        'max_competition_score': 0,
                        'total_overlaps': 0
                    }
            else:
                # No overlaps - isolated club
                club_metrics[club_name] = {
                    'base_competition_index': 0,
                    'max_competition_score': 0,
                    'total_overlaps': 0
                }
        
        return pd.DataFrame.from_dict(club_metrics, orient='index')
    
    def calculate_urban_intensity_score(self):
        """Calculate urban intensity scores."""
        logger.info("Calculating urban intensity scores")
        
        # Get high-density areas
        high_density_clubs = self.overlaps[
            self.overlaps['population_density'] > self.URBAN_DENSITY_THRESHOLD
        ][['club1', 'club2']].values.ravel()
        high_density_clubs = pd.unique(high_density_clubs)
        
        # Get club coordinates
        club_coords = self.clubs[['geometry']].copy()
        club_coords['x'] = club_coords.geometry.x
        club_coords['y'] = club_coords.geometry.y
        
        # Calculate distances between clubs
        distances = cdist(
            club_coords[['x', 'y']],
            club_coords[['x', 'y']]
        )
        
        urban_scores = {}
        for idx, club_name in enumerate(self.clubs['Club']):
            if club_name in high_density_clubs:
                # Calculate proximity to other high-density clubs
                other_high_density = [
                    i for i, name in enumerate(self.clubs['Club'])
                    if name in high_density_clubs and name != club_name
                ]
                
                if other_high_density:
                    # Calculate proximity score (inverse of distance)
                    proximity_scores = 1 / (1 + distances[idx, other_high_density])
                    urban_scores[club_name] = np.mean(proximity_scores)
                else:
                    urban_scores[club_name] = 0
            else:
                urban_scores[club_name] = 0
        
        return pd.Series(urban_scores, name='urban_intensity_score')
    
    def calculate_rural_sustainability_score(self):
        """Calculate rural sustainability scores."""
        logger.info("Calculating rural sustainability scores")
        
        rural_scores = {}
        for club_name in self.clubs['Club']:
            # Get overlaps involving this club
            club_overlaps = self.overlaps[
                (self.overlaps['club1'] == club_name) |
                (self.overlaps['club2'] == club_name)
            ]
            
            if len(club_overlaps) > 0:
                # Calculate average youth proportion
                youth_props = club_overlaps['youth_population'] / club_overlaps['total_population']
                avg_youth_prop = youth_props.mean()
                
                # Calculate distance factor (prefer more isolated clubs)
                avg_density = club_overlaps['population_density'].mean()
                density_factor = min(1, self.URBAN_DENSITY_THRESHOLD / avg_density)
                
                # Combined rural score
                rural_scores[club_name] = (avg_youth_prop / self.BASELINE_YOUTH_PROPORTION) * density_factor
            else:
                rural_scores[club_name] = 0
        
        return pd.Series(rural_scores, name='rural_sustainability_score')
    
    def calculate_accessibility_scores(self):
        """Calculate accessibility scores based on urban/rural context."""
        logger.info("Calculating accessibility scores")
        
        accessibility_scores = {}
        for _, club in self.clubs.iterrows():
            club_name = club['Club']
            
            # Get overlaps involving this club
            club_overlaps = self.overlaps[
                (self.overlaps['club1'] == club_name) |
                (self.overlaps['club2'] == club_name)
            ]
            
            if len(club_overlaps) > 0:
                avg_density = club_overlaps['population_density'].mean()
                
                if avg_density > self.URBAN_DENSITY_THRESHOLD:
                    # Urban accessibility score
                    accessibility_scores[club_name] = {
                        'urban_access_score': 1.0,
                        'rural_access_score': 0.3,
                        'is_urban': True
                    }
                else:
                    # Rural accessibility score
                    isolation_factor = 1 - (avg_density / self.URBAN_DENSITY_THRESHOLD)
                    accessibility_scores[club_name] = {
                        'urban_access_score': 0.3,
                        'rural_access_score': isolation_factor,
                        'is_urban': False
                    }
            else:
                # Isolated club
                accessibility_scores[club_name] = {
                    'urban_access_score': 0.1,
                    'rural_access_score': 1.0,
                    'is_urban': False
                }
        
        return pd.DataFrame.from_dict(accessibility_scores, orient='index')
    
    def calculate_composite_metrics(self):
        """Calculate composite sustainability and development metrics."""
        logger.info("Calculating composite metrics")
        
        # Get component scores
        base_metrics = self.calculate_base_competition_index()
        urban_scores = self.calculate_urban_intensity_score()
        rural_scores = self.calculate_rural_sustainability_score()
        accessibility = self.calculate_accessibility_scores()
        
        # Combine all metrics
        self.metrics = pd.concat([
            base_metrics,
            urban_scores,
            rural_scores,
            accessibility
        ], axis=1)
        
        # Calculate Club Sustainability Index
        self.metrics['sustainability_index'] = (
            0.4 * self.metrics['base_competition_index'] +
            0.3 * np.where(
                accessibility['is_urban'],
                self.metrics['urban_intensity_score'],
                self.metrics['rural_sustainability_score']
            ) +
            0.3 * np.where(
                accessibility['is_urban'],
                accessibility['urban_access_score'],
                accessibility['rural_access_score']
            )
        )
        
        # Calculate Development Priority Score
        self.metrics['development_priority'] = (
            0.4 * (1 - self.metrics['sustainability_index']) +
            0.3 * (1 - self.metrics['base_competition_index']) +
            0.3 * np.where(
                accessibility['is_urban'],
                1 - self.metrics['urban_intensity_score'],
                1 - self.metrics['rural_sustainability_score']
            )
        )
        
        # Normalize scores to [0,1] range
        for col in ['sustainability_index', 'development_priority']:
            self.metrics[col] = (self.metrics[col] - self.metrics[col].min()) / \
                               (self.metrics[col].max() - self.metrics[col].min())
        
        logger.info("Composite metrics calculated")
        
    def generate_reports(self):
        """Generate analysis reports and save results."""
        logger.info("Generating reports")
        
        # Save metrics to CSV
        metrics_path = self.output_dir / 'competition_scores.csv'
        self.metrics.to_csv(metrics_path)
        logger.info(f"Saved competition scores to {metrics_path}")
        
        # Create GeoDataFrame with metrics
        metrics_gdf = self.clubs.merge(
            self.metrics,
            left_on='Club',
            right_index=True
        )
        
        # Save to GeoPackage
        geopackage_path = self.output_dir / 'competition_metrics.gpkg'
        metrics_gdf.to_file(geopackage_path, driver='GPKG')
        logger.info(f"Saved spatial metrics to {geopackage_path}")
        
        # Generate markdown report
        report_path = self.output_dir / 'competition_metrics_report.md'
        with open(report_path, 'w') as f:
            f.write("# Competition Metrics Analysis Report\n\n")
            
            f.write("## Overall Statistics\n\n")
            f.write(f"Total clubs analyzed: {len(self.metrics)}\n")
            f.write(f"Urban clubs: {sum(self.metrics['is_urban'])}\n")
            f.write(f"Rural clubs: {sum(~self.metrics['is_urban'])}\n\n")
            
            f.write("## Sustainability Metrics\n\n")
            f.write("### Top 5 Most Sustainable Clubs\n")
            top_sustainable = self.metrics.nlargest(5, 'sustainability_index')
            for idx, row in top_sustainable.iterrows():
                f.write(f"- {idx}: {row['sustainability_index']:.3f}\n")
            
            f.write("\n### Top 5 Development Priorities\n")
            top_priority = self.metrics.nlargest(5, 'development_priority')
            for idx, row in top_priority.iterrows():
                f.write(f"- {idx}: {row['development_priority']:.3f}\n")
            
            f.write("\n## Urban vs Rural Analysis\n\n")
            f.write("### Urban Clubs\n")
            urban_stats = self.metrics[self.metrics['is_urban']].describe()
            f.write(f"- Average sustainability: {urban_stats.loc['mean', 'sustainability_index']:.3f}\n")
            f.write(f"- Average competition: {urban_stats.loc['mean', 'base_competition_index']:.3f}\n")
            
            f.write("\n### Rural Clubs\n")
            rural_stats = self.metrics[~self.metrics['is_urban']].describe()
            f.write(f"- Average sustainability: {rural_stats.loc['mean', 'sustainability_index']:.3f}\n")
            f.write(f"- Average competition: {rural_stats.loc['mean', 'base_competition_index']:.3f}\n")
        
        logger.info(f"Generated analysis report at {report_path}")
        
    def run_analysis(self):
        """Run the complete competition metrics analysis."""
        self.load_data()
        self.calculate_composite_metrics()
        self.generate_reports()

if __name__ == "__main__":
    calculator = CompetitionMetricsCalculator(
        buffer_overlaps_path="data/analysis/buffer_overlaps.gpkg",
        clubs_path="data/processed/cork_clubs_complete.gpkg",
        sa_path="data/processed/cork_sa_analysis_full.gpkg",
        output_dir="data/analysis"
    )
    calculator.run_analysis() 