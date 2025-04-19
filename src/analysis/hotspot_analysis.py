import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HotspotAnalysis:
    def __init__(self, data_dir: str = "data/analysis"):
        self.data_dir = Path(data_dir)
        self.competition_metrics = None
        self.clubs = None
        self.hotspots = None
        
    def load_data(self) -> None:
        """Load required input data."""
        logger.info("Loading input data...")
        
        # Load competition metrics
        self.competition_metrics = pd.read_csv(
            self.data_dir / "competition_scores.csv"
        )
        
        # Load spatial data
        self.clubs = gpd.read_file(
            self.data_dir / "competition_metrics.gpkg"
        )
        
        logger.info(f"Loaded {len(self.competition_metrics)} competition metrics")
        logger.info(f"Loaded {len(self.clubs)} clubs")
    
    def identify_high_competition_zones(self) -> gpd.GeoDataFrame:
        """Identify high-competition zones based on competition metrics."""
        logger.info("Identifying high-competition zones...")
        
        # Filter for high competition areas
        high_comp = self.clubs[
            (self.clubs['base_competition_index'] > 10.0) &
            (self.clubs['is_urban'] == 1)
        ].copy()
        
        # Create buffer zones around high competition clubs
        high_comp['geometry'] = high_comp.geometry.buffer(2000)  # 2km buffer
        
        # Add metadata before dissolve
        high_comp['zone_type'] = 'high_competition'
        high_comp['competition_level'] = 'high'
        
        # Dissolve overlapping buffers
        high_comp_zones = high_comp.dissolve(by='zone_type')
        
        return high_comp_zones
    
    def identify_underserved_areas(self) -> gpd.GeoDataFrame:
        """Identify underserved areas based on competition metrics."""
        logger.info("Identifying underserved areas...")
        
        # Filter for underserved areas
        underserved = self.clubs[
            (self.clubs['base_competition_index'] < 1.0) &
            (self.clubs['is_urban'] == 0) &
            (self.clubs['sustainability_index'] < 0.3)
        ].copy()
        
        # Create buffer zones around underserved clubs
        underserved['geometry'] = underserved.geometry.buffer(5000)  # 5km buffer
        
        # Add metadata before dissolve
        underserved['zone_type'] = 'underserved'
        underserved['competition_level'] = 'low'
        
        # Dissolve overlapping buffers
        underserved_zones = underserved.dissolve(by='zone_type')
        
        return underserved_zones
    
    def identify_grade_clusters(self) -> gpd.GeoDataFrame:
        """Identify grade-specific clusters."""
        logger.info("Identifying grade clusters...")
        
        # Filter for high-grade clubs
        high_grade = self.clubs[
            (self.clubs['Grade_2024_football_value'] <= 2) |  # Premier Senior or Senior A
            (self.clubs['Grade_2024_hurling_value'] <= 2)
        ].copy()
        
        # Create buffer zones around high-grade clubs
        high_grade['geometry'] = high_grade.geometry.buffer(3000)  # 3km buffer
        
        # Add metadata before dissolve
        high_grade['zone_type'] = 'grade_cluster'
        high_grade['competition_level'] = 'high_grade'
        
        # Dissolve overlapping buffers
        grade_clusters = high_grade.dissolve(by='zone_type')
        
        return grade_clusters
    
    def analyze_urban_rural_differences(self) -> pd.DataFrame:
        """Analyze differences between urban and rural areas."""
        logger.info("Analyzing urban-rural differences...")
        
        # Convert boolean is_urban to True/False for groupby
        self.clubs['is_urban_bool'] = self.clubs['is_urban'] == 1
        
        # Group by urban/rural status
        urban_rural_stats = self.clubs.groupby('is_urban_bool').agg({
            'base_competition_index': ['mean', 'std', 'min', 'max'],
            'sustainability_index': ['mean', 'std', 'min', 'max'],
            'development_priority': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        return urban_rural_stats
    
    def generate_hotspot_analysis(self) -> None:
        """Generate comprehensive hotspot analysis."""
        logger.info("Generating hotspot analysis...")
        
        # Identify different types of hotspots
        high_comp_zones = self.identify_high_competition_zones()
        underserved_zones = self.identify_underserved_areas()
        grade_clusters = self.identify_grade_clusters()
        
        # Combine all hotspot zones
        self.hotspots = pd.concat([
            high_comp_zones,
            underserved_zones,
            grade_clusters
        ])
        
        # Analyze urban-rural differences
        urban_rural_stats = self.analyze_urban_rural_differences()
        
        # Save outputs
        self.save_outputs(urban_rural_stats)
        
        logger.info("Hotspot analysis completed successfully")
    
    def save_outputs(self, urban_rural_stats: pd.DataFrame) -> None:
        """Save analysis outputs."""
        logger.info("Saving analysis outputs...")
        
        # Save hotspot zones
        self.hotspots.to_file(
            self.data_dir / "competition_hotspots.gpkg",
            driver="GPKG"
        )
        
        # Save urban-rural statistics
        urban_rural_stats.to_csv(
            self.data_dir / "hotspot_analysis.csv"
        )
        
        # Generate and save report
        self.generate_report(urban_rural_stats)
    
    def generate_report(self, urban_rural_stats: pd.DataFrame) -> None:
        """Generate hotspot analysis report."""
        logger.info("Generating analysis report...")
        
        # Count zones by type
        zone_counts = {
            'high_competition': len(self.clubs[
                (self.clubs['base_competition_index'] > 10.0) &
                (self.clubs['is_urban'] == 1)
            ]),
            'underserved': len(self.clubs[
                (self.clubs['base_competition_index'] < 1.0) &
                (self.clubs['is_urban'] == 0) &
                (self.clubs['sustainability_index'] < 0.3)
            ]),
            'grade_cluster': len(self.clubs[
                (self.clubs['Grade_2024_football_value'] <= 2) |
                (self.clubs['Grade_2024_hurling_value'] <= 2)
            ])
        }
        
        report = [
            "# Hotspot Analysis Report\n",
            "## 1. High Competition Zones\n",
            f"- Identified {zone_counts['high_competition']} high competition clubs",
            "- Average competition index > 10.0",
            "- Focused on urban areas with high population density\n",
            
            "## 2. Underserved Areas\n",
            f"- Identified {zone_counts['underserved']} underserved clubs",
            "- Competition index < 1.0",
            "- Focused on rural areas with high youth potential\n",
            
            "## 3. Grade Clusters\n",
            f"- Identified {zone_counts['grade_cluster']} high-grade clubs",
            "- Focused on Premier Senior and Senior A clubs\n",
            
            "## 4. Urban-Rural Differences\n",
            "### Urban Areas",
            f"- Average competition index: {urban_rural_stats.loc[True, ('base_competition_index', 'mean')]}",
            f"- Average sustainability: {urban_rural_stats.loc[True, ('sustainability_index', 'mean')]}",
            f"- Average development priority: {urban_rural_stats.loc[True, ('development_priority', 'mean')]}\n",
            
            "### Rural Areas",
            f"- Average competition index: {urban_rural_stats.loc[False, ('base_competition_index', 'mean')]}",
            f"- Average sustainability: {urban_rural_stats.loc[False, ('sustainability_index', 'mean')]}",
            f"- Average development priority: {urban_rural_stats.loc[False, ('development_priority', 'mean')]}\n",
            
            "## 5. Development Recommendations\n",
            "### High Competition Zones",
            "- Focus on resource optimization",
            "- Consider facility upgrades",
            "- Implement youth development programs\n",
            
            "### Underserved Areas",
            "- Prioritize infrastructure development",
            "- Focus on youth engagement",
            "- Consider accessibility improvements\n",
            
            "### Grade Clusters",
            "- Support high-grade club development",
            "- Consider talent development programs",
            "- Focus on maintaining competitive standards"
        ]
        
        with open(self.data_dir / "hotspot_report.md", "w") as f:
            f.write("\n".join(report))

def main():
    """Execute hotspot analysis."""
    try:
        analysis = HotspotAnalysis()
        analysis.load_data()
        analysis.generate_hotspot_analysis()
        logger.info("Hotspot analysis completed successfully")
    except Exception as e:
        logger.error(f"Error in hotspot analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 