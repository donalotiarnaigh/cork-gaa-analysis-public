import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import logging
from pathlib import Path
import sys
from datetime import datetime

# Set up logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'nearest_club_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_data(input_file):
    """Load and validate input data."""
    try:
        # Load Small Areas data
        logging.info(f"Loading Small Areas data from {input_file}...")
        sa_gdf = gpd.read_file(input_file)
        
        # Load club locations
        logging.info("Loading club locations...")
        clubs_gdf = gpd.read_file('data/processed/cork_clubs_complete.gpkg')
        
        # Validate CRS
        if sa_gdf.crs != 'EPSG:2157':
            logging.error(f"Small Areas CRS is {sa_gdf.crs}, expected EPSG:2157")
            raise ValueError("Invalid Small Areas CRS")
            
        # Transform club locations to match Small Areas CRS
        logging.info("Transforming club locations to EPSG:2157...")
        clubs_gdf = clubs_gdf.to_crs('EPSG:2157')
        
        # Validate data completeness
        if len(sa_gdf) == 0:
            raise ValueError("Small Areas dataset is empty")
        if len(clubs_gdf) == 0:
            raise ValueError("Club locations dataset is empty")
            
        logging.info(f"Loaded {len(sa_gdf)} Small Areas and {len(clubs_gdf)} club locations")
        return sa_gdf, clubs_gdf
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def calculate_centroids(sa_gdf):
    """Calculate centroids for Small Areas."""
    try:
        logging.info("Calculating Small Area centroids...")
        centroids = sa_gdf.copy()
        centroids['geometry'] = centroids.geometry.centroid
        return centroids
    except Exception as e:
        logging.error(f"Error calculating centroids: {str(e)}")
        raise

def find_nearest_club(point, clubs_gdf):
    """Find the nearest club to a given point."""
    try:
        # Calculate distances to all clubs
        distances = clubs_gdf.geometry.distance(point)
        
        # Find the index of the nearest club
        nearest_idx = distances.idxmin()
        
        # Get the nearest club data
        nearest_club_data = clubs_gdf.loc[nearest_idx]
        
        # Create a dictionary with all club attributes
        club_data = {
            'nearest_club': nearest_club_data['Club'],
            'distance_meters': distances[nearest_idx],
            'club_latitude': nearest_club_data['Latitude'],
            'club_longitude': nearest_club_data['Longitude'],
            'club_elevation': nearest_club_data['Elevation'],
            'club_annual_rainfall': nearest_club_data['annual_rainfall'],
            'club_rain_days': nearest_club_data['rain_days'],
            'club_grade_2022_football': nearest_club_data['Grade_2022_football'],
            'club_grade_2022_hurling': nearest_club_data['Grade_2022_hurling'],
            'club_grade_2024_football': nearest_club_data['Grade_2024_football'],
            'club_grade_2024_hurling': nearest_club_data['Grade_2024_hurling'],
            'club_grade_2022_football_value': nearest_club_data['Grade_2022_football_value'],
            'club_grade_2022_hurling_value': nearest_club_data['Grade_2022_hurling_value'],
            'club_grade_2024_football_value': nearest_club_data['Grade_2024_football_value'],
            'club_grade_2024_hurling_value': nearest_club_data['Grade_2024_hurling_value'],
            'club_combined_2022': nearest_club_data['combined_2022'],
            'club_combined_2024': nearest_club_data['combined_2024'],
            'club_football_improvement': nearest_club_data['football_improvement'],
            'club_hurling_improvement': nearest_club_data['hurling_improvement'],
            'club_overall_performance': nearest_club_data['overall_performance'],
            'club_football_performance': nearest_club_data['football_performance'],
            'club_hurling_performance': nearest_club_data['hurling_performance'],
            'club_code_balance': nearest_club_data['code_balance'],
            'club_is_dual_2022': nearest_club_data['is_dual_2022'],
            'club_is_dual_2024': nearest_club_data['is_dual_2024']
        }
        
        return club_data
        
    except Exception as e:
        logging.error(f"Error finding nearest clubs: {str(e)}")
        raise

def export_results(sa_gdf, results_df, output_prefix):
    """Export results to GeoPackage and CSV formats."""
    try:
        # Merge results with original Small Areas
        output_gdf = sa_gdf.merge(results_df, on='SA_GUID_2022')
        
        # Export to GeoPackage
        output_gpkg = f"data/processed/{output_prefix}.gpkg"
        output_gdf.to_file(output_gpkg, driver="GPKG")
        logging.info(f"Exported results to {output_gpkg}")
        
        # Export to CSV (excluding geometry)
        output_csv = f"data/processed/{output_prefix}.csv"
        output_gdf.drop(columns=['geometry']).to_csv(output_csv, index=False)
        logging.info(f"Exported results to {output_csv}")
        
        # Generate analysis report only for the key dataset
        if output_prefix == "cork_clubs_nearest_key":
            generate_analysis_report(output_gdf, output_prefix)
        
    except Exception as e:
        logging.error(f"Error exporting results: {str(e)}")
        raise

def generate_analysis_report(gdf, output_prefix):
    """Generate analysis report for the nearest club assignment."""
    try:
        summary = f"""# Nearest Club Assignment Analysis Report

## Summary Statistics
- Total Small Areas: {len(gdf)}
- Total Clubs: {gdf['nearest_club'].nunique()}
- Average Distance: {gdf['distance_meters'].mean():.2f} meters
- Maximum Distance: {gdf['distance_meters'].max():.2f} meters
- Minimum Distance: {gdf['distance_meters'].min():.2f} meters

## Club Assignments
{gdf['nearest_club'].value_counts().to_string()}

## Distance Distribution
{gdf['distance_meters'].describe().to_string()}
"""
        
        with open(f'data/analysis/nearest_assignment_report.md', 'w') as f:
            f.write(summary)
        logging.info("Generated analysis report")
        
    except Exception as e:
        logging.error(f"Error generating analysis report: {str(e)}")
        raise

def process_dataset(input_file, output_prefix):
    """Process a single dataset."""
    try:
        # Load data
        sa_gdf, clubs_gdf = load_data(input_file)
        
        # Calculate centroids
        sa_gdf['centroid'] = sa_gdf.geometry.centroid
        
        # Find nearest clubs
        results = []
        for idx, row in sa_gdf.iterrows():
            club_data = find_nearest_club(row['centroid'], clubs_gdf)
            results.append({
                'SA_GUID_2022': row['SA_GUID_2022'],
                **club_data  # Unpack all club attributes
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Drop centroid column before exporting
        sa_gdf = sa_gdf.drop(columns=['centroid'])
        
        # Export results
        export_results(sa_gdf, results_df, output_prefix)
        
    except Exception as e:
        logging.error(f"Error processing {input_file}: {str(e)}")
        raise

def main():
    """Main function to execute the analysis."""
    try:
        logging.info("Starting nearest club analysis...")
        
        # Process both datasets
        process_dataset('data/processed/cork_sa_analysis_full.gpkg', 'cork_clubs_nearest_full')
        process_dataset('data/processed/cork_sa_analysis_key.gpkg', 'cork_clubs_nearest_key')
        
        logging.info("Nearest club analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 