import geopandas as gpd
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to logger
logger.addHandler(ch)

def validate_assignments(gdf):
    """Validate the assignments GeoDataFrame."""
    # Check for null values in key fields
    null_values = gdf['assigned_club'].isnull().sum()
    if null_values > 0:
        logger.warning(f"Found {null_values} null values in assigned_club field")
    
    # Check area proportions
    invalid_proportions = gdf[~gdf['area_proportion'].between(0, 1)].shape[0]
    if invalid_proportions > 0:
        logger.warning(f"Found {invalid_proportions} invalid area proportions")
    
    # Log validation results
    logger.info(f"Assignments validation complete: {len(gdf)} total records")
    return True

def validate_clubs(gdf):
    """Validate the clubs GeoDataFrame."""
    # Check for unique clubs
    unique_clubs = gdf['Club'].nunique()
    logger.info(f"Found {unique_clubs} unique clubs")
    
    # Check for required columns
    required_cols = ['Club', 'Grade_2024_football', 'Grade_2024_hurling']
    missing_cols = [col for col in required_cols if col not in gdf.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
    
    return True

def load_and_validate_data():
    """Load and validate input data."""
    try:
        # Load assignments data
        logger.info("Loading assignments data...")
        assignments_gdf = gpd.read_file('data/processed/sa_club_assignments.gpkg')
        
        # Load clubs data
        logger.info("Loading clubs data...")
        clubs_gdf = gpd.read_file('data/processed/cork_clubs_complete.gpkg')
        
        # Load full census data
        logger.info("Loading full census data...")
        census_gdf = gpd.read_file('data/processed/cork_sa_analysis_full.gpkg')
        
        # Print column names for debugging
        logger.info("Census data columns:")
        logger.info(census_gdf.columns.tolist())
        
        # Validate assignments data
        logger.info("Validating assignments data...")
        validate_assignments(assignments_gdf)
        
        # Validate clubs data
        logger.info("Validating clubs data...")
        validate_clubs(clubs_gdf)
        
        # Join assignments with census data
        logger.info("Joining assignments with census data...")
        # First, merge census data with assignments, excluding geometry from census
        merged_gdf = assignments_gdf.merge(
            census_gdf.drop(columns=['geometry']),
            on='SA_GUID_2022',
            how='left',
            validate='m:1'
        )
        
        # Print column names after merge for debugging
        logger.info("Merged data columns:")
        logger.info(merged_gdf.columns.tolist())
        
        # Calculate weighted population for split areas
        merged_gdf['weighted_population'] = merged_gdf['T1_1AGETT_x'] * merged_gdf['area_proportion']
        
        # Log basic statistics
        logger.info(f"Total assignments: {len(merged_gdf)}")
        logger.info(f"Unique clubs: {merged_gdf['assigned_club'].nunique()}")
        logger.info(f"Unique Small Areas: {merged_gdf['SA_GUID_2022'].nunique()}")
        logger.info(f"Total population: {merged_gdf['weighted_population'].sum():,.0f}")
        
        return merged_gdf, clubs_gdf
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def calculate_population_metrics(merged_gdf):
    """Calculate population metrics by club."""
    try:
        logger.info("Calculating population metrics...")
        
        # Group by club and calculate metrics
        club_metrics = merged_gdf.groupby('assigned_club').agg({
            'weighted_population': 'sum',
            'SA_GUID_2022': 'nunique',
            'area_proportion': 'sum'
        }).reset_index()
        
        # Rename columns for clarity
        club_metrics = club_metrics.rename(columns={
            'weighted_population': 'total_population',
            'SA_GUID_2022': 'num_small_areas',
            'area_proportion': 'total_area'
        })
        
        # Calculate population density (population per small area)
        club_metrics['population_density'] = club_metrics['total_population'] / club_metrics['num_small_areas']
        
        # Sort by population
        club_metrics = club_metrics.sort_values('total_population', ascending=False)
        
        # Save to CSV
        output_path = 'data/interim/catchment_population_stats.csv'
        club_metrics.to_csv(output_path, index=False)
        logger.info(f"Population metrics saved to {output_path}")
        
        return club_metrics
        
    except Exception as e:
        logger.error(f"Error calculating population metrics: {str(e)}")
        raise

def generate_population_report(club_metrics):
    """Generate population analysis report."""
    try:
        logger.info("Generating population analysis report...")
        
        # Calculate overall statistics
        total_clubs = len(club_metrics)
        total_population = club_metrics['total_population'].sum()
        avg_population = club_metrics['total_population'].mean()
        median_population = club_metrics['total_population'].median()
        avg_density = club_metrics['population_density'].mean()
        
        # Format tables
        def format_table(df, column_map, formatters=None):
            """Format DataFrame as markdown table."""
            if formatters is None:
                formatters = {}
            
            # Create display columns
            display_cols = list(column_map.values())
            data_cols = list(column_map.keys())
            
            # Format the data
            formatted_data = df[data_cols].copy()
            for col, fmt in formatters.items():
                if col in formatted_data.columns:
                    formatted_data[col] = formatted_data[col].apply(fmt)
            
            # Create markdown table
            table = "| " + " | ".join(display_cols) + " |\n"
            table += "| " + " | ".join(["---"] * len(display_cols)) + " |\n"
            
            for _, row in formatted_data.head().iterrows():
                table += "| " + " | ".join(str(row[col]) for col in data_cols) + " |\n"
            
            return table
        
        # Create report content
        report_content = f"""# Population Analysis Report

## Overall Distribution
- Total Clubs: {total_clubs}
- Total Population: {total_population:,.0f}
- Average Population per Club: {avg_population:,.0f}
- Median Population per Club: {median_population:,.0f}
- Average Population Density: {avg_density:,.1f} people per Small Area

## Top 5 Clubs by Population
{format_table(
    club_metrics,
    {
        'assigned_club': 'Club',
        'total_population': 'Population',
        'num_small_areas': 'Small Areas',
        'population_density': 'Density'
    },
    {
        'total_population': lambda x: f"{x:,.0f}",
        'population_density': lambda x: f"{x:.1f}"
    }
)}

## Top 5 Clubs by Population Density
{format_table(
    club_metrics.sort_values('population_density', ascending=False),
    {
        'assigned_club': 'Club',
        'population_density': 'Density',
        'total_population': 'Population',
        'num_small_areas': 'Small Areas'
    },
    {
        'total_population': lambda x: f"{x:,.0f}",
        'population_density': lambda x: f"{x:.1f}"
    }
)}
"""
        
        # Save report
        output_path = 'data/interim/population_analysis_report.md'
        with open(output_path, 'w') as f:
            f.write(report_content)
        logger.info(f"Population analysis report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating population report: {str(e)}")
        raise

def main():
    """Main function to run population analysis."""
    try:
        # Load and validate data
        merged_gdf, clubs_gdf = load_and_validate_data()
        
        # Calculate population metrics
        club_metrics = calculate_population_metrics(merged_gdf)
        
        # Generate report
        generate_population_report(club_metrics)
        
        logger.info("Population analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in population analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 