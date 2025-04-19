import pandas as pd
import geopandas as gpd
import os
import logging
from pathlib import Path
import time
from datetime import datetime

def get_project_root():
    """Get the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def setup_logging():
    """Set up logging configuration."""
    project_root = get_project_root()
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'saps_sa_guid_integration.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_spatial_data():
    """Load and prepare the Small Area boundaries spatial data."""
    logging.info("Loading Small Area boundaries...")
    start_time = time.time()
    
    try:
        # Load the shapefile
        sa_gdf = gpd.read_file('data/raw/SA/sa.shp')
        
        # Log CRS information
        logging.info(f"Detected CRS: {sa_gdf.crs}")
        
        # Filter for Cork records
        cork_mask = sa_gdf['COUNTY_ENG'].str.contains('CORK', case=False, na=False)
        sa_gdf = sa_gdf[cork_mask]
        
        # Ensure required columns are present
        required_cols = ['SA_GUID_20', 'COUNTY_ENG', 'SA_PUB2022', 'ED_ENGLISH']
        missing_cols = [col for col in required_cols if col not in sa_gdf.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logging.info(f"Loaded {len(sa_gdf)} Cork Small Areas")
        logging.info(f"Loading time: {time.time() - start_time:.2f} seconds")
        
        return sa_gdf
    except Exception as e:
        logging.error(f"Error loading spatial data: {e}")
        return None

def load_saps_data(file_path):
    """Load and prepare SAPS data."""
    try:
        saps_df = pd.read_csv(file_path)
        logging.info(f"Loaded SAPS data with {len(saps_df)} records")
        logging.info(f"SAPS columns: {list(saps_df.columns)}")
        return saps_df
    except Exception as e:
        logging.error(f"Error loading SAPS data: {str(e)}")
        return None

def load_sa_boundaries(file_path):
    """Load and prepare SA boundary data."""
    try:
        sa_gdf = gpd.read_file(file_path)
        logging.info(f"Loaded SA boundaries with {len(sa_gdf)} records")
        logging.info(f"SA columns: {list(sa_gdf.columns)}")
        return sa_gdf
    except Exception as e:
        logging.error(f"Error loading SA boundaries: {str(e)}")
        return None

def handle_split_areas(sa_gdf):
    """Handle split areas by creating additional records for each split part."""
    logging.info("\nHandling split areas...")
    
    # Find records with split areas
    split_mask = sa_gdf['SA_PUB2022'].str.contains('/', na=False)
    split_areas = sa_gdf[split_mask].copy()
    non_split_areas = sa_gdf[~split_mask].copy()
    
    if len(split_areas) == 0:
        logging.info("No split areas found")
        return sa_gdf
    
    # Process split areas
    split_records = []
    for _, row in split_areas.iterrows():
        sa_id = row['SA_PUB2022']
        parts = sa_id.split('/')
        
        if len(parts) == 2:
            if parts[1].isdigit():  # Format: ID/01, ID/02
                # Use the base ID
                new_row = row.copy()
                new_row['SA_PUB2022'] = parts[0]
                new_row['split_type'] = 'numbered'
                new_row['split_number'] = parts[1]
                split_records.append(new_row)
            else:  # Format: ID1/ID2
                # Create a record for each part
                for part in parts:
                    new_row = row.copy()
                    new_row['SA_PUB2022'] = part
                    new_row['split_type'] = 'full'
                    new_row['split_number'] = None
                    split_records.append(new_row)
        else:  # Format: ID1/ID2/ID3
            # Create a record for each part
            for part in parts:
                new_row = row.copy()
                new_row['SA_PUB2022'] = part
                new_row['split_type'] = 'full'
                new_row['split_number'] = None
                split_records.append(new_row)
    
    # Convert split records to DataFrame
    if split_records:
        split_df = pd.DataFrame(split_records)
        
        # Log split area statistics
        logging.info(f"Found {len(split_areas)} split areas")
        logging.info(f"Created {len(split_records)} additional records")
        
        split_types = split_df['split_type'].value_counts()
        logging.info("\nSplit types:")
        for split_type, count in split_types.items():
            logging.info(f"{split_type}: {count} records")
        
        # Combine with non-split areas
        result_df = pd.concat([non_split_areas, split_df], ignore_index=True)
    else:
        result_df = sa_gdf
    
    return result_df

def standardize_identifier(id_value):
    """Standardize identifier format."""
    if pd.isna(id_value):
        return None
    
    # Convert to string and remove whitespace
    id_str = str(id_value).strip()
    
    # Remove any special characters
    id_str = ''.join(c for c in id_str if c.isalnum())
    
    # Remove leading zeros from the first 3 digits (county code)
    if len(id_str) >= 3:
        county_code = str(int(id_str[:3]))
        rest_of_id = id_str[3:]
        id_str = county_code + rest_of_id
    
    return id_str

def perform_guid_join(sa_gdf, saps_df):
    """Perform GUID-based join between SA boundaries and SAPS data."""
    try:
        # Rename columns to match
        sa_gdf['GUID'] = sa_gdf['SA_GUID_2022']  # Use SA_GUID_2022 as GUID
        saps_df['GUID'] = saps_df['GUID']  # Already has GUID column
        
        # Ensure GUID columns are in the correct format (string, uppercase)
        sa_gdf['GUID'] = sa_gdf['GUID'].astype(str).str.upper()
        saps_df['GUID'] = saps_df['GUID'].astype(str).str.upper()
        
        # Log sample of GUIDs before join
        logging.info("\nSample of SA GUIDs:")
        logging.info(sa_gdf['GUID'].head().tolist())
        logging.info("\nSample of SAPS GUIDs:")
        logging.info(saps_df['GUID'].head().tolist())
        
        # Perform the join
        merged_gdf = sa_gdf.merge(saps_df, on='GUID', how='left')
        logging.info(f"Joined datasets: {len(merged_gdf)} records")
        
        # Log join statistics
        matched = merged_gdf[~merged_gdf['GEOGID'].isna()]
        unmatched = merged_gdf[merged_gdf['GEOGID'].isna()]
        logging.info(f"Matched records: {len(matched)}")
        logging.info(f"Unmatched records: {len(unmatched)}")
        
        # Ensure all original records are preserved
        if len(merged_gdf) != len(sa_gdf):
            logging.warning(f"Join resulted in {len(merged_gdf)} records, expected {len(sa_gdf)}")
            # Add missing records back
            missing_guids = set(sa_gdf['GUID']) - set(merged_gdf['GUID'])
            if missing_guids:
                missing_records = sa_gdf[sa_gdf['GUID'].isin(missing_guids)]
                merged_gdf = pd.concat([merged_gdf, missing_records], ignore_index=True)
                logging.info(f"Added back {len(missing_records)} missing records")
        
        return merged_gdf
    except Exception as e:
        logging.error(f"Error performing GUID join: {str(e)}")
        return None

def analyze_unmatched(merged_gdf):
    """Analyze records that didn't match in the join."""
    try:
        # Check for unmatched records using GEOGID instead of SMALL_AREA
        unmatched = merged_gdf[merged_gdf['GEOGID'].isna()]
        logging.info(f"Found {len(unmatched)} unmatched records")
        
        # Log sample of unmatched records with both GUID and GEOGID
        if len(unmatched) > 0:
            sample = unmatched[['GUID', 'COUNTY', 'ED']].head()
            logging.info("Sample of unmatched records:")
            for _, row in sample.iterrows():
                logging.info(f"GUID: {row['GUID']}, County: {row['COUNTY']}, ED: {row['ED']}")
        
        return unmatched
    except Exception as e:
        logging.error(f"Error analyzing unmatched records: {str(e)}")
        return None

def export_results(merged_gdf, unmatched):
    """Export the joined results and analysis."""
    try:
        # Save joined dataset in CSV format
        csv_file = os.path.join(get_project_root(), "data/processed/cork_sa_saps_joined_guid.csv")
        merged_gdf.to_csv(csv_file, index=False)
        logging.info(f"Saved joined dataset (CSV) to {csv_file}")
        
        # Save joined dataset in GeoPackage format with CRS
        gpkg_file = os.path.join(get_project_root(), "data/processed/cork_sa_saps_joined_guid.gpkg")
        # Ensure CRS is set to Irish Transverse Mercator (EPSG:2157)
        if merged_gdf.crs is None:
            merged_gdf.set_crs(epsg=2157, inplace=True)
        merged_gdf.to_file(gpkg_file, driver='GPKG')
        logging.info(f"Saved joined dataset (GeoPackage) to {gpkg_file}")
        
        # Save unmatched analysis
        if unmatched is not None:
            unmatched_file = os.path.join(get_project_root(), "data/processed/unmatched_records.csv")
            unmatched.to_csv(unmatched_file, index=False)
            logging.info(f"Saved unmatched analysis to {unmatched_file}")
    except Exception as e:
        logging.error(f"Error exporting results: {str(e)}")

def main():
    # Set up logging
    setup_logging()
    
    # Define file paths relative to project root
    project_root = get_project_root()
    saps_file = os.path.join(project_root, "data/processed/saps_data_cleaned.csv")
    sa_file = os.path.join(project_root, "data/processed/sa_boundaries_final.gpkg")
    output_file = os.path.join(project_root, "data/processed/cork_sa_saps_joined_guid.csv")
    
    # Load and process data
    saps_df = load_saps_data(saps_file)
    sa_gdf = load_sa_boundaries(sa_file)
    
    if saps_df is None or sa_gdf is None:
        logging.error("Failed to load required datasets")
        return
    
    # Handle split areas before join
    sa_gdf = handle_split_areas(sa_gdf)
    
    # Join datasets
    joined_df = perform_guid_join(sa_gdf, saps_df)
    
    if joined_df is None:
        logging.error("Failed to perform join")
        return
    
    # Save results
    export_results(joined_df, analyze_unmatched(joined_df))
    
    # Analyze unmatched records
    analyze_unmatched(joined_df)

if __name__ == "__main__":
    main() 