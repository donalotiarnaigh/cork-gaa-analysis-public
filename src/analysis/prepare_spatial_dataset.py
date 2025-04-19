import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define data directories
DATA_DIR = Path("data/processed")
ANALYSIS_DIR = Path("data/analysis")

def load_data() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Load the required datasets."""
    logger.info("Loading datasets...")
    
    # Load joined dataset and remove duplicates
    joined = gpd.read_file(DATA_DIR / "cork_sa_saps_joined_guid.gpkg")
    joined = joined.drop_duplicates(subset=['SA_GUID_2022'], keep='first')
    logger.info(f"Loaded {len(joined)} unique joined records")
    
    # Load transformed variables
    transformed = pd.read_csv(DATA_DIR / "cork_sa_transformed.csv")
    transformed = transformed.drop_duplicates(subset=['SA_GUID_2022'], keep='first')
    logger.info(f"Loaded {len(transformed)} unique transformed records")
    
    return joined, transformed

def select_variables(transformed_df: pd.DataFrame, saps_df: pd.DataFrame) -> pd.DataFrame:
    """Select variables for spatial analysis."""
    logger.info("Selecting variables for spatial analysis...")
    
    # Define raw variables needed for rate calculations
    raw_vars = [
        # Education variables
        'T10_4_NFT', 'T10_4_PT', 'T10_4_LST',  # Basic education
        'T10_4_UST', 'T10_4_TVT',  # Secondary education
        'T10_4_ACCAT', 'T10_4_HCT', 'T10_4_ODNDT',  # Third level
        'T10_4_HDPQT', 'T10_4_PDT', 'T10_4_DT',
        
        # Employment variables
        'T8_1_WT', 'T8_1_LFFJT', 'T8_1_STUT',
        
        # Social class variables
        'T9_2_PT', 'T9_2_PZ', 'T9_2_PA', 'T9_2_PB',
        'T9_2_PD', 'T9_2_PE', 'T9_2_PF',
        
        # Population variables
        'T1_1AGETT'
    ]
    
    # Add age columns for working age, youth, and gender
    age_cols = []
    for age in range(20):  # 0-19 for youth
        age_cols.extend([
            f'T1_1AGE{age}T',  # Total
            f'T1_1AGE{age}M',  # Male
            f'T1_1AGE{age}F'   # Female
        ])
    raw_vars.extend(age_cols)
    
    # Select rate variables from transformed data
    rate_vars = [
        'basic_education_rate', 'secondary_education_rate', 'third_level_rate',
        'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
        'professional_rate', 'working_class_rate', 'class_verification_rate',
        'youth_proportion', 'school_age_rate', 'youth_gender_ratio'
    ]
    
    # Select identifiers and raw variables from SAPS data
    id_vars = ['SA_GUID_2022', 'SA_GUID_2016', 'SA_PUB2022', 'SA_GEOGID_2022']
    selected_vars = id_vars + raw_vars
    
    # Create final dataset with selected variables
    variables = pd.concat([
        transformed_df[rate_vars],
        saps_df[selected_vars]
    ], axis=1)
    
    logger.info(f"Selected {len(rate_vars)} rate variables and {len(raw_vars)} raw variables")
    return variables

def create_spatial_dataset(joined: gpd.GeoDataFrame, variables: pd.DataFrame) -> gpd.GeoDataFrame:
    """Create the spatial dataset with selected variables."""
    logger.info("Creating spatial dataset...")
    
    # Merge transformed variables with joined dataset
    spatial_data = joined.merge(
        variables,
        on='SA_GUID_2022',  # Use SA_GUID_2022 for both sides
        validate='1:1'
    )
    
    logger.info(f"Created spatial dataset with {len(spatial_data)} features")
    return spatial_data

def validate_spatial_dataset(gdf: gpd.GeoDataFrame) -> bool:
    """Validate the spatial dataset."""
    logger.info("Validating spatial dataset...")
    
    # Check for missing values
    missing = gdf.isnull().sum()
    if missing.any():
        logger.warning("Found missing values:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"- {col}: {count} missing values")
    
    # Check for duplicate identifiers
    duplicates = gdf['SA_GUID_2022'].duplicated()
    if duplicates.any():
        logger.error(f"Found {duplicates.sum()} duplicate SA_GUID_2022 values")
        return False
    
    # Check for invalid geometries
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        logger.error(f"Found {invalid.sum()} invalid geometries")
        return False
    
    logger.info("Validation complete")
    return True

def export_datasets(spatial_data: gpd.GeoDataFrame) -> None:
    """Export the spatial dataset in both GeoPackage and CSV formats."""
    logger.info("Exporting datasets...")
    
    # Define key variables to keep in the key-only version
    key_vars = [
        # Rate variables
        'basic_education_rate', 'secondary_education_rate', 'third_level_rate',
        'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
        'professional_rate', 'working_class_rate', 'class_verification_rate',
        'youth_proportion', 'school_age_rate', 'youth_gender_ratio'
    ]
    
    # Export full version
    spatial_data.to_file(DATA_DIR / "cork_sa_analysis_full.gpkg", driver="GPKG")
    spatial_data.drop(columns=['geometry']).to_csv(DATA_DIR / "cork_sa_analysis_full.csv", index=False)
    logger.info(f"Created full dataset with {len(spatial_data)} records")
    
    # Export key-only version (including all identifiers)
    id_vars = [col for col in spatial_data.columns if col.startswith('SA_') and not col.endswith('_rate')]
    key_data = spatial_data[id_vars + key_vars + ['geometry']]
    key_data.to_file(DATA_DIR / "cork_sa_analysis_key.gpkg", driver="GPKG")
    key_data.drop(columns=['geometry']).to_csv(DATA_DIR / "cork_sa_analysis_key.csv", index=False)
    logger.info(f"Created key-only dataset with {len(key_data)} records")

def create_data_dictionary(variables: pd.DataFrame) -> None:
    """Create a data dictionary for the spatial dataset."""
    logger.info("Creating data dictionary...")
    
    with open(DATA_DIR / "cork_sa_analysis_dictionary.md", "w") as f:
        f.write("# Spatial Analysis Dataset Dictionary\n\n")
        
        f.write("## Overview\n")
        f.write("This dataset combines spatial boundaries with key demographic and socioeconomic variables.\n")
        f.write("Two versions are available:\n")
        f.write("- `cork_sa_analysis_full`: Contains all variables including raw data\n")
        f.write("- `cork_sa_analysis_key`: Contains only key variables for analysis\n\n")
        
        f.write("## Identifiers\n")
        f.write("- `SA_GUID_2022`: Unique identifier for 2022 Small Area\n")
        f.write("- `SA_GUID_2016`: Previous identifier from 2016\n")
        f.write("- `SA_PUB2022`: Public identifier for 2022 Small Area\n")
        f.write("- `SA_GEOGID_2022`: Geographic identifier\n\n")
        
        f.write("## Key Variables (in both versions)\n")
        f.write("The following rate variables were calculated from raw counts:\n\n")
        
        # Document rate variables
        f.write("### Education Variables\n")
        f.write("- `basic_education_rate`: Proportion with basic education as highest level\n")
        f.write("- `secondary_education_rate`: Proportion with secondary education as highest level\n")
        f.write("- `third_level_rate`: Proportion with third level education as highest level\n\n")
        
        f.write("### Employment Variables\n")
        f.write("- `employment_rate`: Proportion of labor force employed\n")
        f.write("- `labor_force_participation_rate`: Proportion of working age in labor force\n")
        f.write("- `unemployment_rate`: Proportion of labor force unemployed\n\n")
        
        f.write("### Social Class Variables\n")
        f.write("- `professional_rate`: Proportion in professional/managerial class\n")
        f.write("- `working_class_rate`: Proportion in working class\n")
        f.write("- `class_verification_rate`: Proportion with verified class\n\n")
        
        f.write("### Youth Variables\n")
        f.write("- `youth_proportion`: Proportion aged 0-19\n")
        f.write("- `school_age_rate`: Proportion aged 5-18\n")
        f.write("- `youth_gender_ratio`: Ratio of male to female youth\n\n")
        
        f.write("## Raw Variables (full version only)\n")
        f.write("The full version includes all raw variables used to calculate the rates:\n\n")
        
        # Document raw variables
        f.write("### Education Variables\n")
        f.write("- Basic education: T10_4_NFT, T10_4_PT, T10_4_LST\n")
        f.write("- Secondary education: T10_4_UST, T10_4_TVT\n")
        f.write("- Third level: T10_4_ACCAT, T10_4_HCT, T10_4_ODNDT, T10_4_HDPQT, T10_4_PDT, T10_4_DT\n\n")
        
        f.write("### Employment Variables\n")
        f.write("- T8_1_WT, T8_1_LFFJT, T8_1_STUT\n\n")
        
        f.write("### Social Class Variables\n")
        f.write("- T9_2_PT, T9_2_PZ, T9_2_PA, T9_2_PB, T9_2_PD, T9_2_PE, T9_2_PF\n\n")
        
        f.write("### Population Variables\n")
        f.write("- T1_1AGETT (total population)\n")
        f.write("- Age-specific variables (T1_1AGE{0-19}{T/M/F})\n\n")
        
        f.write("## Data Quality Notes\n")
        f.write("- Some SA_GUID_2022 values correspond to multiple SA_PUB2022 values due to boundary changes between 2016 and 2022\n")
        f.write("- For these cases, the first occurrence was kept as the rate variables should be identical\n")
        f.write("- This affects 54 records (2.4% of the dataset)\n")
        f.write("- Missing values in urban area names (650) indicate rural areas\n")
        f.write("- Split information is only present for areas that were split between 2016 and 2022\n")

def main():
    """Main function to prepare the spatial analysis dataset."""
    logger.info("Starting spatial dataset preparation...")
    
    # Load all required data
    joined, transformed = load_data()
    
    # Select transformed variables
    variables = select_variables(transformed, joined)
    
    # Create spatial dataset
    spatial_data = create_spatial_dataset(joined, variables)
    
    # Validate the dataset
    if not validate_spatial_dataset(spatial_data):
        logger.error("Spatial dataset validation failed")
        return
    
    # Create data dictionary
    create_data_dictionary(variables)
    
    # Export datasets
    export_datasets(spatial_data)
    
    logger.info("Spatial dataset preparation completed successfully")

if __name__ == "__main__":
    main() 