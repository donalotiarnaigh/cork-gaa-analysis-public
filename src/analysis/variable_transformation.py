import pandas as pd
import numpy as np
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

def load_data() -> pd.DataFrame:
    """Load the SAPS data."""
    logger.info("Loading SAPS data...")
    df = pd.read_csv(DATA_DIR / "cork_sa_saps_joined_guid.csv", low_memory=False)
    logger.info(f"Loaded {len(df)} records")
    return df

def calculate_education_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate education rates using total population as denominator."""
    logger.info("Calculating education rates...")
    
    # Calculate education totals for each level
    basic_education = df['T10_4_NFT'] + df['T10_4_PT'] + df['T10_4_LST']
    secondary_education = df['T10_4_UST'] + df['T10_4_TVT']
    third_level = (
        df['T10_4_ACCAT'] + df['T10_4_HCT'] + df['T10_4_ODNDT'] + 
        df['T10_4_HDPQT'] + df['T10_4_PDT'] + df['T10_4_DT']
    )
    
    # Calculate total education responses
    total_responses = basic_education + secondary_education + third_level
    
    # Calculate rates using total responses as denominator
    df['basic_education_rate'] = basic_education / total_responses
    df['secondary_education_rate'] = secondary_education / total_responses
    df['third_level_rate'] = third_level / total_responses
    
    logger.info("Education rates calculated")
    return df

def calculate_employment_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate employment-related rates."""
    logger.info("Calculating employment rates...")
    
    # Calculate total labor force (employed + unemployed)
    df['labor_force'] = df['T8_1_WT'] + df['T8_1_LFFJT'] + df['T8_1_STUT']
    
    # Calculate working age population (15-64)
    working_age_cols = [col for col in df.columns if any(
        age in col for age in ['15_19', '20_24', '25_29', '30_34', '35_39', 
                              '40_44', '45_49', '50_54', '55_59', '60_64']
    ) and col.endswith('T')]
    df['working_age_pop'] = df[working_age_cols].sum(axis=1)
    
    # Calculate employment rate
    df['employment_rate'] = df['T8_1_WT'] / df['labor_force']
    
    # Calculate labor force participation rate (ensure it doesn't exceed 1)
    df['labor_force_participation_rate'] = np.minimum(
        df['labor_force'] / df['working_age_pop'],
        1.0
    )
    
    # Calculate unemployment rate
    df['unemployment_rate'] = (df['T8_1_LFFJT'] + df['T8_1_STUT']) / df['labor_force']
    
    logger.info("Employment rates calculated")
    return df

def calculate_social_class_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate social class rates."""
    logger.info("Calculating social class rates...")
    
    # Calculate total classified population (excluding Z category)
    df['total_classified'] = df['T9_2_PT'] - df['T9_2_PZ']
    
    # Calculate professional/managerial rate (A + B)
    df['professional_rate'] = (df['T9_2_PA'] + df['T9_2_PB']) / df['total_classified']
    
    # Calculate working class rate (skilled + semi-skilled + unskilled: D + E + F)
    df['working_class_rate'] = (
        df['T9_2_PD'] + df['T9_2_PE'] + df['T9_2_PF']
    ) / df['total_classified']
    
    # Calculate class verification rate
    df['class_verification_rate'] = df['total_classified'] / df['T9_2_PT']
    
    logger.info("Social class rates calculated")
    return df

def calculate_youth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate youth-related rates."""
    logger.info("Calculating youth rates...")
    
    # Calculate total population
    total_pop = df['T1_1AGETT']
    
    # Calculate youth proportion (0-19)
    youth_cols = [col for col in df.columns if any(
        age in col for age in [str(i) for i in range(20)]
    ) and col.endswith('T')]
    df['youth_proportion'] = df[youth_cols].sum(axis=1) / total_pop * 0.01  # Convert to proportion
    
    # Calculate school-age rate (5-18)
    school_age_cols = [col for col in df.columns if any(
        age in col for age in [str(i) for i in range(5, 19)]
    ) and col.endswith('T')]
    df['school_age_rate'] = df[school_age_cols].sum(axis=1) / total_pop * 0.01  # Convert to proportion
    
    # Calculate youth gender ratio (male/female)
    youth_male_cols = [col for col in df.columns if any(
        age in col for age in [str(i) for i in range(20)]
    ) and col.endswith('M')]
    youth_female_cols = [col for col in df.columns if any(
        age in col for age in [str(i) for i in range(20)]
    ) and col.endswith('F')]
    
    df['youth_gender_ratio'] = (
        df[youth_male_cols].sum(axis=1) / df[youth_female_cols].sum(axis=1)
    )
    
    logger.info("Youth rates calculated")
    return df

def validate_rates(df: pd.DataFrame) -> None:
    """Validate calculated rates."""
    logger.info("Validating calculated rates...")
    
    rate_columns = [
        'basic_education_rate', 'secondary_education_rate', 'third_level_rate',
        'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
        'professional_rate', 'working_class_rate', 'class_verification_rate',
        'youth_proportion', 'school_age_rate'
    ]
    
    # Check for rates outside [0,1]
    for col in rate_columns:
        invalid_rates = df[df[col].notna() & ((df[col] < 0) | (df[col] > 1))][col]
        if not invalid_rates.empty:
            logger.warning(f"Found {len(invalid_rates)} invalid values in {col}")
            logger.warning(f"Range: {invalid_rates.min():.3f} to {invalid_rates.max():.3f}")
    
    # Check education rates sum approximately to 1
    education_sum = df[['basic_education_rate', 'secondary_education_rate', 'third_level_rate']].sum(axis=1)
    if not np.allclose(education_sum, 1, rtol=0.1):
        logger.warning("Education rates don't sum to 1 for some areas")
        logger.warning(f"Sum range: {education_sum.min():.3f} to {education_sum.max():.3f}")
    
    logger.info("Rate validation completed")

def main():
    """Main function to run the variable transformations."""
    logger.info("Starting variable transformations...")
    
    # Load data
    df = load_data()
    
    # Calculate rates
    df = calculate_education_rates(df)
    df = calculate_employment_rates(df)
    df = calculate_social_class_rates(df)
    df = calculate_youth_rates(df)
    
    # Validate rates
    validate_rates(df)
    
    # Save transformed data
    output_path = DATA_DIR / "cork_sa_transformed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Transformed data saved to {output_path}")
    
    # Generate summary statistics
    summary_stats = df[[
        'basic_education_rate', 'secondary_education_rate', 'third_level_rate',
        'employment_rate', 'labor_force_participation_rate', 'unemployment_rate',
        'professional_rate', 'working_class_rate', 'class_verification_rate',
        'youth_proportion', 'school_age_rate', 'youth_gender_ratio'
    ]].describe()
    
    # Save summary statistics
    stats_path = ANALYSIS_DIR / "transformed_variables_summary.csv"
    summary_stats.to_csv(stats_path)
    logger.info(f"Summary statistics saved to {stats_path}")
    
    logger.info("Variable transformations completed successfully")

if __name__ == "__main__":
    main() 