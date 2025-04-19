#!/usr/bin/env python3
"""
Script to create a consolidated club demographics file for correlation analysis.
Extracts demographic information from catchment_population_stats.csv and combines 
it with club performance metrics for visualization purposes.
"""

import os
import pandas as pd
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
INPUT_CATCHMENT_PATH = Path('data/interim/catchment_population_stats.csv')
INPUT_BUFFER_PATH = Path('data/interim/buffer_demographics.csv')
INPUT_CLUBS_PATH = Path('data/processed/cork_clubs_transformed.csv')
OUTPUT_PATH = Path('data/processed/club_demographics_processed.csv')

def main():
    """Extract demographic data and create the output file."""
    logger.info("Creating consolidated club demographics file for correlation analysis")
    
    # Load catchment population statistics
    logger.info(f"Loading catchment data from {INPUT_CATCHMENT_PATH}")
    try:
        catchment_df = pd.read_csv(INPUT_CATCHMENT_PATH)
        logger.info(f"Loaded catchment data with {len(catchment_df)} records")
    except FileNotFoundError:
        logger.error(f"Could not find catchment data file: {INPUT_CATCHMENT_PATH}")
        return
    except Exception as e:
        logger.error(f"Error loading catchment data: {str(e)}")
        return
    
    # Load buffer demographics for population density
    logger.info(f"Loading buffer demographic data from {INPUT_BUFFER_PATH}")
    try:
        # Load buffer data and filter to primary tier only
        buffer_df = pd.read_csv(INPUT_BUFFER_PATH)
        primary_buffer_df = buffer_df[buffer_df['tier'] == 'primary']
        
        # Rename club_name to Club and select relevant columns
        primary_buffer_df = primary_buffer_df.rename(columns={'club_name': 'Club'})
        primary_buffer_df = primary_buffer_df[['Club', 'population_density']]
        
        # Get average population density for each club
        primary_buffer_df = primary_buffer_df.groupby('Club').agg({'population_density': 'mean'}).reset_index()
        
        logger.info(f"Loaded buffer data with {len(primary_buffer_df)} unique clubs")
    except FileNotFoundError:
        logger.warning(f"Could not find buffer data file: {INPUT_BUFFER_PATH}")
        primary_buffer_df = pd.DataFrame(columns=['Club', 'population_density'])
    except Exception as e:
        logger.error(f"Error loading buffer data: {str(e)}")
        primary_buffer_df = pd.DataFrame(columns=['Club', 'population_density'])
    
    # Load club data with performance metrics
    logger.info(f"Loading club performance data from {INPUT_CLUBS_PATH}")
    try:
        clubs_df = pd.read_csv(INPUT_CLUBS_PATH)
        logger.info(f"Loaded club data with {len(clubs_df)} records")
    except FileNotFoundError:
        logger.error(f"Could not find club data file: {INPUT_CLUBS_PATH}")
        return
    except Exception as e:
        logger.error(f"Error loading club data: {str(e)}")
        return
    
    # Extract relevant demographic columns from catchment data
    # Rename assigned_club to Club to match the key in clubs_df
    demographic_df = catchment_df.rename(columns={
        'assigned_club': 'Club',
        'third_level_rate': 'education_rate'  # Rename to education_rate for visualization purposes
    })
    
    # Select only the key demographic variables needed for correlation analysis
    demographic_columns = [
        'Club',
        'employment_rate',
        'professional_rate', 
        'education_rate',  # Renamed from third_level_rate
        'youth_proportion'
    ]
    
    # Check which columns actually exist in the data
    available_columns = ['Club']
    for col in demographic_columns[1:]:
        if col in demographic_df.columns:
            available_columns.append(col)
        else:
            logger.warning(f"Column {col} not found in catchment data")
    
    # Extract available columns only
    demographic_df = demographic_df[available_columns]
    
    # Merge with population density from buffer data if available
    if 'population_density' in primary_buffer_df.columns:
        logger.info("Adding population density from buffer data")
        demographic_df = pd.merge(
            demographic_df,
            primary_buffer_df,
            on='Club',
            how='left'
        )
    
    # Choose only the needed performance columns from clubs_df
    performance_columns = [
        'Club',
        'transformed_performance',
        'transformed_football',
        'transformed_hurling',
        'transformed_code_balance'
    ]
    
    # Filter to keep only columns that exist in the dataframe
    clubs_performance_df = clubs_df[
        [col for col in performance_columns if col in clubs_df.columns]
    ]
    
    # Merge with club performance data
    logger.info("Merging demographic data with club performance metrics")
    merged_df = pd.merge(
        clubs_performance_df,
        demographic_df,
        on='Club',
        how='inner'
    )
    
    logger.info(f"Created merged dataset with {len(merged_df)} clubs")
    logger.info(f"Final columns: {list(merged_df.columns)}")
    
    # Save the merged data to the output file
    try:
        merged_df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Successfully saved club demographics data to {OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Error saving output file: {str(e)}")
        return
    
    # Print a summary of the demographic variables in the output file
    demographic_vars = [col for col in merged_df.columns if col in 
                       ['employment_rate', 'professional_rate', 'education_rate', 
                        'youth_proportion', 'population_density']]
    
    logger.info(f"Demographic variables included: {', '.join(demographic_vars)}")
    
    # Log some basic statistics
    for col in demographic_vars:
        if col in merged_df.columns:
            mean_val = merged_df[col].mean()
            min_val = merged_df[col].min()
            max_val = merged_df[col].max()
            logger.info(f"{col}: mean={mean_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
    
    logger.info("Demographics file creation completed")

if __name__ == "__main__":
    main() 