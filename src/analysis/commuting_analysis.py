import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define data directories
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = DATA_DIR / "analysis"

def load_data():
    """Load the SAPS data."""
    logger.info("Loading SAPS data...")
    df = pd.read_csv(PROCESSED_DIR / "cork_sa_saps_joined_guid.csv")
    logger.info(f"Loaded {len(df)} records from SAPS data file")
    return df

def calculate_commuting_metrics(df):
    """Calculate commuting-related metrics."""
    logger.info("Calculating commuting metrics...")
    
    # Calculate mode of transport percentages for work
    work_transport_cols = [
        'T11_1_FW', 'T11_1_BIW', 'T11_1_BUW', 'T11_1_TDLW', 
        'T11_1_MW', 'T11_1_CDW', 'T11_1_CPW', 'T11_1_VW', 
        'T11_1_OTHW', 'T11_1_WMFHW'
    ]
    
    # Calculate journey time percentages
    journey_time_cols = [
        'T11_3_D1', 'T11_3_D2', 'T11_3_D3', 
        'T11_3_D4', 'T11_3_D5', 'T11_3_D6'
    ]
    
    # Calculate departure time percentages
    departure_time_cols = [
        'T11_2_T1', 'T11_2_T2', 'T11_2_T3', 
        'T11_2_T4', 'T11_2_T5', 'T11_2_T6', 
        'T11_2_T7', 'T11_2_T8'
    ]
    
    # Calculate work from home percentage
    wfh_cols = ['T11_4_WFH', 'T11_4_NWFH']
    
    metrics = {}
    
    # Calculate transport mode percentages
    for col in work_transport_cols:
        metrics[f'{col}_pct'] = df[col].sum() / df['T11_1_TW'].sum()
    
    # Calculate journey time percentages
    for col in journey_time_cols:
        metrics[f'{col}_pct'] = df[col].sum() / df['T11_3_T'].sum()
    
    # Calculate departure time percentages
    for col in departure_time_cols:
        metrics[f'{col}_pct'] = df[col].sum() / df['T11_2_T'].sum()
    
    # Calculate work from home percentage
    metrics['wfh_pct'] = df['T11_4_WFH'].sum() / df['T11_4_T'].sum()
    
    # Calculate long commute percentage (over 45 minutes)
    long_commute_cols = ['T11_3_D4', 'T11_3_D5', 'T11_3_D6']
    metrics['long_commute_pct'] = df[long_commute_cols].sum().sum() / df['T11_3_T'].sum()
    
    # Calculate early departure percentage (before 7:30)
    early_departure_cols = ['T11_2_T1', 'T11_2_T2', 'T11_2_T3']
    metrics['early_departure_pct'] = df[early_departure_cols].sum().sum() / df['T11_2_T'].sum()
    
    return metrics

def generate_report(metrics):
    """Generate a markdown report of the commuting analysis."""
    logger.info("Generating commuting analysis report...")
    
    report = """# Commuting Patterns Analysis

## Summary Statistics

### Mode of Transport to Work
- Walking: {:.1%}
- Cycling: {:.1%}
- Public Transport (Bus): {:.1%}
- Public Transport (Train/DART/LUAS): {:.1%}
- Motorcycle/Scooter: {:.1%}
- Car Driver: {:.1%}
- Car Passenger: {:.1%}
- Van: {:.1%}
- Other: {:.1%}
- Work from Home: {:.1%}

### Journey Times
- Under 15 minutes: {:.1%}
- 15-30 minutes: {:.1%}
- 30-45 minutes: {:.1%}
- 45-60 minutes: {:.1%}
- 1-1.5 hours: {:.1%}
- Over 1.5 hours: {:.1%}
- Long Commutes (>45 mins): {:.1%}

### Departure Times
- Before 6:30: {:.1%}
- 6:30-7:00: {:.1%}
- 7:01-7:30: {:.1%}
- 7:31-8:00: {:.1%}
- 8:01-8:30: {:.1%}
- 8:31-9:00: {:.1%}
- 9:01-9:30: {:.1%}
- After 9:30: {:.1%}
- Early Departures (<7:30): {:.1%}

## Impact on GAA Participation

The commuting patterns in the region have several implications for GAA participation:

1. **Time Availability**
   - {:.1%} of commuters have long journeys (>45 minutes)
   - {:.1%} leave early in the morning (<7:30)
   - These patterns may limit evening participation in training and matches

2. **Transport Mode**
   - High car dependency ({:.1%} as drivers, {:.1%} as passengers)
   - Limited public transport usage ({:.1%} by bus, {:.1%} by train)
   - This may affect ability to attend matches in different locations

3. **Work-Life Balance**
   - {:.1%} work from home, potentially offering more flexibility
   - Early departures and long commutes may impact evening training sessions
   - Weekend matches may be more accessible than weekday activities

## Recommendations

1. **Training Schedule Flexibility**
   - Consider later evening training times for areas with high early departure rates
   - Offer weekend training options for those with long commutes

2. **Transport Support**
   - Organize carpooling for matches and training
   - Coordinate with public transport schedules for major matches

3. **Location Planning**
   - Consider transport accessibility when planning new facilities
   - Focus on local community engagement where commuting patterns are most challenging
""".format(
        metrics['T11_1_FW_pct'],
        metrics['T11_1_BIW_pct'],
        metrics['T11_1_BUW_pct'],
        metrics['T11_1_TDLW_pct'],
        metrics['T11_1_MW_pct'],
        metrics['T11_1_CDW_pct'],
        metrics['T11_1_CPW_pct'],
        metrics['T11_1_VW_pct'],
        metrics['T11_1_OTHW_pct'],
        metrics['T11_1_WMFHW_pct'],
        metrics['T11_3_D1_pct'],
        metrics['T11_3_D2_pct'],
        metrics['T11_3_D3_pct'],
        metrics['T11_3_D4_pct'],
        metrics['T11_3_D5_pct'],
        metrics['T11_3_D6_pct'],
        metrics['long_commute_pct'],
        metrics['T11_2_T1_pct'],
        metrics['T11_2_T2_pct'],
        metrics['T11_2_T3_pct'],
        metrics['T11_2_T4_pct'],
        metrics['T11_2_T5_pct'],
        metrics['T11_2_T6_pct'],
        metrics['T11_2_T7_pct'],
        metrics['T11_2_T8_pct'],
        metrics['early_departure_pct'],
        metrics['long_commute_pct'],
        metrics['early_departure_pct'],
        metrics['T11_1_CDW_pct'],
        metrics['T11_1_CPW_pct'],
        metrics['T11_1_BUW_pct'],
        metrics['T11_1_TDLW_pct'],
        metrics['wfh_pct']
    )
    
    # Save report
    report_path = ANALYSIS_DIR / "commuting_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved commuting analysis report to {report_path}")

def create_visualizations(df, metrics):
    """Create visualizations for the commuting analysis."""
    logger.info("Creating commuting analysis visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')  # Using a built-in style that mimics seaborn
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Mode of Transport
    ax1 = fig.add_subplot(gs[0, 0])
    transport_data = {
        'Walking': metrics['T11_1_FW_pct'],
        'Cycling': metrics['T11_1_BIW_pct'],
        'Bus': metrics['T11_1_BUW_pct'],
        'Train/DART': metrics['T11_1_TDLW_pct'],
        'Car Driver': metrics['T11_1_CDW_pct'],
        'Car Passenger': metrics['T11_1_CPW_pct'],
        'Work from Home': metrics['T11_1_WMFHW_pct']
    }
    ax1.bar(transport_data.keys(), transport_data.values())
    ax1.set_title('Mode of Transport to Work')
    ax1.set_ylabel('Percentage')
    plt.xticks(rotation=45)
    
    # 2. Journey Times
    ax2 = fig.add_subplot(gs[0, 1])
    journey_data = {
        '<15min': metrics['T11_3_D1_pct'],
        '15-30min': metrics['T11_3_D2_pct'],
        '30-45min': metrics['T11_3_D3_pct'],
        '45-60min': metrics['T11_3_D4_pct'],
        '1-1.5hr': metrics['T11_3_D5_pct'],
        '>1.5hr': metrics['T11_3_D6_pct']
    }
    ax2.bar(journey_data.keys(), journey_data.values())
    ax2.set_title('Journey Times to Work')
    ax2.set_ylabel('Percentage')
    
    # 3. Departure Times
    ax3 = fig.add_subplot(gs[1, 0])
    departure_data = {
        '<6:30': metrics['T11_2_T1_pct'],
        '6:30-7:00': metrics['T11_2_T2_pct'],
        '7:01-7:30': metrics['T11_2_T3_pct'],
        '7:31-8:00': metrics['T11_2_T4_pct'],
        '8:01-8:30': metrics['T11_2_T5_pct'],
        '8:31-9:00': metrics['T11_2_T6_pct'],
        '>9:00': metrics['T11_2_T7_pct'] + metrics['T11_2_T8_pct']
    }
    ax3.bar(departure_data.keys(), departure_data.values())
    ax3.set_title('Departure Times for Work')
    ax3.set_ylabel('Percentage')
    plt.xticks(rotation=45)
    
    # 4. Work from Home vs Commuting
    ax4 = fig.add_subplot(gs[1, 1])
    wfh_data = {
        'Work from Home': metrics['wfh_pct'],
        'Long Commute (>45min)': metrics['long_commute_pct'],
        'Early Departure (<7:30)': metrics['early_departure_pct']
    }
    ax4.bar(wfh_data.keys(), wfh_data.values())
    ax4.set_title('Work from Home and Challenging Commutes')
    ax4.set_ylabel('Percentage')
    plt.xticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "commuting_analysis.png")
    logger.info("Saved commuting analysis visualizations")

def main():
    """Main function to run the commuting analysis."""
    logger.info("Starting commuting analysis...")
    
    # Load data
    df = load_data()
    
    # Calculate metrics
    metrics = calculate_commuting_metrics(df)
    
    # Generate report
    generate_report(metrics)
    
    # Create visualizations
    create_visualizations(df, metrics)
    
    logger.info("Commuting analysis completed successfully")

if __name__ == "__main__":
    main() 