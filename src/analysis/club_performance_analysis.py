import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up paths
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = WORKSPACE_ROOT / 'data'
OUTPUT_DIR = DATA_DIR / 'analysis'

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the club performance data."""
    df = pd.read_csv(DATA_DIR / 'processed' / 'cork_clubs_complete_graded.csv')
    
    # Convert performance metrics to 7-grade scale where higher is better
    # Original: 1 (Premier Senior) to 6 (NA) where lower is better
    # New: 6 (Premier Senior) to 1 (NA) where higher is better
    grade_columns = [
        'Grade_2022_football_value', 'Grade_2022_hurling_value',
        'Grade_2024_football_value', 'Grade_2024_hurling_value',
        'football_performance', 'hurling_performance'
    ]
    
    for col in grade_columns:
        if col in df.columns:
            df[col] = 7 - df[col]  # Convert 1-6 to 6-1
            
    # Recalculate derived metrics
    df['combined_2022'] = (7 - df['combined_2022']) if 'combined_2022' in df.columns else None
    df['combined_2024'] = (7 - df['combined_2024']) if 'combined_2024' in df.columns else None
    df['overall_performance'] = 7 - df['overall_performance']
    
    # Note: football_improvement and hurling_improvement remain unchanged
    # as they represent relative changes
    
    return df

def analyze_grade_distribution(df):
    """Analyze the distribution of grades across clubs."""
    # Create grade mapping for display
    grade_map = {
        'NA': 1,
        'Junior A': 2,
        'Premier Junior': 2,
        'Intermediate A': 3,
        'Premier Intermediate': 4,
        'Senior A': 5,
        'Premier Senior': 6
    }
    
    # Count grades for each code
    football_grades = df['Grade_2024_football'].value_counts().sort_index()
    hurling_grades = df['Grade_2024_hurling'].value_counts().sort_index()
    
    # Calculate percentages
    total_clubs = len(df)
    football_pct = (football_grades / total_clubs * 100).round(2)
    hurling_pct = (hurling_grades / total_clubs * 100).round(2)
    
    return {
        'football_grades': football_grades,
        'hurling_grades': hurling_grades,
        'football_pct': football_pct,
        'hurling_pct': hurling_pct
    }

def analyze_performance_metrics(df):
    """Analyze club performance metrics."""
    metrics = {
        'total_clubs': len(df),
        'dual_clubs_2024': df['is_dual_2024'].sum(),
        'dual_clubs_2022': df['is_dual_2024'].sum(),
        'avg_overall_performance': df['overall_performance'].mean(),
        'avg_code_balance': df['code_balance'].mean(),
        'improving_clubs': (df['football_improvement'] < 0).sum() + (df['hurling_improvement'] < 0).sum(),
        'declining_clubs': (df['football_improvement'] > 0).sum() + (df['hurling_improvement'] > 0).sum(),
        'stable_clubs': (df['football_improvement'] == 0).sum() + (df['hurling_improvement'] == 0).sum()
    }
    
    return metrics

def generate_performance_report(df, grade_dist, metrics):
    """Generate a comprehensive performance report."""
    report = f"""# Cork GAA Club Performance Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Grading System
1. Premier Senior (Best) - Performance Value: 6
2. Senior A - Performance Value: 5
3. Premier Intermediate - Performance Value: 4
4. Intermediate A - Performance Value: 3
5. Premier Junior/Junior A - Performance Value: 2
6. NA (No Grade) - Performance Value: 1

## Overview
- Total Clubs: {metrics['total_clubs']}
- Dual Clubs (2024): {metrics['dual_clubs_2024']} ({metrics['dual_clubs_2024']/metrics['total_clubs']*100:.1f}%)
- Average Overall Performance: {metrics['avg_overall_performance']:.2f} (higher is better)
- Average Code Balance: {metrics['avg_code_balance']:.2f}

## Grade Distribution (2024)

### Football
{grade_dist['football_grades'].to_string()}
Percentages:
{grade_dist['football_pct'].to_string()}

### Hurling
{grade_dist['hurling_grades'].to_string()}
Percentages:
{grade_dist['hurling_pct'].to_string()}

## Performance Trends
- Improving Clubs: {metrics['improving_clubs']}
- Declining Clubs: {metrics['declining_clubs']}
- Stable Clubs: {metrics['stable_clubs']}

## Top Performing Clubs (2024)
### Overall Performance (Higher is Better)
{df.nlargest(5, 'overall_performance')[['Club', 'overall_performance', 'football_performance', 'hurling_performance', 'is_dual_2024']].to_string()}

### Most Balanced Clubs (Lowest Code Balance)
{df.nsmallest(5, 'code_balance')[['Club', 'code_balance', 'football_performance', 'hurling_performance', 'is_dual_2024']].to_string()}

## Dual Club Analysis
### Dual Clubs Performance
{df[df['is_dual_2024']].describe()[['overall_performance', 'code_balance']].to_string()}

### Single Code Clubs Performance
{df[~df['is_dual_2024']].describe()[['overall_performance', 'code_balance']].to_string()}
"""
    
    return report

def create_visualizations(df):
    """Create performance visualizations."""
    # Set style
    plt.style.use('default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Grade Distribution
    grade_counts = pd.DataFrame({
        'Football': df['Grade_2024_football'].value_counts().sort_index(),
        'Hurling': df['Grade_2024_hurling'].value_counts().sort_index()
    })
    grade_counts.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Grade Distribution by Code')
    axes[0,0].set_xlabel('Grade')
    axes[0,0].set_ylabel('Number of Clubs')
    
    # 2. Performance Distribution
    sns.histplot(data=df, x='overall_performance', bins=20, ax=axes[0,1])
    axes[0,1].set_title('Overall Performance Distribution (Higher is Better)')
    axes[0,1].set_xlabel('Overall Performance Score')
    axes[0,1].set_ylabel('Number of Clubs')
    
    # 3. Code Balance Distribution
    sns.histplot(data=df, x='code_balance', bins=20, ax=axes[1,0])
    axes[1,0].set_title('Code Balance Distribution')
    axes[1,0].set_xlabel('Code Balance Score')
    axes[1,0].set_ylabel('Number of Clubs')
    
    # 4. Dual vs Single Code Performance
    sns.boxplot(data=df, x='is_dual_2024', y='overall_performance', ax=axes[1,1])
    axes[1,1].set_title('Performance: Dual vs Single Code Clubs')
    axes[1,1].set_xlabel('Is Dual Club')
    axes[1,1].set_ylabel('Overall Performance Score (Higher is Better)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'club_performance_analysis.png')
    plt.close()

def main():
    """Main function to run the analysis."""
    # Load data
    df = load_data()
    
    # Analyze grade distribution
    grade_dist = analyze_grade_distribution(df)
    
    # Analyze performance metrics
    metrics = analyze_performance_metrics(df)
    
    # Generate report
    report = generate_performance_report(df, grade_dist, metrics)
    
    # Save report
    with open(OUTPUT_DIR / 'club_performance_report.md', 'w') as f:
        f.write(report)
    
    # Create visualizations
    create_visualizations(df)
    
    print("Analysis complete. Check data/analysis/ for outputs.")

if __name__ == "__main__":
    main() 