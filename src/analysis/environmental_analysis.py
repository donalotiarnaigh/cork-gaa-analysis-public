import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

def load_data():
    """Load the club data with environmental factors."""
    data_path = os.path.join('data', 'processed', 'cork_clubs_complete_graded.csv')
    df = pd.read_csv(data_path)
    return df

def analyze_environmental_correlations(df):
    """Analyze correlations between environmental factors and club performance."""
    # Calculate correlations with overall performance
    env_correlations = df[['Elevation', 'annual_rainfall', 'rain_days', 
                          'overall_performance', 'football_performance', 'hurling_performance']].corr()
    
    # Calculate correlations with grade changes
    grade_changes = df[['football_improvement', 'hurling_improvement', 
                        'Elevation', 'annual_rainfall', 'rain_days']].corr()
    
    return env_correlations, grade_changes

def analyze_grade_distribution_by_environment(df):
    """Analyze grade distribution across different environmental conditions."""
    # Create environmental categories
    df['elevation_category'] = pd.qcut(df['Elevation'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    df['rainfall_category'] = pd.qcut(df['annual_rainfall'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    df['rain_days_category'] = pd.qcut(df['rain_days'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    
    # Calculate average performance by category
    elevation_performance = df.groupby('elevation_category')['overall_performance'].mean()
    rainfall_performance = df.groupby('rainfall_category')['overall_performance'].mean()
    rain_days_performance = df.groupby('rain_days_category')['overall_performance'].mean()
    
    return elevation_performance, rainfall_performance, rain_days_performance

def analyze_dual_clubs_environment(df):
    """Analyze environmental factors for dual vs single-code clubs."""
    # Calculate average environmental conditions for dual and single-code clubs
    dual_clubs = df[df['is_dual_2024']]
    single_clubs = df[~df['is_dual_2024']]
    
    env_comparison = pd.DataFrame({
        'Dual Clubs': dual_clubs[['Elevation', 'annual_rainfall', 'rain_days']].mean(),
        'Single Code Clubs': single_clubs[['Elevation', 'annual_rainfall', 'rain_days']].mean()
    })
    
    return env_comparison

def generate_visualizations(df, env_correlations, grade_changes, 
                          elevation_performance, rainfall_performance, rain_days_performance,
                          env_comparison):
    """Generate visualizations for environmental analysis."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join('output', 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(env_correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Environmental Factors Correlation with Performance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'environmental_correlations.png'))
    plt.close()
    
    # 2. Grade Changes Correlation
    plt.figure(figsize=(10, 6))
    sns.heatmap(grade_changes, annot=True, cmap='coolwarm', center=0)
    plt.title('Environmental Factors Correlation with Grade Changes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grade_changes_correlations.png'))
    plt.close()
    
    # 3. Performance by Environmental Category
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    elevation_performance.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Performance by Elevation')
    axes[0].set_xlabel('Elevation Category')
    axes[0].set_ylabel('Average Performance')
    
    rainfall_performance.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Performance by Annual Rainfall')
    axes[1].set_xlabel('Rainfall Category')
    axes[1].set_ylabel('Average Performance')
    
    rain_days_performance.plot(kind='bar', ax=axes[2])
    axes[2].set_title('Performance by Rain Days')
    axes[2].set_xlabel('Rain Days Category')
    axes[2].set_ylabel('Average Performance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_environment.png'))
    plt.close()
    
    # 4. Dual vs Single Code Environmental Comparison
    plt.figure(figsize=(10, 6))
    env_comparison.plot(kind='bar')
    plt.title('Environmental Conditions: Dual vs Single Code Clubs')
    plt.xlabel('Environmental Factor')
    plt.ylabel('Average Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dual_vs_single_environment.png'))
    plt.close()

def generate_report(df, env_correlations, grade_changes, 
                   elevation_performance, rainfall_performance, rain_days_performance,
                   env_comparison):
    """Generate a markdown report of the environmental analysis."""
    report = f"""# Environmental Factors Impact on GAA Club Performance

## Executive Summary
This report analyzes the relationship between environmental factors (elevation, rainfall, and rain days) and GAA club performance in Cork. The analysis examines correlations with overall performance, grade changes, and differences between dual and single-code clubs.

## Correlation Analysis

### Overall Performance Correlations
{env_correlations['overall_performance'].to_string()}

### Grade Changes Correlations
{grade_changes['football_improvement'].to_string()}

## Performance by Environmental Category

### Elevation Impact
{elevation_performance.to_string()}

### Annual Rainfall Impact
{rainfall_performance.to_string()}

### Rain Days Impact
{rain_days_performance.to_string()}

## Dual vs Single Code Club Environmental Comparison
{env_comparison.to_string()}

## Key Findings

1. **Overall Performance Impact**
   - Strongest correlation between environmental factors and overall performance
   - Impact on individual code performance (football vs hurling)

2. **Grade Changes**
   - Environmental influence on promotion/relegation patterns
   - Differential impact on football vs hurling grades

3. **Environmental Categories**
   - Performance patterns across different elevation levels
   - Impact of rainfall patterns on club success
   - Effect of rain days on performance

4. **Dual vs Single Code Clubs**
   - Environmental conditions for dual clubs vs single-code clubs
   - Impact on club development and success

## Visualizations
The following visualizations have been generated:
1. Environmental Correlations Heatmap
2. Grade Changes Correlation Heatmap
3. Performance by Environmental Category
4. Dual vs Single Code Environmental Comparison

## Recommendations
Based on the analysis, recommendations for club development and planning can be made regarding:
1. Training and competition scheduling
2. Facility development
3. Performance optimization strategies
4. Environmental adaptation measures

"""
    
    # Save the report
    report_path = os.path.join('output', 'analysis', 'environmental_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report_path

def main():
    """Main function to run the environmental analysis."""
    print("Starting environmental analysis...")
    
    # Load data
    df = load_data()
    print(f"Loaded data for {len(df)} clubs")
    
    # Perform analyses
    env_correlations, grade_changes = analyze_environmental_correlations(df)
    elevation_performance, rainfall_performance, rain_days_performance = analyze_grade_distribution_by_environment(df)
    env_comparison = analyze_dual_clubs_environment(df)
    
    # Generate visualizations
    generate_visualizations(df, env_correlations, grade_changes,
                          elevation_performance, rainfall_performance, rain_days_performance,
                          env_comparison)
    
    # Generate report
    report_path = generate_report(df, env_correlations, grade_changes,
                                elevation_performance, rainfall_performance, rain_days_performance,
                                env_comparison)
    
    print(f"Analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    main() 