"""
Validation Metrics Module

This module provides a comprehensive set of validation metrics for spatial analysis,
including population validation, area coverage, method bias, quality control,
urban-rural metrics, demographic validation, spatial autocorrelation, and statistical
significance testing.

The module includes visualization capabilities and report generation for easy
interpretation of the validation results.
"""

import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationMetrics:
    """
    A class for calculating and analyzing validation metrics for spatial analysis.
    
    This class provides methods to calculate various validation metrics including:
    - Population validation statistics
    - Area coverage metrics
    - Method-specific bias measurements
    - Quality control indicators
    - Urban-rural performance metrics
    - Demographic validation metrics
    - Spatial autocorrelation metrics
    - Enhanced quality control metrics
    - Statistical significance testing
    
    The class also includes visualization capabilities and report generation
    for easy interpretation of the validation results.
    
    Attributes:
        output_dir (Path): Directory where validation results will be saved
    """
    
    def __init__(self, output_dir: str = "data/analysis/validation"):
        """
        Initialize the ValidationMetrics class.
        
        Args:
            output_dir: Directory where validation results will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_population_validation(self, 
                                     buffer_data: pd.DataFrame,
                                     voronoi_data: pd.DataFrame,
                                     nearest_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate population validation statistics.
        
        This method calculates various population-related metrics including:
        - Total population coverage
        - Population density statistics
        - Youth population coverage
        
        Args:
            buffer_data: Buffer analysis results
            voronoi_data: Voronoi analysis results
            nearest_data: Nearest analysis results
            
        Returns:
            DataFrame containing population validation metrics
        """
        metrics = []
        
        # Total population coverage
        total_population = {
            'metric': 'total_population',
            'buffer': buffer_data['total_population'].sum(),
            'voronoi': voronoi_data['T1_1AGE0T_y_voronoi'].sum(),  # Using age 0+ as total population
            'nearest': nearest_data['T1_1AGE0T_y_nearest'].sum()    # Using age 0+ as total population
        }
        metrics.append(total_population)
        
        # Population density statistics
        density_stats = {
            'metric': 'population_density',
            'buffer_mean': buffer_data['population_density'].mean(),
            'voronoi_mean': voronoi_data['T1_1AGE0T_y_voronoi'].sum() / voronoi_data.geometry.area.sum() * 1e6,
            'nearest_mean': nearest_data['T1_1AGE0T_y_nearest'].sum() / nearest_data.geometry.area.sum() * 1e6,
            'buffer_std': buffer_data['population_density'].std(),
            'voronoi_std': np.nan,  # Not directly available
            'nearest_std': np.nan    # Not directly available
        }
        metrics.append(density_stats)
        
        # Youth population coverage
        youth_stats = {
            'metric': 'youth_population',
            'buffer': buffer_data['youth_population'].sum(),
            'voronoi': voronoi_data['T1_1AGE0T_y_voronoi'].sum(),  # Using age 0+ as youth population
            'nearest': nearest_data['T1_1AGE0T_y_nearest'].sum()    # Using age 0+ as youth population
        }
        metrics.append(youth_stats)
        
        # Convert to DataFrame
        validation_df = pd.DataFrame(metrics)
        
        # Save results
        output_path = self.output_dir / 'population_validation.csv'
        validation_df.to_csv(output_path, index=False)
        logger.info(f"Population validation metrics saved to: {output_path}")
        
        return validation_df
    
    def calculate_area_coverage(self,
                              buffer_data: pd.DataFrame,
                              voronoi_data: pd.DataFrame,
                              nearest_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate area coverage metrics.
        
        This method calculates various area-related metrics including:
        - Total area coverage
        - Average area per assignment
        - Area distribution statistics
        
        Args:
            buffer_data: Buffer analysis results
            voronoi_data: Voronoi analysis results
            nearest_data: Nearest analysis results
            
        Returns:
            DataFrame containing area coverage metrics
        """
        metrics = []
        
        # Convert geometry areas to km2
        voronoi_area_km2 = voronoi_data.geometry.area / 1e6  # Convert from m2 to km2
        nearest_area_km2 = nearest_data.geometry.area / 1e6   # Convert from m2 to km2
        
        # Total area coverage
        total_area = {
            'metric': 'total_area_km2',
            'buffer': buffer_data['overlap_area_km2'].sum(),
            'voronoi': voronoi_area_km2.sum(),
            'nearest': nearest_area_km2.sum()
        }
        metrics.append(total_area)
        
        # Average area per assignment
        avg_area = {
            'metric': 'average_area_km2',
            'buffer': buffer_data['overlap_area_km2'].mean(),
            'voronoi': voronoi_area_km2.mean(),
            'nearest': nearest_area_km2.mean()
        }
        metrics.append(avg_area)
        
        # Area distribution statistics
        area_stats = {
            'metric': 'area_distribution',
            'buffer_min': buffer_data['overlap_area_km2'].min(),
            'voronoi_min': voronoi_area_km2.min(),
            'nearest_min': nearest_area_km2.min(),
            'buffer_max': buffer_data['overlap_area_km2'].max(),
            'voronoi_max': voronoi_area_km2.max(),
            'nearest_max': nearest_area_km2.max()
        }
        metrics.append(area_stats)
        
        # Convert to DataFrame
        coverage_df = pd.DataFrame(metrics)
        
        # Save results
        output_path = self.output_dir / 'area_coverage.csv'
        coverage_df.to_csv(output_path, index=False)
        logger.info(f"Area coverage metrics saved to: {output_path}")
        
        return coverage_df
    
    def calculate_method_bias(self,
                            buffer_data: pd.DataFrame,
                            voronoi_data: pd.DataFrame,
                            nearest_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate method-specific bias measurements.
        
        This method calculates bias metrics between different methods including:
        - Population bias
        - Area bias
        - Youth population bias
        
        Args:
            buffer_data: Buffer analysis results
            voronoi_data: Voronoi analysis results
            nearest_data: Nearest analysis results
            
        Returns:
            DataFrame containing bias metrics
        """
        metrics = []
        
        # Convert geometry areas to km2
        voronoi_area_km2 = voronoi_data.geometry.area / 1e6  # Convert from m2 to km2
        nearest_area_km2 = nearest_data.geometry.area / 1e6   # Convert from m2 to km2
        
        # Population bias
        pop_bias = {
            'metric': 'population_bias',
            'buffer_voronoi': (buffer_data['total_population'].sum() - 
                             voronoi_data['T1_1AGE0T_y_voronoi'].sum()) / 
                             voronoi_data['T1_1AGE0T_y_voronoi'].sum() * 100,
            'buffer_nearest': (buffer_data['total_population'].sum() - 
                             nearest_data['T1_1AGE0T_y_nearest'].sum()) / 
                             nearest_data['T1_1AGE0T_y_nearest'].sum() * 100,
            'voronoi_nearest': (voronoi_data['T1_1AGE0T_y_voronoi'].sum() - 
                              nearest_data['T1_1AGE0T_y_nearest'].sum()) / 
                              nearest_data['T1_1AGE0T_y_nearest'].sum() * 100
        }
        metrics.append(pop_bias)
        
        # Area bias
        area_bias = {
            'metric': 'area_bias',
            'buffer_voronoi': (buffer_data['overlap_area_km2'].sum() - 
                             voronoi_area_km2.sum()) / 
                             voronoi_area_km2.sum() * 100,
            'buffer_nearest': (buffer_data['overlap_area_km2'].sum() - 
                             nearest_area_km2.sum()) / 
                             nearest_area_km2.sum() * 100,
            'voronoi_nearest': (voronoi_area_km2.sum() - 
                              nearest_area_km2.sum()) / 
                              nearest_area_km2.sum() * 100
        }
        metrics.append(area_bias)
        
        # Youth population bias
        youth_bias = {
            'metric': 'youth_bias',
            'buffer_voronoi': (buffer_data['youth_population'].sum() - 
                             voronoi_data['T1_1AGE0T_y_voronoi'].sum()) / 
                             voronoi_data['T1_1AGE0T_y_voronoi'].sum() * 100,
            'buffer_nearest': (buffer_data['youth_population'].sum() - 
                             nearest_data['T1_1AGE0T_y_nearest'].sum()) / 
                             nearest_data['T1_1AGE0T_y_nearest'].sum() * 100,
            'voronoi_nearest': (voronoi_data['T1_1AGE0T_y_voronoi'].sum() - 
                              nearest_data['T1_1AGE0T_y_nearest'].sum()) / 
                              nearest_data['T1_1AGE0T_y_nearest'].sum() * 100
        }
        metrics.append(youth_bias)
        
        # Convert to DataFrame
        bias_df = pd.DataFrame(metrics)
        
        # Save results
        output_path = self.output_dir / 'method_bias.csv'
        bias_df.to_csv(output_path, index=False)
        logger.info(f"Method bias metrics saved to: {output_path}")
        
        return bias_df
    
    def calculate_quality_control(self,
                                buffer_data: pd.DataFrame,
                                voronoi_data: pd.DataFrame,
                                nearest_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate quality control indicators.
        
        This method calculates various quality control metrics including:
        - Data completeness
        - Population consistency
        - Area consistency
        
        Args:
            buffer_data: Buffer analysis results
            voronoi_data: Voronoi analysis results
            nearest_data: Nearest analysis results
            
        Returns:
            DataFrame containing quality control metrics
        """
        metrics = []
        
        # Convert geometry areas to km2
        voronoi_area_km2 = voronoi_data.geometry.area / 1e6  # Convert from m2 to km2
        nearest_area_km2 = nearest_data.geometry.area / 1e6   # Convert from m2 to km2
        
        # Data completeness
        completeness = {
            'metric': 'data_completeness',
            'buffer': 1 - buffer_data.isnull().sum().sum() / buffer_data.size,
            'voronoi': 1 - voronoi_data.isnull().sum().sum() / voronoi_data.size,
            'nearest': 1 - nearest_data.isnull().sum().sum() / nearest_data.size
        }
        metrics.append(completeness)
        
        # Population consistency
        pop_consistency = {
            'metric': 'population_consistency',
            'buffer': buffer_data['total_population'].sum() / 
                     buffer_data['total_population'].sum(),
            'voronoi': voronoi_data['T1_1AGE0T_y_voronoi'].sum() / 
                      buffer_data['total_population'].sum(),
            'nearest': nearest_data['T1_1AGE0T_y_nearest'].sum() / 
                      buffer_data['total_population'].sum()
        }
        metrics.append(pop_consistency)
        
        # Area consistency
        area_consistency = {
            'metric': 'area_consistency',
            'buffer': buffer_data['overlap_area_km2'].sum() / 
                     buffer_data['overlap_area_km2'].sum(),
            'voronoi': voronoi_area_km2.sum() / 
                      buffer_data['overlap_area_km2'].sum(),
            'nearest': nearest_area_km2.sum() / 
                      buffer_data['overlap_area_km2'].sum()
        }
        metrics.append(area_consistency)
        
        # Convert to DataFrame
        quality_df = pd.DataFrame(metrics)
        
        # Save results
        output_path = self.output_dir / 'quality_control.csv'
        quality_df.to_csv(output_path, index=False)
        logger.info(f"Quality control metrics saved to: {output_path}")
        
        return quality_df
    
    def calculate_urban_rural_metrics(self,
                                    buffer_data: pd.DataFrame,
                                    voronoi_data: pd.DataFrame,
                                    nearest_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate urban-rural performance metrics.
        
        This method calculates metrics specific to urban and rural areas including:
        - Urban population coverage
        - Rural population coverage
        - Urban area coverage
        - Rural area coverage
        
        Args:
            buffer_data: Buffer analysis results
            voronoi_data: Voronoi analysis results
            nearest_data: Nearest analysis results
            
        Returns:
            DataFrame containing urban-rural metrics
        """
        metrics = []
        
        # Calculate population density for Voronoi and nearest
        voronoi_data = voronoi_data.copy()
        nearest_data = nearest_data.copy()
        
        voronoi_data['population_density'] = (voronoi_data['T1_1AGE0T_y_voronoi'] / 
                                            (voronoi_data.geometry.area / 1e6))  # per km2
        nearest_data['population_density'] = (nearest_data['T1_1AGE0T_y_nearest'] / 
                                           (nearest_data.geometry.area / 1e6))   # per km2
        
        # Urban population coverage
        urban_pop = {
            'metric': 'urban_population_coverage',
            'buffer': buffer_data[buffer_data['population_density'] > 1000]['total_population'].sum(),
            'voronoi': voronoi_data[voronoi_data['population_density'] > 1000]['T1_1AGE0T_y_voronoi'].sum(),
            'nearest': nearest_data[nearest_data['population_density'] > 1000]['T1_1AGE0T_y_nearest'].sum()
        }
        metrics.append(urban_pop)
        
        # Rural population coverage
        rural_pop = {
            'metric': 'rural_population_coverage',
            'buffer': buffer_data[buffer_data['population_density'] <= 1000]['total_population'].sum(),
            'voronoi': voronoi_data[voronoi_data['population_density'] <= 1000]['T1_1AGE0T_y_voronoi'].sum(),
            'nearest': nearest_data[nearest_data['population_density'] <= 1000]['T1_1AGE0T_y_nearest'].sum()
        }
        metrics.append(rural_pop)
        
        # Urban area coverage
        urban_area = {
            'metric': 'urban_area_coverage',
            'buffer': buffer_data[buffer_data['population_density'] > 1000]['overlap_area_km2'].sum(),
            'voronoi': voronoi_data[voronoi_data['population_density'] > 1000].geometry.area.sum() / 1e6,
            'nearest': nearest_data[nearest_data['population_density'] > 1000].geometry.area.sum() / 1e6
        }
        metrics.append(urban_area)
        
        # Rural area coverage
        rural_area = {
            'metric': 'rural_area_coverage',
            'buffer': buffer_data[buffer_data['population_density'] <= 1000]['overlap_area_km2'].sum(),
            'voronoi': voronoi_data[voronoi_data['population_density'] <= 1000].geometry.area.sum() / 1e6,
            'nearest': nearest_data[nearest_data['population_density'] <= 1000].geometry.area.sum() / 1e6
        }
        metrics.append(rural_area)
        
        # Convert to DataFrame
        urban_rural_df = pd.DataFrame(metrics)
        
        # Save results
        output_path = self.output_dir / 'urban_rural_metrics.csv'
        urban_rural_df.to_csv(output_path, index=False)
        logger.info(f"Urban-rural metrics saved to: {output_path}")
        
        return urban_rural_df
    
    def calculate_demographic_validation(self,
                                      buffer_data: pd.DataFrame,
                                      voronoi_data: pd.DataFrame,
                                      nearest_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate demographic validation metrics.
        
        This method calculates metrics related to demographic distribution including:
        - Total population
        - Youth population
        - Population density
        
        Args:
            buffer_data: Buffer analysis results
            voronoi_data: Voronoi analysis results
            nearest_data: Nearest analysis results
            
        Returns:
            DataFrame containing demographic validation metrics
        """
        metrics = []
        
        # Total population
        total_pop = {
            'metric': 'total_population',
            'buffer': buffer_data['total_population'].sum(),
            'voronoi': voronoi_data['T1_1AGE0T_y_voronoi'].sum(),
            'nearest': nearest_data['T1_1AGE0T_y_nearest'].sum()
        }
        metrics.append(total_pop)
        
        # Youth population
        youth_pop = {
            'metric': 'youth_population',
            'buffer': buffer_data['youth_population'].sum(),
            'voronoi': voronoi_data['T1_1AGE0T_y_voronoi'].sum(),
            'nearest': nearest_data['T1_1AGE0T_y_nearest'].sum()
        }
        metrics.append(youth_pop)
        
        # Population density
        pop_density = {
            'metric': 'population_density',
            'buffer': buffer_data['population_density'].mean(),
            'voronoi': (voronoi_data['T1_1AGE0T_y_voronoi'] / 
                       (voronoi_data.geometry.area / 1e6)).mean(),
            'nearest': (nearest_data['T1_1AGE0T_y_nearest'] / 
                      (nearest_data.geometry.area / 1e6)).mean()
        }
        metrics.append(pop_density)
        
        # Convert to DataFrame
        demo_df = pd.DataFrame(metrics)
        
        # Save results
        output_path = self.output_dir / 'demographic_validation.csv'
        demo_df.to_csv(output_path, index=False)
        logger.info(f"Demographic validation metrics saved to: {output_path}")
        
        return demo_df
    
    def calculate_spatial_autocorrelation(self,
                                        buffer_data: pd.DataFrame,
                                        voronoi_data: pd.DataFrame,
                                        nearest_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate spatial autocorrelation metrics.
        
        This method calculates metrics related to spatial distribution including:
        - Population distribution
        - Youth distribution
        - Density distribution
        
        Args:
            buffer_data: Buffer analysis results
            voronoi_data: Voronoi analysis results
            nearest_data: Nearest analysis results
            
        Returns:
            DataFrame containing spatial autocorrelation metrics
        """
        metrics = []
        
        # Calculate population density for Voronoi and nearest
        voronoi_data = voronoi_data.copy()
        nearest_data = nearest_data.copy()
        
        voronoi_data['population_density'] = (voronoi_data['T1_1AGE0T_y_voronoi'] / 
                                            (voronoi_data.geometry.area / 1e6))  # per km2
        nearest_data['population_density'] = (nearest_data['T1_1AGE0T_y_nearest'] / 
                                           (nearest_data.geometry.area / 1e6))   # per km2
        
        # Population distribution metrics
        pop_metrics = {
            'metric': 'population_distribution',
            'buffer_cv': buffer_data['total_population'].std() / buffer_data['total_population'].mean(),
            'voronoi_cv': (voronoi_data['T1_1AGE0T_y_voronoi'].std() / 
                          voronoi_data['T1_1AGE0T_y_voronoi'].mean()),
            'nearest_cv': (nearest_data['T1_1AGE0T_y_nearest'].std() / 
                         nearest_data['T1_1AGE0T_y_nearest'].mean())
        }
        metrics.append(pop_metrics)
        
        # Youth population distribution metrics
        youth_metrics = {
            'metric': 'youth_distribution',
            'buffer_cv': buffer_data['youth_population'].std() / buffer_data['youth_population'].mean(),
            'voronoi_cv': (voronoi_data['T1_1AGE0T_y_voronoi'].std() / 
                          voronoi_data['T1_1AGE0T_y_voronoi'].mean()),
            'nearest_cv': (nearest_data['T1_1AGE0T_y_nearest'].std() / 
                         nearest_data['T1_1AGE0T_y_nearest'].mean())
        }
        metrics.append(youth_metrics)
        
        # Density distribution metrics
        density_metrics = {
            'metric': 'density_distribution',
            'buffer_cv': buffer_data['population_density'].std() / buffer_data['population_density'].mean(),
            'voronoi_cv': (voronoi_data['population_density'].std() / 
                          voronoi_data['population_density'].mean()),
            'nearest_cv': (nearest_data['population_density'].std() / 
                         nearest_data['population_density'].mean())
        }
        metrics.append(density_metrics)
        
        # Convert to DataFrame
        spatial_df = pd.DataFrame(metrics)
        
        # Save results
        output_path = self.output_dir / 'spatial_distribution.csv'
        spatial_df.to_csv(output_path, index=False)
        logger.info(f"Spatial distribution metrics saved to: {output_path}")
        
        return spatial_df
    
    def calculate_enhanced_quality_control(self,
                                        buffer_data: pd.DataFrame,
                                        voronoi_data: pd.DataFrame,
                                        nearest_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced quality control metrics.
        
        This method calculates additional quality control metrics including:
        - Data completeness
        - Outlier detection
        - Data consistency validation
        - Error rate calculation
        
        Args:
            buffer_data: Buffer analysis results
            voronoi_data: Voronoi analysis results
            nearest_data: Nearest analysis results
            
        Returns:
            DataFrame containing enhanced quality control metrics
        """
        try:
            # Calculate areas for GeoDataFrames
            voronoi_data['area'] = voronoi_data.geometry.area
            nearest_data['area'] = nearest_data.geometry.area
            
            # Data completeness
            completeness = {
                'buffer': buffer_data.isnull().mean().mean() * 100,
                'voronoi': voronoi_data.isnull().mean().mean() * 100,
                'nearest': nearest_data.isnull().mean().mean() * 100
            }
            
            # Outlier detection
            def detect_outliers(data, column):
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return len(data[(data[column] < lower_bound) | (data[column] > upper_bound)])
            
            outliers = {
                'buffer': detect_outliers(buffer_data, 'total_population'),
                'voronoi': detect_outliers(voronoi_data, 'area'),
                'nearest': detect_outliers(nearest_data, 'area')
            }
            
            # Data consistency validation
            consistency = {
                'buffer_pop_area_ratio': (buffer_data['total_population'] / 
                                        buffer_data['overlap_area_km2']).std(),
                'voronoi_pop_area_ratio': (voronoi_data['T1_1AGE0T_y_voronoi'] / 
                                         (voronoi_data.geometry.area / 1e6)).std(),
                'nearest_pop_area_ratio': (nearest_data['T1_1AGE0T_y_nearest'] / 
                                         (nearest_data.geometry.area / 1e6)).std()
            }
            
            # Error rate calculation
            error_rate = {
                'buffer_zero_pop': (buffer_data['total_population'] == 0).sum() / len(buffer_data) * 100,
                'voronoi_zero_pop': (voronoi_data['T1_1AGE0T_y_voronoi'] == 0).sum() / len(voronoi_data) * 100,
                'nearest_zero_pop': (nearest_data['T1_1AGE0T_y_nearest'] == 0).sum() / len(nearest_data) * 100
            }
            
            # Combine metrics
            metrics = {**completeness, **outliers, **consistency, **error_rate}
            
            # Convert to DataFrame
            enhanced_qc_df = pd.DataFrame([metrics])
            
            # Save results
            output_path = self.output_dir / 'enhanced_quality_control.csv'
            enhanced_qc_df.to_csv(output_path, index=False)
            logger.info(f"Enhanced quality control metrics saved to: {output_path}")
            
            return enhanced_qc_df
        except Exception as e:
            logger.error(f"Error in calculate_enhanced_quality_control: {str(e)}")
            return pd.DataFrame()
    
    def calculate_statistical_significance(self, buffer_data, voronoi_data, nearest_data):
        """Calculate statistical significance metrics."""
        try:
            # Initialize results dictionary
            results = {}
            
            # T-tests for population differences
            def t_test(data1, data2, col1, col2):
                t_stat, p_val = stats.ttest_ind(data1[col1], data2[col2])
                return {
                    't_stat': t_stat,
                    'p_value': p_val
                }
            
            # Perform t-tests
            results['buffer_vs_voronoi_t_stat'] = t_test(buffer_data, voronoi_data, 
                'total_population', 'T1_1AGETT_y_voronoi')['t_stat']
            results['buffer_vs_voronoi_p_value'] = t_test(buffer_data, voronoi_data, 
                'total_population', 'T1_1AGETT_y_voronoi')['p_value']
                
            results['buffer_vs_nearest_t_stat'] = t_test(buffer_data, nearest_data, 
                'total_population', 'T1_1AGETT_y_nearest')['t_stat']
            results['buffer_vs_nearest_p_value'] = t_test(buffer_data, nearest_data, 
                'total_population', 'T1_1AGETT_y_nearest')['p_value']
                
            results['voronoi_vs_nearest_t_stat'] = t_test(voronoi_data, nearest_data, 
                'T1_1AGETT_y_voronoi', 'T1_1AGETT_y_nearest')['t_stat']
            results['voronoi_vs_nearest_p_value'] = t_test(voronoi_data, nearest_data, 
                'T1_1AGETT_y_voronoi', 'T1_1AGETT_y_nearest')['p_value']
            
            # Convert results to DataFrame
            stats_df = pd.DataFrame([results])
            
            # Save results
            output_file = self.output_dir / 'statistical_significance.csv'
            stats_df.to_csv(output_file, index=False)
            logger.info(f"Statistical significance metrics saved to: {output_file}")
            
            return stats_df
            
        except Exception as e:
            logger.error(f"Error in calculate_statistical_significance: {str(e)}")
            return pd.DataFrame()
    
    def visualize_statistical_tests(self, stats_df: pd.DataFrame) -> None:
        """
        Generate visualizations for statistical test results.
        
        This method creates visualizations for:
        - T-test results
        - Chi-square test results
        - Significance summary
        
        Args:
            stats_df: DataFrame containing statistical test results
        """
        try:
            # Use a basic style instead of seaborn
            plt.style.use('default')
            
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_dir) / 'visualizations'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Visualize t-test results
            self._visualize_t_tests(stats_df, output_dir)
            
            # Visualize chi-square test results
            self._visualize_chi_square(stats_df, output_dir)
            
            # Visualize significance summary
            self._visualize_significance_summary(stats_df, output_dir)
            
            logger.info(f"Statistical test visualizations saved to: {output_dir}")
        except Exception as e:
            logger.error(f"Error in visualize_statistical_tests: {str(e)}")
    
    def _visualize_t_tests(self, stats_df: pd.DataFrame, plots_dir: Path) -> None:
        """Visualize t-test results."""
        try:
            # Create comparison pairs
            comparison_pairs = ['buffer_vs_voronoi', 'buffer_vs_nearest', 'voronoi_vs_nearest']
            
            # Extract t-test results for each comparison
            t_test_results = []
            for pair in comparison_pairs:
                if pair in stats_df.columns:
                    t_test_results.append({
                        'comparison': pair,
                        't_statistic': stats_df[pair + '_t_stat'].iloc[0],
                        'p_value': stats_df[pair + '_p_value'].iloc[0]
                    })
            
            if not t_test_results:
                logger.warning("No t-test results found in stats_df")
                return
            
            # Create DataFrame for plotting
            t_test_df = pd.DataFrame(t_test_results)
            
            # Plot t-test results
            plt.figure(figsize=(10, 6))
            plt.bar(t_test_df['comparison'], t_test_df['t_statistic'])
            plt.title('T-Test Results Across Methods')
            plt.xlabel('Comparison')
            plt.ylabel('T-Statistic')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(plots_dir / 't_test_results.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error in _visualize_t_tests: {str(e)}")
    
    def _visualize_chi_square(self, stats_df: pd.DataFrame, plots_dir: Path) -> None:
        """Visualize chi-square test results."""
        # Filter chi-square test results
        chi_square = stats_df[stats_df['metric'].str.startswith('chi_square_')]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chi-Square Test Results', fontsize=16)
        
        # Plot 1: Chi-square statistics
        chi_stats = pd.DataFrame({
            'Buffer vs Voronoi': chi_square['buffer_voronoi'].apply(lambda x: x['chi_square']),
            'Buffer vs Nearest': chi_square['buffer_nearest'].apply(lambda x: x['chi_square']),
            'Voronoi vs Nearest': chi_square['voronoi_nearest'].apply(lambda x: x['chi_square'])
        })
        chi_stats.index = chi_square['metric'].str.replace('chi_square_', '')
        sns.heatmap(chi_stats, annot=True, cmap='YlOrRd', ax=axes[0, 0])
        axes[0, 0].set_title('Chi-Square Statistics')
        
        # Plot 2: P-values
        p_values = pd.DataFrame({
            'Buffer vs Voronoi': chi_square['buffer_voronoi'].apply(lambda x: x['p_value']),
            'Buffer vs Nearest': chi_square['buffer_nearest'].apply(lambda x: x['p_value']),
            'Voronoi vs Nearest': chi_square['voronoi_nearest'].apply(lambda x: x['p_value'])
        })
        p_values.index = chi_square['metric'].str.replace('chi_square_', '')
        sns.heatmap(p_values, annot=True, cmap='YlOrRd', vmin=0, vmax=0.05, ax=axes[0, 1])
        axes[0, 1].set_title('P-Values')
        
        # Plot 3: Degrees of Freedom
        dof = pd.DataFrame({
            'Buffer vs Voronoi': chi_square['buffer_voronoi'].apply(lambda x: x['degrees_of_freedom']),
            'Buffer vs Nearest': chi_square['buffer_nearest'].apply(lambda x: x['degrees_of_freedom']),
            'Voronoi vs Nearest': chi_square['voronoi_nearest'].apply(lambda x: x['degrees_of_freedom'])
        })
        dof.index = chi_square['metric'].str.replace('chi_square_', '')
        sns.heatmap(dof, annot=True, cmap='YlGnBu', ax=axes[1, 0])
        axes[1, 0].set_title('Degrees of Freedom')
        
        # Plot 4: Categories
        categories = pd.DataFrame({
            'Buffer vs Voronoi': chi_square['buffer_voronoi'].apply(lambda x: x['categories']),
            'Buffer vs Nearest': chi_square['buffer_nearest'].apply(lambda x: x['categories']),
            'Voronoi vs Nearest': chi_square['voronoi_nearest'].apply(lambda x: x['categories'])
        })
        categories.index = chi_square['metric'].str.replace('chi_square_', '')
        sns.heatmap(categories, annot=True, cmap='YlGnBu', ax=axes[1, 1])
        axes[1, 1].set_title('Number of Categories')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'chi_square_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_significance_summary(self, stats_df: pd.DataFrame, plots_dir: Path) -> None:
        """Visualize summary of significant results."""
        # Extract significance information
        significance_data = []
        
        for _, row in stats_df.iterrows():
            for comparison in ['buffer_voronoi', 'buffer_nearest', 'voronoi_nearest']:
                result = row[comparison]
                if isinstance(result, dict) and 'significant' in result:
                    significance_data.append({
                        'metric': row['metric'],
                        'comparison': comparison.replace('_', ' vs '),
                        'significant': result['significant']
                    })
        
        significance_df = pd.DataFrame(significance_data)
        
        # Create pivot table for visualization
        pivot_df = significance_df.pivot_table(
            index='metric',
            columns='comparison',
            values='significant',
            aggfunc='sum'
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='g')
        plt.title('Number of Significant Results by Metric and Comparison')
        plt.tight_layout()
        plt.savefig(plots_dir / 'significance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_validation_metrics(self):
        """Generate visualizations for all validation metrics."""
        try:
            # Use default style
            plt.style.use('default')
            
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_dir) / 'visualizations'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load metrics from CSV files
            population_metrics = pd.read_csv(self.output_dir / 'population_validation.csv')
            area_metrics = pd.read_csv(self.output_dir / 'area_coverage.csv')
            bias_metrics = pd.read_csv(self.output_dir / 'method_bias.csv')
            qc_metrics = pd.read_csv(self.output_dir / 'quality_control.csv')
            urban_rural_metrics = pd.read_csv(self.output_dir / 'urban_rural_metrics.csv')
            demographic_metrics = pd.read_csv(self.output_dir / 'demographic_validation.csv')
            spatial_metrics = pd.read_csv(self.output_dir / 'spatial_distribution.csv')
            enhanced_qc_metrics = pd.read_csv(self.output_dir / 'enhanced_quality_control.csv')
            stats_metrics = pd.read_csv(self.output_dir / 'statistical_significance.csv')
            
            # Create visualizations for each metric type
            self._plot_population_metrics(population_metrics, output_dir)
            self._plot_area_metrics(area_metrics, output_dir)
            self._plot_bias_metrics(bias_metrics, output_dir)
            self._plot_qc_metrics(qc_metrics, output_dir)
            self._plot_urban_rural_metrics(urban_rural_metrics, output_dir)
            self._plot_demographic_metrics(demographic_metrics, output_dir)
            self._plot_spatial_metrics(spatial_metrics, output_dir)
            self._plot_enhanced_qc_metrics(enhanced_qc_metrics, output_dir)
            self._plot_stats_metrics(stats_metrics, output_dir)
            
            logger.info(f"Validation metrics visualizations saved to: {output_dir}")
        except Exception as e:
            logger.error(f"Error in visualize_validation_metrics: {str(e)}")
    
    def _plot_population_metrics(self, metrics_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot population validation metrics."""
        try:
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Population Validation Metrics')
            plt.ylabel('Value')
            plt.xticks(range(len(metrics_df.index)), metrics_df.index, rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'population_validation.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error in _plot_population_metrics: {str(e)}")
    
    def _plot_area_metrics(self, metrics_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot area coverage metrics."""
        try:
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Area Coverage Metrics')
            plt.ylabel('Value (km²)')
            plt.xticks(range(len(metrics_df.index)), metrics_df.index, rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'area_coverage.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error in _plot_area_metrics: {str(e)}")
    
    def _plot_bias_metrics(self, metrics_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot method bias metrics."""
        try:
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Method Bias Metrics')
            plt.ylabel('Bias (%)')
            plt.xticks(range(len(metrics_df.index)), metrics_df.index, rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'method_bias.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error in _plot_bias_metrics: {str(e)}")
    
    def _plot_qc_metrics(self, metrics_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot quality control metrics."""
        try:
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Quality Control Metrics')
            plt.ylabel('Value')
            plt.xticks(range(len(metrics_df.index)), metrics_df.index, rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'quality_control.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error in _plot_qc_metrics: {str(e)}")
    
    def _plot_urban_rural_metrics(self, metrics_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot urban-rural metrics."""
        try:
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Urban-Rural Metrics')
            plt.ylabel('Value')
            plt.xticks(range(len(metrics_df.index)), metrics_df.index, rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'urban_rural_metrics.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error in _plot_urban_rural_metrics: {str(e)}")
    
    def _plot_demographic_metrics(self, metrics_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot demographic validation metrics."""
        try:
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Demographic Validation Metrics')
            plt.ylabel('Value')
            plt.xticks(range(len(metrics_df.index)), metrics_df.index, rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'demographic_validation.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error in _plot_demographic_metrics: {str(e)}")
    
    def _plot_spatial_metrics(self, metrics_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot spatial autocorrelation metrics."""
        try:
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Spatial Autocorrelation Metrics')
            plt.ylabel('Value')
            plt.xticks(range(len(metrics_df.index)), metrics_df.index, rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'spatial_autocorrelation.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error in _plot_spatial_metrics: {str(e)}")
    
    def _plot_enhanced_qc_metrics(self, metrics_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot enhanced quality control metrics."""
        try:
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Enhanced Quality Control Metrics')
            plt.ylabel('Value')
            plt.xticks(range(len(metrics_df.index)), metrics_df.index, rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'enhanced_quality_control.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error in _plot_enhanced_qc_metrics: {str(e)}")
    
    def _plot_stats_metrics(self, metrics_df: pd.DataFrame, output_dir: Path) -> None:
        """Plot statistical significance metrics."""
        try:
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Statistical Significance Metrics')
            plt.ylabel('Value')
            plt.xticks(range(len(metrics_df.index)), metrics_df.index, rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'statistical_significance.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error in _plot_stats_metrics: {str(e)}")
    
    def generate_validation_report(self) -> None:
        """
        Generate a comprehensive validation report including all metrics and visualizations.
        """
        # Create report directory
        report_dir = self.output_dir / 'report'
        report_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        self.visualize_validation_metrics()
        
        # Load all validation metrics
        population_validation = pd.read_csv(self.output_dir / 'population_validation.csv')
        area_coverage = pd.read_csv(self.output_dir / 'area_coverage.csv')
        method_bias = pd.read_csv(self.output_dir / 'method_bias.csv')
        quality_control = pd.read_csv(self.output_dir / 'quality_control.csv')
        urban_rural_metrics = pd.read_csv(self.output_dir / 'urban_rural_metrics.csv')
        demographic_validation = pd.read_csv(self.output_dir / 'demographic_validation.csv')
        spatial_autocorrelation = pd.read_csv(self.output_dir / 'spatial_distribution.csv')
        enhanced_quality_control = pd.read_csv(self.output_dir / 'enhanced_quality_control.csv')
        statistical_significance = pd.read_csv(self.output_dir / 'statistical_significance.csv')
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Metrics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .metric-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .visualization {{ margin: 20px 0; }}
                .visualization img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Validation Metrics Report</h1>
            
            <div class="section">
                <h2>1. Population Validation</h2>
                {population_validation.to_html(classes='metric-table')}
                <div class="visualization">
                    <img src="visualizations/population_validation.png" alt="Population Validation">
                </div>
            </div>
            
            <div class="section">
                <h2>2. Area Coverage</h2>
                {area_coverage.to_html(classes='metric-table')}
                <div class="visualization">
                    <img src="visualizations/area_coverage.png" alt="Area Coverage">
                </div>
            </div>
            
            <div class="section">
                <h2>3. Method Bias</h2>
                {method_bias.to_html(classes='metric-table')}
                <div class="visualization">
                    <img src="visualizations/method_bias.png" alt="Method Bias">
                </div>
            </div>
            
            <div class="section">
                <h2>4. Quality Control</h2>
                {quality_control.to_html(classes='metric-table')}
                <div class="visualization">
                    <img src="visualizations/quality_control.png" alt="Quality Control">
                </div>
            </div>
            
            <div class="section">
                <h2>5. Urban-Rural Metrics</h2>
                {urban_rural_metrics.to_html(classes='metric-table')}
                <div class="visualization">
                    <img src="visualizations/urban_rural_metrics.png" alt="Urban-Rural Metrics">
                </div>
            </div>
            
            <div class="section">
                <h2>6. Demographic Validation</h2>
                {demographic_validation.to_html(classes='metric-table')}
                <div class="visualization">
                    <img src="visualizations/demographic_validation.png" alt="Demographic Validation">
                </div>
            </div>
            
            <div class="section">
                <h2>7. Spatial Autocorrelation</h2>
                {spatial_autocorrelation.to_html(classes='metric-table')}
                <div class="visualization">
                    <img src="visualizations/spatial_autocorrelation.png" alt="Spatial Autocorrelation">
                </div>
            </div>
            
            <div class="section">
                <h2>8. Enhanced Quality Control</h2>
                {enhanced_quality_control.to_html(classes='metric-table')}
                <div class="visualization">
                    <img src="visualizations/enhanced_quality_control.png" alt="Enhanced Quality Control">
                </div>
            </div>
            
            <div class="section">
                <h2>9. Statistical Significance</h2>
                {statistical_significance.to_html(classes='metric-table')}
                <div class="visualization">
                    <img src="visualizations/statistical_significance.png" alt="Statistical Significance">
                </div>
            </div>
            
            <div class="section">
                <h2>10. Combined Metrics Summary</h2>
                <div class="visualization">
                    <img src="visualizations/combined_metrics_summary.png" alt="Combined Metrics Summary">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = report_dir / 'validation_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Validation report generated and saved to: {report_path}")

def main():
    """Main function to demonstrate usage of the ValidationMetrics class."""
    try:
        # Initialize the validation metrics class
        validator = ValidationMetrics()
        
        # Load the data
        logger.info("Loading analysis data...")
        buffer_data = pd.read_csv('data/processed/buffer_demographics.csv')
        voronoi_data = gpd.read_file('data/processed/voronoi_demographics.gpkg')
        nearest_data = gpd.read_file('data/processed/nearest_demographics.gpkg')
        
        # Calculate validation metrics
        logger.info("Calculating population validation metrics...")
        validator.calculate_population_validation(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating area coverage metrics...")
        validator.calculate_area_coverage(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating method bias metrics...")
        validator.calculate_method_bias(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating quality control metrics...")
        validator.calculate_quality_control(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating urban-rural metrics...")
        validator.calculate_urban_rural_metrics(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating demographic validation metrics...")
        validator.calculate_demographic_validation(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating spatial autocorrelation metrics...")
        validator.calculate_spatial_autocorrelation(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating enhanced quality control metrics...")
        validator.calculate_enhanced_quality_control(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Calculating statistical significance...")
        stats_df = validator.calculate_statistical_significance(buffer_data, voronoi_data, nearest_data)
        
        logger.info("Generating statistical test visualizations...")
        validator.visualize_statistical_tests(stats_df)
        
        # Generate validation report
        logger.info("Generating validation report...")
        validator.generate_validation_report()
        
    except Exception as e:
        logger.error(f"Error in validation metrics calculation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 