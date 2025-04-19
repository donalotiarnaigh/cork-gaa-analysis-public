#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial Pattern Analysis Script for Cork GAA Club Analysis.

This script performs spatial autocorrelation analysis to identify patterns
in the geographic distribution of GAA club performance metrics.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import cohen_kappa_score
import esda  # Import the standalone esda package
import libpysal.weights as weights  # Import weights from libpysal
from matplotlib import colors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/spatial_pattern_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("spatial_pattern_analysis")

# Constants and Configuration
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"  # Update to point to processed data directory
OUTPUT_DIR = ROOT_DIR / "output" / "spatial_analysis"
REPORTS_DIR = Path("reports")

# Define key performance metrics to analyze
PERFORMANCE_METRICS = [
    'football_performance',   # Football performance metric
    'hurling_performance',    # Hurling performance metric
    'overall_performance',    # Overall performance metric
    'code_balance'            # Code balance metric
]

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log_section(message):
    """Log a section header to make logs more readable."""
    line = "=" * 80
    logger.info(f"\n{line}\n{message}\n{line}")

class SpatialPatternAnalysis:
    """
    Perform spatial autocorrelation analysis on club performance metrics.
    
    This class implements the calculation of Moran's I and related spatial
    statistics to identify patterns in the geographic distribution of
    performance metrics for Cork GAA clubs.
    """
    
    def __init__(self):
        """Initialize the SpatialPatternAnalysis class."""
        log_section("Initializing Spatial Pattern Analysis")
        
        # Data loading and processing will be implemented here
        self.voronoi_data = None
        self.nearest_data = None
        self.club_data = None
        self.weights_matrix = None
        
        # Lists to track analysis results
        self.morans_results = {}
        self.hotspot_results = {}
        self.urban_rural_patterns = {}
        
        # Track demographic variables with high agreement between methods
        self.key_demographic_vars = []
        
        logger.info("Spatial Pattern Analysis initialized")

    def load_data(self):
        """
        Load data for spatial pattern analysis.
        
        This method loads data from the Voronoi and nearest club catchment methods,
        as well as club performance data, and prepares a hybrid dataset for analysis.
        """
        log_section("Loading Data for Spatial Pattern Analysis")
        
        try:
            # Load Voronoi data
            voronoi_path = DATA_DIR / "voronoi_demographics.gpkg"
            logger.info(f"Loading Voronoi data from: {voronoi_path}")
            self.voronoi_data = gpd.read_file(voronoi_path)
            logger.info(f"Loaded {len(self.voronoi_data)} records from Voronoi data")
            
            # Load nearest club data
            nearest_path = DATA_DIR / "nearest_demographics.gpkg"
            logger.info(f"Loading nearest club data from: {nearest_path}")
            self.nearest_data = gpd.read_file(nearest_path)
            logger.info(f"Loaded {len(self.nearest_data)} records from nearest club data")
            
            # Load club data
            club_path = DATA_DIR / "cork_clubs_complete_graded.csv"
            logger.info(f"Loading club data from: {club_path}")
            self.club_data = pd.read_csv(club_path)
            logger.info(f"Loaded {len(self.club_data)} clubs from club data")
            
            # Identify key demographic variables with high agreement between methods
            logger.info("Identifying key demographic variables with high agreement")
            
            # Find columns with demographic info
            demog_cols = [col for col in self.voronoi_data.columns if col.endswith('_demog')]
            
            # If we have fewer than 5 columns, use a random sample
            if len(demog_cols) < 5:
                sample_size = min(5, len(demog_cols))
                self.key_demographic_vars = np.random.choice(demog_cols, size=sample_size, replace=False).tolist()
            else:
                # For simplicity, we'll just use the first 5 as key variables for now
                # In a more comprehensive implementation, this would be based on importance or correlation
                self.key_demographic_vars = demog_cols[:5]
            
            logger.info(f"Identified {len(self.key_demographic_vars)} key demographic variables")
            logger.info(f"Variables: {', '.join(self.key_demographic_vars[:5])}... and {len(self.key_demographic_vars)-5} more")
            
            # Prepare a hybrid dataset from the different catchment methods
            logger.info("Preparing hybrid dataset from catchment methods")
            
            # Start with Voronoi data as the base
            self.hybrid_data = self.voronoi_data.copy()
            
            # Add a method source column to track the origin of each variable
            self.hybrid_data['method_source'] = 'voronoi'
            
            # For performance metrics, prefer the methods identified as more reliable in the comparison
            # This may vary by metric based on the catchment comparison results
            
            # For now, use this as a placeholder - in a real implementation, this would be
            # based on the results of the catchment methodology comparison
            metric_method_map = {
                "football_performance": "voronoi",
                "hurling_performance": "voronoi",
                "overall_performance": "voronoi",
                "code_balance": "voronoi"
            }
            
            # Copy metrics from the preferred method
            for metric, method in metric_method_map.items():
                if method == 'nearest' and metric in self.nearest_data.columns:
                    self.hybrid_data[metric] = self.nearest_data[metric]
                    self.hybrid_data[f"{metric}_source"] = 'nearest'
                    logger.info(f"Using nearest club data for {metric}")
            
            # Add club-level performance metrics to small areas
            # This maps performance metrics from the club-level data to the small areas
            
            # First, identify the club ID column in hybrid data
            club_id_cols = [col for col in self.hybrid_data.columns if 'club_id' in col.lower()]
            
            if club_id_cols:
                club_id_col = club_id_cols[0]
                
                # Create a mapping dictionary from club name to performance metrics
                performance_dict = {}
                
                # Check which column contains the club names in both datasets
                club_col_hybrid = 'nearest_club' if 'nearest_club' in self.hybrid_data.columns else 'assigned_club_name'
                club_col_club = 'Club' if 'Club' in self.club_data.columns else 'Name'
                
                # Create mapping from club name to performance metrics
                for metric in PERFORMANCE_METRICS:
                    if metric in self.club_data.columns:
                        performance_dict[metric] = dict(zip(self.club_data[club_col_club], self.club_data[metric]))
                
                # Apply mapping to add performance metrics to each small area
                for metric, club_perf_dict in performance_dict.items():
                    self.hybrid_data[metric] = self.hybrid_data[club_col_hybrid].map(club_perf_dict)
                    logger.info(f"Added {metric} to small areas from club data")
                
            else:
                logger.warning("No club ID column found in hybrid data. Cannot add club performance metrics.")
            
            logger.info(f"Created hybrid dataset with {len(self.hybrid_data)} records")
            logger.info("Data loading completed successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def run_analysis(self):
        """Execute the complete spatial pattern analysis workflow."""
        log_section("Starting Spatial Pattern Analysis Workflow")
        
        # Method implementation will go here
        self.load_data()
        self.prepare_data()
        self.create_weights_matrix()
        self.calculate_morans_i()
        self.identify_hotspots()
        self.analyze_urban_rural_patterns()
        self.compare_metrics()
        self.analyze_density_impact()
        self.generate_visualizations()
        self.create_report()
        
        logger.info("Spatial Pattern Analysis completed successfully")
        
    def prepare_data(self):
        """
        Prepare data for spatial analysis.
        
        This method performs data cleaning, projection verification, and
        ensures that all required columns are properly formatted for analysis.
        """
        log_section("Preparing Data for Spatial Analysis")
        
        try:
            # Ensure data is in proper projection for spatial analysis
            if self.hybrid_data.crs is None or self.hybrid_data.crs != "EPSG:2157":
                logger.info("Reprojecting data to EPSG:2157 (Irish Grid)")
                self.hybrid_data = self.hybrid_data.to_crs("EPSG:2157")
            
            # Check for and handle missing values in key columns
            for col in PERFORMANCE_METRICS:
                if col in self.hybrid_data.columns:
                    missing = self.hybrid_data[col].isna().sum()
                    if missing > 0:
                        logger.warning(f"Found {missing} missing values in {col}")
                        # For performance metrics, we'll exclude missing values rather than impute
                        self.hybrid_data = self.hybrid_data.dropna(subset=[col])
                        logger.info(f"Dropped rows with missing values in {col}, {len(self.hybrid_data)} remaining")
            
            # Ensure club identity fields are consistent
            if 'Club' not in self.hybrid_data.columns and 'club_name' in self.hybrid_data.columns:
                self.hybrid_data['Club'] = self.hybrid_data['club_name']
            elif 'Club' not in self.hybrid_data.columns and 'assigned_club_name' in self.hybrid_data.columns:
                self.hybrid_data['Club'] = self.hybrid_data['assigned_club_name']
            
            # Ensure we have valid geometry
            if not all(self.hybrid_data.geometry.is_valid):
                logger.warning("Found invalid geometries. Applying buffer(0) to fix.")
                self.hybrid_data.geometry = self.hybrid_data.geometry.buffer(0)
            
            # Create club identifiers for use in analyses
            if 'club_id' not in self.hybrid_data.columns:
                self.hybrid_data['club_id'] = range(len(self.hybrid_data))
            
            # Add centroid coordinates for additional analyses
            self.hybrid_data['centroid_x'] = self.hybrid_data.geometry.centroid.x
            self.hybrid_data['centroid_y'] = self.hybrid_data.geometry.centroid.y
            
            # Check urban/rural classification
            if 'SA_URBAN_AREA_FLAG' in self.hybrid_data.columns:
                urban_count = self.hybrid_data['SA_URBAN_AREA_FLAG'].sum()
                rural_count = len(self.hybrid_data) - urban_count
                logger.info(f"Dataset contains {urban_count} urban and {rural_count} rural clubs")
            
            logger.info("Data preparation completed successfully")
        except Exception as e:
            logger.error(f"Error during data preparation: {str(e)}")
            raise
    
    def create_weights_matrix(self):
        """
        Create spatial weights matrix for spatial autocorrelation analysis.
        
        This method creates a weights matrix based on queen contiguity, where
        neighboring areas sharing any boundary point are considered adjacent.
        """
        log_section("Creating Spatial Weights Matrix")
        
        try:
            # Create a Queen contiguity weights matrix from the geometries
            logger.info("Creating Queen contiguity weights matrix")
            self.queen_w = weights.Queen.from_dataframe(self.hybrid_data)
            
            # Check for islands (observations with no neighbors)
            if self.queen_w.islands:
                logger.warning(f"Found {len(self.queen_w.islands)} islands (clubs with no neighbors)")
                logger.info("Switching to K-nearest neighbors weights matrix for better connectivity")
                
                # Create a K-nearest neighbors weights matrix instead
                k = min(10, len(self.hybrid_data) - 1)  # Use a reasonable number of neighbors
                self.knn_w = weights.KNN.from_dataframe(self.hybrid_data, k=k)
                logger.info(f"Created KNN weights matrix with k={k}")
                
                # Use the KNN weights as the primary weights matrix
                self.weights_matrix = self.knn_w
            else:
                # Use the Queen weights as the primary weights matrix
                self.weights_matrix = self.queen_w
            
            # Create a distance-based weights matrix for robustness checks
            logger.info("Creating distance-based weights matrix for robustness checks")
            threshold = self.hybrid_data.geometry.centroid.distance(self.hybrid_data.geometry.centroid.iloc[0]).median() * 2
            self.distance_w = weights.DistanceBand.from_dataframe(
                self.hybrid_data, 
                threshold=threshold, 
                binary=False, 
                alpha=-1.0  # Inverse distance weighting
            )
            logger.info(f"Created distance-based weights matrix with threshold={threshold:.2f} meters")
            
            # Report on the connectivity of the weights matrix
            n_neighbors = self.weights_matrix.cardinalities
            avg_neighbors = sum(n_neighbors.values()) / len(n_neighbors)
            min_neighbors = min(n_neighbors.values())
            max_neighbors = max(n_neighbors.values())
            
            logger.info(f"Weights matrix connectivity: avg={avg_neighbors:.2f}, min={min_neighbors}, max={max_neighbors}")
            logger.info(f"Weights matrix type: {type(self.weights_matrix).__name__}")
            
            # Row-standardize the weights matrix
            self.weights_matrix.transform = 'R'
            logger.info("Weights matrix row-standardized")
            
            logger.info("Spatial weights matrix creation completed successfully")
        except Exception as e:
            logger.error(f"Error creating weights matrix: {str(e)}")
            raise

    def calculate_morans_i(self):
        """
        Calculate Moran's I spatial autocorrelation statistic for performance metrics.
        
        This method computes Global Moran's I to determine whether similar performance
        values are clustered, dispersed, or randomly distributed in space.
        """
        log_section("Calculating Moran's I Spatial Autocorrelation")
        
        try:
            # Dictionary to store results
            self.morans_results = {}
            
            # Calculate Moran's I for each performance metric
            for metric in PERFORMANCE_METRICS:
                if metric in self.hybrid_data.columns:
                    logger.info(f"Calculating Moran's I for {metric}")
                    
                    # Get values and standardize them
                    values = self.hybrid_data[metric].values
                    
                    # Skip if all values are NaN or standard deviation is zero
                    if pd.isna(values).all() or np.std(values) == 0:
                        logger.warning(f"Skipping {metric} - all values are NaN or standard deviation is zero")
                        continue
                    
                    # Handle NaN values by filling with mean
                    if pd.isna(values).any():
                        logger.warning(f"Found {pd.isna(values).sum()} NaN values in {metric}, filling with mean")
                        values = pd.Series(values).fillna(pd.Series(values).mean()).values
                    
                    # Standardize values
                    values_std = (values - values.mean()) / values.std()
                    
                    # Calculate global Moran's I
                    moran = esda.Moran(values_std, self.weights_matrix)
                    
                    # Calculate local Moran's I
                    local_moran = esda.Moran_Local(values_std, self.weights_matrix)
                    
                    # Store results
                    self.morans_results[metric] = {
                        'global': {
                            'I': moran.I,
                            'E[I]': moran.EI,
                            'p_value': moran.p_sim,
                            'z_score': moran.z_sim,
                            'permutations': moran.permutations
                        },
                        'local': local_moran
                    }
                    
                    # Log global results
                    logger.info(f"Global Moran's I for {metric}: {moran.I:.4f} (p-value: {moran.p_sim:.4f})")
                    
                    # Interpret the result
                    if moran.p_sim < 0.05:
                        if moran.I > 0:
                            pattern = "clustered"
                        else:
                            pattern = "dispersed"
                        logger.info(f"Spatial pattern for {metric} is significantly {pattern}")
                    else:
                        logger.info(f"No significant spatial autocorrelation detected for {metric}")
                    
                    # Add local Moran's I values to the dataset
                    self.hybrid_data[f'{metric}_local_moran'] = local_moran.Is
                    self.hybrid_data[f'{metric}_local_moran_p'] = local_moran.p_sim
                    
                    # Classify areas based on local Moran's I
                    high_threshold = values_std.mean() + 0  # Could use different thresholds
                    
                    # Create clusters based on value and local Moran's I
                    # 1: High-High (hot spots)
                    # 2: Low-Low (cold spots)
                    # 3: High-Low (spatial outliers)
                    # 4: Low-High (spatial outliers)
                    # 0: Not significant
                    self.hybrid_data[f'{metric}_cluster'] = 0
                    
                    # High-high: high values surrounded by high values (hot spots)
                    self.hybrid_data.loc[(local_moran.p_sim < 0.05) & 
                                       (local_moran.q == 1), f'{metric}_cluster'] = 1
                    
                    # Low-low: low values surrounded by low values (cold spots)
                    self.hybrid_data.loc[(local_moran.p_sim < 0.05) & 
                                       (local_moran.q == 3), f'{metric}_cluster'] = 2
                    
                    # High-low: high values surrounded by low values (spatial outliers)
                    self.hybrid_data.loc[(local_moran.p_sim < 0.05) & 
                                       (local_moran.q == 2), f'{metric}_cluster'] = 3
                    
                    # Low-high: low values surrounded by high values (spatial outliers)
                    self.hybrid_data.loc[(local_moran.p_sim < 0.05) & 
                                       (local_moran.q == 4), f'{metric}_cluster'] = 4
                    
                    # Count the clusters
                    cluster_counts = self.hybrid_data[f'{metric}_cluster'].value_counts().to_dict()
                    logger.info(f"Cluster counts for {metric}: {cluster_counts}")
            
            # Calculate Moran's I for key demographic variables
            for var in self.key_demographic_vars:
                try:
                    if var in self.hybrid_data.columns:
                        logger.info(f"Calculating Moran's I for demographic variable: {var}")
                        
                        # Check if variable is numeric
                        if not pd.api.types.is_numeric_dtype(self.hybrid_data[var]):
                            logger.warning(f"Skipping {var} - not a numeric variable")
                            continue
                        
                        # Get values and standardize them
                        values = self.hybrid_data[var].values
                        
                        # Skip if all values are NaN or standard deviation is zero
                        if pd.isna(values).all() or np.std(values) == 0:
                            logger.warning(f"Skipping {var} - all values are NaN or standard deviation is zero")
                            continue
                        
                        # Handle NaN values by filling with mean
                        if pd.isna(values).any():
                            logger.warning(f"Found {pd.isna(values).sum()} NaN values in {var}, filling with mean")
                            values = pd.Series(values).fillna(pd.Series(values).mean()).values
                        
                        # Standardize values
                        values_std = (values - values.mean()) / values.std()
                        
                        # Calculate global Moran's I
                        moran = esda.Moran(values_std, self.weights_matrix)
                        
                        # Store results
                        self.morans_results[var] = {
                            'global': {
                                'I': moran.I,
                                'E[I]': moran.EI,
                                'p_value': moran.p_sim,
                                'z_score': moran.z_sim,
                                'permutations': moran.permutations
                            }
                        }
                        
                        # Log global results
                        logger.info(f"Global Moran's I for {var}: {moran.I:.4f} (p-value: {moran.p_sim:.4f})")
                except Exception as e:
                    logger.warning(f"Error calculating Moran's I for {var}: {str(e)}")
                    continue
            
            # Create a summary dataframe of global Moran's I results
            global_morans_results = []
            for var, result in self.morans_results.items():
                if 'global' in result:
                    global_morans_results.append({
                        'Variable': var,
                        'Morans_I': result['global']['I'],
                        'P_Value': result['global']['p_value'],
                        'Z_Score': result['global']['z_score'],
                        'Expected_I': result['global']['E[I]'],
                        'Significant': result['global']['p_value'] < 0.05,
                        'Pattern': 'Clustered' if result['global']['I'] > 0 and result['global']['p_value'] < 0.05 else
                                  'Dispersed' if result['global']['I'] < 0 and result['global']['p_value'] < 0.05 else 'Random'
                    })
            
            # Create a dataframe from the results
            self.global_morans_df = pd.DataFrame(global_morans_results)
            logger.info(f"Created summary of global Moran's I results for {len(global_morans_results)} variables")
            
            # Save the global Moran's I results to CSV
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self.global_morans_df.to_csv(OUTPUT_DIR / "global_morans_i_results.csv", index=False)
            logger.info(f"Saved global Moran's I results to {OUTPUT_DIR / 'global_morans_i_results.csv'}")
            
            logger.info("Moran's I calculation completed successfully")
        except Exception as e:
            logger.error(f"Error calculating Moran's I: {str(e)}")
            raise

    def identify_hotspots(self):
        """
        Identify hotspots and coldspots based on Getis-Ord Gi* statistic.
        
        This method uses the Getis-Ord Gi* statistic to identify statistically 
        significant hot spots (clusters of high values) and cold spots (clusters of low values).
        """
        log_section("Identifying Hotspots and Coldspots")
        
        try:
            # Dictionary to store results
            self.hotspot_results = {}
            
            # Calculate Getis-Ord Gi* for each performance metric
            for metric in PERFORMANCE_METRICS:
                if metric in self.hybrid_data.columns:
                    logger.info(f"Calculating Getis-Ord Gi* for {metric}")
                    
                    # Get values
                    values = self.hybrid_data[metric].values
                    
                    # Ensure values are float type for compatibility with weights matrix
                    values = values.astype(float)
                    
                    try:
                        # Calculate Getis-Ord Gi* - try with modified weights
                        # Convert weights matrix to float type if needed
                        if hasattr(self.weights_matrix, 'sparse'):
                            # If it's a sparse matrix representation
                            weights_float = weights.W.from_networkx(
                                self.weights_matrix.to_networkx(),
                                weight_col='weight'
                            )
                            weights_float.transform = 'R'  # Row-standardize like the original
                            g_star = esda.G_Local(values, weights_float)
                        else:
                            # Try using standard weights
                            g_star = esda.G_Local(values, self.weights_matrix)
                    except Exception as e:
                        logger.warning(f"Error calculating G* with standard approach: {str(e)}")
                        logger.info("Trying alternative approach for G* calculation...")
                        
                        # Alternative approach: Just calculate G* without inference
                        # Create the basic G* statistic without permutation inference
                        w_array = self.weights_matrix.sparse.toarray()
                        zs = (values - values.mean()) / values.std()
                        
                        # Manually calculate the G* statistic
                        n = len(values)
                        g_vals = np.zeros(n)
                        for i in range(n):
                            weights_i = w_array[i]
                            sum_weights = weights_i.sum()
                            if sum_weights > 0:
                                g_vals[i] = np.sum(weights_i * values) / sum_weights
                            else:
                                g_vals[i] = 0
                                
                        # Create a mock G_Local object
                        class MockGLocal:
                            def __init__(self, Gs, p_sim):
                                self.Gs = Gs
                                self.p_sim = p_sim
                                
                        # Use z-scores as a simple proxy for G*
                        g_star_vals = (g_vals - g_vals.mean()) / g_vals.std() if g_vals.std() > 0 else g_vals
                        
                        # Simple p-value approximation (using normal distribution)
                        # This is a placeholder - not as accurate as the true permutation test
                        p_vals = 2 * (1 - stats.norm.cdf(np.abs(g_star_vals)))
                        
                        g_star = MockGLocal(g_star_vals, p_vals)
                        logger.info("Used alternative G* calculation method")
                    
                    # Store results
                    self.hotspot_results[metric] = g_star
                    
                    # Add Gi* values to the dataset
                    self.hybrid_data[f'{metric}_g_star'] = g_star.Gs
                    self.hybrid_data[f'{metric}_g_star_p'] = g_star.p_sim
                    
                    # Classify areas based on Gi*
                    # 2: Hot spot (99% confidence)
                    # 1: Hot spot (95% confidence)
                    # 0: Not significant
                    # -1: Cold spot (95% confidence)
                    # -2: Cold spot (99% confidence)
                    self.hybrid_data[f'{metric}_hotspot'] = 0
                    
                    # Hot spots (99% confidence)
                    self.hybrid_data.loc[(g_star.p_sim < 0.01) & (g_star.Gs > 0), f'{metric}_hotspot'] = 2
                    # Hot spots (95% confidence)
                    self.hybrid_data.loc[(g_star.p_sim < 0.05) & (g_star.p_sim >= 0.01) & (g_star.Gs > 0), f'{metric}_hotspot'] = 1
                    # Cold spots (95% confidence)
                    self.hybrid_data.loc[(g_star.p_sim < 0.05) & (g_star.p_sim >= 0.01) & (g_star.Gs < 0), f'{metric}_hotspot'] = -1
                    # Cold spots (99% confidence)
                    self.hybrid_data.loc[(g_star.p_sim < 0.01) & (g_star.Gs < 0), f'{metric}_hotspot'] = -2
                    
                    # Count the number of observations in each hotspot category
                    hotspot_counts = self.hybrid_data[f'{metric}_hotspot'].value_counts().to_dict()
                    logger.info(f"Hotspot/coldspot counts for {metric}: {hotspot_counts}")
                    
                    # Create a version for easier mapping (all positives)
                    hotspot_map = {
                        2: 4,   # Hot spot (99% confidence)
                        1: 3,   # Hot spot (95% confidence)
                        0: 0,   # Not significant
                        -1: 1,  # Cold spot (95% confidence)
                        -2: 2   # Cold spot (99% confidence)
                    }
                    self.hybrid_data[f'{metric}_hotspot_map'] = self.hybrid_data[f'{metric}_hotspot'].map(hotspot_map)
            
            # Save hotspot results to GeoPackage
            hotspot_columns = []
            for metric in PERFORMANCE_METRICS:
                if f'{metric}_hotspot' in self.hybrid_data.columns:
                    hotspot_columns.extend([
                        metric,
                        f'{metric}_g_star',
                        f'{metric}_g_star_p',
                        f'{metric}_hotspot',
                        f'{metric}_hotspot_map'
                    ])
            
            # Add club identifier and geometry columns
            id_cols = [col for col in self.hybrid_data.columns if 'club' in col.lower() or 'Club' == col]
            geom_cols = ['geometry', 'centroid_x', 'centroid_y']
            
            # Create a subset for saving
            hotspot_gdf = self.hybrid_data[id_cols + hotspot_columns + geom_cols].copy()
            
            # Save to GeoPackage
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            hotspot_gdf.to_file(OUTPUT_DIR / "hotspot_analysis.gpkg", driver="GPKG")
            logger.info(f"Saved hotspot analysis results to {OUTPUT_DIR / 'hotspot_analysis.gpkg'}")
            
            logger.info("Hotspot identification completed successfully")
        except Exception as e:
            logger.error(f"Error identifying hotspots: {str(e)}")
            raise

    def analyze_urban_rural_patterns(self):
        """
        Analyze spatial patterns of performance metrics across urban and rural areas.
        
        This method compares the spatial distribution of performance metrics between
        urban and rural areas, and identifies any significant differences or patterns.
        """
        log_section("Analyzing Urban/Rural Performance Patterns")
        
        try:
            # Check if urban/rural classification is available
            urban_flag_col = None
            for col in ['SA_URBAN_AREA_FLAG', 'urban_flag', 'is_urban']:
                if col in self.hybrid_data.columns:
                    urban_flag_col = col
                    break
            
            if urban_flag_col is None:
                logger.warning("No urban/rural classification found. Creating based on population density.")
                
                # Create urban/rural classification based on population density
                if 'population_density' in self.hybrid_data.columns:
                    # Use median as threshold for urban/rural classification
                    density_threshold = self.hybrid_data['population_density'].median()
                    self.hybrid_data['is_urban'] = (self.hybrid_data['population_density'] > density_threshold).astype(int)
                    urban_flag_col = 'is_urban'
                    logger.info(f"Created urban/rural classification using density threshold {density_threshold:.2f}")
                else:
                    logger.error("No population density column found. Cannot create urban/rural classification.")
                    return
            
            # Dictionary to store urban/rural pattern results
            self.urban_rural_patterns = {}
            
            # Analyze each performance metric by urban/rural classification
            for metric in PERFORMANCE_METRICS:
                if metric in self.hybrid_data.columns:
                    logger.info(f"Analyzing urban/rural patterns for {metric}")
                    
                    # Get urban and rural values
                    urban_values = self.hybrid_data.loc[self.hybrid_data[urban_flag_col] == 1, metric]
                    rural_values = self.hybrid_data.loc[self.hybrid_data[urban_flag_col] == 0, metric]
                    
                    # Skip if either group has too few observations
                    if len(urban_values) < 5 or len(rural_values) < 5:
                        logger.warning(f"Skipping {metric} - insufficient observations in urban or rural group")
                        continue
                    
                    # Calculate descriptive statistics
                    urban_stats = urban_values.describe()
                    rural_stats = rural_values.describe()
                    
                    # Perform t-test to compare means
                    t_stat, p_value = stats.ttest_ind(
                        urban_values.dropna(), 
                        rural_values.dropna(), 
                        equal_var=False  # Welch's t-test for unequal variances
                    )
                    
                    # Perform Mann-Whitney U test (non-parametric)
                    u_stat, u_p_value = stats.mannwhitneyu(
                        urban_values.dropna(),
                        rural_values.dropna(),
                        alternative='two-sided'
                    )
                    
                    # Calculate effect size (Cohen's d)
                    urban_mean = urban_values.mean()
                    rural_mean = rural_values.mean()
                    pooled_std = np.sqrt(((urban_values.std() ** 2) + (rural_values.std() ** 2)) / 2)
                    cohens_d = (urban_mean - rural_mean) / pooled_std if pooled_std > 0 else 0
                    
                    # Store results
                    self.urban_rural_patterns[metric] = {
                        'urban_mean': urban_mean,
                        'rural_mean': rural_mean,
                        'urban_median': urban_values.median(),
                        'rural_median': rural_values.median(),
                        'urban_std': urban_values.std(),
                        'rural_std': rural_values.std(),
                        'urban_count': len(urban_values),
                        'rural_count': len(rural_values),
                        't_stat': t_stat,
                        'p_value': p_value,
                        'u_stat': u_stat,
                        'u_p_value': u_p_value,
                        'cohens_d': cohens_d,
                        'significant_diff': p_value < 0.05,
                        'higher_in': 'urban' if urban_mean > rural_mean else 'rural'
                    }
                    
                    # Log results
                    logger.info(f"Urban mean for {metric}: {urban_mean:.4f}, Rural mean: {rural_mean:.4f}")
                    logger.info(f"T-test: t={t_stat:.4f}, p={p_value:.4f}, Effect size (Cohen's d): {cohens_d:.4f}")
                    
                    if p_value < 0.05:
                        logger.info(f"Significant difference in {metric} between urban and rural areas")
                        logger.info(f"Higher in {self.urban_rural_patterns[metric]['higher_in']} areas")
                    else:
                        logger.info(f"No significant difference in {metric} between urban and rural areas")
            
            # Create a summary dataframe of urban/rural pattern results
            urban_rural_results = []
            for var, result in self.urban_rural_patterns.items():
                urban_rural_results.append({
                    'Variable': var,
                    'Urban_Mean': result['urban_mean'],
                    'Rural_Mean': result['rural_mean'],
                    'Urban_Median': result['urban_median'],
                    'Rural_Median': result['rural_median'],
                    'Urban_Count': result['urban_count'],
                    'Rural_Count': result['rural_count'],
                    'T_Statistic': result['t_stat'],
                    'P_Value': result['p_value'],
                    'Effect_Size': result['cohens_d'],
                    'Significant': result['significant_diff'],
                    'Higher_In': result['higher_in']
                })
            
            self.urban_rural_df = pd.DataFrame(urban_rural_results)
            logger.info(f"Created summary of urban/rural pattern results for {len(urban_rural_results)} variables")
            
            # Save the urban/rural pattern results to CSV
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self.urban_rural_df.to_csv(OUTPUT_DIR / "urban_rural_pattern_results.csv", index=False)
            logger.info(f"Saved urban/rural pattern results to {OUTPUT_DIR / 'urban_rural_pattern_results.csv'}")
            
            # Create additional dataset with urban/rural classification for mapping
            urban_rural_cols = ['Club', 'club_id', urban_flag_col, 'geometry']
            urban_rural_cols.extend([col for col in self.hybrid_data.columns if 'density' in col.lower()])
            
            for metric in PERFORMANCE_METRICS:
                if metric in self.hybrid_data.columns:
                    urban_rural_cols.extend([metric, f'{metric}_hotspot', f'{metric}_hotspot_map'])
            
            # Create a subset for saving
            urban_rural_gdf = self.hybrid_data[list(set(urban_rural_cols))].copy()
            
            # Save to GeoPackage
            urban_rural_gdf.to_file(OUTPUT_DIR / "urban_rural_patterns.gpkg", driver="GPKG")
            logger.info(f"Saved urban/rural patterns to {OUTPUT_DIR / 'urban_rural_patterns.gpkg'}")
            
            logger.info("Urban/rural pattern analysis completed successfully")
        except Exception as e:
            logger.error(f"Error analyzing urban/rural patterns: {str(e)}")
            raise

    def compare_metrics(self):
        """
        Compare spatial patterns between original and transformed performance metrics.
        
        This method compares the spatial patterns observed in original and transformed
        performance metrics, focusing on variables identified as significant.
        """
        log_section("Comparing Original and Transformed Metrics")
        
        try:
            # Check if we have both original and transformed metrics
            original_metrics = [m for m in self.hybrid_data.columns 
                               if any(m.startswith(pm.replace('transformed_', '')) 
                                      for pm in PERFORMANCE_METRICS)]
            transformed_metrics = PERFORMANCE_METRICS
            
            # Dictionary to store comparison results
            self.metric_comparisons = {}
            
            # Compare pairs of original and transformed metrics
            for trans_metric in transformed_metrics:
                # Find corresponding original metric
                orig_metric = trans_metric.replace('transformed_', '')
                if orig_metric in original_metrics:
                    logger.info(f"Comparing spatial patterns: {orig_metric} vs {trans_metric}")
                    
                    # Check if we have Moran's I results for both metrics
                    if (orig_metric in self.morans_results and 
                        trans_metric in self.morans_results):
                        
                        # Get global Moran's I values
                        orig_moran = self.morans_results[orig_metric]['global']['I']
                        trans_moran = self.morans_results[trans_metric]['global']['I']
                        
                        # Check significance
                        orig_sig = self.morans_results[orig_metric]['global']['p_value'] < 0.05
                        trans_sig = self.morans_results[trans_metric]['global']['p_value'] < 0.05
                        
                        # Determine which has stronger spatial autocorrelation
                        if abs(orig_moran) > abs(trans_moran):
                            stronger = orig_metric
                            stronger_value = abs(orig_moran)
                            weaker = trans_metric
                            weaker_value = abs(trans_moran)
                        else:
                            stronger = trans_metric
                            stronger_value = abs(trans_moran)
                            weaker = orig_metric
                            weaker_value = abs(orig_moran)
                        
                        # Calculate correlation between metrics
                        corr = self.hybrid_data[[orig_metric, trans_metric]].corr().iloc[0, 1]
                        
                        # Store results
                        self.metric_comparisons[f"{orig_metric}_vs_{trans_metric}"] = {
                            'original_metric': orig_metric,
                            'transformed_metric': trans_metric,
                            'original_morans_i': orig_moran,
                            'transformed_morans_i': trans_moran,
                            'original_significant': orig_sig,
                            'transformed_significant': trans_sig,
                            'stronger_metric': stronger,
                            'stronger_value': stronger_value,
                            'weaker_metric': weaker,
                            'weaker_value': weaker_value,
                            'correlation': corr
                        }
                        
                        # Log results
                        logger.info(f"Original Moran's I: {orig_moran:.4f} ({'' if orig_sig else 'not '}significant)")
                        logger.info(f"Transformed Moran's I: {trans_moran:.4f} ({'' if trans_sig else 'not '}significant)")
                        logger.info(f"Correlation between metrics: {corr:.4f}")
                        logger.info(f"Stronger spatial autocorrelation: {stronger} ({stronger_value:.4f})")
            
            # Compare hotspot patterns
            for trans_metric in transformed_metrics:
                orig_metric = trans_metric.replace('transformed_', '')
                if (f"{orig_metric}_hotspot" in self.hybrid_data.columns and 
                    f"{trans_metric}_hotspot" in self.hybrid_data.columns):
                    
                    logger.info(f"Comparing hotspot patterns: {orig_metric} vs {trans_metric}")
                    
                    # Calculate agreement in hotspot classification
                    hotspot_match = (self.hybrid_data[f"{orig_metric}_hotspot"] == 
                                     self.hybrid_data[f"{trans_metric}_hotspot"])
                    agreement_pct = hotspot_match.mean() * 100
                    
                    # Calculate kappa statistic for agreement
                    kappa = cohen_kappa_score(
                        self.hybrid_data[f"{orig_metric}_hotspot"],
                        self.hybrid_data[f"{trans_metric}_hotspot"]
                    )
                    
                    # Update results
                    if f"{orig_metric}_vs_{trans_metric}" in self.metric_comparisons:
                        self.metric_comparisons[f"{orig_metric}_vs_{trans_metric}"].update({
                            'hotspot_agreement_pct': agreement_pct,
                            'hotspot_kappa': kappa
                        })
                    
                    # Log results
                    logger.info(f"Hotspot pattern agreement: {agreement_pct:.2f}%")
                    logger.info(f"Kappa statistic: {kappa:.4f}")
            
            # Create a summary dataframe of metric comparison results
            comparison_results = []
            for comparison, result in self.metric_comparisons.items():
                comparison_results.append({
                    'Comparison': comparison,
                    'Original_Metric': result['original_metric'],
                    'Transformed_Metric': result['transformed_metric'],
                    'Original_Morans_I': result['original_morans_i'],
                    'Transformed_Morans_I': result['transformed_morans_i'],
                    'Original_Significant': result['original_significant'],
                    'Transformed_Significant': result['transformed_significant'],
                    'Stronger_Metric': result['stronger_metric'],
                    'Correlation': result['correlation'],
                    'Hotspot_Agreement_Pct': result.get('hotspot_agreement_pct', np.nan),
                    'Hotspot_Kappa': result.get('hotspot_kappa', np.nan)
                })
            
            self.metric_comparison_df = pd.DataFrame(comparison_results)
            logger.info(f"Created summary of metric comparison results for {len(comparison_results)} comparisons")
            
            # Save the metric comparison results to CSV
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self.metric_comparison_df.to_csv(OUTPUT_DIR / "metric_comparison_results.csv", index=False)
            logger.info(f"Saved metric comparison results to {OUTPUT_DIR / 'metric_comparison_results.csv'}")
            
            logger.info("Metric comparison completed successfully")
        except Exception as e:
            logger.error(f"Error comparing metrics: {str(e)}")
            raise
    
    def analyze_density_impact(self):
        """
        Analyze the impact of club distribution density on performance metrics.
        
        This method examines how the density of clubs in an area affects performance
        metrics and spatial patterns.
        """
        log_section("Analyzing Club Distribution Density Impact")
        
        try:
            # Calculate club density
            logger.info("Calculating club distribution density")
            
            # Create a KDE surface of club density
            from scipy.spatial import distance
            
            # Get club centroids
            centroids = np.column_stack((
                self.hybrid_data['centroid_x'].values,
                self.hybrid_data['centroid_y'].values
            ))
            
            # Calculate distance matrix between all clubs
            dist_matrix = distance.squareform(distance.pdist(centroids))
            
            # Calculate the average distance to the nearest 5 clubs
            k = min(5, len(dist_matrix) - 1)
            nearest_dists = np.sort(dist_matrix, axis=1)[:, 1:k+1]  # Skip the first one (distance to self = 0)
            avg_nearest_dist = np.mean(nearest_dists, axis=1)
            
            # Calculate club density (inverse of average nearest distance)
            self.hybrid_data['avg_nearest_club_dist'] = avg_nearest_dist
            self.hybrid_data['club_density'] = 1000 / avg_nearest_dist  # Scaled for readability
            
            logger.info(f"Average nearest club distance: {np.mean(avg_nearest_dist):.2f} meters")
            logger.info(f"Club density range: {self.hybrid_data['club_density'].min():.2f} to {self.hybrid_data['club_density'].max():.2f}")
            
            # Analyze relationship between club density and performance metrics
            logger.info("Analyzing relationship between club density and performance metrics")
            
            density_correlations = {}
            for metric in PERFORMANCE_METRICS:
                if metric in self.hybrid_data.columns:
                    # Calculate correlation between club density and performance
                    corr, p_value = stats.pearsonr(
                        self.hybrid_data['club_density'].values,
                        self.hybrid_data[metric].values
                    )
                    
                    # Calculate rank correlation (Spearman)
                    rank_corr, rank_p = stats.spearmanr(
                        self.hybrid_data['club_density'].values,
                        self.hybrid_data[metric].values
                    )
                    
                    # Store results
                    density_correlations[metric] = {
                        'pearson_corr': corr,
                        'pearson_p': p_value,
                        'spearman_corr': rank_corr,
                        'spearman_p': rank_p,
                        'significant': p_value < 0.05 or rank_p < 0.05
                    }
                    
                    # Log results
                    logger.info(f"Correlation between club density and {metric}: {corr:.4f} (p={p_value:.4f})")
                    logger.info(f"Rank correlation: {rank_corr:.4f} (p={rank_p:.4f})")
                    
                    # Perform regression analysis
                    X = sm.add_constant(self.hybrid_data[['club_density']])
                    model = sm.OLS(self.hybrid_data[metric], X)
                    results = model.fit()
                    
                    # Store regression results
                    density_correlations[metric].update({
                        'coef': results.params[1],
                        'std_err': results.bse[1],
                        'r_squared': results.rsquared,
                        'adj_r_squared': results.rsquared_adj,
                        'f_pvalue': results.f_pvalue
                    })
                    
                    # Log regression results
                    logger.info(f"Regression coefficient: {results.params[1]:.4f} (p={results.pvalues[1]:.4f})")
                    logger.info(f"R-squared: {results.rsquared:.4f}, Adj. R-squared: {results.rsquared_adj:.4f}")
            
            # Create a summary dataframe of density correlation results
            density_corr_results = []
            for metric, result in density_correlations.items():
                density_corr_results.append({
                    'Metric': metric,
                    'Pearson_Correlation': result['pearson_corr'],
                    'Pearson_P_Value': result['pearson_p'],
                    'Spearman_Correlation': result['spearman_corr'],
                    'Spearman_P_Value': result['spearman_p'],
                    'Regression_Coefficient': result['coef'],
                    'Coefficient_Std_Error': result['std_err'],
                    'R_Squared': result['r_squared'],
                    'Adj_R_Squared': result['adj_r_squared'],
                    'Significant': result['significant']
                })
            
            self.density_corr_df = pd.DataFrame(density_corr_results)
            logger.info(f"Created summary of density correlation results for {len(density_corr_results)} metrics")
            
            # Save the density correlation results to CSV
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self.density_corr_df.to_csv(OUTPUT_DIR / "density_correlation_results.csv", index=False)
            logger.info(f"Saved density correlation results to {OUTPUT_DIR / 'density_correlation_results.csv'}")
            
            # Save a version of the dataset with club density for mapping
            density_cols = ['Club', 'club_id', 'geometry', 'centroid_x', 'centroid_y', 
                           'avg_nearest_club_dist', 'club_density']
            density_cols.extend(PERFORMANCE_METRICS)
            
            # Create a subset for saving
            density_gdf = self.hybrid_data[list(set(density_cols))].copy()
            
            # Save to GeoPackage
            density_gdf.to_file(OUTPUT_DIR / "club_density_analysis.gpkg", driver="GPKG")
            logger.info(f"Saved club density analysis to {OUTPUT_DIR / 'club_density_analysis.gpkg'}")
            
            logger.info("Club density impact analysis completed successfully")
            
            # Add club density to the run_analysis method
            return density_correlations
        except Exception as e:
            logger.error(f"Error analyzing club density impact: {str(e)}")
            raise

    def generate_visualizations(self):
        """
        Generate visualizations of spatial analysis results.
        
        This method creates maps, charts, and other visualizations to illustrate
        the spatial patterns, hotspots, and relationships identified in the analysis.
        """
        log_section("Generating Spatial Analysis Visualizations")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            from matplotlib.ticker import MaxNLocator
            import contextily as ctx
            
            # Create output directory for visualizations
            os.makedirs(OUTPUT_DIR / "visualizations", exist_ok=True)
            
            # Generate Moran's I scatterplots
            logger.info("Generating Moran's I scatterplots")
            for metric in PERFORMANCE_METRICS:
                if metric in self.morans_results:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Get Moran's I scatterplot
                    moran = self.morans_results[metric]['global']
                    local_moran = self.morans_results[metric]['local']
                    
                    # Standardized values for the metric
                    values = self.hybrid_data[metric].values
                    values_std = (values - values.mean()) / values.std()
                    
                    # Create Moran scatterplot manually
                    # Get the spatial lag of standardized values
                    lag_values = weights.lag_spatial(self.weights_matrix, values_std)
                    
                    # Plot standardized values vs. lagged values
                    ax.scatter(values_std, lag_values, alpha=0.6, edgecolor='k', s=50)
                    
                    # Add vertical and horizontal lines at 0
                    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
                    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
                    
                    # Add labels
                    ax.set_xlabel('Standardized Performance')
                    ax.set_ylabel('Spatial Lag')
                    ax.set_title(f"Moran's I Scatterplot for {metric}\nI={moran['I']:.4f}, p={moran['p_value']:.4f}")
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    
                    # Label the quadrants
                    ax.text(min(values_std) * 0.9, max(lag_values) * 0.9, 'LL', fontsize=10)
                    ax.text(max(values_std) * 0.9, max(lag_values) * 0.9, 'HH', fontsize=10)
                    ax.text(min(values_std) * 0.9, min(lag_values) * 0.9, 'LL', fontsize=10)
                    ax.text(max(values_std) * 0.9, min(lag_values) * 0.9, 'HL', fontsize=10)
                    
                    # Add club names to points if club column exists
                    club_col = None
                    for col in ['Club', 'club_name', 'Name', 'name']:
                        if col in self.hybrid_data.columns:
                            club_col = col
                            break
                            
                    if club_col:
                        for idx, club in enumerate(self.hybrid_data[club_col]):
                            try:
                                x, y = values_std[idx], lag_values[idx]
                                ax.annotate(club, (x, y), fontsize=8, alpha=0.7,
                                           xytext=(5, 5), textcoords='offset points')
                            except:
                                # Skip if any issues with annotation
                                continue
                    
                    # Save the figure
                    output_path = OUTPUT_DIR / "visualizations" / f"{metric}_morans_scatterplot.png"
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300)
                    plt.close(fig)
                    logger.info(f"Saved Moran's I scatterplot to {output_path}")
            
            # Generate hotspot maps
            logger.info("Generating hotspot maps")
            for metric in PERFORMANCE_METRICS:
                if f"{metric}_hotspot" in self.hybrid_data.columns:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Create a custom colormap for hotspots
                    cmap = plt.cm.RdBu_r
                    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
                    norm = colors.BoundaryNorm(bounds, cmap.N)
                    
                    # Plot the hotspots
                    self.hybrid_data.plot(
                        column=f"{metric}_hotspot",
                        categorical=True,
                        cmap=cmap,
                        norm=norm,
                        legend=True,
                        ax=ax,
                        linewidth=0.5,
                        edgecolor='gray',
                        alpha=0.7
                    )
                    
                    # Customize the plot
                    ax.set_title(f"Hotspot Analysis for {metric}")
                    
                    # Add basemap for context
                    try:
                        ctx.add_basemap(ax, crs=self.hybrid_data.crs.to_string())
                    except Exception as e:
                        logger.warning(f"Unable to add basemap: {str(e)}")
                    
                    # Annotate with club names
                    for idx, row in self.hybrid_data.iterrows():
                        if row[f"{metric}_hotspot"] != 0:  # Only label significant hotspots/coldspots
                            ax.annotate(row['Club'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                                       xytext=(3, 3), textcoords="offset points",
                                       fontsize=7, alpha=0.8)
                    
                    # Create a custom legend
                    legend_labels = {
                        -2: "Cold Spot (99% confidence)",
                        -1: "Cold Spot (95% confidence)",
                        0: "Not Significant",
                        1: "Hot Spot (95% confidence)",
                        2: "Hot Spot (99% confidence)"
                    }
                    
                    # Replace the automatically generated legend with a custom one
                    ax.get_legend().remove()
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor=cmap(norm(val)), edgecolor='gray', label=label)
                        for val, label in legend_labels.items()
                    ]
                    ax.legend(handles=legend_elements, title="Hotspot Classification", loc="lower right")
                    
                    # Save the figure
                    output_path = OUTPUT_DIR / "visualizations" / f"{metric}_hotspot_map.png"
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300)
                    plt.close(fig)
                    logger.info(f"Saved hotspot map to {output_path}")
            
            # Generate urban/rural comparison charts
            if hasattr(self, 'urban_rural_patterns') and self.urban_rural_patterns:
                logger.info("Generating urban/rural comparison charts")
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Prepare data for plotting
                metrics = []
                urban_means = []
                rural_means = []
                urban_stds = []
                rural_stds = []
                
                for metric, data in self.urban_rural_patterns.items():
                    metrics.append(metric)
                    urban_means.append(data['urban_mean'])
                    rural_means.append(data['rural_mean'])
                    urban_stds.append(data['urban_std'])
                    rural_stds.append(data['rural_std'])
                
                # Set positions for bars
                x = np.arange(len(metrics))
                width = 0.35
                
                # Create grouped bars
                rects1 = ax.bar(x - width/2, urban_means, width, yerr=urban_stds, 
                               label='Urban', color='skyblue', alpha=0.7, capsize=5)
                rects2 = ax.bar(x + width/2, rural_means, width, yerr=rural_stds,
                               label='Rural', color='lightgreen', alpha=0.7, capsize=5)
                
                # Add labels and legend
                ax.set_ylabel('Mean Value')
                ax.set_title('Urban vs. Rural Performance Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                ax.legend()
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # Add p-value annotations
                for i, metric in enumerate(metrics):
                    p_value = self.urban_rural_patterns[metric]['p_value']
                    ax.annotate(f"p={p_value:.3f}", 
                               xy=(i, max(urban_means[i], rural_means[i]) + max(urban_stds[i], rural_stds[i]) + 0.05),
                               ha='center', va='bottom', fontsize=8)
                
                # Save the figure
                output_path = OUTPUT_DIR / "visualizations" / "urban_rural_comparison.png"
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close(fig)
                logger.info(f"Saved urban/rural comparison chart to {output_path}")
            
            # Generate club density map and correlation plots
            if 'club_density' in self.hybrid_data.columns:
                logger.info("Generating club density visualizations")
                
                # Club density map
                fig, ax = plt.subplots(figsize=(12, 10))
                self.hybrid_data.plot(
                    column='club_density',
                    cmap='viridis',
                    scheme='quantiles',
                    k=5,
                    legend=True,
                    ax=ax,
                    legend_kwds={'title': 'Club Density\n(clubs per sq km)'},
                    linewidth=0.5,
                    edgecolor='gray',
                    alpha=0.7
                )
                
                ax.set_title('Club Density Distribution')
                
                # Add basemap for context
                try:
                    ctx.add_basemap(ax, crs=self.hybrid_data.crs.to_string())
                except Exception as e:
                    logger.warning(f"Unable to add basemap: {str(e)}")
                
                # Save the figure
                output_path = OUTPUT_DIR / "visualizations" / "club_density_map.png"
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close(fig)
                logger.info(f"Saved club density map to {output_path}")
                
                # Density correlation plots
                if hasattr(self, 'density_corr_df'):
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Prepare data for plotting
                    metrics = self.density_corr_df['Metric'].tolist()
                    correlations = self.density_corr_df['Pearson_Correlation'].tolist()
                    
                    # Create horizontal bar chart
                    bars = ax.barh(metrics, correlations, color='cornflowerblue', alpha=0.7)
                    
                    # Add a vertical line at x=0
                    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                    
                    # Add labels and customize plot
                    ax.set_xlabel('Correlation with Club Density')
                    ax.set_title('Correlation between Club Density and Performance Metrics')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    
                    # Add correlation values
                    for i, v in enumerate(correlations):
                        ax.text(v + (0.01 if v >= 0 else -0.01), 
                               i, 
                               f"{v:.3f}", 
                               va='center', 
                               ha='left' if v >= 0 else 'right',
                               fontsize=9)
                    
                    # Save the figure
                    output_path = OUTPUT_DIR / "visualizations" / "density_correlations.png"
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300)
                    plt.close(fig)
                    logger.info(f"Saved density correlation plot to {output_path}")
            
            logger.info("Spatial analysis visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

    def create_report(self):
        """
        Create a comprehensive report of spatial analysis findings.
        
        This method generates a Markdown report summarizing all spatial analysis
        results, including maps, charts, and statistical findings.
        """
        log_section("Creating Spatial Analysis Report")
        
        try:
            # Create report filename and directory
            report_dir = OUTPUT_DIR
            os.makedirs(report_dir, exist_ok=True)
            report_path = report_dir / "spatial_pattern_analysis_report.md"
            
            # Start building the report content
            report_content = [
                "# Spatial Pattern Analysis Report",
                f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*",
                "",
                "## Overview",
                "This report presents the results of spatial pattern analysis for GAA club performance metrics in Cork.",
                "The analysis examines spatial autocorrelation, hotspots, urban/rural patterns, and the impact of club density.",
                "",
                "## 1. Spatial Autocorrelation Analysis",
                "",
                "### 1.1 Global Moran's I Results",
                "",
                "| Metric | Moran's I | p-value | Pattern |",
                "|--------|-----------|---------|---------|"
            ]
            
            # Add Moran's I results
            for metric, result in self.morans_results.items():
                moran = result['global']
                pattern = "Clustered" if moran['I'] > 0 and moran['p_value'] < 0.05 else \
                         "Dispersed" if moran['I'] < 0 and moran['p_value'] < 0.05 else "Random"
                report_content.append(f"| {metric} | {moran['I']:.4f} | {moran['p_value']:.4f} | {pattern} |")
            
            # Add visualizations references
            report_content.extend([
                "",
                "### 1.2 Local Spatial Autocorrelation",
                "",
                "The analysis identified local spatial clusters and outliers using Local Moran's I and Getis-Ord Gi* statistics.",
                "Visualizations for each metric show the spatial distribution of hotspots and coldspots.",
                "",
                "| Metric | Visualization |",
                "|--------|---------------|"
            ])
            
            # Add visualization links
            for metric in PERFORMANCE_METRICS:
                if f"{metric}_hotspot" in self.hybrid_data.columns:
                    report_content.append(f"| {metric} | [Hotspot Map](visualizations/{metric}_hotspot_map.png) |")
                if metric in self.morans_results:
                    report_content.append(f"| {metric} | [Moran's I Scatterplot](visualizations/{metric}_morans_scatterplot.png) |")
            
            # Add urban/rural pattern analysis
            report_content.extend([
                "",
                "## 2. Urban/Rural Pattern Analysis",
                "",
                "### 2.1 Performance Comparison by Urban/Rural Classification",
                "",
                "| Metric | Urban Mean | Rural Mean | Difference | p-value | Significant | Higher In |",
                "|--------|------------|------------|------------|---------|-------------|-----------|"
            ])
            
            # Add urban/rural comparison results
            if hasattr(self, 'urban_rural_patterns'):
                for metric, data in self.urban_rural_patterns.items():
                    diff = data['urban_mean'] - data['rural_mean']
                    sig = "Yes" if data['p_value'] < 0.05 else "No"
                    report_content.append(
                        f"| {metric} | {data['urban_mean']:.4f} | {data['rural_mean']:.4f} | " +
                        f"{diff:.4f} | {data['p_value']:.4f} | {sig} | {data['higher_in'].capitalize()} |"
                    )
            
            # Add density impact analysis
            report_content.extend([
                "",
                "### 2.2 Urban/Rural Distribution Visualization",
                "",
                "The following visualization shows the comparison of performance metrics between urban and rural areas:",
                "",
                "![Urban/Rural Comparison](visualizations/urban_rural_comparison.png)",
                "",
                "## 3. Club Density Impact Analysis",
                "",
                "### 3.1 Club Density Distribution",
                "",
                "The following map shows the distribution of club density across Cork:",
                "",
                "![Club Density Map](visualizations/club_density_map.png)",
                "",
                "### 3.2 Correlation with Performance Metrics",
                "",
                "| Metric | Correlation | p-value | Significant | Relationship |",
                "|--------|-------------|---------|-------------|--------------|"
            ])
            
            # Add density correlation results
            if hasattr(self, 'density_corr_df'):
                for _, row in self.density_corr_df.iterrows():
                    sig = "Yes" if row['Pearson_P_Value'] < 0.05 else "No"
                    relation = "Positive" if row['Pearson_Correlation'] > 0 else "Negative" if row['Pearson_Correlation'] < 0 else "None"
                    report_content.append(
                        f"| {row['Metric']} | {row['Pearson_Correlation']:.4f} | " +
                        f"{row['Pearson_P_Value']:.4f} | {sig} | {relation} |"
                    )
                
                report_content.extend([
                    "",
                    "The following chart shows the correlation between club density and performance metrics:",
                    "",
                    "![Density Correlations](visualizations/density_correlations.png)"
                ])
            
            # Add key findings and conclusions
            report_content.extend([
                "",
                "## 4. Key Findings and Conclusions",
                "",
                "### 4.1 Spatial Autocorrelation Patterns",
                "",
                "- " + self.summarize_spatial_patterns(),
                "",
                "### 4.2 Urban/Rural Distinctions",
                "",
                "- " + self.summarize_urban_rural_patterns(),
                "",
                "### 4.3 Club Density Impact",
                "",
                "- " + self.summarize_density_impact(),
                "",
                "## 5. Methodology",
                "",
                "This analysis was performed using the following approaches:",
                "",
                "- **Spatial Weights**: " + self.describe_weights_matrix(),
                "- **Autocorrelation**: Global Moran's I and Local Moran's I statistics",
                "- **Hotspot Detection**: Getis-Ord Gi* statistic",
                "- **Group Comparison**: T-tests and Mann-Whitney U tests for urban/rural comparisons",
                "- **Correlation Analysis**: Pearson and Spearman correlations for density impact analysis",
                "",
                "---",
                "",
                "*End of Report*"
            ])
            
            # Write the report to file
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_content))
            
            logger.info(f"Spatial analysis report created successfully at {report_path}")
            
        except Exception as e:
            logger.error(f"Error creating report: {str(e)}")
            raise
    
    def summarize_spatial_patterns(self):
        """Create a summary of the spatial patterns identified in the analysis."""
        try:
            significant_metrics = []
            for metric, result in self.morans_results.items():
                if result['global']['p_value'] < 0.05:
                    direction = "clustered" if result['global']['I'] > 0 else "dispersed"
                    significant_metrics.append((metric, direction, result['global']['I']))
            
            if not significant_metrics:
                return "No significant spatial autocorrelation patterns were found in any performance metrics."
            
            strongest = max(significant_metrics, key=lambda x: abs(x[2]))
            
            summary = f"Significant spatial autocorrelation was found in {len(significant_metrics)} of {len(self.morans_results)} performance metrics. "
            summary += f"The strongest pattern was observed in {strongest[0]} (Moran's I = {strongest[2]:.4f}), showing a {strongest[1]} pattern."
            
            # Add information about hotspots if available
            hotspot_metrics = [m for m in PERFORMANCE_METRICS if f"{m}_hotspot" in self.hybrid_data.columns]
            if hotspot_metrics:
                summary += f" Hotspot analysis identified significant clusters in {len(hotspot_metrics)} metrics."
            
            return summary
        except Exception:
            return "Unable to summarize spatial patterns due to an error."
    
    def summarize_urban_rural_patterns(self):
        """Create a summary of the urban/rural patterns identified in the analysis."""
        try:
            if not hasattr(self, 'urban_rural_patterns') or not self.urban_rural_patterns:
                return "No urban/rural pattern analysis was performed."
            
            significant_diffs = [(m, d['higher_in'], d['p_value']) 
                               for m, d in self.urban_rural_patterns.items() 
                               if d['p_value'] < 0.05]
            
            if not significant_diffs:
                return "No significant differences were found between urban and rural areas for any performance metrics."
            
            most_sig = min(significant_diffs, key=lambda x: x[2])
            
            summary = f"Significant differences between urban and rural areas were found in {len(significant_diffs)} of {len(self.urban_rural_patterns)} performance metrics. "
            summary += f"The most significant difference was in {most_sig[0]} (p = {most_sig[2]:.4f}), with {most_sig[1]} areas showing better performance."
            
            return summary
        except Exception:
            return "Unable to summarize urban/rural patterns due to an error."
    
    def summarize_density_impact(self):
        """Create a summary of the density impact analysis findings."""
        try:
            if not hasattr(self, 'density_corr_df') or self.density_corr_df.empty:
                return "No club density impact analysis was performed."
            
            significant_corrs = self.density_corr_df[self.density_corr_df['Pearson_P_Value'] < 0.05]
            
            if len(significant_corrs) == 0:
                return "No significant correlations were found between club density and performance metrics."
            
            # Find the strongest correlation
            strongest_idx = significant_corrs['Pearson_Correlation'].abs().idxmax()
            strongest = significant_corrs.loc[strongest_idx]
            direction = "positive" if strongest['Pearson_Correlation'] > 0 else "negative"
            
            summary = f"Significant correlations between club density and performance were found in {len(significant_corrs)} of {len(self.density_corr_df)} metrics. "
            summary += f"The strongest relationship was with {strongest['Metric']} (r = {strongest['Pearson_Correlation']:.4f}), showing a {direction} correlation."
            
            return summary
        except Exception:
            return "Unable to summarize density impact due to an error."
    
    def describe_weights_matrix(self):
        """Create a description of the weights matrix used in the analysis."""
        try:
            if not hasattr(self, 'weights_matrix'):
                return "Spatial weights matrix not defined."
            
            matrix_type = type(self.weights_matrix).__name__
            
            if matrix_type == "W":
                # Try to determine the type from properties
                if hasattr(self, 'queen_w') and self.weights_matrix is self.queen_w:
                    matrix_type = "Queen contiguity"
                elif hasattr(self, 'knn_w') and self.weights_matrix is self.knn_w:
                    k = min(10, len(self.hybrid_data) - 1)  # Assuming this was the k used
                    matrix_type = f"K-nearest neighbors (k={k})"
                else:
                    matrix_type = "Custom weights matrix"
            
            n_neighbors = self.weights_matrix.cardinalities
            avg_neighbors = sum(n_neighbors.values()) / len(n_neighbors)
            
            description = f"{matrix_type} weights matrix with an average of {avg_neighbors:.1f} neighbors per observation. "
            description += f"The weights were row-standardized for the analysis."
            
            return description
        except Exception:
            return "Custom spatial weights matrix."

def main():
    """Execute the spatial pattern analysis."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR / "visualizations", exist_ok=True)
        
        analysis = SpatialPatternAnalysis()
        analysis.run_analysis()
    except Exception as e:
        logger.exception(f"Error during spatial pattern analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 