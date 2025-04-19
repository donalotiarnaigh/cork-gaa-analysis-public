import os
import logging
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemographicDistribution:
    """Class to create demographic distribution visualizations."""
    
    def __init__(self):
        """Initialize with data paths and load necessary data."""
        self.data_dir = Path('data/processed')
        self.output_dir = Path('data/analysis/visualizations/demographics')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load demographic data
        self.demographic_data = gpd.read_file(self.data_dir / 'cork_sa_analysis_full.gpkg')
        
        # Set style for all plots
        plt.style.use('default')
        sns.set_style("whitegrid")
        
    def create_population_density_map(self):
        """Create population density distribution map."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot population density
        self.demographic_data.plot(
            column='T1_1AGETT_x',
            ax=ax,
            legend=True,
            legend_kwds={
                'label': 'Total Population',
                'orientation': 'vertical',
                'shrink': 0.8
            },
            cmap='YlOrRd',
            missing_kwds={'color': 'lightgrey'},
            edgecolor='white',
            linewidth=0.2
        )
        
        # Customize the plot
        ax.set_title('Population Density Distribution in Cork', pad=20, fontsize=14)
        ax.axis('off')
        
        # Save the plot
        output_path = self.output_dir / 'population_density.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Population density map saved to {output_path}")
        return output_path
        
    def create_youth_distribution_map(self):
        """Create youth population distribution map."""
        # Calculate youth proportion using age groups 0-18
        youth_cols = [f'T1_1AGE{i}T_x' for i in range(19)]  # 0 to 18 years
        self.demographic_data['youth_total'] = self.demographic_data[youth_cols].sum(axis=1)
        self.demographic_data['youth_proportion'] = (
            self.demographic_data['youth_total'] / self.demographic_data['T1_1AGETT_x']
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot youth distribution
        self.demographic_data.plot(
            column='youth_proportion',
            ax=ax,
            legend=True,
            legend_kwds={
                'label': 'Youth Population Proportion',
                'orientation': 'vertical',
                'shrink': 0.8
            },
            cmap='viridis',
            missing_kwds={'color': 'lightgrey'},
            edgecolor='white',
            linewidth=0.2
        )
        
        # Customize the plot
        ax.set_title('Youth Population Distribution in Cork', pad=20, fontsize=14)
        ax.axis('off')
        
        # Save the plot
        output_path = self.output_dir / 'youth_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Youth distribution map saved to {output_path}")
        return output_path
        
    def create_socioeconomic_maps(self):
        """Create maps for various socioeconomic indicators."""
        indicators = {
            'education_level': {
                'column': 'third_level_rate',
                'title': 'Third Level Education Distribution',
                'label': 'Population with Third Level Education (%)'
            },
            'social_class': {
                'column': 'professional_rate',
                'title': 'Professional Class Distribution',
                'label': 'Professional Population (%)'
            }
        }
        
        output_paths = {}
        for indicator, config in indicators.items():
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot indicator distribution
            self.demographic_data.plot(
                column=config['column'],
                ax=ax,
                legend=True,
                legend_kwds={
                    'label': config['label'],
                    'orientation': 'vertical',
                    'shrink': 0.8
                },
                cmap='RdYlBu',
                missing_kwds={'color': 'lightgrey'},
                edgecolor='white',
                linewidth=0.2
            )
            
            # Customize the plot
            ax.set_title(config['title'], pad=20, fontsize=14)
            ax.axis('off')
            
            # Save the plot
            output_path = self.output_dir / f'{indicator}_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths[indicator] = output_path
            logger.info(f"{indicator} distribution map saved to {output_path}")
            
        return output_paths
        
    def create_urban_rural_visualization(self):
        """Create visualization showing urban-rural patterns."""
        # Use the urban area flag from the data
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a categorical color map
        colors = {'1': '#1f77b4', '0': '#2ca02c'}
        
        # Plot urban-rural distribution
        self.demographic_data.plot(
            column='SA_URBAN_AREA_FLAG',
            ax=ax,
            legend=True,
            categorical=True,
            legend_kwds={
                'title': 'Area Type',
                'labels': ['Rural', 'Urban']
            },
            cmap='Set2',
            missing_kwds={'color': 'lightgrey'},
            edgecolor='white',
            linewidth=0.2
        )
        
        # Customize the plot
        ax.set_title('Urban-Rural Distribution in Cork', pad=20, fontsize=14)
        ax.axis('off')
        
        # Save the plot
        output_path = self.output_dir / 'urban_rural_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Urban-rural distribution map saved to {output_path}")
        return output_path
        
    def create_method_comparison_maps(self):
        """Create comparative maps between different analysis methods."""
        # Load data for different methods
        nearest = gpd.read_file(self.data_dir / 'nearest_demographics.gpkg')
        voronoi = gpd.read_file(self.data_dir / 'voronoi_demographics.gpkg')
        
        # Create subplots for comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot nearest neighbor assignments
        nearest.plot(
            column='T1_1AGETT_x_nearest',
            ax=axes[0],
            legend=True,
            legend_kwds={
                'label': 'Population',
                'orientation': 'vertical',
                'shrink': 0.8
            },
            cmap='YlOrRd',
            edgecolor='white',
            linewidth=0.2
        )
        axes[0].set_title('Population Distribution - Nearest Method', fontsize=14)
        axes[0].axis('off')
        
        # Plot Voronoi assignments
        voronoi.plot(
            column='T1_1AGETT_x_voronoi',
            ax=axes[1],
            legend=True,
            legend_kwds={
                'label': 'Population',
                'orientation': 'vertical',
                'shrink': 0.8
            },
            cmap='YlOrRd',
            edgecolor='white',
            linewidth=0.2
        )
        axes[1].set_title('Population Distribution - Voronoi Method', fontsize=14)
        axes[1].axis('off')
        
        plt.suptitle('Comparison of Population Distribution Methods', fontsize=16, y=1.05)
        
        # Save the plot
        output_path = self.output_dir / 'method_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Method comparison maps saved to {output_path}")
        return output_path

    def generate_all_visualizations(self):
        """Generate all demographic distribution visualizations."""
        try:
            logger.info("Generating demographic distribution visualizations...")
            
            # Create all visualizations
            population_map = self.create_population_density_map()
            youth_map = self.create_youth_distribution_map()
            socioeconomic_maps = self.create_socioeconomic_maps()
            urban_rural_map = self.create_urban_rural_visualization()
            method_comparison = self.create_method_comparison_maps()
            
            # Return paths to all generated visualizations
            return {
                'population_density': population_map,
                'youth_distribution': youth_map,
                'socioeconomic': socioeconomic_maps,
                'urban_rural': urban_rural_map,
                'method_comparison': method_comparison
            }
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

def main():
    """Main function to generate all demographic distribution visualizations."""
    try:
        visualizer = DemographicDistribution()
        output_paths = visualizer.generate_all_visualizations()
        logger.info("Successfully generated all demographic distribution visualizations")
        logger.info(f"Output files: {output_paths}")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 