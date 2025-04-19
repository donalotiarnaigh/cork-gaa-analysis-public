import os
import logging
import folium
import geopandas as gpd
import pandas as pd
from pathlib import Path
from folium import plugins

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InteractiveMap:
    """Class to create an interactive map of Cork GAA club catchment areas."""
    
    def __init__(self):
        """Initialize the map with data paths."""
        self.data_dir = Path('data/processed')
        self.output_dir = Path('data/analysis/visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data with simplified geometries
        self.nearest_data = gpd.read_file(self.data_dir / 'nearest_demographics.gpkg')
        self.nearest_data['geometry'] = self.nearest_data['geometry'].simplify(tolerance=0.001)
        
        self.clubs = gpd.read_file(self.data_dir / 'cork_clubs_complete.gpkg')
        
        # Create base map centered on Cork with a lighter tile layer
        self.m = folium.Map(
            location=[51.8985, -8.4756],  # Cork city center coordinates
            zoom_start=10,
            tiles='OpenStreetMap',  # Lighter tile layer
            control_scale=True,
            prefer_canvas=True  # Use canvas for better performance
        )
        
    def add_catchment_layers(self):
        """Add catchment area layers to the map."""
        # Add nearest club assignments with simplified styling
        folium.GeoJson(
            self.nearest_data,
            name='Club Catchment Areas',
            style_function=lambda x: {
                'fillColor': '#FF9999',
                'color': '#FF9999',
                'weight': 1,
                'fillOpacity': 0.3
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['nearest_club', 'T1_1AGETT_x_nearest'],
                aliases=['Club:', 'Population:'],
                localize=True,
                style=('background-color: white; color: black; font-family: arial; font-size: 12px; padding: 10px;')
            ),
            smooth_factor=1.0  # Reduce smoothing for better performance
        ).add_to(self.m)
        
    def add_club_locations(self):
        """Add GAA club locations to the map."""
        for idx, row in self.clubs.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                popup=folium.Popup(
                    f"""
                    <h4>{row['Club']}</h4>
                    <p>Address: {row.get('Address', 'N/A')}</p>
                    """,
                    max_width=200
                ),
                color='red',
                fill=True,
                fill_color='red',
                tooltip=row['Club']
            ).add_to(self.m)
            
    def add_demographic_layer(self):
        """Add a single demographic layer to the map."""
        # Add population density layer with simplified styling
        population_density = folium.Choropleth(
            geo_data=self.nearest_data,
            name='Population Density',
            data=self.nearest_data,
            columns=['SA_GUID_2016_x_demog', 'T1_1AGETT_x_demog'],
            key_on='feature.properties.SA_GUID_2016_x_demog',
            fill_color='YlOrRd',
            fill_opacity=0.5,
            line_opacity=0.2,
            legend_name='Population Density',
            highlight=False,  # Disable highlight for better performance
            smooth_factor=1.0  # Reduce smoothing for better performance
        )
        population_density.add_to(self.m)
        
    def add_controls(self):
        """Add essential map controls only."""
        folium.LayerControl().add_to(self.m)
        plugins.Fullscreen().add_to(self.m)
        
    def generate_map(self):
        """Generate the interactive map."""
        try:
            logger.info("Generating interactive map...")
            
            # Add essential layers and controls
            self.add_catchment_layers()
            self.add_club_locations()
            self.add_demographic_layer()
            self.add_controls()
            
            # Save the map
            output_path = self.output_dir / 'interactive_map.html'
            self.m.save(str(output_path))
            
            logger.info(f"Interactive map saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating interactive map: {str(e)}")
            raise

def main():
    """Main function to generate the interactive map."""
    try:
        map_generator = InteractiveMap()
        output_path = map_generator.generate_map()
        logger.info(f"Successfully generated interactive map at {output_path}")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 