#!/usr/bin/env python3

from pathlib import Path
from src.utils.visualization_organizer import VisualizationOrganizer

def main():
    """Organize visualization outputs."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'analysis'
    
    # Initialize and run the organizer
    organizer = VisualizationOrganizer(data_dir)
    
    print("Creating directory structure...")
    organizer.create_directory_structure()
    
    print("Organizing visualization files...")
    organizer.organize_files()
    
    print("Creating documentation...")
    organizer.create_readme()
    
    print("Visualization organization complete!")

if __name__ == "__main__":
    main() 