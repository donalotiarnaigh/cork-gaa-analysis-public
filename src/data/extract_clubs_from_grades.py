"""
Script to extract all unique clubs from grade markdown files.
This ensures we have a complete list of clubs before merging with grades.
"""

import pandas as pd
from pathlib import Path
import logging
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Set up file paths for data processing."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir

def extract_clubs_from_markdown(file_path: Path) -> list:
    """
    Extract club names from a grade markdown file.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        List of club names
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Split content by grade headers
        grades = re.split(r'\*\*.*?\*\*\n\n', content)
        # Remove empty strings and strip whitespace
        grades = [grade.strip() for grade in grades if grade.strip()]
        
        # Extract club names from each grade section
        clubs = []
        for grade in grades:
            # Split by newlines and clean up
            grade_clubs = [club.strip() for club in grade.split('\n') if club.strip()]
            clubs.extend(grade_clubs)
            
        return clubs
        
    except Exception as e:
        logger.error(f"Error extracting clubs from {file_path}: {str(e)}")
        raise

def create_complete_club_list(raw_dir: Path) -> pd.DataFrame:
    """
    Create a complete list of unique clubs from all grade files.
    
    Args:
        raw_dir: Directory containing raw data files
        
    Returns:
        DataFrame with unique club names
    """
    try:
        # Extract clubs from all grade files
        all_clubs = set()
        
        # 2022 grades
        football_2022 = extract_clubs_from_markdown(raw_dir / "clubs" / "2022_football_grades.md")
        hurling_2022 = extract_clubs_from_markdown(raw_dir / "clubs" / "2022_hurling_grades.md")
        
        # 2024 grades
        football_2024 = extract_clubs_from_markdown(raw_dir / "clubs" / "2024_football_grades.md")
        hurling_2024 = extract_clubs_from_markdown(raw_dir / "clubs" / "2024_hurling_grades.md")
        
        # Combine all clubs
        all_clubs.update(football_2022)
        all_clubs.update(hurling_2022)
        all_clubs.update(football_2024)
        all_clubs.update(hurling_2024)
        
        # Convert to DataFrame
        df = pd.DataFrame({'Club': sorted(list(all_clubs))})
        
        # Log statistics
        logger.info(f"Total unique clubs found: {len(df)}")
        logger.info(f"Clubs in 2022 football: {len(football_2022)}")
        logger.info(f"Clubs in 2022 hurling: {len(hurling_2022)}")
        logger.info(f"Clubs in 2024 football: {len(football_2024)}")
        logger.info(f"Clubs in 2024 hurling: {len(hurling_2024)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating complete club list: {str(e)}")
        raise

def main():
    """Main function to run the club extraction process."""
    try:
        # Set up paths
        raw_dir, processed_dir = setup_paths()
        
        # Create complete club list
        clubs_df = create_complete_club_list(raw_dir)
        
        # Save to CSV
        output_file = processed_dir / "cork_clubs_complete.csv"
        clubs_df.to_csv(output_file, index=False)
        logger.info(f"Saved complete club list to {output_file}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 