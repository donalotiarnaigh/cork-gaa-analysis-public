"""
Script to extract club grades from markdown files and create CSV files.
This script:
1. Reads grade markdown files
2. Extracts club names and their grades
3. Creates CSV files with the correct grade assignments
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

# Define grade hierarchy (1 is highest)
GRADE_HIERARCHY = {
    'Premier Senior': 1,
    'Senior A': 2,
    'Premier Intermediate': 3,
    'Intermediate A': 4,
    'Premier Junior': 5,
    'Junior A': 5
}

# Define club name standardization mappings
CLUB_NAME_MAPPINGS = {
    'St Finbarrs': "St. Finbarr's",
    'St Catherines': "St. Catherine's",
    'St Nicholas': "St. Nicholas",
    'St Michaels': "St. Michael's",
    'St Vincents': "St. Vincent's",
    'St James': "St. James'",
    'St Colums': "St. Colum's",
    'St Itas': "St. Ita's",
    'St Marys': "St. Mary's",
    'St Oliver Plunketts': "St. Oliver Plunkett's",
    'Erins Own': "Erin's Own",
    'Fr O Neills': "Fr. O' Neills",
    'O Donovan Rossa': "O' Donovan Rossa"
}

def standardize_club_name(name: str) -> str:
    """Standardize club name to match core data format."""
    return CLUB_NAME_MAPPINGS.get(name.strip(), name.strip())

def setup_paths():
    """Set up file paths for data processing."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw" / "clubs"  # Updated path
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def extract_grades_from_markdown(file_path: Path) -> pd.DataFrame:
    """
    Extract club names and grades from a grade markdown file.
    If a club appears in multiple grades, takes the highest grade.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        DataFrame with club names and grades
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Split content by grade headers
        grade_sections = re.split(r'\*\*(.*?)\*\*\n\n', content)
        # Remove empty strings
        grade_sections = [section for section in grade_sections if section.strip()]
        
        # Process each grade section
        clubs_data = {}  # Use dict to handle duplicates
        current_grade = None
        
        for section in grade_sections:
            if section in GRADE_HIERARCHY:
                current_grade = section
            else:
                # Split by newlines and clean up
                clubs = [standardize_club_name(club) for club in section.split('\n') if club.strip()]
                for club in clubs:
                    # If club already exists, only update if new grade is higher
                    if club in clubs_data:
                        current_hierarchy = GRADE_HIERARCHY[clubs_data[club]]
                        new_hierarchy = GRADE_HIERARCHY[current_grade]
                        if new_hierarchy < current_hierarchy:  # Lower number is higher grade
                            clubs_data[club] = current_grade
                            logger.info(f"Updated {club} from {clubs_data[club]} to {current_grade}")
                    else:
                        clubs_data[club] = current_grade
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {'club_name': club, 'grade': grade}
            for club, grade in clubs_data.items()
        ])
        
        # Log statistics
        logger.info(f"Extracted {len(df)} clubs from {file_path}")
        logger.info(f"Grade distribution:\n{df['grade'].value_counts()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error extracting grades from {file_path}: {str(e)}")
        raise

def create_grade_csvs(raw_dir: Path):
    """Create CSV files with club grades for each year and code."""
    try:
        # Extract grades from all markdown files
        football_2022 = extract_grades_from_markdown(raw_dir / "2022_football_grades.md")
        hurling_2022 = extract_grades_from_markdown(raw_dir / "2022_hurling_grades.md")
        football_2024 = extract_grades_from_markdown(raw_dir / "2024_football_grades.md")
        hurling_2024 = extract_grades_from_markdown(raw_dir / "2024_hurling_grades.md")
        
        # Create 2022 grades DataFrame
        grades_2022 = pd.DataFrame({'club_name': sorted(set(football_2022['club_name'].unique()) | 
                                                       set(hurling_2022['club_name'].unique()))})
        grades_2022 = grades_2022.merge(
            football_2022[['club_name', 'grade']].rename(columns={'grade': 'football_grade'}),
            on='club_name',
            how='left'
        )
        grades_2022 = grades_2022.merge(
            hurling_2022[['club_name', 'grade']].rename(columns={'grade': 'hurling_grade'}),
            on='club_name',
            how='left'
        )
        
        # Create 2024 grades DataFrame
        grades_2024 = pd.DataFrame({'club_name': sorted(set(football_2024['club_name'].unique()) | 
                                                       set(hurling_2024['club_name'].unique()))})
        grades_2024 = grades_2024.merge(
            football_2024[['club_name', 'grade']].rename(columns={'grade': 'football_grade'}),
            on='club_name',
            how='left'
        )
        grades_2024 = grades_2024.merge(
            hurling_2024[['club_name', 'grade']].rename(columns={'grade': 'hurling_grade'}),
            on='club_name',
            how='left'
        )
        
        # Add dual status
        for df in [grades_2022, grades_2024]:
            df['dual_status'] = df.apply(
                lambda row: 'Yes' if pd.notna(row['football_grade']) and pd.notna(row['hurling_grade']) 
                else 'No',
                axis=1
            )
        
        # Add club_id
        club_ids = pd.read_csv(raw_dir / "cork_clubs_unique.csv")[['Club']]
        club_ids = club_ids.reset_index().rename(columns={'index': 'club_id', 'Club': 'club_name'})
        club_ids['club_id'] = club_ids['club_id'] + 1
        
        # Add club_id to grade DataFrames
        grades_2022 = grades_2022.merge(club_ids, on='club_name', how='left')
        grades_2024 = grades_2024.merge(club_ids, on='club_name', how='left')
        
        # Reorder columns
        column_order = ['club_id', 'club_name', 'football_grade', 'hurling_grade', 'dual_status']
        grades_2022 = grades_2022[column_order]
        grades_2024 = grades_2024[column_order]
        
        # Save to CSV
        grades_2022.to_csv(raw_dir / "cork_clubs_grades_2022.csv", index=False)
        grades_2024.to_csv(raw_dir / "cork_clubs_grades_2024.csv", index=False)
        
        logger.info(f"Saved grade data for {len(grades_2022)} clubs (2022) and {len(grades_2024)} clubs (2024)")
        
    except Exception as e:
        logger.error(f"Error creating grade CSV files: {str(e)}")
        raise

def main():
    """Main function to run the grade extraction process."""
    try:
        # Set up paths
        raw_dir = setup_paths()
        
        # Create grade CSV files
        create_grade_csvs(raw_dir)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 