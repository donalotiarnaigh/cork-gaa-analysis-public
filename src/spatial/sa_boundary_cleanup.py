"""
Clean up duplicate and intermediate files in the processed directory.
This script removes old versions of files and keeps only the latest versions.
"""

import os
from pathlib import Path
import logging
import re
from datetime import datetime
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Set up file paths."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    processed_dir = project_root / "data" / "processed"
    archive_dir = processed_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir, archive_dir

def parse_timestamp(filename):
    """Extract timestamp from filename."""
    match = re.search(r'_(\d{8}_\d{6})', filename)
    if match:
        timestamp_str = match.group(1)
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    return None

def group_files(processed_dir):
    """Group files by their base name without timestamp."""
    file_groups = {}
    
    for file_path in processed_dir.glob('*'):
        if file_path.is_file():
            filename = file_path.name
            
            # Skip certain files
            if filename in ['.gitkeep']:
                continue
                
            # Extract base name without timestamp
            base_name = re.sub(r'_\d{8}_\d{6}', '', filename)
            
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_path)
    
    return file_groups

def cleanup_files(processed_dir, archive_dir):
    """Clean up duplicate and intermediate files."""
    file_groups = group_files(processed_dir)
    
    # Files to keep (latest version only)
    keep_patterns = [
        'sa_boundaries_final',
        'sa_boundaries_attributes',
        'sa_boundary_export_report',
        'sa_boundary_processing_documentation',
        'sa_boundary_dictionary',
        'cork_saps_2022_cleaned',
        'data_dictionary',
        'column_name_mapping'
    ]
    
    # Process each group
    for base_name, files in file_groups.items():
        # If this is a timestamped group we want to keep
        keep_this_group = any(pattern in base_name for pattern in keep_patterns)
        
        if keep_this_group and len(files) > 1:
            # Sort files by timestamp, newest first
            files.sort(key=lambda x: parse_timestamp(x.name) if parse_timestamp(x.name) else datetime.min, reverse=True)
            
            # Keep the newest file, move others to archive
            for file_path in files[1:]:
                archive_path = archive_dir / file_path.name
                logger.info(f"Moving {file_path.name} to archive")
                shutil.move(str(file_path), str(archive_path))
        
        # For intermediate files we don't need to keep
        elif not keep_this_group:
            for file_path in files:
                archive_path = archive_dir / file_path.name
                logger.info(f"Moving {file_path.name} to archive")
                shutil.move(str(file_path), str(archive_path))

def main():
    """Main function to clean up processed directory."""
    try:
        processed_dir, archive_dir = setup_paths()
        cleanup_files(processed_dir, archive_dir)
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise

if __name__ == "__main__":
    main() 