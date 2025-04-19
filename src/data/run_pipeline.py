#!/usr/bin/env python3
"""
Pipeline script to run the complete data processing workflow.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Configure logging
def setup_logging():
    """Configure logging for the pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("../../logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

class Pipeline:
    """Pipeline class to manage the data processing workflow."""
    
    def __init__(self, data_dir: str = "../../data"):
        """Initialize the pipeline.
        
        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Store paths to output files
        self.output_files = {}
        
    def run_saps_extraction(self) -> bool:
        """Run the SAPS data extraction and cleaning step."""
        try:
            logging.info("Starting SAPS data extraction and cleaning...")
            
            # Import and run the extraction script
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            import saps_data_extraction
            
            # Run the extraction and cleaning
            saps_data_extraction.main()
            
            # Store output file path
            output_file = self.processed_dir / "saps_data_cleaned.csv"
            self.output_files['saps_extracted'] = output_file
            
            logging.info(f"SAPS extraction and cleaning completed. Output saved to: {output_file}")
            return True
        except Exception as e:
            logging.error(f"Error in SAPS extraction and cleaning: {str(e)}")
            return False
    
    def run_saps_validation(self) -> bool:
        """Run the SAPS data validation step."""
        try:
            logging.info("Starting SAPS data validation...")
            
            # Import and run the validation script
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            import validate_saps_data
            
            # Store output file paths
            validation_report = self.processed_dir / "saps_data_validation_report.md"
            validation_results = self.processed_dir / "saps_data_validation_results.json"
            
            self.output_files['saps_validation_report'] = validation_report
            self.output_files['saps_validation_results'] = validation_results
            
            logging.info(f"SAPS validation completed. Reports saved to:")
            logging.info(f"  - Report: {validation_report}")
            logging.info(f"  - Results: {validation_results}")
            return True
        except Exception as e:
            logging.error(f"Error in SAPS validation: {str(e)}")
            return False
    
    def run_sa_preparation(self) -> bool:
        """Run the SA boundary preparation step."""
        try:
            logging.info("Starting SA boundary preparation...")
            
            # Import and run the preparation script
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            spatial_dir = os.path.join(os.path.dirname(current_dir), 'spatial')
            if spatial_dir not in sys.path:
                sys.path.insert(0, spatial_dir)
            import sa_boundary_preparation
            
            # Store output file paths
            boundaries_gpkg = self.processed_dir / "sa_boundaries_cleaned.gpkg"
            preparation_report = self.processed_dir / "sa_boundary_preparation_report.md"
            
            self.output_files['sa_boundaries_gpkg'] = boundaries_gpkg
            self.output_files['sa_preparation_report'] = preparation_report
            
            logging.info(f"SA boundary preparation completed. Files saved to:")
            logging.info(f"  - Boundaries: {boundaries_gpkg}")
            logging.info(f"  - Report: {preparation_report}")
            return True
        except Exception as e:
            logging.error(f"Error in SA preparation: {str(e)}")
            return False
    
    def run_sa_cleaning(self) -> bool:
        """Run the SA boundary cleaning step."""
        try:
            logging.info("Starting SA boundary cleaning...")
            
            # Import and run the cleaning script
            import sys
            sys.path.append('../spatial')
            import sa_boundary_cleaning
            
            # Store output file paths
            cleaning_report = self.processed_dir / "sa_boundary_cleaning.md"
            boundaries_gpkg = self.processed_dir / "sa_boundaries_cleaned.gpkg"
            boundaries_csv = self.processed_dir / "sa_boundaries_cleaned.csv"
            
            self.output_files['sa_cleaning_report'] = cleaning_report
            self.output_files['sa_boundaries_gpkg'] = boundaries_gpkg
            self.output_files['sa_boundaries_csv'] = boundaries_csv
            
            logging.info(f"SA boundary cleaning completed. Files saved to:")
            logging.info(f"  - Report: {cleaning_report}")
            logging.info(f"  - Boundaries (GPKG): {boundaries_gpkg}")
            logging.info(f"  - Boundaries (CSV): {boundaries_csv}")
            return True
        except Exception as e:
            logging.error(f"Error in SA cleaning: {str(e)}")
            return False
    
    def run_sa_export(self) -> bool:
        """Run the SA boundary export step."""
        try:
            logging.info("Starting SA boundary export...")
            
            # Import and run the export script
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            spatial_dir = os.path.join(os.path.dirname(current_dir), 'spatial')
            if spatial_dir not in sys.path:
                sys.path.insert(0, spatial_dir)
            import sa_boundary_export
            
            # Store output file paths
            export_report = self.processed_dir / "sa_boundary_export_report.md"
            boundaries_final_gpkg = self.processed_dir / "sa_boundaries_final.gpkg"
            boundaries_final_geojson = self.processed_dir / "sa_boundaries_final.geojson"
            boundaries_final_csv = self.processed_dir / "sa_boundaries_attributes.csv"
            processing_doc = self.processed_dir / "sa_boundary_processing_documentation.md"
            
            self.output_files['sa_export_report'] = export_report
            self.output_files['sa_boundaries_final_gpkg'] = boundaries_final_gpkg
            self.output_files['sa_boundaries_final_geojson'] = boundaries_final_geojson
            self.output_files['sa_boundaries_final_csv'] = boundaries_final_csv
            self.output_files['sa_processing_doc'] = processing_doc
            
            logging.info(f"SA boundary export completed. Files saved to:")
            logging.info(f"  - Report: {export_report}")
            logging.info(f"  - Boundaries (GPKG): {boundaries_final_gpkg}")
            logging.info(f"  - Boundaries (GeoJSON): {boundaries_final_geojson}")
            logging.info(f"  - Boundaries (CSV): {boundaries_final_csv}")
            logging.info(f"  - Documentation: {processing_doc}")
            return True
        except Exception as e:
            logging.error(f"Error in SA export: {str(e)}")
            return False
    
    def run_club_grade_extraction(self) -> bool:
        """Run the club grade extraction step."""
        try:
            logging.info("Starting club grade extraction...")
            
            # Import and run the extraction script
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            import extract_grades_from_markdown
            
            # Run the extraction
            extract_grades_from_markdown.main()
            
            # Store output file paths
            grades_2022 = self.processed_dir / "cork_clubs_grades_2022.csv"
            grades_2024 = self.processed_dir / "cork_clubs_grades_2024.csv"
            
            self.output_files['club_grades_2022'] = grades_2022
            self.output_files['club_grades_2024'] = grades_2024
            
            logging.info(f"Club grade extraction completed. Files saved to:")
            logging.info(f"  - 2022 grades: {grades_2022}")
            logging.info(f"  - 2024 grades: {grades_2024}")
            return True
        except Exception as e:
            logging.error(f"Error in club grade extraction: {str(e)}")
            return False

    def run_club_data_merging(self) -> bool:
        """Run the club data merging step."""
        try:
            logging.info("Starting club data merging...")
            
            # Import and run the merging script
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            import merge_club_data_complete
            
            # Run the merging
            merge_club_data_complete.main()
            
            # Store output file paths
            complete_csv = self.processed_dir / "cork_clubs_complete.csv"
            complete_graded = self.processed_dir / "cork_clubs_complete_graded.csv"
            complete_gpkg = self.processed_dir / "cork_clubs_complete.gpkg"
            
            self.output_files['clubs_complete'] = complete_csv
            self.output_files['clubs_complete_graded'] = complete_graded
            self.output_files['clubs_complete_gpkg'] = complete_gpkg
            
            logging.info(f"Club data merging completed. Files saved to:")
            logging.info(f"  - Complete data: {complete_csv}")
            logging.info(f"  - Graded data: {complete_graded}")
            logging.info(f"  - GeoPackage: {complete_gpkg}")
            return True
        except Exception as e:
            logging.error(f"Error in club data merging: {str(e)}")
            return False
    
    def run_data_integration(self) -> bool:
        """Run the data integration step."""
        try:
            logging.info("Starting data integration...")
            
            # Import and run the integration script
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            import join_saps_sa_guid
            
            # Store output file paths
            joined_data = self.processed_dir / "cork_sa_saps_joined_guid.csv"
            unmatched_records = self.processed_dir / "unmatched_records.csv"
            
            self.output_files['joined_data'] = joined_data
            self.output_files['unmatched_records'] = unmatched_records
            
            # Run the join script
            join_saps_sa_guid.main()
            
            logging.info(f"Data integration completed. Files saved to:")
            logging.info(f"  - Joined data: {joined_data}")
            logging.info(f"  - Unmatched records: {unmatched_records}")
            return True
        except Exception as e:
            logging.error(f"Error in data integration: {str(e)}")
            return False
    
    def run(self) -> bool:
        """Run the complete pipeline."""
        logging.info("Starting data processing pipeline...")
        
        # SAPS Processing
        if not self.run_saps_extraction():
            return False
        if not self.run_saps_validation():
            return False
        
        # SA Boundary Processing
        if not self.run_sa_preparation():
            return False
        if not self.run_sa_cleaning():
            return False
        if not self.run_sa_export():
            return False
        
        # Club Processing
        if not self.run_club_grade_extraction():
            return False
        if not self.run_club_data_merging():
            return False
        
        # SA/SAPS Integration (separate from club data)
        if not self.run_data_integration():
            return False
        
        logging.info("Pipeline completed successfully!")
        return True

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the data processing pipeline")
    parser.add_argument("--data-dir", default="../../data",
                      help="Path to the data directory")
    parser.add_argument("--clean", action="store_true",
                      help="Clean intermediate files after processing")
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    
    # Initialize and run pipeline
    pipeline = Pipeline(args.data_dir)
    success = pipeline.run()
    
    if success:
        logging.info("Pipeline completed successfully!")
    else:
        logging.error("Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 