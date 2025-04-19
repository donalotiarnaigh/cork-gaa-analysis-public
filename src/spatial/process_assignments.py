import logging
import geopandas as gpd
from pathlib import Path
import pandas as pd
from shapely.validation import make_valid
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load required datasets."""
    logger.info("Loading input data...")
    
    # Load Small Area assignments
    assignments = gpd.read_file(
        "data/processed/sa_club_assignments.gpkg",
        layer="sa_club_assignments"
    )
    
    # Load club points
    clubs = gpd.read_file(
        "data/processed/cork_clubs_complete.gpkg",
        layer="cork_clubs_complete"
    )
    clubs = clubs.to_crs(assignments.crs)  # Convert to same CRS
    
    # Load Small Areas
    small_areas = gpd.read_file(
        "data/processed/sa_boundaries_final.gpkg",
        layer="sa_boundaries_final"
    )
    small_areas = small_areas.to_crs(assignments.crs)  # Convert to same CRS
    
    return assignments, clubs, small_areas

def calculate_metadata(assignments, small_areas, clubs):
    """Calculate metadata for assignments."""
    logger.info("Calculating assignment metadata...")
    
    # Calculate distances from Small Area centroids to club locations
    sa_centroids = small_areas.copy()
    sa_centroids.geometry = sa_centroids.geometry.centroid
    
    # Create a lookup of club locations
    club_locations = clubs.set_index('Club')[['geometry']]
    
    # Calculate distances for each assignment
    distances = []
    for idx, row in assignments.iterrows():
        sa_guid = row['SA_GUID_2022']
        club_name = row['Club']
        
        try:
            sa_centroid = sa_centroids.loc[sa_centroids['SA_GUID_2022'] == sa_guid, 'geometry'].iloc[0]
            club_point = club_locations.loc[club_name, 'geometry']
            distance = sa_centroid.distance(club_point)
            distances.append(distance)
        except (KeyError, IndexError):
            logger.warning(f"Could not calculate distance for SA {sa_guid} and Club {club_name}")
            distances.append(None)
    
    assignments['distance_to_club'] = distances
    
    # Calculate area proportions if not already present
    if 'area_proportion' not in assignments.columns:
        assignments['area_proportion'] = assignments['intersection_area'] / assignments.geometry.area
    
    # Ensure assigned_club matches Club field
    logger.info("Checking assigned_club values...")
    null_assigned = assignments['assigned_club'].isna().sum()
    if null_assigned > 0:
        logger.info(f"Found {null_assigned} records with NULL assigned_club")
        assignments['assigned_club'] = assignments['Club']
        logger.info("Updated assigned_club field with Club values")
    
    return assignments

def validate_assignments(assignments, clubs, small_areas):
    """Validate assignments against various criteria."""
    logger.info("Validating assignments...")
    
    validation_results = {
        'total_assignments': len(assignments),
        'unique_small_areas': assignments['SA_GUID_2022'].nunique(),
        'unique_clubs': assignments['Club'].nunique(),
        'orphaned_small_areas': [],
        'orphaned_clubs': [],
        'distance_issues': [],
        'null_assignments': [],
        'distance_stats': {},
        'adjusted_assignments': []
    }
    
    # Check for orphaned Small Areas
    all_small_areas = set(small_areas['SA_GUID_2022'])
    assigned_small_areas = set(assignments['SA_GUID_2022'])
    validation_results['orphaned_small_areas'] = list(all_small_areas - assigned_small_areas)
    
    # Check for orphaned clubs
    all_clubs = set(clubs['Club'])
    assigned_clubs = set(assignments['Club'])
    validation_results['orphaned_clubs'] = list(all_clubs - assigned_clubs)
    
    # Check for NULL assignments
    null_assignments = assignments[assignments['assigned_club'].isna()]
    if len(null_assignments) > 0:
        validation_results['null_assignments'] = null_assignments[['SA_GUID_2022', 'Club', 'area_proportion']].to_dict('records')
    
    # Analyze distance distribution
    distances = assignments['distance_to_club'].dropna()
    if len(distances) > 0:
        # Calculate distance statistics
        validation_results['distance_stats'] = {
            'mean': distances.mean(),
            'median': distances.median(),
            'std': distances.std(),
            'min': distances.min(),
            'max': distances.max(),
            'q1': distances.quantile(0.25),
            'q3': distances.quantile(0.75),
            'iqr': distances.quantile(0.75) - distances.quantile(0.25)
        }
        
        # Set maximum distance threshold (10km)
        max_distance = 10000  # meters
        
        # Calculate dynamic area proportion thresholds based on distance
        def get_area_threshold(distance):
            if distance <= 5000:  # Urban/suburban areas
                return 0.05  # 5%
            elif distance <= 7500:  # Semi-rural areas
                return 0.10  # 10%
            else:  # Rural areas
                return 0.15  # 15%
        
        # Get distance anomalies
        distance_issues = assignments[
            (assignments['distance_to_club'] > max_distance) |
            (assignments['distance_to_club'] > validation_results['distance_stats']['q3'] + (1.5 * validation_results['distance_stats']['iqr']))
        ].copy()
        
        # Add context for each anomaly and suggest adjustments
        for idx, row in distance_issues.iterrows():
            sa_guid = row['SA_GUID_2022']
            club = row['Club']
            distance = row['distance_to_club']
            area_prop = row['area_proportion']
            
            # Get all assignments for this Small Area
            sa_assignments = assignments[assignments['SA_GUID_2022'] == sa_guid]
            
            # Get the closest club for this Small Area
            closest_club = sa_assignments.loc[sa_assignments['distance_to_club'].idxmin()]
            
            # Calculate dynamic threshold for this distance
            area_threshold = get_area_threshold(distance)
            
            issue = {
                'SA_GUID_2022': sa_guid,
                'Club': club,
                'distance_to_club': distance,
                'area_proportion': area_prop,
                'closest_club': closest_club['Club'],
                'closest_distance': closest_club['distance_to_club'],
                'total_assignments': len(sa_assignments),
                'area_threshold': area_threshold,
                'suggested_action': 'review' if area_prop < area_threshold else 'keep'
            }
            
            # If area proportion is below threshold and distance is high, suggest reassignment
            if area_prop < area_threshold and distance > max_distance:
                issue['suggested_action'] = 'reassign'
                issue['suggested_club'] = closest_club['Club']
            
            validation_results['distance_issues'].append(issue)
            
            # Track suggested adjustments
            if issue['suggested_action'] == 'reassign':
                validation_results['adjusted_assignments'].append({
                    'SA_GUID_2022': sa_guid,
                    'current_club': club,
                    'suggested_club': closest_club['Club'],
                    'current_distance': distance,
                    'suggested_distance': closest_club['distance_to_club'],
                    'area_proportion': area_prop
                })
    
    return validation_results

def generate_validation_report(validation_results):
    """Generate a validation report."""
    logger.info("Generating validation report...")
    
    report = [
        "# Assignment Validation Report",
        f"\nGenerated: {pd.Timestamp.now()}",
        "\n## Assignment Statistics",
        f"- Total assignments: {validation_results['total_assignments']}",
        f"- Unique Small Areas: {validation_results['unique_small_areas']}",
        f"- Unique clubs: {validation_results['unique_clubs']}",
        "\n## Validation Results",
        f"- Orphaned Small Areas: {len(validation_results['orphaned_small_areas'])}",
        f"- Orphaned clubs: {len(validation_results['orphaned_clubs'])}",
        f"- Distance anomalies: {len(validation_results['distance_issues'])}",
        f"- NULL assignments: {len(validation_results['null_assignments'])}",
        f"- Suggested reassignments: {len(validation_results['adjusted_assignments'])}",
    ]
    
    # Add distance statistics
    if validation_results['distance_stats']:
        stats = validation_results['distance_stats']
        report.extend([
            "\n## Distance Statistics",
            f"- Mean distance: {stats['mean']:.2f} meters",
            f"- Median distance: {stats['median']:.2f} meters",
            f"- Standard deviation: {stats['std']:.2f} meters",
            f"- Minimum distance: {stats['min']:.2f} meters",
            f"- Maximum distance: {stats['max']:.2f} meters",
            f"- Q1 (25th percentile): {stats['q1']:.2f} meters",
            f"- Q3 (75th percentile): {stats['q3']:.2f} meters",
            f"- IQR: {stats['iqr']:.2f} meters",
            f"- Distance threshold for anomalies: {stats['q3'] + (1.5 * stats['iqr']):.2f} meters",
            f"- Maximum distance threshold: 10,000 meters"
        ])
    
    report.append("\n## Detailed Issues")
    
    if validation_results['orphaned_small_areas']:
        report.append("\n### Orphaned Small Areas")
        for sa_id in validation_results['orphaned_small_areas']:
            report.append(f"- Small Area ID: {sa_id}")
    
    if validation_results['orphaned_clubs']:
        report.append("\n### Orphaned Clubs")
        for club_id in validation_results['orphaned_clubs']:
            report.append(f"- Club ID: {club_id}")
    
    if validation_results['null_assignments']:
        report.append("\n### NULL Assignments")
        for issue in validation_results['null_assignments']:
            report.append(
                f"- Small Area {issue['SA_GUID_2022']} with Club {issue['Club']} "
                f"(area proportion: {issue['area_proportion']:.2%})"
            )
    
    if validation_results['distance_issues']:
        report.append("\n### Distance Anomalies")
        for issue in validation_results['distance_issues']:
            action = issue['suggested_action']
            report.append(
                f"- Small Area {issue['SA_GUID_2022']} assigned to Club {issue['Club']} "
                f"(distance: {issue['distance_to_club']:.2f} meters, "
                f"area proportion: {issue['area_proportion']:.2%})\n"
                f"  - This Small Area has {issue['total_assignments']} total assignments\n"
                f"  - Closest club is {issue['closest_club']} at {issue['closest_distance']:.2f} meters\n"
                f"  - Area threshold for this distance: {issue['area_threshold']:.2%}\n"
                f"  - Suggested action: {action}"
            )
            if action == 'reassign':
                report.append(f"  - Suggested reassignment to: {issue['suggested_club']}")
    
    if validation_results['adjusted_assignments']:
        report.append("\n### Suggested Reassignments")
        for adj in validation_results['adjusted_assignments']:
            report.append(
                f"- Small Area {adj['SA_GUID_2022']}\n"
                f"  - Current: {adj['current_club']} ({adj['current_distance']:.2f} meters, {adj['area_proportion']:.2%})\n"
                f"  - Suggested: {adj['suggested_club']} ({adj['suggested_distance']:.2f} meters)"
            )
    
    return "\n".join(report)

def apply_reassignments(assignments, validation_results):
    """Apply the suggested reassignments to the assignments DataFrame."""
    logger.info("Applying suggested reassignments...")
    
    # Create a copy of the assignments to avoid modifying the original
    updated_assignments = assignments.copy()
    
    # Track reassignments for logging
    reassignment_count = 0
    
    # Process each suggested reassignment
    for adj in validation_results['adjusted_assignments']:
        sa_guid = adj['SA_GUID_2022']
        current_club = adj['current_club']
        suggested_club = adj['suggested_club']
        
        # Find the index of the assignment to update
        mask = (updated_assignments['SA_GUID_2022'] == sa_guid) & (updated_assignments['Club'] == current_club)
        if mask.any():
            idx = mask.idxmax()
            # Update the club assignment
            updated_assignments.at[idx, 'Club'] = suggested_club
            updated_assignments.at[idx, 'assigned_club'] = suggested_club
            reassignment_count += 1
            
            logger.info(
                f"Reassigned Small Area {sa_guid} from {current_club} to {suggested_club} "
                f"(distance reduced from {adj['current_distance']:.2f}m to {adj['suggested_distance']:.2f}m)"
            )
    
    logger.info(f"Completed {reassignment_count} reassignments")
    return updated_assignments

def generate_anomaly_summary(validation_results):
    """Generate a focused summary report on the remaining distance anomalies."""
    logger.info("Generating anomaly summary report...")
    
    # Group anomalies by suggested action
    review_cases = []
    keep_cases = []
    
    for issue in validation_results['distance_issues']:
        if issue.get('suggested_action') == 'review':
            review_cases.append(issue)
        else:
            keep_cases.append(issue)
    
    # Calculate IQR-based threshold from distance stats
    stats = validation_results['distance_stats']
    iqr_threshold = stats['q3'] + (1.5 * stats['iqr'])
    
    # Generate the report
    report = [
        "# Distance Anomaly Summary Report",
        f"Generated: {datetime.now()}",
        "",
        "## Definition of Anomaly",
        "An assignment is considered anomalous if it meets either of these criteria:",
        f"1. The distance between Small Area and assigned club exceeds 10,000 meters (absolute threshold)",
        f"2. The distance exceeds {iqr_threshold:.2f} meters (statistical threshold: Q3 + 1.5*IQR)",
        "",
        "Additionally, anomalies are categorized as either:",
        "- **Review Required**: Cases where the area proportion is below the distance-based threshold:",
        "  - Urban areas (≤5km): 5% threshold",
        "  - Semi-rural areas (5-7.5km): 10% threshold",
        "  - Rural areas (>7.5km): 15% threshold",
        "- **Keep As Is**: Cases where the area proportion exceeds the threshold, indicating a significant",
        "  portion of the Small Area belongs to the club despite the distance",
        "",
        "## Overview",
        f"- Total anomalies: {len(validation_results['distance_issues'])}",
        f"- Cases requiring review: {len(review_cases)}",
        f"- Cases to keep: {len(keep_cases)}",
        "",
    ]
    
    if review_cases:
        # Calculate statistics for review cases
        distances = []
        areas = []
        for issue in review_cases:
            try:
                dist = float(str(issue.get('distance_to_club', '0')).replace('meters', '').strip())
                area = float(str(issue.get('area_proportion', '0')).replace('%', '').strip())
                distances.append(dist)
                areas.append(area)
            except (ValueError, AttributeError) as e:
                logger.warning(f"Error processing issue stats: {e}")
                continue
        
        if distances and areas:
            report.extend([
                "## Review Cases Statistics",
                f"- Average distance: {np.mean(distances):.2f} meters",
                f"- Median distance: {np.median(distances):.2f} meters",
                f"- Average area proportion: {np.mean(areas):.2f}%",
                f"- Median area proportion: {np.median(areas):.2f}%",
                "",
            ])
        
        report.append("## Cases Requiring Review")
        
        # Sort review cases by distance
        review_cases.sort(key=lambda x: float(str(x.get('distance_to_club', '0')).replace('meters', '').strip()), reverse=True)
        
        for case in review_cases:
            report.extend([
                f"### Small Area: {case.get('SA_GUID_2022', 'Unknown')}",
                f"- Current club: {case.get('Club', 'Unknown')}",
                f"- Distance: {case.get('distance_to_club', 'Unknown')} meters",
                f"- Area proportion: {case.get('area_proportion', 'Unknown')}%",
                f"- Total assignments: {case.get('total_assignments', 'Unknown')}",
                f"- Closest club: {case.get('closest_club', 'Unknown')} ({case.get('closest_distance', 'Unknown')} meters)",
                f"- Area threshold: {case.get('area_threshold', 'Unknown')}%",
                f"- Anomaly type: {'Distance exceeds maximum threshold' if float(str(case.get('distance_to_club', '0')).replace('meters', '').strip()) > 10000 else 'Statistical outlier'}",
                ""
            ])
    
    # Add cases to keep for reference
    report.extend([
        "## Cases to Keep",
        "These cases have been reviewed and should be kept as is due to significant area proportions:",
    ])
    
    for case in keep_cases:
        report.extend([
            f"- {case.get('SA_GUID_2022', 'Unknown')} -> {case.get('Club', 'Unknown')} "
            f"(distance: {case.get('distance_to_club', 'Unknown')} meters, area: {case.get('area_proportion', 'Unknown')}%)"
        ])
    
    return "\n".join(report)

def main():
    """Main function to process and validate assignments."""
    try:
        # Load data
        assignments, clubs, small_areas = load_data()
        
        # Calculate metadata
        assignments = calculate_metadata(assignments, small_areas, clubs)
        
        # Validate assignments
        validation_results = validate_assignments(assignments, clubs, small_areas)
        
        # Apply suggested reassignments
        assignments = apply_reassignments(assignments, validation_results)
        
        # Recalculate metadata after reassignments
        assignments = calculate_metadata(assignments, small_areas, clubs)
        
        # Validate assignments again after reassignments
        validation_results = validate_assignments(assignments, clubs, small_areas)
        
        # Generate validation report
        report = generate_validation_report(validation_results)
        
        # Generate anomaly summary
        anomaly_summary = generate_anomaly_summary(validation_results)
        
        # Save updated assignments
        logger.info("Saving updated assignments...")
        assignments.to_file(
            "data/processed/sa_club_assignments.gpkg",
            layer="assignments",
            driver="GPKG"
        )
        
        # Save validation report
        report_path = Path("data/analysis/assignment_validation_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        
        # Save anomaly summary
        summary_path = Path("data/analysis/anomaly_summary_report.md")
        summary_path.write_text(anomaly_summary)
        
        logger.info("Assignment processing complete!")
        
    except Exception as e:
        logger.error(f"Error in assignment processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 