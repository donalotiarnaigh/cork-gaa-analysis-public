import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

class QualityAssurance:
    """Performs quality assurance checks on analysis outputs."""
    
    def __init__(self, base_dir: Path):
        """Initialize QA with base directory."""
        self.base_dir = Path(base_dir)
        self.viz_dir = self.base_dir / 'visualizations'
        self.output_dir = self.base_dir / 'output'
        self.logger = logging.getLogger(__name__)
        
        # Load visualization index
        with open(self.viz_dir / 'visualization_index.yaml', 'r') as f:
            self.viz_index = yaml.safe_load(f)
    
    def verify_visualizations(self) -> Dict[str, List[str]]:
        """Verify all visualizations exist and are valid."""
        results = {
            'missing_files': [],
            'invalid_files': [],
            'valid_files': []
        }
        
        for viz_name, metadata in self.viz_index.items():
            file_path = self.viz_dir / metadata['category'] / metadata['filename']
            
            if not file_path.exists():
                results['missing_files'].append(str(file_path))
                self.logger.error(f"Missing visualization: {file_path}")
            elif file_path.stat().st_size == 0:
                results['invalid_files'].append(str(file_path))
                self.logger.error(f"Empty visualization file: {file_path}")
            else:
                results['valid_files'].append(str(file_path))
                self.logger.info(f"Valid visualization: {file_path}")
        
        return results
    
    def validate_statistical_accuracy(self) -> Dict[str, List[Dict]]:
        """Validate statistical accuracy of analysis outputs."""
        results = {
            'validation_metrics': [],
            'demographic_stats': [],
            'method_comparison': []
        }
        
        # Check validation metrics
        validation_file = self.output_dir / 'csv' / 'validation_results.csv'
        if validation_file.exists():
            df = pd.read_csv(validation_file)
            results['validation_metrics'].extend(self._check_validation_metrics(df))
        
        # Check demographic statistics
        demo_file = self.output_dir / 'csv' / 'demographic_statistics.csv'
        if demo_file.exists():
            df = pd.read_csv(demo_file)
            results['demographic_stats'].extend(self._check_demographic_stats(df))
        
        # Check method comparison
        method_file = self.output_dir / 'csv' / 'method_comparison.csv'
        if method_file.exists():
            df = pd.read_csv(method_file)
            results['method_comparison'].extend(self._check_method_comparison(df))
        
        return results
    
    def _check_validation_metrics(self, df: pd.DataFrame) -> List[Dict]:
        """Check validation metrics for consistency and accuracy."""
        checks = []
        
        # Check population coverage
        if 'population_coverage' in df.columns:
            coverage = df['population_coverage'].mean()
            checks.append({
                'metric': 'population_coverage',
                'value': coverage,
                'status': 'PASS' if 0.95 <= coverage <= 1.05 else 'FAIL',
                'message': f"Population coverage: {coverage:.2%}"
            })
        
        # Check area coverage
        if 'area_coverage' in df.columns:
            coverage = df['area_coverage'].mean()
            checks.append({
                'metric': 'area_coverage',
                'value': coverage,
                'status': 'PASS' if 0.95 <= coverage <= 1.05 else 'FAIL',
                'message': f"Area coverage: {coverage:.2%}"
            })
        
        return checks
    
    def _check_demographic_stats(self, df: pd.DataFrame) -> List[Dict]:
        """Check demographic statistics for consistency."""
        checks = []
        
        # Check population totals
        if 'total_population' in df.columns:
            total = df['total_population'].sum()
            checks.append({
                'metric': 'total_population',
                'value': total,
                'status': 'PASS' if total > 0 else 'FAIL',
                'message': f"Total population: {total:,.0f}"
            })
        
        # Check youth population percentage
        if 'youth_population' in df.columns and 'total_population' in df.columns:
            youth_pct = df['youth_population'].sum() / df['total_population'].sum()
            checks.append({
                'metric': 'youth_population_pct',
                'value': youth_pct,
                'status': 'PASS' if 0.1 <= youth_pct <= 0.3 else 'FAIL',
                'message': f"Youth population percentage: {youth_pct:.2%}"
            })
        
        return checks
    
    def _check_method_comparison(self, df: pd.DataFrame) -> List[Dict]:
        """Check method comparison results for consistency."""
        checks = []
        
        # Check correlation between methods
        if 'buffer_score' in df.columns and 'voronoi_score' in df.columns:
            correlation = df['buffer_score'].corr(df['voronoi_score'])
            checks.append({
                'metric': 'method_correlation',
                'value': correlation,
                'status': 'PASS' if correlation > 0.7 else 'FAIL',
                'message': f"Method correlation: {correlation:.3f}"
            })
        
        # Check score distributions
        for col in ['buffer_score', 'voronoi_score', 'nearest_score']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                checks.append({
                    'metric': f'{col}_distribution',
                    'value': {'mean': mean, 'std': std},
                    'status': 'PASS' if 0 <= mean <= 1 and std > 0 else 'FAIL',
                    'message': f"{col}: mean={mean:.3f}, std={std:.3f}"
                })
        
        return checks
    
    def check_report_completeness(self) -> Dict[str, List[str]]:
        """Check if all required reports are present and complete."""
        results = {
            'missing_reports': [],
            'incomplete_reports': [],
            'complete_reports': []
        }
        
        required_reports = [
            'demographic_profiles_report.md',
            'method_bias_report.md',
            'method_comparison_report.md',
            'method_reliability_report.md',
            'pattern_identification_report.md',
            'urban_rural_analysis_report.md',
            'validation_documentation.md'
        ]
        
        for report in required_reports:
            report_path = self.output_dir / 'documentation' / report
            if not report_path.exists():
                results['missing_reports'].append(report)
                self.logger.error(f"Missing report: {report}")
            else:
                with open(report_path, 'r') as f:
                    content = f.read()
                    if len(content.strip()) < 100:  # Arbitrary minimum length
                        results['incomplete_reports'].append(report)
                        self.logger.warning(f"Incomplete report: {report}")
                    else:
                        results['complete_reports'].append(report)
                        self.logger.info(f"Complete report: {report}")
        
        return results
    
    def review_documentation(self) -> Dict[str, List[str]]:
        """Review documentation for completeness and accuracy."""
        results = {
            'missing_docs': [],
            'incomplete_docs': [],
            'complete_docs': []
        }
        
        # Check visualization documentation
        if not (self.viz_dir / 'README.md').exists():
            results['missing_docs'].append('visualizations/README.md')
        else:
            results['complete_docs'].append('visualizations/README.md')
        
        # Check visualization index
        if not (self.viz_dir / 'visualization_index.yaml').exists():
            results['missing_docs'].append('visualization_index.yaml')
        else:
            results['complete_docs'].append('visualization_index.yaml')
        
        return results
    
    def perform_final_validation(self) -> Dict[str, Dict]:
        """Perform final validation checks."""
        results = {
            'visualization_checks': self.verify_visualizations(),
            'statistical_checks': self.validate_statistical_accuracy(),
            'report_checks': self.check_report_completeness(),
            'documentation_checks': self.review_documentation()
        }
        
        # Generate summary
        total_checks = sum(len(v) for v in results.values())
        passed_checks = sum(
            len([x for x in v if isinstance(x, dict) and x.get('status') == 'PASS'])
            for v in results.values()
        )
        
        results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 0
        }
        
        return results 