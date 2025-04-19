import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

class ReportGenerator:
    def __init__(self, analysis_dir: Path, output_dir: Path):
        """
        Initialize the ReportGenerator.
        
        Args:
            analysis_dir (Path): Directory containing analysis results
            output_dir (Path): Directory to save the report
        """
        self.analysis_dir = Path(analysis_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_validation_metrics(self) -> Dict[str, pd.DataFrame]:
        """Load all validation metrics CSV files."""
        validation_dir = self.analysis_dir / 'validation'
        metrics = {}
        
        for csv_file in validation_dir.glob('*.csv'):
            try:
                metrics[csv_file.stem] = pd.read_csv(csv_file)
                self.logger.info(f"Loaded {csv_file.stem}")
            except Exception as e:
                self.logger.error(f"Error loading {csv_file}: {str(e)}")
                
        return metrics
    
    def _load_analysis_reports(self) -> Dict[str, str]:
        """Load all analysis report markdown files."""
        reports = {}
        
        for md_file in self.analysis_dir.glob('*_report.md'):
            try:
                with open(md_file, 'r') as f:
                    reports[md_file.stem] = f.read()
                self.logger.info(f"Loaded {md_file.stem}")
            except Exception as e:
                self.logger.error(f"Error loading {md_file}: {str(e)}")
                
        return reports
    
    def _generate_summary_statistics(self, metrics: Dict[str, pd.DataFrame]) -> str:
        """Generate summary statistics section."""
        summary = "# Summary Statistics\n\n"
        
        # Population Validation Summary
        if 'population_validation' in metrics:
            pop_metrics = metrics['population_validation']
            summary += "## Population Validation\n\n"
            for _, row in pop_metrics.iterrows():
                metric = row['metric']
                if pd.notna(row.get('buffer_mean')):  # For metrics with mean/std
                    summary += f"### {metric}\n\n"
                    summary += f"- Buffer Method Mean: {row['buffer_mean']:.2f} (std: {row['buffer_std']:.2f})\n"
                    summary += f"- Voronoi Method Mean: {row['voronoi_mean']:.2f} (std: {row['voronoi_std']:.2f})\n"
                    summary += f"- Nearest Method Mean: {row['nearest_mean']:.2f} (std: {row['nearest_std']:.2f})\n\n"
                else:  # For metrics with direct values
                    summary += f"### {metric}\n\n"
                    summary += f"- Buffer Method: {row['buffer']:.2f}\n"
                    summary += f"- Voronoi Method: {row['voronoi']:.2f}\n"
                    summary += f"- Nearest Method: {row['nearest']:.2f}\n\n"
        
        # Area Coverage Summary
        if 'area_coverage' in metrics:
            area_metrics = metrics['area_coverage']
            summary += "## Area Coverage\n\n"
            for _, row in area_metrics.iterrows():
                metric = row['metric']
                summary += f"### {metric}\n\n"
                summary += f"- Buffer Method: {row['buffer']:.2f}\n"
                summary += f"- Voronoi Method: {row['voronoi']:.2f}\n"
                summary += f"- Nearest Method: {row['nearest']:.2f}\n\n"
        
        # Method Bias Summary
        if 'method_bias' in metrics:
            bias_metrics = metrics['method_bias']
            summary += "## Method Bias\n\n"
            for _, row in bias_metrics.iterrows():
                metric = row['metric']
                summary += f"### {metric}\n\n"
                summary += f"- Buffer vs Voronoi: {row['buffer_voronoi']:.2f}\n"
                summary += f"- Buffer vs Nearest: {row['buffer_nearest']:.2f}\n"
                summary += f"- Voronoi vs Nearest: {row['voronoi_nearest']:.2f}\n\n"
        
        # Quality Control Summary
        if 'quality_control' in metrics:
            qc_metrics = metrics['quality_control']
            summary += "## Quality Control\n\n"
            for _, row in qc_metrics.iterrows():
                metric = row['metric']
                summary += f"### {metric}\n\n"
                summary += f"- Buffer Method: {row['buffer']:.4f}\n"
                summary += f"- Voronoi Method: {row['voronoi']:.4f}\n"
                summary += f"- Nearest Method: {row['nearest']:.4f}\n\n"
        
        # Urban-Rural Metrics Summary
        if 'urban_rural_metrics' in metrics:
            ur_metrics = metrics['urban_rural_metrics']
            summary += "## Urban-Rural Distribution\n\n"
            for _, row in ur_metrics.iterrows():
                metric = row['metric']
                summary += f"### {metric}\n\n"
                summary += f"- Buffer Method: {row['buffer']:.2f}\n"
                summary += f"- Voronoi Method: {row['voronoi']:.2f}\n"
                summary += f"- Nearest Method: {row['nearest']:.2f}\n\n"
        
        # Statistical Significance Summary
        if 'statistical_significance' in metrics:
            stats = metrics['statistical_significance'].iloc[0]
            summary += "## Statistical Significance\n\n"
            summary += "### Method Comparisons\n\n"
            summary += f"- Buffer vs Voronoi: t-stat = {stats['buffer_vs_voronoi_t_stat']:.4f}, p-value = {stats['buffer_vs_voronoi_p_value']:.4e}\n"
            summary += f"- Buffer vs Nearest: t-stat = {stats['buffer_vs_nearest_t_stat']:.4f}, p-value = {stats['buffer_vs_nearest_p_value']:.4e}\n"
            summary += f"- Voronoi vs Nearest: t-stat = {stats['voronoi_vs_nearest_t_stat']:.4f}, p-value = {stats['voronoi_vs_nearest_p_value']:.4e}\n\n"
        
        return summary
    
    def _generate_visualization_documentation(self) -> str:
        """Generate visualization methodology documentation."""
        doc = "# Visualization Methodology\n\n"
        
        doc += "## Overview\n\n"
        doc += "The visualizations in this report were generated using the following methodology:\n\n"
        doc += "- All plots were created using Python's matplotlib and seaborn libraries\n"
        doc += "- Consistent color schemes were used across all visualizations\n"
        doc += "- Interactive plots were generated using Plotly for web-based viewing\n"
        doc += "- Static plots were saved in PNG format for documentation\n\n"
        
        doc += "## Plot Types\n\n"
        doc += "1. **Bar Plots**: Used for comparing metrics across different methods\n"
        doc += "2. **Box Plots**: Used for showing distribution of values\n"
        doc += "3. **Scatter Plots**: Used for showing relationships between variables\n"
        doc += "4. **Heatmaps**: Used for showing spatial patterns and correlations\n\n"
        
        doc += "## Data Processing\n\n"
        doc += "- All data was normalized before visualization\n"
        doc += "- Outliers were handled using the IQR method\n"
        doc += "- Missing values were imputed using median values\n\n"
        
        return doc
    
    def _generate_limitations_section(self) -> str:
        """Generate limitations and recommendations section."""
        limitations = "# Limitations and Recommendations\n\n"
        
        limitations += "## Limitations\n\n"
        limitations += "1. **Data Quality**:\n"
        limitations += "   - Some census data may be outdated\n"
        limitations += "   - Small area population statistics have inherent uncertainty\n\n"
        
        limitations += "2. **Methodological Limitations**:\n"
        limitations += "   - Buffer method assumes uniform population distribution\n"
        limitations += "   - Voronoi method doesn't account for physical barriers\n"
        limitations += "   - Nearest method may not reflect actual travel patterns\n\n"
        
        limitations += "3. **Spatial Resolution**:\n"
        limitations += "   - Analysis is limited by the resolution of input data\n"
        limitations += "   - Small-scale variations may be missed\n\n"
        
        limitations += "## Recommendations\n\n"
        limitations += "1. **Data Collection**:\n"
        limitations += "   - Implement regular data updates\n"
        limitations += "   - Collect more detailed demographic information\n\n"
        
        limitations += "2. **Methodological Improvements**:\n"
        limitations += "   - Combine multiple assignment methods\n"
        limitations += "   - Incorporate transport network analysis\n"
        limitations += "   - Add temporal dimension to analysis\n\n"
        
        limitations += "3. **Future Research**:\n"
        limitations += "   - Conduct sensitivity analysis\n"
        limitations += "   - Validate against actual club membership data\n"
        limitations += "   - Study impact of new club developments\n\n"
        
        return limitations
    
    def _include_visualizations(self) -> str:
        """Include visualizations in the report."""
        viz_section = "# Visualizations\n\n"
        
        # Create visualizations directory in reports
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy validation visualizations
        validation_viz_dir = self.analysis_dir / 'validation' / 'visualizations'
        if validation_viz_dir.exists():
            viz_section += "## Validation Metrics Visualizations\n\n"
            for viz_file in validation_viz_dir.glob('*.png'):
                # Copy file to reports visualization directory
                dest_file = viz_dir / viz_file.name
                import shutil
                shutil.copy2(viz_file, dest_file)
                
                # Add to report with relative path
                title = viz_file.stem.replace('_', ' ').title()
                viz_section += f"### {title}\n\n"
                viz_section += f"![{title}](visualizations/{viz_file.name})\n\n"
        
        # Copy other analysis visualizations
        for viz_file in self.analysis_dir.glob('*.png'):
            if 'analysis' in viz_file.name.lower():
                # Copy file to reports visualization directory
                dest_file = viz_dir / viz_file.name
                import shutil
                shutil.copy2(viz_file, dest_file)
                
                # Add to report with relative path
                title = viz_file.stem.replace('_', ' ').title()
                viz_section += f"### {title}\n\n"
                viz_section += f"![{title}](visualizations/{viz_file.name})\n\n"
        
        return viz_section
        
    def _generate_toc(self, content: str) -> str:
        """Generate table of contents from markdown content."""
        toc = "# Table of Contents\n\n"
        
        # Split content into lines
        lines = content.split('\n')
        
        # Track current section level and numbering
        current_level = [0] * 6  # h1 to h6
        
        for line in lines:
            if line.startswith('#'):
                # Count heading level
                level = len(line.split()[0])
                
                if level > 1:  # Skip the main title
                    # Update numbering
                    current_level[level-1] += 1
                    for i in range(level, 6):
                        current_level[i] = 0
                    
                    # Create section number
                    section_number = '.'.join(str(n) for n in current_level[1:level] if n > 0)
                    
                    # Extract heading text
                    heading = line.strip('#').strip()
                    
                    # Create link
                    link = heading.lower().replace(' ', '-').replace(':', '').replace('(', '').replace(')', '')
                    
                    # Add to TOC with proper indentation
                    indent = '    ' * (level - 2)
                    toc += f"{indent}- [{section_number} {heading}](#{link})\n"
        
        return toc + "\n"
        
    def generate_report(self, generate_html: bool = False) -> None:
        """Generate the complete analysis report.
        
        Args:
            generate_html (bool): Whether to generate an HTML version of the report.
        """
        self.logger.info("Starting report generation")
        
        # Load all data
        metrics = self._load_validation_metrics()
        reports = self._load_analysis_reports()
        
        # Generate report sections
        report_content = []
        
        # Add title and timestamp
        report_content.append(f"# GAA Cork Spatial Analysis Report\n\n")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Generate initial content without TOC
        content_sections = []
        
        # Add summary statistics
        content_sections.append(self._generate_summary_statistics(metrics))
        
        # Add visualization documentation
        content_sections.append(self._generate_visualization_documentation())
        
        # Add visualizations
        content_sections.append(self._include_visualizations())
        
        # Add existing reports
        content_sections.append("# Detailed Analysis Reports\n\n")
        for report_name, content in reports.items():
            content_sections.append(f"## {report_name.replace('_', ' ').title()}\n\n")
            content_sections.append(content)
            content_sections.append("\n\n")
        
        # Add limitations and recommendations
        content_sections.append(self._generate_limitations_section())
        
        # Join all content
        main_content = '\n'.join(content_sections)
        
        # Generate and add table of contents
        report_content.append(self._generate_toc(main_content))
        
        # Add main content
        report_content.append(main_content)
        
        # Write the report
        report_path = self.output_dir / 'comprehensive_analysis_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
            
        self.logger.info(f"Report generated successfully at {report_path}")
        
        # Generate HTML version if requested
        if generate_html:
            try:
                self._generate_html_report(report_path)
            except ImportError as e:
                self.logger.warning(f"Could not generate HTML report: {str(e)}")
                self.logger.warning("Install required packages with: pip install markdown beautifulsoup4")
        
    def _generate_html_report(self, markdown_path: Path) -> None:
        """Convert markdown report to HTML with styling."""
        import markdown
        from bs4 import BeautifulSoup
        
        # Read markdown content
        with open(markdown_path, 'r') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add styling
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GAA Cork Spatial Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Write HTML file
        html_path = self.output_dir / 'comprehensive_analysis_report.html'
        with open(html_path, 'w') as f:
            f.write(html_template)
            
        self.logger.info(f"HTML report generated at {html_path}")

if __name__ == "__main__":
    # Set up paths
    analysis_dir = Path("data/analysis")
    output_dir = Path("data/analysis/reports")
    
    # Generate report
    generator = ReportGenerator(analysis_dir, output_dir)
    generator.generate_report() 