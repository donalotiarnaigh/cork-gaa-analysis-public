# Cork GAA Club Analysis

## Project Overview

This repository contains code and documentation for analyzing the relationship between demographic characteristics and GAA club success in Cork County, Ireland. The project integrates spatial analysis with demographic data to identify patterns and factors associated with club performance in both Gaelic football and hurling.

## Key Features

- **Catchment Area Analysis**: Implementation of multiple methodologies (Voronoi tessellation, nearest-club assignment, buffer analysis) to define club catchment areas
- **Demographic Integration**: Techniques for integrating Small Area Population Statistics (SAPS) with spatial boundaries
- **Statistical Modeling**: Ordinary Least Squares (OLS) and Geographically Weighted Regression (GWR) implementations to analyze relationships between demographics and performance
- **Spatial Pattern Analysis**: Hotspot analysis and spatial autocorrelation techniques to identify geographic patterns in club performance
- **Visualization Tools**: Specialized visualization scripts for both statistical and spatial data

## Repository Structure

```
├── data/                # Data directory (not included - see Data section below)
├── docs/                # Documentation
│   ├── data/            # Data integration and flow documentation
│   └── methodology/     # Analysis methodology documentation
├── src/                 # Source code
│   ├── analysis/        # Statistical analysis scripts
│   ├── data/            # Data processing utilities
│   ├── scripts/         # Utility scripts
│   ├── spatial/         # Spatial analysis tools
│   └── visualization/   # Visualization scripts
└── README.md            # This file
```

## Data

This repository contains code only and does not include the data used in the analysis. The analysis requires:

1. **Small Area Population Statistics (SAPS)** from the Central Statistics Office (CSO) 2022 Census
2. **Small Area (SA) boundary files** for Cork County and City
3. **GAA club data** including locations and championship grades

To use this codebase with your own data, please structure your data files according to the paths expected in the scripts (see `docs/data/data_flow.md`).

## Installation

### Requirements

- Python 3.8+
- Required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- geopandas
- pandas
- numpy
- matplotlib
- scikit-learn
- mgwr (for Geographically Weighted Regression)
- pysal/esda (for spatial statistics)
- shapely (for geometric operations)

## Usage

### Data Processing Pipeline

1. **SAPS Data Processing**:
```bash
python src/data/process_saps_data.py
```

2. **SA Boundary Processing**:
```bash
python src/spatial/sa_boundary_preparation.py
```

3. **SAPS-SA Integration**:
```bash
python src/spatial/process_assignments.py
```

4. **Catchment Area Analysis**:
```bash
python src/spatial/voronoi_catchment_analysis.py
python src/spatial/nearest_club_analysis.py
python src/spatial/buffer_generation.py
```

5. **Statistical Analysis**:
```bash
python src/analysis/demographic_correlation_analysis.py
python src/analysis/linear_regression_models.py
python src/analysis/gwr_analysis.py
```

### Generating Visualizations

To generate key visualizations:
```bash
python src/visualization/statistical_visualization.py
python src/visualization/gwr_spatial_visualizations.py
```

## Methodology Overview

The project employs a three-stage methodology:

1. **Club Catchment Area Delineation**: Three approaches are implemented (Voronoi tessellation, nearest-club, and buffer methods) to define the geographic areas served by each club

2. **Demographic Profiling**: Catchment areas are enriched with demographic data from the 2022 Census, including education levels, youth population, social class distribution, and employment patterns

3. **Statistical Analysis**: Relationships between demographic characteristics and club performance are analyzed using both global (OLS) and local (GWR) regression techniques

For detailed methodology documentation, see the `docs/methodology/` directory.

## Key Findings

- Significant spatial clustering of high-performance clubs, particularly in urban areas
- Youth population shows consistent positive association with club performance
- Education levels (particularly third-level education) correlate with performance
- Spatial variation in these relationships, with different factors important in different areas
- Competition density shows complex relationships with club performance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or methodology in your research, please cite:

```
Tierney, D. (2025). Analyzing the relationship between demographic characteristics 
and GAA club success in Cork, Ireland.
```

## Contact

For questions or collaborations, please contact:
- daniel@curlew.ie
