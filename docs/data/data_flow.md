# GAA Cork Analysis - Data Flow Documentation

## Overview
This document outlines the data flow and processing pipeline for the GAA Cork Club Analysis project. It details the relationships between input data sources, processing scripts, and output files.

## Data Sources
1. **Census Data**
   - Small Area Population Statistics (SAPS) from the 2022 Census
   - Contains demographic information at Small Area level
   - Provides socioeconomic indicators (education, employment, social class)

2. **Geographic Data**
   - Small Area (SA) boundary files for Cork
   - Provides spatial information for each Small Area
   - Base layer for spatial analysis and visualization

3. **Club Data**
   - GAA club information (location, grades, codes)
   - Championship performance data (2022-2024)
   - Club facilities and resources information

## Processing Pipeline

### 1. Data Acquisition and Preparation
```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│ SAPS Data   │────>│ SA Boundary │────>│ Club Data    │
│ Processing  │     │ Processing  │     │ Processing   │
└─────────────┘     └─────────────┘     └──────────────┘
       │                  │                    │
       ▼                  ▼                    ▼
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│ Cleaned     │     │ Processed   │     │ Standardized │
│ SAPS Data   │     │ Boundaries  │     │ Club Data    │
└─────────────┘     └─────────────┘     └──────────────┘
```

### 2. Spatial Integration
```
┌─────────────┐     ┌─────────────┐
│ Cleaned     │     │ Processed   │
│ SAPS Data   │────>│ Boundaries  │
└─────────────┘     └─────────────┘
       │                  │
       └──────────┬───────┘
                  ▼
        ┌───────────────────┐
        │ SAPS-SA Integrated│
        │ Dataset           │
        └───────────────────┘
```

### 3. Catchment Area Analysis
```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Standardized  │     │ SAPS-SA       │     │ Cork Boundary │
│ Club Data     │────>│ Integrated    │────>│ Data          │
└───────────────┘     │ Dataset       │     └───────────────┘
       │              └───────────────┘            │
       │                      │                    │
       ▼                      ▼                    ▼
┌─────────────┐     ┌───────────────────┐     ┌──────────────┐
│ Voronoi     │     │ Nearest Club      │     │ Buffer       │
│ Catchments  │     │ Analysis          │     │ Analysis     │
└─────────────┘     └───────────────────┘     └──────────────┘
       │                      │                    │
       └──────────────┬───────┴────────────┬──────┘
                      ▼                    ▼
          ┌─────────────────────┐  ┌────────────────────┐
          │ Catchment Comparison│  │ Catchment-Based    │
          │ Analysis            │  │ Demographics        │
          └─────────────────────┘  └────────────────────┘
```

### 4. Statistical Analysis
```
┌────────────────────┐     ┌─────────────────────┐
│ Catchment-Based    │────>│ Correlation Analysis│
│ Demographics       │     └─────────────────────┘
└────────────────────┘             │
       │                           ▼
       │                  ┌─────────────────────┐
       │                  │ Regression Analysis │
       │                  └─────────────────────┘
       ▼                           │
┌─────────────────┐                ▼
│ Spatial Pattern │       ┌─────────────────────┐
│ Analysis        │<──────│ GWR Analysis        │
└─────────────────┘       └─────────────────────┘
```

### 5. Visualization and Reporting
```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│ Statistical     │     │ Spatial Pattern     │     │ Demographic     │
│ Results         │────>│ Results            │────>│ Insights        │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
       │                          │                        │
       └──────────────┬───────────┴────────────┬──────────┘
                      ▼                        ▼
          ┌─────────────────────┐    ┌─────────────────────┐
          │ Data Visualization  │    │ Final Report        │
          └─────────────────────┘    └─────────────────────┘
```

## Key Files and Locations

### Input Data
- `data/raw/saps/`: Raw SAPS data from CSO
- `data/raw/sa/`: Raw Small Area boundary files
- `data/raw/clubs/`: Raw club data files

### Intermediate Data
- `data/processed/saps/`: Processed SAPS data
- `data/processed/sa/`: Processed SA boundary files
- `data/processed/clubs/`: Processed club data
- `data/processed/integrated/`: Integrated SAPS-SA data

### Analysis Outputs
- `data/analysis/catchments/`: Catchment area analysis results
- `data/analysis/statistics/`: Statistical analysis results
- `data/analysis/spatial/`: Spatial pattern analysis results

### Visualization Outputs
- `report_visualizations/`: Visualizations for the final report
- `report_visualizations/Best/`: Final visualizations selected for publication

## Processing Scripts

### Data Processing
- `src/data/`: Data processing scripts
  - SAPS data cleaning and transformation
  - SA boundary preparation
  - Club data standardization

### Spatial Analysis
- `src/spatial/`: Spatial analysis scripts
  - Voronoi catchment generation
  - Nearest club analysis
  - Buffer analysis
  - Spatial data validation

### Statistical Analysis
- `src/analysis/`: Statistical analysis scripts
  - Correlation analysis
  - Regression modeling
  - GWR implementation
  - Model validation

### Visualization
- `src/visualization/`: Visualization scripts
  - Statistical visualizations
  - Spatial visualizations
  - Interactive maps
  - Report-ready graphics

## Data Flow Dependencies
- SAPS-SA integration must precede catchment analysis
- Catchment analysis must precede demographic aggregation
- Demographic aggregation must precede statistical analysis
- Statistical analysis must precede spatial pattern analysis
- All analyses must complete before final visualization 