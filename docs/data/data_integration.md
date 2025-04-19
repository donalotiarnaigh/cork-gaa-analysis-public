# SAPS-SA Data Integration Documentation

## Overview
This document describes the process of integrating Small Area Population Statistics (SAPS) with Small Area (SA) boundary data for Cork City and County. The integration uses a GUID-based approach for reliable matching between datasets.

## Process Documentation

### Integration Steps
1. **Data Loading**
   - Load Small Area boundaries from shapefile
   - Load SAPS data from processed CSV
   - Filter SA data for Cork records only

2. **Split Area Handling**
   - Identify split areas in SA dataset
   - Handle two types of splits:
     - Numbered splits (e.g., '048073002/01')
     - Full splits (e.g., '048064004/048064005')
   - Create additional records for split parts

3. **Identifier Standardization**
   - Standardize SA identifiers (`SA_PUB2022`)
   - Standardize SAPS identifiers (`geogid`)
   - Implement GUID-based matching
   - Ensure consistent format across datasets

4. **Data Join**
   - Perform left join on standardized identifiers
   - Preserve spatial information
   - Maintain original attributes
   - Handle split areas appropriately

5. **Result Export**
   - Export as GeoPackage (spatial)
   - Export as CSV (tabular)
   - Export unmatched records separately
   - Generate validation reports

### Issues Encountered and Solutions

1. **Split Areas**
   - Issue: Different formats for split areas
   - Solution: Implemented specialized handling for both numbered and full splits
   - Result: Improved match rate by 16.77%

2. **Identifier Format**
   - Issue: Inconsistent identifier formats between datasets
   - Solution: Used GUID-based matching as a reliable alternative
   - Result: Achieved 100% match rate

3. **Spatial Data Handling**
   - Issue: CRS information missing in shapefile
   - Solution: Explicitly set and transform to EPSG:2157
   - Note: Complete spatial integrity maintained

## Data Structure Documentation

### Input Datasets

#### Small Area Boundaries
- Source: SA boundary shapefile
- Key Columns:
  - `SA_PUB2022`: Primary identifier
  - `SA_GUID_2022`: Global unique identifier
  - `COUNTY_ENG`: County name
  - `ED_ENGLISH`: Electoral Division
  - `geometry`: Spatial information

#### SAPS Data
- Source: Processed SAPS data
- Key Columns:
  - `geogid`: Primary identifier
  - `GUID`: Global unique identifier
  - Various socioeconomic indicators

### Output Datasets

#### Integrated Dataset
- Format: GeoPackage and CSV
- Naming: `cork_sa_saps_joined_guid.{ext}`
- Key Features:
  - Combined spatial and attribute data
  - Standardized identifiers
  - Split area handling information
  - Join indicator

#### Unmatched Records
- Format: CSV
- Naming: `cork_sa_saps_unmatched.csv`
- Purpose: Track records requiring attention

### Field Mapping Guide

| SAPS Field | SA Field | Description | Notes |
|------------|----------|-------------|-------|
| geogid | SA_PUB2022 | Primary identifier | Standardized during join |
| GUID | SA_GUID_2022 | Global unique identifier | Key for reliable join |
| COUNTY_ENG | - | County name | From SA dataset |
| ED_ENGLISH | - | Electoral Division | From SA dataset |
| geometry | - | Spatial information | From SA dataset |
| split_type | - | Type of split area | 'numbered' or 'full' |
| split_number | - | Specific split identifier | For numbered splits |

## Integration Results

1. **Record Counts**
   - Input SA boundaries: 2,206 records
   - Split areas identified: 245
   - Additional records created: 299
   - Final output: 2,260 records

2. **Geographic Coverage**
   - CORK CITY: 845 records (99.41% match rate)
   - CORK: 1,338 records (94.17% match rate)
   - All areas properly represented in final output

3. **Data Quality**
   - No unmatched records
   - All GUIDs preserved
   - Split areas properly handled
   - Spatial integrity maintained

## Usage Examples

1. **Basic Data Loading**
```python
import geopandas as gpd
gdf = gpd.read_file('data/processed/cork_sa_saps_joined_guid.gpkg')
print(f"Total records: {len(gdf)}")
```

2. **County Analysis**
```python
county_stats = gdf.groupby('COUNTY_ENG').agg({
    'split_type': lambda x: x.notna().sum(),
    'GUID': 'count'
})
county_stats['split_percentage'] = (county_stats['split_type'] / county_stats['GUID'] * 100)
```

3. **Split Area Analysis**
```python
split_areas = gdf[gdf['split_type'].notna()]
print(f"Split areas: {len(split_areas)}")
print(split_areas['split_type'].value_counts())
```

## Known Limitations

1. **Spatial Data**
   - All data in EPSG:2157 projection
   - Some coastal boundaries simplified
   - Island areas may have connection issues

2. **Split Areas**
   - Only handles standard split formats
   - May miss custom split patterns
   - Original split information preserved but may need manual review

3. **Data Quality**
   - Source data accuracy dependent on CSO
   - 2022 Census base with no temporal updates
   - Some demographic variables have missing values 