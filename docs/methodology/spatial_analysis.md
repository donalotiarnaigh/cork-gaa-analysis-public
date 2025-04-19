# Spatial Analysis Methodology

## Catchment Area Analysis

### Voronoi Tessellation Method

The Voronoi tessellation approach creates club catchment areas by generating polygons where every point within the polygon is closer to its club than to any other club.

#### Implementation Details
1. **Polygon Generation**
   - Uses club locations as seed points
   - Creates Voronoi polygons using computational geometry
   - Clips polygons to the Cork county boundary
   - Handles edge effects at boundary regions

2. **Technical Specifications**
   - Projection: EPSG:2157 (Irish Transverse Mercator)
   - Topology preservation during clipping
   - Edge case handling for coastal areas
   - Validation of polygon integrity

3. **Advantages**
   - Complete coverage of study area
   - Mathematically precise boundaries
   - No overlapping catchment areas
   - Efficient computation

4. **Limitations**
   - Does not account for travel routes
   - Assumes equal access in all directions
   - May not reflect natural boundaries
   - Does not consider population distribution

### Nearest Club Method

The nearest club method assigns each Small Area to its geographically closest GAA club based on center-to-center distance calculations.

#### Implementation Details
1. **Assignment Process**
   - Calculates centroids for each Small Area
   - Computes distance to all club locations
   - Assigns each Small Area to nearest club
   - Handles edge cases where multiple clubs are equidistant

2. **Technical Specifications**
   - Distance calculation: Euclidean in projected space
   - Projection: EPSG:2157 (Irish Transverse Mercator)
   - Spatial indexing for performance optimization
   - Complete assignment validation

3. **Advantages**
   - Intuitive assignment logic
   - Accounts for population distribution
   - Preserves administrative boundaries
   - Less sensitive to outlier club locations

4. **Limitations**
   - Assumes centroids represent population centers
   - Does not account for natural barriers
   - Disregards road networks and travel time
   - Binary assignment does not model overlapping influence

### Buffer Analysis Method

The buffer analysis method creates concentric zones of influence around each club, allowing for overlapping catchment areas with distance-based weighting.

#### Implementation Details
1. **Buffer Creation**
   - Primary zone: 0-2km (weight = 1.0)
   - Secondary zone: 2-5km (weight = 0.7)
   - Tertiary zone: 5-10km (weight = 0.3)
   - Overlap handling with weighted influence

2. **Technical Specifications**
   - Buffer generation in projected space (EPSG:2157)
   - Intersection calculation for overlapping areas
   - Population-weighted attribution for demographic data
   - Statistical aggregation with tiered weighting

3. **Advantages**
   - Models gradual decrease in club influence
   - Accounts for overlapping catchment areas
   - Reflects realistic spheres of influence
   - Allows for competition analysis

4. **Limitations**
   - Arbitrary buffer distances
   - Circular buffers may not reflect actual travel patterns
   - Complex overlap handling increases computational requirements
   - Weighted statistics interpretation challenges

## Method Comparison

### Comparative Analysis Framework

A comprehensive comparison of catchment area methodologies was conducted, evaluating:

1. **Population Coverage**
   - Total population captured
   - Population density distributions
   - Urban/rural population balance
   - Youth population capture

2. **Spatial Characteristics**
   - Total area and distribution
   - Boundary complexity
   - Overlap characteristics (for buffer method)
   - Edge effects

3. **Demographic Characteristics**
   - Education level distributions
   - Employment patterns
   - Social class representation
   - Housing characteristics

4. **Performance Evaluation**
   - Method bias identification
   - Statistical robustness
   - Sensitivity to outliers
   - Practical implementation considerations

### Key Findings

1. **Method Agreement**
   - High agreement in urban areas with dense club distribution
   - Significant differences in rural areas with sparse clubs
   - Buffer method captures more realistic spheres of influence
   - Voronoi method provides clearest boundaries

2. **Demographic Insights**
   - Consistent demographic patterns across methods
   - Buffer method best captures demographic gradient effects
   - Nearest method best preserves population statistics
   - Voronoi method provides balanced geographic representation

3. **Implementation Recommendations**
   - Use multiple methods for robust analysis
   - Buffer method preferred for competition analysis
   - Voronoi method preferred for administrative assignment
   - Nearest method preferred for demographic analysis

## Spatial Pattern Analysis

### Hotspot Analysis

Spatial hotspot analysis was implemented to identify statistically significant clusters of high and low performance clubs.

#### Implementation Details
1. **Analysis Framework**
   - Local Moran's I for spatial autocorrelation
   - Getis-Ord Gi* for hotspot identification
   - K-function analysis for cluster distance evaluation
   - Multiple testing correction to prevent false positives

2. **Technical Specifications**
   - Spatial weights matrix: distance-based (5km threshold)
   - Significance threshold: p < 0.05 with FDR correction
   - Permutation approach: 999 randomizations
   - Software implementation: PySAL/esda package

3. **Key Findings**
   - Significant spatial clustering of high-performance clubs
   - Urban concentration of premier-grade clubs
   - Rural clusters of stable intermediate-grade clubs
   - Distance decay effect in performance metrics

### Geographically Weighted Regression (GWR)

GWR analysis was implemented to model spatial non-stationarity in the relationship between demographic factors and club performance.

#### Implementation Details
1. **Model Specification**
   - Dependent variable: Club performance score
   - Independent variables: Key demographic indicators
   - Bandwidth selection: Adaptive (bi-square kernel)
   - Calibration method: Cross-validation

2. **Technical Specifications**
   - Software implementation: mgwr package
   - Spatial weights: Adaptive bi-square kernel
   - Bandwidth optimization: Golden section search
   - Model comparison: AICc criterion

3. **Key Findings**
   - Significant spatial variation in model parameters
   - Local R² values ranging from 0.42 to 0.78
   - Education effects strongest in urban areas
   - Youth population effects strongest in suburban areas
   - Improved model fit compared to global regression 