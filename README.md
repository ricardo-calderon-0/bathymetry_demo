# Topobathymetry + Watershed Integration Pipeline
Workflow integrating bathymetry and HydroBASINS data to compute topobathymetry, slopes, reef-like shallow features, and watershed outlet momentum vectors.

### Overview

This repository contains a geospatial processing pipeline that integrates regional bathymetry, HydroSHEDS terrain, and HydroBASINS watershed boundaries to produce harmonized topobathymetric surfaces, terrain slopes, shallow-reef morphological features, and watershed outlet discharge/momentum vectors.

The workflow:

- Aligns GEBCO 2025 bathymetry with HydroSHEDS DEMs

- Constructs a fused topobathymetric DEM with consistent CRS, nodata handling, and aligned grids

- Computes terrain slopes from the merged DEM

- Identifies basin–sea outlet points via raster-to-vector water boundary intersection

- Derives synthetic discharge, velocity, and momentum vectors for each watershed outlet

 - Detects shallow, reef-like bathymetry structures using depth–slope morphological filters

- Exports processed layers as compressed GeoTIFFs and GeoParquets, suitable for large-scale analytics

- Generates geographic visualizations integrating all layers

### Study Area & Data
The demonstration focuses on a subsection of Puget Sound (Mt. Vernon → Olympia), using:

- GEBCO 2025 regional bathymetry (40–50°N, 130–120°W)

- HydroSHEDS/HydroBASINS terrain and watershed boundaries

- Synthetic outlet hydrodynamics for illustrative purposes
