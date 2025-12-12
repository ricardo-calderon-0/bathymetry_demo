# config.py
import os

# Paths
BATHY_RAW_PATH = "data/raw/bathymetry/gebco_2025_n50.0_s40.0_w-130.0_e-120.0.tif"
BATHY_PATH = "data/intermediate/bathymetry/puget_sound_bathy_utm.tif"
HYDRO_PATH = "data/raw/hydrosheds/n40w130_con.tif"
BASINS_PATH = "data/raw/hydrobasins/hybas_na_lev06_v1c.shp"

TOPO_PATH = "data/intermediate/topobathy/puget_sound_topobathy_utm30m_clean.tif"
SLOPE_PATH = "data/intermediate/topobathy/puget_sound_topobathy_slope_deg.tif"

REEF_PARQUET_PATH = "data/processed/reef/puget_sound_shallow_reef_candidates.parquet"
BASINS_PARQUET_PATH = "data/processed/basins/puget_sound_basins_with_slopes.parquet"
METADATA_PATH = "data/intermediate/puget_sound_metadata_summary.json"
QC_PNG_PATH = "data/processed/puget_sound_qc_map.png"

# Make sure directories exist
os.makedirs("data/raw/bathymetry", exist_ok=True)
os.makedirs("data/raw/hydrosheds", exist_ok=True)
os.makedirs("data/raw/hydrobasins", exist_ok=True)

os.makedirs("data/intermediate/bathymetry", exist_ok=True)
os.makedirs("data/intermediate/topobathy", exist_ok=True)
os.makedirs("data/intermediate/reef", exist_ok=True)
os.makedirs("data/intermediate/basins", exist_ok=True)

os.makedirs("data/processed/reef", exist_ok=True)
os.makedirs("data/processed/basins", exist_ok=True)

# Params
HYDRO_NODATA = -9999.0

# Reef-like detection thresholds
SHALLOW_MIN_DEPTH = -30.0  # meters (negative = below sea level)
SHALLOW_MAX_DEPTH = 0.0    # meters
REEF_SLOPE_MIN_DEG = 5.0
REEF_SLOPE_MAX_DEG = 30.0

# Discharge model parameter
Q_SPEC_M3S_PER_KM2 = 0.02  # m^3/s per km^2 (synthetic)

# Plotting CRS
PLOT_CRS = "EPSG:4326"
