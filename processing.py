# processing.py
import json
from typing import Tuple

import os
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes, rasterize

import geopandas as gpd
from shapely.geometry import box, shape

from config import (
    BATHY_RAW_PATH,
    BATHY_PATH,
    HYDRO_PATH,
    BASINS_PATH,
    TOPO_PATH,
    SLOPE_PATH,
    REEF_PARQUET_PATH,
    BASINS_PARQUET_PATH,
    METADATA_PATH,
    HYDRO_NODATA,
    SHALLOW_MIN_DEPTH,
    SHALLOW_MAX_DEPTH,
    REEF_SLOPE_MIN_DEG,
    REEF_SLOPE_MAX_DEG,
    Q_SPEC_M3S_PER_KM2,
)


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop NaN, empty, and invalid geometries."""
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()
    return gdf

def cdf_to_geotiff():
    ds = xr.open_dataset(BATHY_RAW_PATH)
    bathy = ds["Band1"]

    # setting dimension names
    bathy = bathy.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace = False)
    bathy = bathy.rio.write_crs("EPSG:32610", inplace=False)  # UTM zone 10N
    print("In-memory CRS:", bathy.rio.crs)   # checking if geotiff will contain crs

    #sanity_check
    print(bathy.rio.crs)
    print(bathy.rio.bounds())

    # export as GeoTIFF
    return bathy.rio.to_raster(BATHY_PATH, compress='LZW')

# 1. Load bathymetry and HydroSHEDS, align, build topobathy
def build_topobathy() -> Tuple[xr.DataArray, xr.DataArray]:
    # if not os.path.exists(BATHY_RAW_PATH):
    #     raise FileNotFoundError(
    #         f"Raw bathymetry not found at {BATHY_RAW_PATH}. "
    #         "Place your GEBCO subset there or update BATHY_RAW_PATH in config.py."
    #     )
    # # Load raw bathymetry
    # bathy_raw = rioxarray.open_rasterio(BATHY_RAW_PATH).squeeze()
    # print("Raw bathy CRS:", bathy_raw.rio.crs)
    # print("Raw bathy shape:", bathy_raw.shape)
    # print("Raw bathy bounds:", bathy_raw.rio.bounds())

    # hydro = rioxarray.open_rasterio(HYDRO_PATH).squeeze()
    # hydro = hydro.rio.write_crs("EPSG:4326", inplace=False)
    # hydro = hydro.rio.write_nodata(HYDRO_NODATA, inplace=False)

    # hxmin, hymin, hxmax, hymax = hydro.rio.bounds()
    # bxmin, bymin, bxmax, bymax = bathy_raw.rio.bounds()

    # print("Hydro bounds:", hxmin, hymin, hxmax, hymax)
    # print("Bathy raw bounds:", bxmin, bymin, bxmax, bymax)

    # # Compute intersection ROI
    # roi_minx = max(bxmin, hxmin)
    # roi_miny = max(bymin, hymin)
    # roi_maxx = min(bxmax, hxmax)
    # roi_maxy = min(bymax, hymax)

    # if (roi_minx >= roi_maxx) or (roi_miny >= roi_maxy):
    #     raise RuntimeError(
    #         "HydroSHEDS and raw bathymetry have no overlapping region.\n"
    #         f"Hydro bounds: {hxmin, hymin, hxmax, hymax}\n"
    #         f"Bathy bounds: {bxmin, bymin, bxmax, bymax}\n"
    #         "Check that HYDRO_PATH and BATHY_RAW_PATH refer to matching tiles "
    #         "(e.g., both 40–50N, 130–120W)."
    #     )

    # bathy_raw_clipped = bathy_raw.rio.clip_box(
    #     minx=roi_minx,
    #     miny=roi_miny,
    #     maxx=roi_maxx,
    #     maxy=roi_maxy,
    # )

    # print("Clipped bathy shape:", bathy_raw_clipped.shape)
    # print("Clipped bathy bounds:", bathy_raw_clipped.rio.bounds())

    # # Reproject to a working UTM CRS
    # target_crs = "EPSG:32610"

    # bathy = bathy_raw_clipped.rio.reproject(
    #     target_crs,
    #     resampling=Resampling.bilinear,
    # )

    # print("Bathy (UTM) CRS:", bathy.rio.crs)
    # print("Bathy (UTM) shape:", bathy.shape)
    # print("Bathy (UTM) bounds:", bathy.rio.bounds())

    # # Save UTM bathy as intermediate product
    # bathy.rio.to_raster(BATHY_PATH, compress="LZW")
    # print("Wrote UTM bathy:", BATHY_PATH)

    # # Load HydroSHEDS and align to bathy grid
    # hydro = rioxarray.open_rasterio(HYDRO_PATH).squeeze()
    # print("Hydro dims:", hydro.dims)
    # print("Hydro CRS before:", hydro.rio.crs)

    # hydro = hydro.rio.write_crs("EPSG:4326", inplace=False)
    # hydro = hydro.rio.write_nodata(HYDRO_NODATA, inplace=False)

    # hydro_utm = hydro.rio.reproject_match(
    #     bathy,
    #     resampling=Resampling.bilinear,
    # )

    # print("Hydro_utm CRS:", hydro_utm.rio.crs)
    # print("Hydro_utm shape:", hydro_utm.shape)

    # # Merge into topobathy
    # land_mask = hydro_utm > 0
    # topobathy = xr.where(land_mask, hydro_utm, bathy)

    # print("Topobathy stats:")
    # print("  min:", float(topobathy.min()))
    # print("  max:", float(topobathy.max()))

    # topobathy_clean = topobathy.where(topobathy != HYDRO_NODATA)
    # topobathy_clean = topobathy_clean.rio.write_nodata(HYDRO_NODATA, inplace=False)

    # print(
    #     "Cleaned topobathy stats:",
    #     float(topobathy_clean.min().values),
    #     float(topobathy_clean.max().values),
    # )

    # topobathy_clean.rio.to_raster(TOPO_PATH, compress="LZW")
    # print("Wrote topobathy:", TOPO_PATH)

    # return bathy, topobathy_clean

    bathy = rioxarray.open_rasterio(BATHY_PATH).squeeze()
    print("Bathy CRS:", bathy.rio.crs)
    print("Bathy shape:", bathy.shape)
    print("Bathy res:", bathy.rio.resolution())
    print("Bathy bounds:", bathy.rio.bounds())

    # HydroSHEDS conditioned DEM (lat/lon)
    hydro = rioxarray.open_rasterio(HYDRO_PATH).squeeze()
    print("Hydro dims:", hydro.dims)
    print("Hydro CRS before:", hydro.rio.crs)

    # Set CRS and nodata for HydroSHEDS
    hydro = hydro.rio.write_crs("EPSG:4326", inplace=False)
    hydro = hydro.rio.write_nodata(HYDRO_NODATA, inplace=False)

    # Reproject HydroSHEDS to match bathy grid
    hydro_utm = hydro.rio.reproject_match(
        bathy,
        resampling=Resampling.bilinear,
    )

    print("Hydro_utm CRS:", hydro_utm.rio.crs)
    print("Hydro_utm shape:", hydro_utm.shape)
    print("Hydro_utm bounds:", hydro_utm.rio.bounds())

    # Land mask and merged topobathy
    land_mask = hydro_utm > 0
    print("Land mask type:", land_mask.dtype)

    topobathy = xr.where(
        land_mask,
        hydro_utm,  # land elevations (if any)
        bathy,      # underwater bathymetry
    )

    print("Topobathy stats:")
    print("  min:", float(topobathy.min()))
    print("  max:", float(topobathy.max()))
    print("Hydro_utm stats:", float(hydro_utm.min()), float(hydro_utm.max()))
    print("Bathy stats:", float(bathy.min()), float(bathy.max()))

    # Clean nodata from topobathy
    topobathy_clean = topobathy.where(topobathy != HYDRO_NODATA)
    topobathy_clean = topobathy_clean.rio.write_nodata(HYDRO_NODATA, inplace=False)

    print(
        "Cleaned topobathy stats:",
        float(topobathy_clean.min().values),
        float(topobathy_clean.max().values),
    )

    topobathy_clean.rio.to_raster(TOPO_PATH, compress="LZW")
    print("Wrote topobathy:", TOPO_PATH)

    return bathy, topobathy_clean


# 2. HydroBASINS: clip to bathy extent, find outlets, hydrology
def process_basins_and_outlets(
    bathy: xr.DataArray,
    topobathy_clean: xr.DataArray,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    basins = gpd.read_file(BASINS_PATH)
    print("Basins CRS before:", basins.crs)

    # Reproject to match bathy/topobathy CRS
    basins_utm = basins.to_crs(bathy.rio.crs)
    print("Basins CRS after:", basins_utm.crs)

    # Clip basins to bathymetry bounding box
    minx, miny, maxx, maxy = bathy.rio.bounds()
    bbox = gpd.GeoDataFrame(
        geometry=[box(minx, miny, maxx, maxy)],
        crs=bathy.rio.crs,
    )
    basins_clip = gpd.overlay(basins_utm, bbox, how="intersection")
    print("Basins clipped:", len(basins_clip))

    # Water mask and union polygon
    water_mask = (topobathy_clean < 0).fillna(False)
    water_shapes = shapes(
        water_mask.astype("uint8").values,
        mask=None,
        transform=bathy.rio.transform(),
    )
    water_polys = [shape(geom) for geom, val in water_shapes if val == 1]

    water_gdf = gpd.GeoDataFrame(
        geometry=water_polys,
        crs=bathy.rio.crs,
    )
    print("Raw water polygons:", len(water_gdf))

    # Dissolve into a single water geometry
    water_union = water_gdf.union_all()
    water_union_gdf = gpd.GeoDataFrame(
        geometry=[water_union],
        crs=bathy.rio.crs,
    )

    # Derive outlet points (basin–sea intersections)
    outlet_records = []
    for idx, row in basins_clip.iterrows():
        basin_geom = row.geometry
        inter = basin_geom.boundary.intersection(water_union)
        if inter.is_empty:
            continue

        try:
            outlet_point = inter.representative_point()
        except Exception:
            outlet_point = inter.centroid

        outlet_records.append(
            {
                "basin_id": row.get("HYBAS_ID", idx),
                "geometry": outlet_point,
            }
        )

    outlets_gdf = gpd.GeoDataFrame(outlet_records, crs=bathy.rio.crs)
    outlets_gdf = outlets_gdf.set_crs(bathy.rio.crs, allow_override=True)
    print("Number of outlets:", len(outlets_gdf))

    # Basin areas
    basins_clip["area_m2"] = basins_clip.geometry.area
    basins_clip["area_km2"] = basins_clip["area_m2"] / 1e6

    # Centroids
    centroids = basins_clip.geometry.centroid
    basins_clip["centroid_x"] = centroids.x
    basins_clip["centroid_y"] = centroids.y

    print(basins_clip[["HYBAS_ID", "area_km2"]].head())

    # Standardize ID name
    basins_clip = basins_clip.rename(columns={"HYBAS_ID": "basin_id"})

    print("Basins columns:", basins_clip.columns)
    print("Outlets columns:", outlets_gdf.columns)

    # Join basin attributes onto outlets
    outlets = outlets_gdf.merge(
        basins_clip[["basin_id", "area_km2", "centroid_x", "centroid_y"]],
        on="basin_id",
        how="left",
    )

    # Synthetic discharge: Q [m^3/s]
    q_spec = Q_SPEC_M3S_PER_KM2
    outlets["discharge_m3s"] = q_spec * outlets["area_km2"]

    print(outlets[["basin_id", "area_km2", "discharge_m3s"]].head())

    # Outlet coordinates
    outlets["out_x"] = outlets.geometry.x
    outlets["out_y"] = outlets.geometry.y

    # Direction vector: centroid -> outlet (land to sea)
    dx = outlets["out_x"] - outlets["centroid_x"]
    dy = outlets["out_y"] - outlets["centroid_y"]
    length = np.sqrt(dx**2 + dy**2)
    length = length.replace(0, np.nan)

    outlets["dir_x"] = dx / length
    outlets["dir_y"] = dy / length

    # Channel geometry assumptions
    channel_width_m = 100.0
    channel_depth_m = 3.0
    cross_section_area = channel_width_m * channel_depth_m  # m²

    # Velocity magnitude [m/s]
    outlets["velocity_ms"] = outlets["discharge_m3s"] / cross_section_area
    outlets["velocity_ms"] = outlets["velocity_ms"].clip(upper=5.0)

    outlets["vel_x"] = outlets["velocity_ms"] * outlets["dir_x"]
    outlets["vel_y"] = outlets["velocity_ms"] * outlets["dir_y"]

    rho = 1000.0  # kg/m^3
    outlets["momentum"] = rho * outlets["discharge_m3s"] * outlets["velocity_ms"]
    outlets["mom_x"] = outlets["momentum"] * outlets["dir_x"]
    outlets["mom_y"] = outlets["momentum"] * outlets["dir_y"]

    return basins_clip, outlets, water_union_gdf


# 3. Slope, zonal stats, reef detection, exports, metadata
def compute_slope_and_zonal_stats(
    bathy: xr.DataArray,
    topobathy_clean: xr.DataArray,
    basins_clip: gpd.GeoDataFrame,
) -> Tuple[xr.DataArray, gpd.GeoDataFrame]:
    print("Bathy CRS:", bathy.rio.crs)
    print("Bathy res:", bathy.rio.resolution())
    print("bathy shape:", bathy.shape)
    print("topobathy_clean shape:", topobathy_clean.shape)

    dem = bathy.copy(data=topobathy_clean.values)
    dem.name = "topobathy"

    print("DEM CRS:", dem.rio.crs)
    print("DEM res:", dem.rio.resolution())

    dem_masked = dem.where(dem != HYDRO_NODATA)

    res_x, res_y = dem_masked.rio.resolution()
    dx = abs(res_x)
    dy = abs(res_y)

    dz_dy, dz_dx = np.gradient(dem_masked.values, dy, dx)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.degrees(slope_rad)

    slope = dem_masked.copy(data=slope_deg)
    slope.name = "slope_deg"

    print("Slope stats (deg):", float(np.nanmin(slope)), float(np.nanmax(slope)))
    print("Slope CRS:", slope.rio.crs)
    print("Slope res:", slope.rio.resolution())

    slope.rio.to_raster(SLOPE_PATH, compress="LZW")
    print("Wrote slope raster:", SLOPE_PATH)

    with rasterio.open(SLOPE_PATH) as src:
        slope_arr = src.read(1, masked=True).filled(np.nan)
        transform = src.transform
        slope_crs = src.crs
        slope_bounds_geom = box(*src.bounds)

    print("Slope array shape:", slope_arr.shape)
    print("Slope CRS:", slope_crs)

    basins_clip = basins_clip.set_crs(slope_crs, allow_override=True)
    basins_clip = basins_clip[basins_clip.geometry.notna()].copy()
    basins_clip = basins_clip[~basins_clip.geometry.is_empty].copy()
    basins_clip = basins_clip[basins_clip.geometry.is_valid].copy()
    basins_clip = basins_clip[basins_clip.intersects(slope_bounds_geom)].copy()
    basins_clip = basins_clip.reset_index(drop=True)

    print("Basins going into manual zonal stats:", len(basins_clip))

    mean_list = []
    median_list = []
    max_list = []
    std_list = []

    for geom in basins_clip.geometry:
        mask = rasterize(
            [(geom, 1)],
            out_shape=slope_arr.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )

        vals = slope_arr[mask == 1]
        vals = vals[~np.isnan(vals)]

        if vals.size == 0:
            mean_list.append(np.nan)
            median_list.append(np.nan)
            max_list.append(np.nan)
            std_list.append(np.nan)
        else:
            mean_list.append(float(np.nanmean(vals)))
            median_list.append(float(np.nanmedian(vals)))
            max_list.append(float(np.nanmax(vals)))
            std_list.append(float(np.nanstd(vals)))

    basins_clip["slope_mean_deg"] = mean_list
    basins_clip["slope_median_deg"] = median_list
    basins_clip["slope_max_deg"] = max_list
    basins_clip["slope_std_deg"] = std_list

    print(basins_clip[["basin_id", "slope_mean_deg", "slope_max_deg"]].head())

    return slope, basins_clip


def detect_reef_features(
    topobathy_clean: xr.DataArray,
    slope: xr.DataArray,
    water_union_gdf: gpd.GeoDataFrame,
    basins_clip: gpd.GeoDataFrame,
):
    depth = topobathy_clean.where(topobathy_clean != HYDRO_NODATA)

    # Shallow band
    shallow = (depth < SHALLOW_MAX_DEPTH) & (depth >= SHALLOW_MIN_DEPTH)

    # Slope band
    reef_slope = (slope >= REEF_SLOPE_MIN_DEG) & (slope <= REEF_SLOPE_MAX_DEG)

    reef_mask = shallow & reef_slope
    print(
        "Reef-like cell fraction:",
        float(reef_mask.sum() / np.isfinite(depth).sum()),
    )

    reef_uint8 = reef_mask.fillna(False).astype("uint8")

    reef_shapes = shapes(
        reef_uint8.values,
        mask=None,
        transform=topobathy_clean.rio.transform(),
    )
    reef_geoms = [shape(geom) for geom, val in reef_shapes if val == 1]

    reef_gdf = gpd.GeoDataFrame(
        geometry=reef_geoms,
        crs=topobathy_clean.rio.crs,
    )
    print("Number of reef-like polygons:", len(reef_gdf))

    # Keep only reef polygons in water
    reef_gdf = gpd.overlay(reef_gdf, water_union_gdf, how="intersection")

    # Reef area by basin
    reef_by_basin = gpd.overlay(basins_clip, reef_gdf, how="intersection")
    reef_by_basin["reef_area_m2"] = reef_by_basin.geometry.area

    reef_stats = (
        reef_by_basin.groupby("basin_id")["reef_area_m2"].sum().reset_index()
    )

    basins_clip = basins_clip.merge(reef_stats, on="basin_id", how="left")
    basins_clip["reef_area_m2"] = basins_clip["reef_area_m2"].fillna(0)

    return reef_gdf, basins_clip, reef_mask, depth


def export_data(
    reef_gdf: gpd.GeoDataFrame,
    basins_clip: gpd.GeoDataFrame,
):
    reef_gdf.to_parquet(REEF_PARQUET_PATH)
    basins_clip.to_parquet(BASINS_PARQUET_PATH)

    print("Wrote reef GeoParquet:", REEF_PARQUET_PATH)
    print("Wrote basins GeoParquet:", BASINS_PARQUET_PATH)


def write_metadata(
    basins_clip: gpd.GeoDataFrame,
    outlets: gpd.GeoDataFrame,
    reef_gdf: gpd.GeoDataFrame,
    reef_mask,
    depth,
    channel_width_m: float = 100.0,
    channel_depth_m: float = 3.0,
):
    metadata = {
        "topobathy_tif": TOPO_PATH,
        "slope_tif": SLOPE_PATH,
        "reef_parquet": REEF_PARQUET_PATH,
        "basins_parquet": BASINS_PARQUET_PATH,
        "num_basins": int(len(basins_clip)),
        "num_outlets": int(len(outlets)),
        "num_reef_polygons": int(len(reef_gdf)),
        "reef_fraction_cells": float(reef_mask.sum() / np.isfinite(depth).sum()),
        "q_spec_m3s_per_km2": float(Q_SPEC_M3S_PER_KM2),
        "channel_width_m": float(channel_width_m),
        "channel_depth_m": float(channel_depth_m),
        "velocity_clip_ms": 5.0,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Wrote metadata summary:", METADATA_PATH)
