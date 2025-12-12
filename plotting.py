# plotting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import geopandas as gpd
import xarray as xr

from config import HYDRO_NODATA, PLOT_CRS, QC_PNG_PATH


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()
    return gdf


def plot_qc_map(
    bathy: xr.DataArray,
    topobathy_clean: xr.DataArray,
    basins_clip: gpd.GeoDataFrame,
    reef_gdf: gpd.GeoDataFrame,
    outlets: gpd.GeoDataFrame,
):
    # Reproject for plotting
    reef_gdf = reef_gdf.set_crs(bathy.rio.crs, allow_override=True)
    basins_clip = basins_clip.set_crs(bathy.rio.crs, allow_override=True)
    outlets = outlets.set_crs(bathy.rio.crs, allow_override=True)

    topobathy_clean = topobathy_clean.rio.write_crs(bathy.rio.crs, inplace=False)
    topo_ll = topobathy_clean.rio.reproject(PLOT_CRS)

    basins_ll = clean_geometries(basins_clip.to_crs(PLOT_CRS))
    reef_ll = clean_geometries(reef_gdf.to_crs(PLOT_CRS))
    outlets_ll = clean_geometries(outlets.to_crs(PLOT_CRS))

    fig, ax = plt.subplots(figsize=(6, 8))

    depth_ll = topo_ll.values
    depth_ll = np.where(depth_ll == HYDRO_NODATA, np.nan, depth_ll)

    minx, miny, maxx, maxy = topo_ll.rio.bounds()

    valid_min = np.nanmin(depth_ll)
    valid_max = np.nanmax(depth_ll)
    print("Depth bounds (clean):", valid_min, valid_max)

    img = ax.imshow(
        depth_ll,
        cmap="terrain",
        extent=[minx, maxx, miny, maxy],
        origin="upper",
        alpha=0.7,
        vmin=valid_min,
        vmax=valid_max,
        zorder=1,
    )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Topobathymetry, Basins, Reefs, and Outlets")

    plt.colorbar(img, ax=ax, label="Elevation / Depth (m)")

    # Basins
    if basins_ll is not None and not basins_ll.empty:
        basins_ll.boundary.plot(
            ax=ax,
            linewidth=0.5,
            alpha=0.7,
            color="black",
            zorder=3,
        )

    # Reefs
    if reef_ll is not None and not reef_ll.empty:
        reef_ll.plot(
            ax=ax,
            facecolor="none",
            edgecolor="magenta",
            linewidth=0.3,
            alpha=1.0,
            zorder=4,
        )
    else:
        print("No reef polygons to plot after cleaning.")

    # Outlets
    if outlets_ll is not None and not outlets_ll.empty:
        outlets_ll.plot(
            ax=ax,
            color="yellow",
            edgecolor="black",
            markersize=20,
            zorder=5,
        )

    # Legend via proxy artists
    legend_handles = []
    if basins_ll is not None and not basins_ll.empty:
        legend_handles.append(
            Line2D([], [], color="black", linewidth=0.5, label="Basins")
        )
    if reef_ll is not None and not reef_ll.empty:
        legend_handles.append(
            Line2D([], [], color="magenta", linewidth=0.8, label="Reef-like shallow areas")
        )
    if outlets_ll is not None and not outlets_ll.empty:
        legend_handles.append(
            Line2D(
                [], [],
                linestyle="none",
                marker="o",
                markerfacecolor="yellow",
                markeredgecolor="black",
                markersize=6,
                label="Outlets",
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    plt.savefig(QC_PNG_PATH, dpi=300)
    plt.close(fig)

    print("Wrote QC map:", QC_PNG_PATH)
