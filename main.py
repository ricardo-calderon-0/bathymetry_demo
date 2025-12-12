# main.py
from processing import (
    build_topobathy,
    process_basins_and_outlets,
    compute_slope_and_zonal_stats,
    detect_reef_features,
    export_data,
    write_metadata,
)
from plotting import plot_qc_map


def main() -> None:
    # 1. Topobathy
    bathy, topobathy_clean = build_topobathy()

    # 2. Basins and outlets
    basins_clip, outlets, water_union_gdf = process_basins_and_outlets(
        bathy, topobathy_clean
    )

    # 3. Slope and zonal stats
    slope, basins_with_slope = compute_slope_and_zonal_stats(
        bathy, topobathy_clean, basins_clip
    )

    # 4. Reef detection
    reef_gdf, basins_final, reef_mask, depth = detect_reef_features(
        topobathy_clean, slope, water_union_gdf, basins_with_slope
    )

    # 5. Exports
    export_data(reef_gdf, basins_final)

    # 6. Plot QC map
    plot_qc_map(bathy, topobathy_clean, basins_final, reef_gdf, outlets)

    # 7. Metadata summary
    write_metadata(
        basins_final,
        outlets,
        reef_gdf,
        reef_mask,
        depth,
    )


if __name__ == "__main__":
    main()
