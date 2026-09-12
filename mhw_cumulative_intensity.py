# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# MEAN MHW CUMULATIVE INTENSITY AND TREND
#
# Region:
#   0-30°N, 60-10°W
#
# Climatological baseline:
#   1981-2010
#
# Analysis:
#   1982-2024
#
# MHW definition:
#   SST > daily P90 for >= 5 consecutive days
#
# Climatology / P90:
#   +/- 5-day climatological window
#   31-day circular smoothing
#
# Cumulative intensity of ONE event:
#
#   sum(SST - daily climatological mean)
#
#   over all days belonging to that MHW event.
#
# OUTPUT:
#
# (g) Mean MHW Cumulative Intensity
#     Units = °C days / event
#
# (h) MHW Cumulative Intensity Trend
#     Units = °C days / decade
#
# IMPORTANT:
#   P90 determines whether a MHW exists.
#
#   SST - climatological mean determines MHW intensity.
#
#   Events crossing Dec-Jan remain one event and are assigned
#   to the year in which the event STARTS.
# ==============================================================


import os
import glob
import gc
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.ticker import (
    LongitudeFormatter,
    LatitudeFormatter
)

warnings.filterwarnings("ignore")


# ==============================================================
# 1. SETTINGS
# ==============================================================

DATA_DIR = r"C:\Users\Aina Ajibola\Desktop\oisst_data"

LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0


# --------------------------------------------------------------
# Climatological baseline
# --------------------------------------------------------------

BASE_START = "1981-01-01"
BASE_END = "2010-12-31"


# --------------------------------------------------------------
# Analysis
# --------------------------------------------------------------

START_YEAR = 1982
END_YEAR = 2024

ANALYSIS_END = "2024-12-31"


# --------------------------------------------------------------
# MHW parameters
# --------------------------------------------------------------

PERCENTILE = 90

HALF_WINDOW = 5

SMOOTH_WINDOW = 31

MIN_DURATION = 5


# --------------------------------------------------------------
# Memory control
# --------------------------------------------------------------

LAT_BLOCK_SIZE = 5


# ==============================================================
# 2. PREPROCESS OISST
# ==============================================================

def preprocess(ds):

    rename = {}

    for old, new in {
        "latitude": "lat",
        "longitude": "lon",
        "Time": "time",
        "TIME": "time"
    }.items():

        if old in ds.coords or old in ds.dims:
            rename[old] = new


    if rename:
        ds = ds.rename(rename)


    # Remove singleton vertical dimensions
    for dim in [
        "zlev",
        "depth",
        "lev",
        "level"
    ]:

        if (
            dim in ds.dims
            and ds.sizes[dim] == 1
        ):

            ds = ds.squeeze(
                dim,
                drop=True
            )


    if "sst" not in ds.data_vars:

        raise KeyError(
            "Variable 'sst' was not found."
        )


    ds = ds[["sst"]]


    # Convert 0-360 longitude to -180...180
    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )


    ds = ds.sortby("lat")
    ds = ds.sortby("lon")


    # Tropical North East Atlantic
    ds = ds.sel(

        lat=slice(
            LAT_MIN,
            LAT_MAX
        ),

        lon=slice(
            LON_MIN,
            LON_MAX
        )

    )


    return ds


# ==============================================================
# 3. CLIMATOLOGICAL DAY
#
# Maps every calendar date to leap year 2000.
# ==============================================================

def get_clim_day(dates):

    dates = pd.DatetimeIndex(
        dates
    )


    reference = pd.to_datetime(
        {
            "year":
                np.full(
                    len(dates),
                    2000
                ),

            "month":
                dates.month,

            "day":
                dates.day
        }
    )


    return (
        pd.DatetimeIndex(reference)
        .dayofyear
        .to_numpy(dtype=np.int16)
    )


# ==============================================================
# 4. CIRCULAR 31-DAY SMOOTHING
# ==============================================================

def circular_smooth_3d(
    values,
    window=31
):

    half = window // 2


    extended = np.concatenate(
        [
            values[-half:, :, :],
            values,
            values[:half, :, :]
        ],
        axis=0
    )


    valid = np.isfinite(
        extended
    )


    filled = np.where(
        valid,
        extended,
        0.0
    )


    csum = np.cumsum(
        filled,
        axis=0,
        dtype=np.float64
    )


    ccount = np.cumsum(
        valid.astype(np.int32),
        axis=0
    )


    csum = np.concatenate(
        [
            np.zeros(
                (
                    1,
                    csum.shape[1],
                    csum.shape[2]
                ),
                dtype=np.float64
            ),

            csum
        ],
        axis=0
    )


    ccount = np.concatenate(
        [
            np.zeros(
                (
                    1,
                    ccount.shape[1],
                    ccount.shape[2]
                ),
                dtype=np.int32
            ),

            ccount
        ],
        axis=0
    )


    smoothed = np.full(
        values.shape,
        np.nan,
        dtype=np.float32
    )


    for d in range(366):

        start = d
        end = d + window


        total = (
            csum[end]
            -
            csum[start]
        )


        count = (
            ccount[end]
            -
            ccount[start]
        )


        np.divide(
            total,
            count,
            out=smoothed[d],
            where=(count > 0)
        )


    return smoothed


# ==============================================================
# 5. ANNUAL MEAN CUMULATIVE INTENSITY
#
# above:
#   True where SST > P90
#
# intensity:
#   SST - climatological mean
#
# For each confirmed >=5-day event:
#
# cumulative intensity =
# sum of daily SST-climatology anomalies
#
# Example:
#
# Daily intensities:
#
# 1.2, 1.4, 1.8, 1.5, 1.1, 0.9 °C
#
# cumulative intensity:
#
# 7.9 °C days
#
# If a year has multiple events:
#
# annual mean cumulative intensity =
# mean(cumulative intensity of its events)
#
# No-event years = NaN
# ==============================================================

def calculate_annual_mean_cumulative_intensity(
    above,
    intensity,
    dates,
    years
):

    n_time, n_lat, n_lon = (
        above.shape
    )


    n_cells = (
        n_lat
        *
        n_lon
    )


    hot = above.reshape(
        n_time,
        n_cells
    )


    intensity_flat = intensity.reshape(
        n_time,
        n_cells
    )


    cumulative_sum = np.zeros(
        (
            len(years),
            n_cells
        ),
        dtype=np.float32
    )


    event_count = np.zeros(
        (
            len(years),
            n_cells
        ),
        dtype=np.int16
    )


    # ----------------------------------------------------------
    # Process each grid cell
    # ----------------------------------------------------------

    for cell in range(
        n_cells
    ):

        series = hot[
            :,
            cell
        ]


        intensity_series = intensity_flat[
            :,
            cell
        ]


        start = None


        for t in range(
            n_time
        ):

            is_hot = bool(
                series[t]
            )


            # --------------------------------------------------
            # Start exceedance run
            # --------------------------------------------------

            if (
                is_hot
                and
                start is None
            ):

                start = t


            # --------------------------------------------------
            # End exceedance run
            # --------------------------------------------------

            if start is not None:

                run_finished = (

                    (not is_hot)

                    or

                    (
                        t
                        ==
                        n_time - 1
                    )

                )


                if run_finished:

                    if (
                        is_hot
                        and
                        t == n_time - 1
                    ):

                        end = t

                    else:

                        end = t - 1


                    duration = (
                        end
                        -
                        start
                        +
                        1
                    )


                    # ------------------------------------------
                    # Confirm MHW
                    # ------------------------------------------

                    if duration >= MIN_DURATION:

                        event_year = int(
                            dates[start].year
                        )


                        if (
                            START_YEAR
                            <=
                            event_year
                            <=
                            END_YEAR
                        ):

                            # ----------------------------------
                            # Cumulative intensity of event
                            # ----------------------------------

                            event_intensity = (
                                intensity_series[
                                    start:end + 1
                                ]
                            )


                            if np.all(
                                np.isfinite(
                                    event_intensity
                                )
                            ):

                                event_cumulative = float(
                                    np.sum(
                                        event_intensity
                                    )
                                )


                                year_index = (
                                    event_year
                                    -
                                    START_YEAR
                                )


                                cumulative_sum[
                                    year_index,
                                    cell
                                ] += event_cumulative


                                event_count[
                                    year_index,
                                    cell
                                ] += 1


                    start = None


    # ----------------------------------------------------------
    # Annual mean cumulative intensity per event
    # ----------------------------------------------------------

    annual_mean = np.full(
        cumulative_sum.shape,
        np.nan,
        dtype=np.float32
    )


    np.divide(

        cumulative_sum,

        event_count,

        out=annual_mean,

        where=(
            event_count > 0
        )

    )


    return (

        annual_mean.reshape(
            len(years),
            n_lat,
            n_lon
        ),

        cumulative_sum.reshape(
            len(years),
            n_lat,
            n_lon
        ),

        event_count.reshape(
            len(years),
            n_lat,
            n_lon
        )

    )


# ==============================================================
# 6. TREND PER DECADE
# ==============================================================

def calculate_trend_per_decade(
    annual_values,
    years
):

    n_years, n_lat, n_lon = (
        annual_values.shape
    )


    trend = np.full(
        (
            n_lat,
            n_lon
        ),
        np.nan,
        dtype=np.float32
    )


    x = np.asarray(
        years,
        dtype=np.float64
    )


    for i in range(
        n_lat
    ):

        for j in range(
            n_lon
        ):

            y = annual_values[
                :,
                i,
                j
            ].astype(
                np.float64
            )


            valid = np.isfinite(
                y
            )


            # Need at least 3 annual observations
            if np.sum(valid) < 3:

                continue


            xv = x[
                valid
            ]


            yv = y[
                valid
            ]


            xv_centered = (
                xv
                -
                np.mean(xv)
            )


            yv_centered = (
                yv
                -
                np.mean(yv)
            )


            denominator = np.sum(
                xv_centered ** 2
            )


            if denominator == 0:

                continue


            slope_per_year = (

                np.sum(
                    xv_centered
                    *
                    yv_centered
                )

                /

                denominator

            )


            trend[
                i,
                j
            ] = (
                slope_per_year
                *
                10.0
            )


    return trend


# ==============================================================
# 7. FIND FILES
# ==============================================================

files = sorted(

    glob.glob(

        os.path.join(
            DATA_DIR,
            "*_oisst.nc"
        )

    )

)


if not files:

    files = sorted(

        glob.glob(

            os.path.join(
                DATA_DIR,
                "*.nc"
            )

        )

    )


if not files:

    raise FileNotFoundError(
        f"No NetCDF files found in:\n"
        f"{DATA_DIR}"
    )


print("=" * 90)

print(
    "TROPICAL NORTH EAST ATLANTIC "
    "MHW CUMULATIVE INTENSITY"
)

print("=" * 90)


print(
    f"\nSST files found: "
    f"{len(files):,}"
)


print(
    f"First file: "
    f"{os.path.basename(files[0])}"
)


print(
    f"Last file: "
    f"{os.path.basename(files[-1])}"
)


# ==============================================================
# 8. OPEN OISST
# ==============================================================

print(
    "\nOpening OISST..."
)


ds = xr.open_mfdataset(

    files,

    combine="by_coords",

    preprocess=preprocess,

    parallel=False,

    data_vars="minimal",

    coords="minimal",

    compat="override",

    join="outer",

    engine="netcdf4"

)


ds = ds.sortby(
    "time"
)


sst = ds[
    "sst"
]


# ==============================================================
# 9. NORMALIZE TIME
# ==============================================================

time_index = pd.DatetimeIndex(
    sst.time.values
).normalize()


sst = sst.assign_coords(
    time=time_index
)


keep = np.where(

    ~time_index.duplicated(
        keep="first"
    )

)[0]


sst = sst.isel(
    time=keep
)


sst = sst.sortby(
    "time"
)


# ==============================================================
# 10. UNIT CHECK
# ==============================================================

sample = float(

    sst.isel(

        time=slice(
            0,
            min(
                10,
                sst.sizes["time"]
            )
        )

    )

    .mean(
        skipna=True
    )

    .compute()

)


if sample > 100:

    print(
        "\nConverting SST from Kelvin to °C..."
    )

    sst = sst - 273.15


else:

    print(
        "\nSST already appears to be °C."
    )


# ==============================================================
# 11. BASELINE
# ==============================================================

baseline = sst.sel(

    time=slice(
        BASE_START,
        BASE_END
    )

)


baseline_dates = pd.DatetimeIndex(
    baseline.time.values
)


if len(baseline_dates) == 0:

    raise ValueError(
        "No baseline observations found."
    )


print(
    f"\nBaseline actually available: "
    f"{baseline_dates[0].date()} to "
    f"{baseline_dates[-1].date()}"
)


if baseline_dates[0] > pd.Timestamp(
    BASE_START
):

    print(
        "\nWARNING:"
    )

    print(
        "The local OISST archive does not contain "
        "the complete 1981 climatological year."
    )


baseline_clim_day = get_clim_day(
    baseline_dates
)


# ==============================================================
# 12. ANALYSIS PERIOD
# ==============================================================

analysis_start = pd.Timestamp(
    sst.time.values[0]
)


analysis = sst.sel(

    time=slice(
        analysis_start,
        ANALYSIS_END
    )

)


analysis_dates = pd.DatetimeIndex(
    analysis.time.values
)


analysis_clim_day = get_clim_day(
    analysis_dates
)


years = np.arange(
    START_YEAR,
    END_YEAR + 1
)


n_years = len(
    years
)


print(
    f"\nAnalysis period: "
    f"{START_YEAR}-{END_YEAR}"
)


print(
    f"Number of analysis years: "
    f"{n_years}"
)


# ==============================================================
# 13. GRID SIZE
# ==============================================================

n_lat = sst.sizes[
    "lat"
]


n_lon = sst.sizes[
    "lon"
]


# ==============================================================
# 14. OUTPUT ARRAYS
# ==============================================================

annual_mean_cumulative = np.full(

    (
        n_years,
        n_lat,
        n_lon
    ),

    np.nan,

    dtype=np.float32

)


annual_cumulative_sum = np.zeros(

    (
        n_years,
        n_lat,
        n_lon
    ),

    dtype=np.float32

)


annual_event_count = np.zeros(

    (
        n_years,
        n_lat,
        n_lon
    ),

    dtype=np.int16

)


# ==============================================================
# 15. BLOCK PROCESSING
# ==============================================================

total_blocks = int(

    np.ceil(
        n_lat
        /
        LAT_BLOCK_SIZE
    )

)


print(
    f"\nProcessing "
    f"{total_blocks} latitude blocks..."
)


for block_number, block_start in enumerate(

    range(
        0,
        n_lat,
        LAT_BLOCK_SIZE
    ),

    start=1

):


    block_end = min(

        block_start
        +
        LAT_BLOCK_SIZE,

        n_lat

    )


    block_lat = (
        block_end
        -
        block_start
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        f"BLOCK "
        f"{block_number}/{total_blocks}"
    )


    print(
        f"Latitude rows: "
        f"{block_start + 1}-{block_end}"
    )


    print(
        "=" * 72
    )


    # ==========================================================
    # 15A. LOAD BASELINE
    # ==========================================================

    print(
        "Loading baseline SST..."
    )


    base_block = np.asarray(

        baseline.isel(

            lat=slice(
                block_start,
                block_end
            )

        ).values,

        dtype=np.float32

    )


    # ==========================================================
    # 15B. CALCULATE DAILY CLIMATOLOGY AND P90
    # ==============================================================

    print(
        "Calculating daily climatology and P90..."
    )


    daily_climatology = np.full(

        (
            366,
            block_lat,
            n_lon
        ),

        np.nan,

        dtype=np.float32

    )


    daily_p90 = np.full(

        (
            366,
            block_lat,
            n_lon
        ),

        np.nan,

        dtype=np.float32

    )


    for day in range(
        1,
        367
    ):

        # ------------------------------------------------------
        # Circular distance between climatological days
        # ------------------------------------------------------

        distance = np.abs(
            baseline_clim_day
            -
            day
        )


        distance = np.minimum(
            distance,
            366
            -
            distance
        )


        selected = (
            distance
            <=
            HALF_WINDOW
        )


        selected_sst = base_block[
            selected,
            :,
            :
        ]


        # ------------------------------------------------------
        # Daily climatological mean
        # ------------------------------------------------------

        daily_climatology[
            day - 1,
            :,
            :
        ] = np.nanmean(

            selected_sst,

            axis=0

        )


        # ------------------------------------------------------
        # Daily P90
        # ------------------------------------------------------

        daily_p90[
            day - 1,
            :,
            :
        ] = np.nanpercentile(

            selected_sst,

            PERCENTILE,

            axis=0

        )


    # ==========================================================
    # 15C. 31-DAY CIRCULAR SMOOTHING
    # ==============================================================

    print(
        "Applying 31-day circular smoothing..."
    )


    daily_climatology = circular_smooth_3d(

        daily_climatology,

        SMOOTH_WINDOW

    )


    daily_p90 = circular_smooth_3d(

        daily_p90,

        SMOOTH_WINDOW

    )


    # ==========================================================
    # 15D. LOAD ANALYSIS SST
    # ==============================================================

    print(
        "Loading analysis SST..."
    )


    analysis_block = np.asarray(

        analysis.isel(

            lat=slice(
                block_start,
                block_end
            )

        ).values,

        dtype=np.float32

    )


    # ==========================================================
    # 15E. MATCH CLIMATOLOGY AND P90 TO ACTUAL DATES
    # ==============================================================

    climatology_block = daily_climatology[
        analysis_clim_day - 1,
        :,
        :
    ]


    threshold_block = daily_p90[
        analysis_clim_day - 1,
        :,
        :
    ]


    # ==========================================================
    # 15F. VALID DATA
    # ==============================================================

    valid = (

        np.isfinite(
            analysis_block
        )

        &

        np.isfinite(
            threshold_block
        )

        &

        np.isfinite(
            climatology_block
        )

    )


    # ==========================================================
    # 15G. DETECT P90 EXCEEDANCE
    # ==============================================================

    above = (

        valid

        &

        (
            analysis_block
            >
            threshold_block
        )

    )


    # ==========================================================
    # 15H. DAILY MHW INTENSITY
    #
    # Intensity = SST - climatological mean
    # ==============================================================

    intensity = (

        analysis_block
        -
        climatology_block

    )


    intensity[
        ~valid
    ] = np.nan


    # ==========================================================
    # 15I. DETECT EVENTS AND CUMULATIVE INTENSITY
    # ==============================================================

    print(
        "Detecting MHW events and cumulative intensities..."
    )


    (
        block_mean_cumulative,
        block_cumulative_sum,
        block_event_count

    ) = calculate_annual_mean_cumulative_intensity(

        above,

        intensity,

        analysis_dates,

        years

    )


    annual_mean_cumulative[
        :,
        block_start:block_end,
        :
    ] = block_mean_cumulative


    annual_cumulative_sum[
        :,
        block_start:block_end,
        :
    ] = block_cumulative_sum


    annual_event_count[
        :,
        block_start:block_end,
        :
    ] = block_event_count


    # ==========================================================
    # 15J. CLEAN MEMORY
    # ==============================================================

    del base_block

    del daily_climatology
    del daily_p90

    del analysis_block

    del climatology_block
    del threshold_block

    del valid
    del above
    del intensity

    del block_mean_cumulative
    del block_cumulative_sum
    del block_event_count

    gc.collect()


    print(
        "Block completed."
    )


# ==============================================================
# 16. OCEAN MASK
# ==============================================================

print(
    "\nCreating ocean mask..."
)


ocean_mask = np.isfinite(

    sst.sel(

        time="2000-01-01",

        method="nearest"

    ).values

)


annual_mean_cumulative[
    :,
    ~ocean_mask
] = np.nan


# ==============================================================
# 17. MEAN MHW CUMULATIVE INTENSITY
#
# Event-weighted mean across entire 1982-2024 period:
#
# sum cumulative intensity of ALL events
# --------------------------------------
# total number of MHW events
#
# Units = °C days/event
# ==============================================================

print(
    "\nCalculating mean MHW cumulative intensity..."
)


total_cumulative_intensity = np.nansum(

    annual_cumulative_sum,

    axis=0

)


total_events = np.sum(

    annual_event_count,

    axis=0

)


mean_cumulative_intensity = np.full(

    (
        n_lat,
        n_lon
    ),

    np.nan,

    dtype=np.float32

)


np.divide(

    total_cumulative_intensity,

    total_events,

    out=mean_cumulative_intensity,

    where=(
        total_events > 0
    )

)


mean_cumulative_intensity[
    ~ocean_mask
] = np.nan


# ==============================================================
# 18. CUMULATIVE INTENSITY TREND
#
# Linear trend of annual mean cumulative intensity.
#
# Years without an MHW are NaN and are not treated as zero.
#
# Units = °C days/decade
# ==============================================================

print(
    "Calculating cumulative intensity trend..."
)


cumulative_intensity_trend = calculate_trend_per_decade(

    annual_mean_cumulative,

    years

)


cumulative_intensity_trend[
    ~ocean_mask
] = np.nan


# ==============================================================
# 19. RESULTS
# ==============================================================

print(
    "\n"
    + "=" * 90
)


print(
    "MHW CUMULATIVE INTENSITY RESULTS"
)


print(
    "=" * 90
)


print(
    f"\nAnalysis period: "
    f"{START_YEAR}-{END_YEAR}"
)


print(
    f"Number of years: "
    f"{n_years}"
)


print(
    f"\nSpatial mean cumulative intensity: "
    f"{np.nanmean(mean_cumulative_intensity):.1f} °C days/event"
)


print(
    f"Minimum mean cumulative intensity: "
    f"{np.nanmin(mean_cumulative_intensity):.1f} °C days/event"
)


print(
    f"Maximum mean cumulative intensity: "
    f"{np.nanmax(mean_cumulative_intensity):.1f} °C days/event"
)


print(
    f"\nSpatial mean cumulative intensity trend: "
    f"{np.nanmean(cumulative_intensity_trend):+.1f} °C days/decade"
)


print(
    f"Minimum cumulative intensity trend: "
    f"{np.nanmin(cumulative_intensity_trend):+.1f} °C days/decade"
)


print(
    f"Maximum cumulative intensity trend: "
    f"{np.nanmax(cumulative_intensity_trend):+.1f} °C days/decade"
)


# ==============================================================
# 20. MAP COORDINATES
# ==============================================================

lon_values = sst.lon.values

lat_values = sst.lat.values


LON, LAT = np.meshgrid(

    lon_values,

    lat_values

)


projection = ccrs.PlateCarree()


# ==============================================================
# 21. COLORMAP
#
# Same blue -> yellow -> red style
# ==============================================================

cmap = plt.get_cmap(
    "turbo"
)


# ==============================================================
# 22. MEAN CUMULATIVE INTENSITY SCALE
# ==============================================================

cum_min = 0.0


cum_max = float(

    np.nanpercentile(

        mean_cumulative_intensity,

        99

    )

)


cum_max = np.ceil(
    cum_max
)


if cum_max <= 0:

    cum_max = 1.0


cum_levels = np.linspace(

    cum_min,

    cum_max,

    16

)


cum_ticks = np.linspace(

    cum_min,

    cum_max,

    7

)


# ==============================================================
# 23. TREND SCALE
#
# Symmetric around zero
# ==============================================================

trend_abs = float(

    np.nanpercentile(

        np.abs(
            cumulative_intensity_trend
        ),

        99

    )

)


trend_abs = np.ceil(
    trend_abs
)


if trend_abs <= 0:

    trend_abs = 1.0


trend_levels = np.linspace(

    -trend_abs,

    trend_abs,

    17

)


trend_ticks = np.linspace(

    -trend_abs,

    trend_abs,

    7

)


trend_norm = mcolors.TwoSlopeNorm(

    vmin=-trend_abs,

    vcenter=0.0,

    vmax=trend_abs

)


# ==============================================================
# 24. CREATE FIGURE
# ==============================================================

fig, axes = plt.subplots(

    1,
    2,

    figsize=(
        16,
        6
    ),

    subplot_kw={
        "projection":
            projection
    }

)


# ==============================================================
# 25. PANEL G
# ==============================================================

cf1 = axes[0].contourf(

    LON,

    LAT,

    mean_cumulative_intensity,

    levels=cum_levels,

    cmap=cmap,

    extend="max",

    transform=projection

)


# ==============================================================
# 26. PANEL H
# ==============================================================

cf2 = axes[1].contourf(

    LON,

    LAT,

    cumulative_intensity_trend,

    levels=trend_levels,

    cmap=cmap,

    norm=trend_norm,

    extend="both",

    transform=projection

)


# ==============================================================
# 27. MAP FORMATTING
# ==============================================================

longitude_ticks = [
    -60,
    -50,
    -40,
    -30,
    -20,
    -10
]


latitude_ticks = [
    0,
    5,
    10,
    15,
    20,
    25,
    30
]


for ax in axes:

    ax.set_extent(

        [
            LON_MIN,
            LON_MAX,
            LAT_MIN,
            LAT_MAX
        ],

        crs=projection

    )


    ax.add_feature(

        cfeature.LAND.with_scale(
            "50m"
        ),

        facecolor="0.70",

        edgecolor="black",

        linewidth=0.45,

        zorder=10

    )


    ax.coastlines(

        resolution="50m",

        linewidth=0.7,

        zorder=11

    )


    ax.add_feature(

        cfeature.BORDERS.with_scale(
            "50m"
        ),

        linewidth=0.3,

        edgecolor="black",

        zorder=11

    )


    ax.set_xticks(
        longitude_ticks,
        crs=projection
    )


    ax.set_yticks(
        latitude_ticks,
        crs=projection
    )


    ax.xaxis.set_major_formatter(

        LongitudeFormatter(
            degree_symbol="°"
        )

    )


    ax.yaxis.set_major_formatter(

        LatitudeFormatter(
            degree_symbol="°"
        )

    )


    ax.tick_params(
        labelsize=9
    )


    ax.gridlines(

        crs=projection,

        draw_labels=False,

        xlocs=longitude_ticks,

        ylocs=latitude_ticks,

        linewidth=0.3,

        linestyle="--",

        alpha=0.35

    )


# ==============================================================
# 28. PANEL TITLES - G AND H
# ==============================================================

axes[0].set_title(

    "(g) Mean MHW Cumulative Intensity",

    fontsize=13,

    fontweight="bold",

    pad=10

)


axes[1].set_title(

    "(h) MHW Cumulative Intensity Trend",

    fontsize=13,

    fontweight="bold",

    pad=10

)


# ==============================================================
# 29. MAIN TITLE
# ==============================================================

fig.suptitle(

    "Tropical North East Atlantic Marine Heatwave "
    "Cumulative Intensity and Trend (1982–2024)",

    fontsize=16,

    fontweight="bold",

    y=0.97

)


# ==============================================================
# 30. SPACING
# ==============================================================

fig.subplots_adjust(

    left=0.05,

    right=0.91,

    bottom=0.10,

    top=0.86,

    wspace=0.25

)


# ==============================================================
# 31. PANEL G COLORBAR
# ==============================================================

pos1 = axes[0].get_position()


cax1 = fig.add_axes(

    [
        pos1.x1 + 0.008,
        pos1.y0,
        0.013,
        pos1.height
    ]

)


cbar1 = fig.colorbar(

    cf1,

    cax=cax1,

    orientation="vertical",

    ticks=cum_ticks

)


cbar1.set_label(

    "°C·Days/Event",

    fontsize=10,

    fontweight="bold"

)


cbar1.ax.yaxis.set_major_formatter(

    mticker.FormatStrFormatter(
        "%.1f"
    )

)


# ==============================================================
# 32. PANEL H COLORBAR
# ==============================================================

pos2 = axes[1].get_position()


cax2 = fig.add_axes(

    [
        pos2.x1 + 0.008,
        pos2.y0,
        0.013,
        pos2.height
    ]

)


cbar2 = fig.colorbar(

    cf2,

    cax=cax2,

    orientation="vertical",

    ticks=trend_ticks

)


cbar2.set_label(

    "°C·Days/Decade",

    fontsize=10,

    fontweight="bold"

)


cbar2.ax.yaxis.set_major_formatter(

    mticker.FormatStrFormatter(
        "%.1f"
    )

)


# ==============================================================
# 33. SHOW
# ==============================================================

plt.show()


# ==============================================================
# 34. CLOSE
# ==============================================================

ds.close()


print(
    "\nMHW cumulative intensity analysis "
    "completed successfully."
)