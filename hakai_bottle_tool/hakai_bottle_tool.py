import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hakai_api import Client
from loguru import logger

from hakai_bottle_tool import endpoints

load_dotenv()

client = Client(credentials=os.getenv("HAKAI_API_TOKEN"))
CONFIG_DIR = Path(__file__).parent / "config"

# List of variables to ignore from the sample data
IGNORED_VARIABLES = [
    "action",
    "rn",
    "sampling_bout",
    "analyzed",
    "preserved",
    "technician",
    "lab_technician",
    "source",
    "volume",
    "dna_volume_te_wash",
    "before_acid",
    "after_acid",
    "calibration_slope",
    "acid_ratio_correction_factor",
    "acid_coefficient",
    "acetone_volumne_ml",
    "fluorometer_serial_no",
    "calibration",
    "analyzing_lab",
]

# Variable list to ignore from the CTD data
CTD_VARIABLES = [
    "hakai_id",
    "device_model",
    "device_sn",
    "work_area",
    "cruise",
    "station",
    "station_longitude",
    "station_latitude",
    "distance_from_station",
    "latitude",
    "longitude",
    "start_dt",
    "bottom_dt",
    "end_dt",
    "direction_flag",
    "measurement_dt",
    "conductivity",
    "conductivity_flag",
    "temperature",
    "temperature_flag",
    "depth",
    "depth_flag",
    "pressure",
    "pressure_flag",
    "par",
    "par_flag",
    "flc",
    "flc_flag",
    "turbidity",
    "turbidity_flag",
    "ph",
    "ph_flag",
    "salinity",
    "salinity_flag",
    "spec_cond",
    "spec_cond_flag",
    "dissolved_oxygen_ml_l",
    "dissolved_oxygen_ml_l_flag",
    "rinko_do_ml_l",
    "rinko_do_ml_l_flag",
    "dissolved_oxygen_percent",
    "dissolved_oxygen_percent_flag",
    "c_star_at",
    "c_star_at_flag",
    "sos_un",
    "sos_un_flag",
    "backscatter_beta",
    "backscatter_beta_flag",
    "cdom_ppb",
    "cdom_ppb_flag",
]

INDEXED_VARIABLES = [
    "organization",
    "work_area",
    "survey",
    "site_id",
    "event_pk",
    "line_out_depth",
    "collected",
]
AGG_FUNCS = {
    (float, int): [np.ptp, "mean", "std", "count"],
    (str, object): ["count", ",".join],
}

AGG_NAME_MAPPING = {
    "count": ["nReplicates"],
    "ptp": ["range"],
    "<lambda_0>": [],
    "mean": [],
    "std": ["std"],
}


def rename_sample_columns(endpoint, columns):
    """Rename columns based on endpoint and predefined aggregate name mapping

    Rename stats variable based on predefined mapping, reverse other of
    tuples, join by underscore and replace white space by underscore
    """
    return [
        "_".join(
            [endpoint.rsplit("/", 1)[1]]
            + [col[0]]
            + AGG_NAME_MAPPING[col[1]]
            + ([col[2]] if len(col) == 3 else [])
        ).replace(" ", "_")
        for col in columns
    ]


def get_aggregate_function(df, ignore):
    agg = {}
    for var in df.columns:
        if var in ignore:
            continue

        if df[var].dtype in (object, str, bool):
            agg[var] = ["count", lambda x: ",".join([str(y) for y in x])]
        else:
            agg[var] = [np.ptp, "mean", "std", "count"]
    return agg


def rename_agg_column(columns, prefix, ignore):
    new_columns = []
    for column in columns:
        if column in ignore:
            new_columns += [column]
            continue
        new_columns += [
            prefix
            + "_"
            + "_".join(
                [
                    item
                    for item in map(lambda x: AGG_NAME_MAPPING(x, x), columns)
                    if item
                ]
            )
        ]

    return new_columns


def join_sample_data(
    station, time_min=None, time_max=None, bottle_matching_timedelta="1hour"
):
    """join_sample_data

    Args:
        station (str): Station for which to download the sample data
        time_min (str, optional):  Minimal Time for which to download the sample data. Defaults to None.
        time_max (str, optional):  Maximal Time for which to download the sample data. Defaults to None.
        bottle_matching_timedelta (str, optional): Time delta to match bottle data. Defaults to "1hour".

    Returns:
        dataframe: joined sample data
    """

    df = pd.DataFrame()
    filter_url = f"limit=-1&site_id={station}"
    if time_min:
        filter_url += f"&collected>={time_min}"
    if time_max:
        filter_url += f"&collected<={time_max}"

    for endpoint, attrs in endpoints.SAMPLES.items():
        url = f"{client.api_root}/{endpoint}?{filter_url}"
        # query_filter input if exist
        if "query_filter" in attrs:
            url += attrs["query_filter"]

        logger.info(f"Download data from: {url}")
        response = client.get(url)
        if response.status_code != 200:
            response.raise_for_status()
        df_endpoint = pd.DataFrame(response.json()).astype(
            {"pressure_transducer_depth": float}
        )

        if df_endpoint.empty:
            logger.warning("Failed to retrieve any data")
            continue

        logger.info(f"Downloaded: {len(df_endpoint)} records")

        # Drop variables we don't want
        df_endpoint = df_endpoint.drop(columns=IGNORED_VARIABLES, errors="ignore")

        # Rename
        if "map_values" in attrs:
            df_endpoint = df_endpoint.replace(attrs["map_values"])

        # Regroup data and apply pivot
        index_list = list(set(INDEXED_VARIABLES) & set(df_endpoint.columns))
        df_endpoint = df_endpoint.pivot_table(
            index=index_list,
            columns=attrs.get("pivot"),
            aggfunc=get_aggregate_function(df_endpoint, index_list),
        )

        # Transform resulting dataframe
        df_endpoint.columns = rename_sample_columns(endpoint, df_endpoint.columns)
        df_endpoint = df_endpoint.reset_index().sort_values("collected")
        df_endpoint["collected"] = pd.to_datetime(df_endpoint["collected"])

        # Merge data to previously downloaded one
        if df.empty:
            df = df_endpoint
            continue

        # Add a column to review which new samples were matched
        df_endpoint = df_endpoint.reset_index().rename(
            columns={"index": "endpoint_index"}
        )
        df = pd.merge_asof(
            df.sort_values("collected"),
            df_endpoint.sort_values("collected"),
            by=index_list,
            on="collected",
            tolerance=pd.Timedelta(bottle_matching_timedelta),
            allow_exact_matches=True,
        )
        # Include back samples that were not matched
        unmatched_samples = df_endpoint.loc[
            ~df_endpoint["endpoint_index"].isin(df["endpoint_index"])
        ]
        df = df.drop(columns=["endpoint_index"])
        unmatched_samples = unmatched_samples.drop(columns=["endpoint_index"])
        logger.info(
            f"{len(unmatched_samples)} samples were not matched to the previous ones. Add them on their own"
        )
        df = pd.concat([df, unmatched_samples], ignore_index=True)

    # Define bottle_depth, find list of pressure_transducer_depth columns and averaged
    pressure_transducer_depth_columns = df.filter(
        like="pressure_transducer_depth"
    ).columns
    df["pressure_transducer_depth"] = df.filter(
        regex="pressure_transducer_depth$"
    ).mean(axis="columns")
    df["bottle_depth"] = df["pressure_transducer_depth"].fillna(df["line_out_depth"])
    df = df.drop(columns=pressure_transducer_depth_columns)

    df = df.replace({"None": None})
    return df


def join_ctd_data(
    df_bottle,
    station,
    time_min=None,
    time_max=None,
    bin_size=1,
    bottle_depth_variable="bottle_depth",
):
    """join_ctd_data
    Matching Algorithm use to match CTD Profile to bottle data. The algorithm
    always match data for the same station id on both side (Bottle and CTD).
    Then then matching will be done by in the following other:
        1. Bottle data will be matched to the Closest CTD profile in time
            (before or after) and matched to an exact
            CTD profile depth bin if available.
            If no exact depth bin is available.
            This bottle will be ignored from this step.
        2. Unmatched bottles will then be matched to the closest
            profile and closest depth
        bin as long as the difference between the bottle and the
            matched CTD depth bin is within the tolerance.
        3. Unmatched bottles will then be matched to the closest
            CTD depth bin available within
        the considered time range as long as the difference in
            depth between the two remains within the tolerance.
        4. Bottle data will be not matched to any CTD data.

    Args:
        df_bottle (dataframe): Joined bottle sample data
        station (str): station used to merge data
        time_min (str, optional): Minimum time range to look for.
            Defaults to None.
        time_max (str, optional): Maximum time range to look for.
            Defaults to None.
        bin_size (int, optional): Size of the vertical bins
            to which bottle should be merged. Defaults to 1m.
    """

    def _within_depth_tolerance(bottle_depth, ctd_depth):
        depth_tolerance_range = 3
        depth_tolerance_ratio = 0.15
        dD = np.abs(ctd_depth - bottle_depth)
        return (dD < depth_tolerance_range) | (
            np.abs(np.divide(ctd_depth, bottle_depth) - 1) < depth_tolerance_ratio
        )

    # Get CTD Data
    logger.info("Download CTD data")
    filter_url = (
        "limit=-1&pressure!=null&salinity!=-9.99e-29&"
        f"station={station}&direction_flag=d"
    )
    if time_min:
        filter_url += f"&measurement_dt>{time_min}"
    if time_max:
        filter_url += f"&measurement_dt<{time_max}"
    filter_url += f"&fields={','.join(CTD_VARIABLES)}"
    url = f"{client.api_root}/{endpoints.CTD_DATA}?{filter_url}"
    logger.info(url)
    response = client.get(url)
    df_ctd = pd.DataFrame(response.json())
    if df_ctd.empty:
        return df_bottle

    # Add ctd suffix
    df_ctd = df_ctd.add_prefix("ctd_")

    # Generate matching depth and time variables
    df_bottle["matching_depth"] = (
        df_bottle[bottle_depth_variable].div(bin_size).round().astype("int64")
    )
    df_ctd["matching_depth"] = df_ctd["ctd_depth"].div(bin_size).round().astype("int64")
    df_bottle["matching_time"] = df_bottle["collected"]
    df_ctd["matching_time"] = pd.to_datetime(df_ctd["ctd_measurement_dt"], utc=True)

    # N Bottles samples
    n_bottles = len(df_bottle)

    # Find closest profile with the exact same depth
    logger.info(f"Match {n_bottles} bottles to CTD Profile data: ")
    df_bottles_closest_time_depth = pd.merge_asof(
        df_bottle.sort_values("matching_time"),
        df_ctd.sort_values(["matching_time"]),
        on="matching_time",
        left_by=["work_area", "site_id", "matching_depth"],
        right_by=["ctd_work_area", "ctd_station", "matching_depth"],
        tolerance=pd.Timedelta("4h"),
        allow_exact_matches=True,
        direction="nearest",
    )

    # Retrieve bottle data with matching depths and remove those
    # not matching from df_bottles
    in_tolerance = _within_depth_tolerance(
        df_bottles_closest_time_depth["ctd_depth"],
        df_bottles_closest_time_depth[bottle_depth_variable],
    )
    df_not_matched = df_bottles_closest_time_depth.loc[~in_tolerance, df_bottle.columns]
    df_bottles_matched = df_bottles_closest_time_depth[in_tolerance]
    logger.info(f"{len(df_bottles_matched)} were matched to exact ctd pressure bin.")
    n_bottles -= len(df_bottles_matched)

    # First try to retrieve to the closest profile in time
    #  and then closest depth
    if len(df_not_matched) > 0:
        df_bottles_time = pd.merge_asof(
            df_not_matched.sort_values("matching_time"),
            df_ctd.sort_values(["matching_time"])[
                ["ctd_station", "ctd_hakai_id", "matching_time"]
            ],
            on="matching_time",
            left_by="site_id",
            right_by="ctd_station",
            tolerance=pd.Timedelta("4h"),
            allow_exact_matches=True,
            direction="nearest",
        )

        df_bottles_time = pd.merge_asof(
            df_bottles_time.sort_values("matching_depth").drop(
                columns=["matching_time"]
            ),
            df_ctd.sort_values(["matching_depth"]),
            on="matching_depth",
            by=["ctd_station", "ctd_hakai_id"],
            allow_exact_matches=True,
            direction="nearest",
        )
        # Verify if matched data is within tolerance
        in_tolerance = _within_depth_tolerance(
            df_bottles_time["ctd_depth"],
            df_bottles_time[bottle_depth_variable],
        )
        df_not_matched = df_bottles_time.loc[~in_tolerance][df_bottle.columns]
        df_bottles_matched = pd.concat(
            [df_bottles_matched, df_bottles_time[in_tolerance]]
        )
    logger.info(
        f"{len(df_bottles_time[in_tolerance])} were matched to nearest ctd profile."
    )
    n_bottles = n_bottles - len(df_bottles_matched)

    # Then try to match whatever closest depth sample depth within
    # the allowed time range
    dt = pd.Timedelta("4h")
    df_bottles_depth = pd.DataFrame()
    drop_unmatched = []
    if len(df_not_matched) > 0:
        # Loop through each collected times find any data collected
        # nearby by time and try to match nearest depth
        for collected, drop in df_not_matched.drop(columns=["matching_time"]).groupby(
            "collected"
        ):
            # Filter CTD data for that time period
            df_ctd_filtered = df_ctd.loc[
                (df_ctd["matching_time"] > (collected - dt))
                & (df_ctd["matching_time"] < (collected + dt))
            ]

            # No profile is available around that bottle sample
            if df_ctd_filtered.empty:
                drop_unmatched += [drop]
                continue

            # Match the bottles drop to the closest data within this time range
            df_bottles_depth = pd.concat(
                [
                    df_bottles_depth,
                    pd.merge_asof(
                        drop,
                        df_ctd_filtered.sort_values(["matching_depth"]),
                        on="matching_depth",
                        left_by="site_id",
                        right_by="ctd_station",
                        allow_exact_matches=True,
                        direction="nearest",
                    ),
                ]
            )

        if len(df_bottles_depth) > 0:
            in_tolerance = _within_depth_tolerance(
                df_bottles_depth["ctd_depth"],
                df_bottles_depth[bottle_depth_variable],
            )
            df_bottles_matched = pd.concat(
                [df_bottles_matched, df_bottles_depth[in_tolerance]]
            )

            df_not_matched = df_bottles_depth[~in_tolerance][df_bottle.columns]
            logger.info(
                f"{len(df_bottles_depth[in_tolerance])} were matched to the closest in depth"
                f" ctd profile collected within ±{dt} period of the sample collection time."
            )
            n_bottles -= len(df_bottles_matched)
        else:
            df_not_matched = pd.DataFrame()

        df_not_matched = pd.concat([df_not_matched] + drop_unmatched)
        no_matched_collected_times = (
            df_not_matched["collected"]
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            .drop_duplicates()
            .tolist()
        )
        logger.info(f"{len(df_not_matched)} failed to be matched to any profiles.")
        logger.info(f"Failed Collection Time list: {no_matched_collected_times}")

    # Finally, keep unmatched bottle data with no possible match
    if len(df_not_matched) > 0:
        df_bottles_matched = pd.concat([df_bottles_matched, df_not_matched])

    # Sort data
    df_bottles_matched = df_bottles_matched.sort_values(
        ["site_id", "collected", "line_out_depth"]
    )
    return df_bottles_matched


def create_aggregated_meta_variables(df, bottle_depth_variable):
    # Split lat and gather lat
    df = df.assign(
        time=df["collected"],
        depth=df[bottle_depth_variable],
        depth_difference=df[bottle_depth_variable] - df["ctd_depth"],
        latitude=df.filter(regex="_lat$").median(axis=1),
        longitude=df.filter(regex="_long$").median(axis=1),
        preciseLat=df.filter(regex="_gather_lat$").median(axis=1),
        preciseLon=df.filter(regex="_gather_long$").median(axis=1),
    )

    # Remove columns that have been aggregated we assume that all have the same values
    _drop_regex = (
        r"_survey$|_lat$|_long$|_gather_lat$|_gather_long$|_pressure_transducer_depth$"
    )
    df = df[df.columns.drop(list(df.filter(regex=_drop_regex)))]
    return df


def get_bottle_data(
    station, time_min=None, time_max=None, bottle_depth_variable="bottle_depth"
):
    """get_bottle_data

    Args:
        station (str): [description]
        time_min (str, optional): Minimum time range (ex: '2020-01-01').
            Defaults to None.
        time_max (str, optional): Maximum time range (ex: '2021-01-01').
            Defaults to None.
        match_bottle_depth_with (str, optional): Depth parameter to match
          bottle data with CTD data.
            - 'bottle_depth' (default): Mixed of pressure_transducer_depth
                (if available) and line_out_depth.
            - 'line_out_depth': Line out depth.

    Returns:
        dataframe: Bottle data with sample and ctd dataset.
    """
    if bottle_depth_variable not in [
        "bottle_depth",
        "line_out_depth",
    ]:
        raise ValueError(
            "match_bottle_depth_with can only be 'bottle_depth' or "
            "'line_out_depth'."
            f" Got {bottle_depth_variable}"
        )
    # Samples matched by bottles
    df = join_sample_data(station, time_min, time_max)
    # Matched to ctd data
    df = join_ctd_data(
        df, station, time_min, time_max, bottle_depth_variable=bottle_depth_variable
    )
    df = df.dropna(axis="columns", how="all")
    df = create_aggregated_meta_variables(df, bottle_depth_variable)
    return df


def filter_bottle_variables(df, filter_variables):
    """filter_bottle_variables

    Args:
        df (dataframe): bottle data dataframe
        filter_variables (str): "Reduced" or "Complete" which make reference
            to the predefine list of variables present within the package.

    Returns:
        [dataframe]: [description]
    """
    # Filter columns
    if filter_variables == "Reduced":
        variable_list = (
            (CONFIG_DIR / "Reduced_variable_list.csv").read_text().split("\n")
        )
    elif filter_variables == "Complete":
        variable_list = (
            (CONFIG_DIR / "Complete_variable_list.csv").read_text().split("\n")
        )
    else:
        raise RuntimeError(
            f"Can't recognize filter_variable {filter_variables}. Can only be Reduced or Complete"
        )

    return df.filter(items=variable_list)


def export_to_netcdf(df, output_path=None):
    """AI is creating summary for save_bottle_to

    Args:
        df (dataframe): Bottle matched data
        station (str): Station used within bottle data
        output_path (str, optional): Path where to save file (Default: "")
        output_format (str, optional): Format to save the bottle data to ("csv" or "netcdf" [default]).

    Returns:
       str : path to saved file
    """
    # Default output_path to local directory
    if output_path is None:
        output_path = "."
    output_path = Path(output_path)
    # Convert datetime variables to timezone unaware datetime64 format in UTC
    for var, var_type in df.dtypes.to_dict().items():
        if "datetime" in f"{var_type}":
            df[var] = df[var].dt.tz_convert("UTC").dt.tz_localize(None)

    ds = df.to_xarray()
    attributes = json.loads((CONFIG_DIR / "bottle_netcdf_attributes.json").read_text())

    # Add Global Attributes
    ds.attrs = attributes["NC_GLOBAL"]

    # Add Variable Attributes
    for var in ds:
        if var in attributes:
            ds[var].attrs = attributes[var]

    # Loop through each groups and save grouped files
    for work_area, ds_work_area in ds.groupby("work_area"):
        for site_id, ds_site in ds_work_area.groupby("site_id"):
            logger.info(
                f"Save bottle data by work_area={work_area}, station={site_id} and by date"
            )
            for collected, ds_collected in ds_site.groupby("collected.date"):
                # Format name
                subdir = output_path / work_area / site_id
                filename = f"Hakai_Bottle_{work_area}_{site_id}_{collected}.nc".replace(
                    ":", ""
                )
                filename = re.sub(r"\:|\-|\.0+", "", filename)
                # Generate subfolder if it doesnt' exist yet
                subdir.mkdir(parents=True, exist_ok=True)

                # Save NetCDF
                new_file = subdir / filename
                logger.info(f"Save: {new_file}")
                ds_collected.to_netcdf(new_file)
