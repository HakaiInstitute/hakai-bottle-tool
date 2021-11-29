import pandas as pd
import numpy as np

import json
import re
import warnings

from hakai_api import Client

import os

client = Client()
module_path = os.path.dirname(os.path.abspath(__file__))

# Define each sample type endpoint and the need transformations needed per endpoint
bottle_sample_endpoints = {
    "eims/views/output/nutrients": {
        'query_filter': "&(no2_no3_flag!=SVD|no2_no3_flag=null)&(po4_flag!=SVD|po4_flag=null)&(sio2_flag!=SVD|sio2_flag=null)"
    },
    "eims/views/output/microbial": {"pivot": "microbial_sample_type"},
    "eims/views/output/hplc": {},
    "eims/views/output/poms": {
        "map_values": {"acidified": {True: "Acidified", False: "nonAcidified"}},
        "pivot": "acidified",
    },
    "eims/views/output/ysi": {},
    "eims/views/output/chlorophyll": {
        "query_filter": "&(chla_flag!=SVD|chla_flag=null)&(phaeo_flag!=SVD|phaeo_flag=null)",
        "map_values": {"filter_type": {"GF/F": "GF_F", "Bulk GF/F": "Bulk_GF_F"}},
        "pivot": "filter_type",
    },
}

ctd_endpoint = "ctd/views/file/cast/data"

# List of variables to ignore from the sample data
ignored_variable_list = [
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
ctd_variable_to_ignore = [
    "device_firmware",
    "file_processing_stage",
    "shutoff",
    "ctd_file_pk",
    "ctd_data_pk",
    "v_main",
    "v_lith",
    "i_pump",
    "i_ext01",
    "i_ext2345",
    "cast_processing_stage",
    "status",
    "duration",
    "start_depth",
    "bottom_depth",
    "target_depth",
    "drop_speed",
    "vessel",
    "operators",
    "pruth_air_pressure_before_cast",
    "min_pressure_before_cast",
    "min_depth_before_cast",
    "min_pressure_after_cast",
    "min_depth_after_cast",
    "estimated_air_pressure",
    "estimated_depth_shift",
    "original_start_dt",
    "original_bottom_dt",
    "original_start_depth",
    "original_bottom_depth",
    "spec_cond",
    "spec_cond_flag",
    "oxygen_voltage",
    "oxygen_voltage_flag",
    "cast_number",
]

index_default_list = [
    "work_area",
    "site_id",
    "event_pk",
    "collected",
    "line_out_depth",
    "pressure_transducer_depth",
]
agg_funcs = [np.ptp, "mean", "std", "count", ",".join]

agg_rename = {
    "count": "nReplicates",
    "ptp": "range",
    "join": "",
    "mean": "",
    "std" : "std"
    }


def join_sample_data(station, time_min=None, time_max=None):
    """join_sample_data

    Args:
        station (str): Station for which to download the sample data
        time_min (str, optional):  Minimal Time for which to download the sample data. Defaults to None.
        time_max (str, optional):  Maximal Time for which to download the sample data. Defaults to None.

    Returns:
        dataframe: joined sample data
    """

    df = pd.DataFrame()
    filter_url = f"limit=-1&site_id={station}"
    if time_min:
        filter_url += f"&collected>={time_min}"
    if time_max:
        filter_url += f"&collected<={time_max}"


    for endpoint, attrs in bottle_sample_endpoints.items():
        url = f"{client.api_root}/{endpoint}?{filter_url}"
        # query_filter input if exist 
        if 'query_filter' in attrs:
            url += attrs['query_filter']

        print(f"Download data from: {url}")
        response = client.get(url)

        # If failed to connect raise status
        if response.status_code != 200:
            response.raise_for_status()

        df_endpoint = pd.DataFrame(response.json())

        if df_endpoint.empty:
            warnings.warn("Failed to retrieve any data", UserWarning)
            continue
        else:
            print(f"Downloaded: {len(df_endpoint)} records")

        # Drop variables we don't want
        df_endpoint = df_endpoint.drop(columns=ignored_variable_list, errors="ignore")
        index_list = [
            item if item in df_endpoint else print(f"Missing : {item}")
            for item in index_default_list
        ]
        # Rename
        if "map_values" in attrs:
            for var_map, mapping in attrs["map_values"].items():
                df_endpoint[var_map] = df_endpoint[var_map].replace(mapping)

        # Regroup data and apply pivot
        if "pivot" not in attrs:
            attrs["pivot"] = None
        df_endpoint = df_endpoint.pivot_table(
            index=index_list,
            columns=attrs["pivot"],
            aggfunc=agg_funcs,
        )

        # Rename columns
        # Rename stats variable based on predefined mapping, reverse other of tuples, join by underscore and replace white space by underscore
        prefix = endpoint.rsplit("/", 1)[1]
        new_columns = []
        for col in df_endpoint.columns:
            stats = agg_rename[col[0]]
            name = list(col[::-1])[:-1]
            if stats == "":
                new_var = prefix + '_' + '_'.join(name)
            else:
                new_var = prefix + '_' + '_'.join(name + [stats])

            new_columns += [new_var.replace(' ','_')]
        df_endpoint.columns = new_columns

        # Convert collected to datetime
        # FIXME Hakai EIMS return time data in America/Vancouver timezone eventhough the format output has the 'Z' letter following which suggests it is in UTC.
        #   Because of that, we have to read and drop the timezone and apply the correct one and finally convert to UTC
        df_endpoint = df_endpoint.reset_index().sort_values("collected")
        df_endpoint["collected"] = (
            pd.to_datetime(df_endpoint["collected"])
            .dt.tz_localize(None)
            .dt.tz_localize("America/Vancouver")
            .dt.tz_convert("UTC")
        )

        # Merge data to previously downloaded one
        if df.empty:
            df = df_endpoint
        else:
            df = pd.merge_asof(
                df,
                df_endpoint,
                by=index_list,
                on="collected",
                tolerance=pd.Timedelta("5minutes"),
                allow_exact_matches=True,
            )

    # Define bottle_depth
    df["bottle_depth"] = df["pressure_transducer_depth"].fillna(df["line_out_depth"])
    return df


def join_ctd_data(df_bottle, station, time_min=None, time_max=None, bin_size=1):
    """join_ctd_data
    Matching Algorithm use to match CTD Profile to bottle data. The algorithm always match data for the same
    station id on both side (Bottle and CTD). Then then matching will be done by in the following other:
        1. Bottle data will be matched to the Closest CTD profile in time (before or after) and matched to an exact
        CTD profile depth bin if available.
        If no exact depth bin is available. This bottle will be ignored from this step.
        2. Unmatched bottles will then be matched to the closest profile and closest depth
        bin as long as the difference between the bottle and the matched CTD depth bin is within the tolerance.
        3. Unmatched bottles will then be matched to the closest CTD depth bin available within
        the considered time range as long as the difference in depth between the two remains within the tolerance.
        4. Bottle data will be not matched to any CTD data.

    Args:
        df_bottle (dataframe): Joined bottle sample data
        station (str): station used to merge data
        time_min (str, optional): Minimum time range to look for. Defaults to None.
        time_max (str, optional): Maximum time range to look for. Defaults to None.
        bin_size (int, optional): Size of the vertical bins to which bottle should be merged.. Defaults to 1.
    """

    def _within_depth_tolerance(bottle_depth, ctd_depth):
        depth_tolerance_range = 3
        depth_tolerance_ratio = 0.15
        dD = np.abs(ctd_depth - bottle_depth)
        return (dD < depth_tolerance_range) | (
            np.abs(np.divide(ctd_depth, bottle_depth) - 1) < depth_tolerance_ratio
        )

    # Get CTD Data
    print("Download CTD data")
    filter_url = f"limit=-1&pressure!=null&station={station}"
    if time_min:
        filter_url += f"&measurement_dt>{time_min}"
    if time_max:
        filter_url += f"&measurement_dt<{time_max}"
    url = f"{client.api_root}/{ctd_endpoint}?{filter_url}"
    response = client.get(url)
    df_ctd = pd.DataFrame(response.json())
    if df_ctd.empty:
        return df_bottle

    # Add ctd suffix
    df_ctd = df_ctd.add_prefix("ctd_")

    # Generate matching depth and time variables
    df_bottle["matching_depth"] = (
        df_bottle["bottle_depth"].div(bin_size).round().astype("int64")
    )
    df_ctd["matching_depth"] = df_ctd["ctd_depth"].div(bin_size).round().astype("int64")
    df_bottle["matching_time"] = df_bottle["collected"]
    df_ctd["matching_time"] = pd.to_datetime(df_ctd["ctd_measurement_dt"], utc=True)

    # N Bottles samples
    n_bottles = len(df_bottle)

    # Find closest profile with the exact same depth
    print(f"Match {n_bottles} bottles to CTD Profile data: ")
    df_bottles_closest_time_depth = pd.merge_asof(
        df_bottle.sort_values("matching_time"),
        df_ctd.sort_values(["matching_time"]),
        on="matching_time",
        left_by=["site_id", "matching_depth"],
        right_by=["ctd_station", "matching_depth"],
        tolerance=pd.Timedelta("3h"),
        allow_exact_matches=True,
        direction="nearest",
    )

    # Retrieve bottle data with matching depths and remove those not matching from df_bottles
    in_tolerance = _within_depth_tolerance(
        df_bottles_closest_time_depth["ctd_depth"],
        df_bottles_closest_time_depth["bottle_depth"],
    )
    df_not_matched = df_bottles_closest_time_depth.loc[in_tolerance == False][
        df_bottle.columns
    ]
    df_bottles_matched = df_bottles_closest_time_depth[in_tolerance]
    print(f"{len(df_bottles_matched)} were matched to exact ctd pressure bin.")
    n_bottles -= len(df_bottles_matched)

    # First try to retrieve to the closest profile in time and then closest depth
    if len(df_not_matched) > 0:
        df_bottles_time = pd.merge_asof(
            df_not_matched.sort_values("matching_time"),
            df_ctd.sort_values(["matching_time"])[
                ["ctd_station", "ctd_hakai_id", "matching_time"]
            ],
            on="matching_time",
            left_by="site_id",
            right_by="ctd_station",
            tolerance=pd.Timedelta("1d"),
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
            df_bottles_time["bottle_depth"],
        )
        df_not_matched = df_bottles_time.loc[in_tolerance == False][df_bottle.columns]
        df_bottles_matched = pd.concat(
            [df_bottles_matched, df_bottles_time[in_tolerance]]
        )
    print(f"{len(df_bottles_time[in_tolerance])} were matched to nearest ctd profile.")
    n_bottles = n_bottles - len(df_bottles_matched)

    # Then try to match whatever closest depth sample depth within the allowed time range
    dt = pd.Timedelta("1d")
    df_bottles_depth = pd.DataFrame()
    df_bottles_drop_unmatched = pd.DataFrame()
    if len(df_not_matched) > 0:
        # Loop through each collected times find any data collected nearby by time and try to match nearest depth
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
                df_bottles_drop_unmatched = df_bottles_drop_unmatched.append(drop)
                continue

            # Match the bottles drop to the closest data within this time range
            df_bottles_depth = df_bottles_depth.append(
                pd.merge_asof(
                    drop,
                    df_ctd_filtered.sort_values(["matching_depth"]),
                    on="matching_depth",
                    left_by="site_id",
                    right_by="ctd_station",
                    allow_exact_matches=True,
                    direction="nearest",
                )
            )

        if len(df_bottles_depth) > 0:
            in_tolerance = _within_depth_tolerance(
                df_bottles_depth["ctd_depth"],
                df_bottles_depth["bottle_depth"],
            )
            df_bottles_matched = pd.concat(
                [df_bottles_matched, df_bottles_depth[in_tolerance]]
            )

            df_not_matched = df_bottles_depth[in_tolerance == False][df_bottle.columns]
            print(
                f"{len(df_bottles_depth[in_tolerance])} were matched to the closest in depth ctd profile collected within ±{dt} period of the sample collection time."
            )
            n_bottles -= len(df_bottles_matched)
        else:
            df_not_matched = pd.DataFrame()

        df_not_matched = df_not_matched.append(df_bottles_drop_unmatched)
        no_matched_collected_times = (
            df_not_matched["collected"]
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            .drop_duplicates()
            .tolist()
        )
        print(f"{len(df_not_matched)} failed to be matched to any profiles.")
        print(f"Failed Collection Time list: {no_matched_collected_times}")

    # Finally, keep unmatched bottle data with no possible match
    if len(df_not_matched) > 0:
        df_bottles_matched = pd.concat([df_bottles_matched, df_not_matched])

    return df_bottles_matched


def create_aggregated_meta_variables(df):
    def get_unique_string(x):
        x = ",".join(x)
        return ",".join(set([item for item in x.split(",") if item]))

    df["site_id"] = (
        df.filter(regex="site_id$")
        .replace(np.nan, "")
        .aggregate(get_unique_string, axis=1)
    )
    df["work_area"] = (
        df.filter(regex="work_area$")
        .replace(np.nan, "")
        .aggregate(get_unique_string, axis=1)
    )
    df["organization"] = (
        df.filter(regex="organization$")
        .replace(np.nan, "")
        .aggregate(get_unique_string, axis=1)
    )
    df["survey"] = (
        df.filter(regex="survey$")
        .replace(np.nan, "")
        .aggregate(get_unique_string, axis=1)
    )

    # Split lat and gather lat
    df["latitude"] = df.filter(regex="_lat$").median(axis=1)
    df["longitude"] = df.filter(regex="_long$").median(axis=1)
    df["preciseLat"] = df.filter(regex="_gather_lat$").median(axis=1)
    df["preciseLong"] = df.filter(regex="_gather_long$").median(axis=1)

    df["pressure_transducer_depth"] = (
        df.filter(regex="pressure_transducer_depth$")
        .select_dtypes("number")
        .aggregate(["median"], axis=1)
    )
    df["time"] = df["collected"]
    df["depth"] = df["pressure_transducer_depth"].fillna(df["line_out_depth"])
    df['depth_difference'] = df['depth'] - df['ctd_depth']

    # Remove columns that have been aggregated we assume that all have the same values
    _drop_regex = (
        r"_survey$|_lat$|_long$|_gather_lat$|_gather_long$|_pressure_transducer_depth$"
    )
    df = df[df.columns.drop(list(df.filter(regex=_drop_regex)))]
    return df


def get_bottle_data(
    station,
    time_min=None,
    time_max=None,
):
    """get_bottle_data

    Args:
        station (str): [description]
        time_min (str, optional): Minimum time range (ex: '2020-01-01'). Defaults to None.
        time_max (str, optional): Maximum time range (ex: '2021-01-01'). Defaults to None.

    Returns:
        dataframe: Bottle data with sample and ctd dataset.
    """
    # Samples matched by bottles
    df = join_sample_data(station, time_min, time_max)
    # Matched to ctd data
    df = join_ctd_data(df, station, time_min, time_max)

    df = create_aggregated_meta_variables(df)
    df["time"] = df["collected"]
    df["depth"] = df["pressure_transducer_depth"].fillna(df["line_out_depth"])
    return df


def filter_bottle_variables(df, filter_variables):
    """filter_bottle_variables

    Args:
        df (dataframe): bottle data dataframe
        filter_variables (str): "Reduced" or "Complete" which make reference to the predefine list of variables present within the package.

    Returns:
        [dataframe]: [description]
    """
    # Filter columns
    if filter_variables == "Reduced":
        variable_list = read_variable_list_file(
            os.path.join(module_path, "config", "Reduced_variable_list.csv")
        )
    elif filter_variables == "Complete":
        variable_list = read_variable_list_file(
            os.path.join(module_path, "config", "Complete_variable_list.csv")
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
        output_path = ''

    # Convert datetime variables to timezone unaware datetime64 format in UTC
    for var, var_type in df.dtypes.to_dict().items():
        if "datetime" in f"{var_type}":
            df[var] = df[var].dt.tz_convert("UTC").dt.tz_localize(None)

    ds = df.to_xarray()
    with open(
        os.path.join(module_path, "config", "bottle_netcdf_attributes.json"), "r"
    ) as f:
        attributes = json.loads(f.read())

    # Add Global Attributes
    ds.attrs = attributes["NC_GLOBAL"]

    # Add Variable Attributes
    for var in ds:
        if var in attributes:
            ds[var].attrs = attributes[var]

    # Loop through each groups and save grouped files
    for work_area, ds_work_area in ds.groupby('work_area'):
        for site_id, ds_site in ds_work_area.groupby('site_id'):
            print(f'Save bottle data by work_area={work_area}, station={site_id} and by date')
            for collected, ds_collected in ds_site.groupby('collected.date'):
                #Format name
                subdir = os.path.join(work_area,site_id)
                filename = f"Hakai_Bottle_{work_area}_{site_id}_{collected}.nc".replace(':','')
                filename = re.sub('\:|\-|\.0+','',filename)
                # Generate subfolder if it doesnt' exist yet
                if not os.path.exists(os.path.join(output_path,subdir)):
                    os.makedirs(os.path.join(output_path,subdir))
                
                # Save NetCDF
                new_file = os.path.join(output_path,subdir,filename)
                print(f"Save: {new_file}")
                ds_collected.to_netcdf(new_file)


def read_variable_list_file(path):
    with open(path, "r") as f:
        var_list = f.read().split("\n")
    return var_list
