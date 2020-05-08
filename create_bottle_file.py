import numpy as np
import pandas as pd
import re
import datetime as dt
import xarray as xr
import json

from hakai_api import Client


# Endpoint list to get data from
endpoint_list = {
    1: {'endpoint': 'eims/views/output/nutrients'},
    2: {'endpoint': 'eims/views/output/microbial', 'pivot_variable': 'microbial_sample_type'},
    3: {'endpoint': 'eims/views/output/hplc'},
    4: {'endpoint': 'eims/views/output/o18'},
    5: {'endpoint': 'eims/views/output/poms', 'pivot_variable': 'acidified'},
    6: {'endpoint': 'eims/views/output/ysi'},
    7: {'endpoint': 'eims/views/output/chlorophyll', 'pivot_variable': 'filter_type'},
    8: {'endpoint': 'eims/views/output/doc'}
}

# List of variables to use as index for matching of the different sample data sets
index_variable_list = ['organization', 'work_area', 'site_id', 'event_pk', 'collected', 'line_out_depth']

# List of columns to index, associate with Metadata, and ignore
meta_variable_list = ['organization', 'work_area', 'site_id', 'date', 'survey', 'pressure_transducer_depth',
                      'line_out_depth']
meta_variable_list = []

# List of variables to ignore from the sample data
ignored_variable_list = ['action', 'rn', 'sampling_bout',
                         'analyzed', 'preserved',
                         'technician', 'lab_technician', 'source', 'volume', 'dna_volume_te_wash',
                         'before_acid', 'after_acid', 'calibration_slope', 'acid_ratio_correction_factor',
                         'acid_coefficient', 'acetone_volumne_ml', 'fluorometer_serial_no', 'calibration',
                         'analyzing_lab']

# regex keys used to identify the time variables each item is then separated by | in the regex query
time_variable_list = ['date', 'collected', 'preserved', 'analyzed','_dt','time']

# Variable list to ignore from the CTD data
ctd_variable_list_to_ignore = ['device_firmware', 'file_processing_stage', 'shutoff', 'ctd_file_pk', 'ctd_data_pk',
                               'v_main', 'v_lith', 'i_pump', 'i_ext01', 'i_ext2345', 'cast_processing_stage', 'status',
                               'duration', 'start_depth', 'bottom_depth', 'target_depth', 'drop_speed', 'vessel',
                               'operators',
                               'pruth_air_pressure_before_cast', 'min_pressure_before_cast', 'min_depth_before_cast',
                               'min_pressure_after_cast', 'min_depth_after_cast', 'estimated_air_pressure',
                               'estimated_depth_shift', 'original_start_dt', 'original_bottom_dt',
                               'original_start_depth', 'original_bottom_depth', 'direction_flag',
                               'descent_rate', 'spec_cond', 'spec_cond_flag', 'oxygen_voltage', 'oxygen_voltage_flag',
                               'cast_number']

# regex keys used to identify text columns from the different data sets
string_columns_regexp = ['flag', 'comments', 'hakai_id', 'survey', 'method', 'organization', 'site_id',
                         'work_area', 'quality_level', 'quality_log', 'row_flag', 'serial_no', 'filename',
                         'device_sn', 'cruise', 'filter_type', 'units', 'project_specific_id', 'station',
                         'device_model']

# List of Expressions to rename from the different variable names
rename_variables_dict = {'poms.True':'poms.Acidified', 'poms.False': 'poms.nonAcidified'}

# Variable order at the end
variables_final_order = ['bottle_profile_id', 'organization', 'work_area', 'site_id', 'latitude', 'longitude',
                         'gather_lat', 'gather_long',
                         'event_pk', 'time', 'depth', 'collected', 'line_out_depth', 'pressure_transducer_depth',
                         'matching_depth']

#####################################################################################
# Transformation tools to apply on Pandas DataFrames


def get_hakai_data(endpoint_url, filter_url):
    # Get Hakai Data
    # Get Data from Hakai API
    client = Client()  # Follow stdout prompts to get an API token

    # Make a data request for sampling stations
    url = '%s/%s?%s' % (client.api_root, endpoint_url, filter_url)
    response = client.get(url)
    df = pd.DataFrame(response.json())

    # Get Metadata
    url_meta = '%s/%s?%s' % (client.api_root, endpoint_url, 'meta')
    response = client.get(url_meta)
    meta = pd.DataFrame(response.json())

    return df, url, meta


def generate_depth_matching_variable_for_sample(df, index_variable_list):
    # Find median values for pressure transducer
    sample_depth = df.filter(like='pressure_transducer_depth').median(axis=1)

    # Get Pressure Transducer data first if available
    df['sample_matching_depth'] = sample_depth

    # Fill missing values with line_out_depth values
    df = df.reset_index()  # remove indexes to have access to line_out_depth
    df['sample_matching_depth'] = df['sample_matching_depth'].fillna(df['line_out_depth'])
    df, considered_indexes = set_index_from_list(df, index_variable_list)  # Reapply indexes
    return df


def separate_constant_variables(df, index_list):
    constant_columns = df.columns[df.nunique() <= 1]

    # Remove constant value, keep one for each index
    df_meta = df.filter(items=constant_columns).reset_index().drop_duplicates().set_index(index_list)

    # Get Variables that aren't constant
    df_data = df.drop(constant_columns, axis=1)
    return df_data, df_meta, constant_columns


def set_index_from_list(df, index_variable_list):
    # Remove any index if there some already
    df = df.reset_index()

    # Keep only the column to become index that are available and have no NaN/empty values associated
    considered_indexes = np.array(index_variable_list)[np.in1d(index_variable_list,  df.columns)].tolist()
    considered_indexes = np.array(considered_indexes)[~np.array(df[considered_indexes].isna().any(axis=0))].tolist()

    # Define new indexes and sort
    df = df.set_index(considered_indexes).sort_index().drop(['index'], axis=1)

    return df, considered_indexes


def reduce_to_index(df):
    index_names = df.index.names
    df_reduced = df.reset_index().drop_duplicates().set_index(index_names)
    return df_reduced


def remove_variable(df, var_list_to_remove):
    df_without_variables = df.drop(var_list_to_remove, axis=1, errors='ignore')
    df_removed_variables = df.filter(items=var_list_to_remove)
    return df_without_variables, df_removed_variables


def remove_empty_columns(df):
    df = df.dropna(axis=1, how='all')
    return df


def sort_column_order(df, column_to_order):
    # Keep only columns that exist in df
    keep = []
    for keys in column_to_order:
        if keys in df.columns:
            keep.append(keys)

    # Set order and append any variables that are not listed after
    new_columns = keep + (df.columns.drop(keep).tolist())
    df = df[new_columns]
    return df


def regroup_data_by_index_and_pivot(df, index_variable_list, variable_to_pivot=''):
    if variable_to_pivot in df.columns:
        # Get pivoted data frames
        df_str = pd.pivot_table(df, index=index_variable_list, columns=variable_to_pivot,
                                aggfunc=lambda x: ', '.join(np.unique(x)))
        df_mean = pd.pivot_table(df, index=index_variable_list, columns=variable_to_pivot,
                                 aggfunc='mean')

        # Merge Column Names into one layer
        df_str.columns = [f'{j}_{i}' for i, j in df_str.columns]
        df_mean.columns = [f'{j}_{i}' for i, j in df_mean.columns]

    else:
        # Regroup data by index
        df_str = df.groupby(by=index_variable_list).transform(lambda x: ', '.join(np.unique(x))).drop_duplicates()
        df_mean = df.groupby(by=index_variable_list).mean()

    # Merge Strings with Averaged Values
    df_out = df_str.join(df_mean, how='outer')

    if variable_to_pivot in df.columns:
        # Find if there's any duplicate numerical values
        number_columns = [variable_to_pivot] + df.select_dtypes(['number']).columns.to_list()

        # Count how many values are superposed
        df_count = pd.pivot_table(df[number_columns], index=index_variable_list, columns=variable_to_pivot,
                                  aggfunc='count')

        # Merge Column Names into count Data Frame to be similar to others
        df_count.columns = [f'{j}_{i}' for i, j in df_count.columns]
        # TODO leverage groups in NetCDFs by replacing separator  '_' by '.'
    else:
        # Find if there's any duplicate numerical values
        number_columns = df.select_dtypes(['number']).columns.to_list()

        df_count = df[number_columns].groupby(by=index_variable_list).count()

    # Get Min, Max and Count values for replicate numbers
    if any(df_count > 1):

        # Combined duplicated data either with a pivot on one column or by grouping
        if variable_to_pivot in df.columns:
            df_min = pd.pivot_table(df[number_columns], index=index_variable_list, columns=variable_to_pivot,
                                    aggfunc='min')
            df_max = pd.pivot_table(df[number_columns], index=index_variable_list, columns=variable_to_pivot,
                                    aggfunc='max')

            # Merge Column Names into count Data Frame to be similar to others
            df_min.columns = [f'{j}_{i}' for i, j in df_min.columns]
            df_max.columns = [f'{j}_{i}' for i, j in df_max.columns]

        else:
            df_min = df[number_columns].groupby(by=index_variable_list).min()
            df_max = df[number_columns].groupby(by=index_variable_list).max()

        # If there's any duplicates with variations in values, get there min/max values too
        if any(df_max-df_min != 0):
            # Find columns where there's replicates and different values
            columns_with_duplicates = df_min.columns[df_min.where(df_max-df_min != 0).any(axis=0)]

            # Remove rows with no replicates
            df_min[df_count <= 1] = np.nan
            df_max[df_count <= 1] = np.nan
            df_count[df_count <= 1] = np.nan

            # Add suffix to all columns to force it
            df_min = df_min.filter(columns_with_duplicates).add_suffix('_min')
            df_max = df_max.filter(columns_with_duplicates).add_suffix('_max')
            df_count_reduced = df_count.filter(columns_with_duplicates).add_suffix('_nReplicates')

            # Merge stats together
            df_stats = df_min.join(df_max, how='outer')
            df_stats = df_stats.join(df_count_reduced, how='outer')

            # Sort columns to have the similar next to each other
            df_stats = df_stats[sorted(df_stats.columns, reverse=True)]

            # Merge with the data
            df_out = df_out.join(df_stats, how='outer')

    return df_out


def convert_columns_to_datetime(df, regexp_string):
    regexp_look = '|'.join(regexp_string)
    columns_to_convert = df.filter(regex=regexp_look).columns

    for item in columns_to_convert:
        if item in df.columns:
            df[item] = pd.to_datetime(df[item])
    return df, columns_to_convert


def standardize_object_type(df, regexp_string, type, empty_value_prior, empty_value_after):
    regexp_look = '|'.join(regexp_string)
    columnToModify = df.filter(regex=regexp_look).columns

    # Replace empty values by another
    df[columnToModify] = df[columnToModify].replace(empty_value_prior, empty_value_after)

    # Change type of the columns selected
    for var in columnToModify:
        if type == '|S':
            df[var] = df[var].str.encode("ascii", errors='ignore').astype('|S')
        else:
            df[var] = df[var].astype(type)

    return df


def get_prefix_name_from_hakai_endpoint_url(url):
    out = re.search(r'[a-zA-Z]+\Z', url)
    return out[0]


def create_aggregated_meta_variables(df):
    # df['site_id'] = df.filter(like='site_id').replace(np.nan, '').aggregate(['unique'], axis=1)
    # df['work_area'] = df.filter(like='work_area').replace(np.nan, '').aggregate(['unique'], axis=1)
    # df['organization'] = df.filter(like='organization').replace(np.nan, '').aggregate(['unique'], axis=1)
    # df['survey'] = df.filter(like='survey').replace(np.nan, '').aggregate(['unique'], axis=1)

    df['latitude'] = df.filter(regex='lat$').aggregate(['median'], axis=1)
    df['longitude'] = df.filter(regex='long$').aggregate(['median'], axis=1)
    df['gather_lat'] = df.filter(regex='gather_lat$').aggregate(['median'], axis=1)
    df['gather_long'] = df.filter(regex='gather_long$').aggregate(['median'], axis=1)

    df['pressure_transducer_depth'] = df.filter(regex='pressure_transducer_depth$').aggregate(['median'], axis=1)
    # TODO should we remove the extra variables once combined

    # Remove columns that have been aggregated we assume that all have the same values
    df = df[df.columns.drop(list(df.filter(
        regex=r'_survey$|_lat$|_long$|_gather_lat$|_gather_long$|_pressure_transducer_depth$')))]

    return df


def process_sample_data(event_pk,
                        endpoint_list,
                        index_variable_list,
                        meta_variable_list,
                        ignored_variable_list,
                        time_variable_list,
                        string_columns_regexp):

    filter_url = 'event_pk=' + str(event_pk)

    # Get Data
    df_raw, url, metadata = get_hakai_data(endpoint_list['endpoint'], filter_url)

    # If there's no output from API, just past back None values otherwise apply transformations
    if not df_raw.empty:
        # Convert time data to a datetime object
        df_data, converted_columns = convert_columns_to_datetime(df_raw, time_variable_list)

        # Index data
        df_data, temp_index_list = set_index_from_list(df_data, index_variable_list)

        # Remove Drop Metadata related variables
        df_data, df_drop_meta = remove_variable(df_data, meta_variable_list)

        # Remove empty columns and ignored ones
        df_data = remove_empty_columns(df_data)
        df_data, df_ignored = remove_variable(df_data, ignored_variable_list)
        metadata, meta_ignored = remove_variable(metadata, ignored_variable_list)

        # Do pivot for some variables
        if 'pivot_variable' in endpoint_list:
            df_data = regroup_data_by_index_and_pivot(df_data,
                                                      temp_index_list,
                                                      endpoint_list['pivot_variable'])
        else:
            df_data = regroup_data_by_index_and_pivot(df_data,
                                                      temp_index_list)

        # Add data type prefix to variable name
        df_data = df_data.add_prefix(get_prefix_name_from_hakai_endpoint_url(endpoint_list['endpoint']) + '_')
        metadata = metadata.add_prefix(get_prefix_name_from_hakai_endpoint_url(endpoint_list['endpoint']) + '*')
        # TODO leverage group from NetCDF files by replacing the separator '_'  by '.'

        # Merged Back Meta Columns with Data Columns
        df_out = df_drop_meta.join(df_data, how='outer').drop_duplicates()
        # TODO review do we still need that metadata recombination

    else: # No Sample is available for event_pk and endpoint
        print('No data available from: ' + endpoint_list['endpoint'])
        df_out = None
        metadata = None

    return df_out, metadata


def combine_data_from_hakai_endpoints(event_pk,
                                      endpoint_dict,
                                      index_variable_list,
                                      meta_variable_list,
                                      ignored_variable_list,
                                      string_columns_regexp):

    print('Process: ' + str(event_pk))
    for ii in endpoint_dict:

        print(endpoint_dict[ii]['endpoint'])
        df_temp, metadata = process_sample_data(event_pk, endpoint_dict[ii], index_variable_list,
                                                meta_variable_list, ignored_variable_list, time_variable_list,
                                                string_columns_regexp)

        if df_temp is not None:
            if 'df_joined' not in locals():
                df_joined = df_temp
                metadata_joined = metadata
            else:
                # new_index = index_variable_list + meta_variable_list
                df_joined = df_joined.merge(df_temp, left_index=True, right_index=True, how='outer')
                metadata_joined = metadata_joined.merge(metadata, left_index=True, right_index=True, how='outer')

    # Add aggregated meta variables
    df_joined = create_aggregated_meta_variables(df_joined)

    # Generate Matching Depth Variable and add indexing
    df_joined = generate_depth_matching_variable_for_sample(df_joined, index_variable_list)

    # Make all missing value being np.nan and remove all empty columns
    df_joined.replace('', np.nan, inplace=True)
    df_joined = remove_empty_columns(df_joined)

    # Remove indexation
    df_joined = df_joined.reset_index()

    # Replace special characters in column names with underscore ' ' '\'
    df_joined.columns = df_joined.columns.str.replace(' |/', '_', regex=True)

    return df_joined, metadata_joined


# Get the list of event_pk for a specific site
def get_event_pks_for_a_site(endpoint_list, station_name):
    filter_url = 'fields=site_id,event_pk,collected&limit=-1&distinct&site_id=' + station_name

    for ii in endpoint_list:
        df, url, meta = get_hakai_data(endpoint_list[ii]['endpoint'], filter_url)
        if df is not None:
            if 'df_joined' in locals():
                df_joined = df_joined.append(df)
            else:
                df_joined = df

    return df_joined


def get_matching_ctd_data(df_bottles):
    # Get Matching CTD
    # Get the list of event_pk for a specific site
    # Get Time Range around Sample Collection time
    TIME_RANGE = dt.timedelta(days=1)  # days #TODO review interval to get the CTD data
    station_name = df_bottles['site_id'].unique()
    start_time_range = (df_bottles['collected'].min() - TIME_RANGE).strftime("%Y-%m-%d")
    end_time_range = (df_bottles['collected'].max() + TIME_RANGE).strftime("%Y-%m-%d")

    filter_url = 'fields=station,ctd_cast_pk,start_dt,bottom_dt,end_dt&station=' + station_name[0] + '&start_dt>' + \
                 start_time_range + '&start_dt<' + end_time_range + '&limit=-1&distinct'
    # TODO handle multiple sites: Not sure if that's necessary

    endpoint_url = 'ctd/views/file/cast/data'

    # Find closest Cast PK associated to a profile
    df_ctd_site_specific_drops, url, ctd_metadata = get_hakai_data(endpoint_url, filter_url)

    # If any profile is available merge it to the bottle data
    if len(df_ctd_site_specific_drops) > 0:
        # Extract matching time from CTD data
        df_ctd_site_specific_drops['matching_time'] = pd.to_datetime(df_ctd_site_specific_drops['start_dt'])

        # Sort profiles in time
        df_ctd_site_specific_drops = df_ctd_site_specific_drops.sort_values(['matching_time'])

        # Find closest profile
        df_bottles = pd.merge_asof(df_bottles, df_ctd_site_specific_drops[{'matching_time', 'ctd_cast_pk'}],
                                   left_on=['collected'], right_on=['matching_time'], allow_exact_matches=True,
                                   direction='nearest')

        selected_cast_pks = df_bottles['ctd_cast_pk'].unique()

        # Get the corresponding data from the API (ignore CTD with depth flagged: Seabird Flag is -9.99E-29)
        filter_url = 'ctd_cast_pk=' + str(selected_cast_pks[0]) + '&direction_flag=d&depth!=-9.99E-29&limit=-1'
        df_ctd_profile, url, ctd_metadata = get_hakai_data(endpoint_url, filter_url)
        # FIXME Make the download compatible with multiple cast pks

        # Remove unnecessary columns
        df_ctd_profile, removed_ctd_vars = remove_variable(df_ctd_profile, ctd_variable_list_to_ignore)

        # Rename add CTD_ prefix to variables
        df_ctd_profile = df_ctd_profile.add_prefix('CTD_')
        ctd_metadata = ctd_metadata.add_prefix('CTD*')

        # Sort by depth
        df_ctd_profile = df_ctd_profile.sort_values(['CTD_depth'])
        df_bottles = df_bottles.sort_values(['sample_matching_depth'])

        # Make sure that the sample matching depth is same type as CTD.depth
        df_bottles['sample_matching_depth'] = df_bottles['sample_matching_depth'].astype('float64')
        # Match CTD with Samples
        df_with_matched_ctd = pd.merge_asof(df_bottles, df_ctd_profile,
                                            left_on='sample_matching_depth', right_on='CTD_depth',
                                            left_by='ctd_cast_pk', right_by='CTD_ctd_cast_pk', direction='nearest')


        # TODO add range limit in meters that depth values can be with be matched with.
        # Could be done after the fact with a flag based on percentage of the depth
    else:
        # If no CTD data available just give back the bottle data still.
        df_with_matched_ctd = df_bottles

    return df_with_matched_ctd, ctd_metadata

##################


def convert_bottle_data_to_xarray(df,
                                  netcdf_file_name,
                                  metadata_for_xarray,
                                  string_columns_regexp,
                                  time_variable_list):
    # Standardize the data types
    # Text data to Strings
    df = standardize_object_type(df, string_columns_regexp, '|S', np.nan, '')

    # Date time objects to datetime64s UTC
    df, converted_time_variables = convert_columns_to_datetime(df, time_variable_list)

    print('Convert DataFrame to Xarray')
    # Convert to a xarray
    ds = df.to_xarray()

    # Add metadata and documentation to xarray data from the database metadata
    ds = add_metadata_to_xarray(ds, metadata_for_xarray, converted_time_variables)

    #TODO add documentation to data. Specify all the variable attributes.

    # Save xarray to netcdf
    print('Save to '+netcdf_file_name)
    ds.to_netcdf(netcdf_file_name)
    return ds


def add_metadata_to_xarray(ds, metadata, time_variable_list):
    # Give standard attributes to time variables
    for time_variable in time_variable_list:
        if time_variable in ds:
            ds[time_variable].encoding['units'] = 'seconds since 1970-01-01T00:00:00Z'
            ds[time_variable].attrs['timezone'] = 'UTC'

    # Loop through all the metadata variable listed and find any matching ones in the xarray
    for keys in metadata.columns:
        searchKey = keys.replace('*', '.*?')+'($|_min|_max)'  # Consider values and min/max columns have the metadata
        regexp = re.compile(searchKey)
        matching_variables = list(filter(regexp.search, ds.keys()))

        # Loop through each similar variables and add metadata
        for variable in matching_variables:
            if metadata[keys].variable_name_long is not None:
                ds[variable].attrs['name_long'] = metadata[keys].variable_name_long
            if metadata[keys].variable_units is not None:
                ds[variable].attrs['units'] = metadata[keys].variable_units
            if metadata[keys].variable_definition is not None:
                ds[variable].attrs['definition'] = metadata[keys].variable_definition

    return ds


def create_bottle_netcdf(event_pk):
    print('Collect Sample data for event pk: '+str(event_pk))
    df_bottles, metadata = combine_data_from_hakai_endpoints(event_pk,
                                                             endpoint_list,
                                                             index_variable_list,
                                                             meta_variable_list,
                                                             ignored_variable_list,
                                                             string_columns_regexp)

    # There's multiple collected time per event_pk loop through each of them
    uniqueDrops = df_bottles[['event_pk', 'collected']].drop_duplicates()

    # Generate a profile_id variable
    df_bottles['bottle_profile_id'] = df_bottles['organization'] + \
                                      '_' + df_bottles['work_area'] +\
                                      '_' + df_bottles['site_id'] + \
                                      '_' + df_bottles['collected'].dt.strftime('%Y%m%d_%H%M%S%Z') + \
                                      '_EventPk' + df_bottles['event_pk'].apply(str)

    # Loop through the different collected time
    for profile_id in df_bottles['bottle_profile_id'].unique():
        df_bottle_temp = df_bottles[df_bottles['bottle_profile_id'] == profile_id]

        print('Retrieve Corresponding CTD Data')
        df_matched, ctd_metadata = get_matching_ctd_data(df_bottle_temp)

        # Rename variables for ERDDAP time, and depth
        df_matched['time'] = df_matched['collected']
        df_matched['depth'] = df_matched['sample_matching_depth']

        # Rename some of the variables
        for key in rename_variables_dict.keys():
            df_matched = df_matched.rename(columns=lambda x: re.sub(key, rename_variables_dict[key], x))

        # Sort columns
        df_matched = sort_column_order(df_matched, variables_final_order)

        # Add Index for output dimensions
        df_matched = df_matched.set_index(['depth'])

        # Merge metadata from bottles and CTD to fill up the netcdf attributes
        metadata_for_xarray = metadata.merge(ctd_metadata, left_index=True, right_index=True, how='outer')

        # Create netcdf by converting the pandas DataFrame to an xarray
        ds = convert_bottle_data_to_xarray(df_matched, profile_id+'.nc', metadata_for_xarray,
                                           string_columns_regexp, time_variable_list)

        meta_dict = compile_netcdf_variable_and_attributes(ds, 'Hakai_bottle_files_variables.csv')

    return


def get_site_netcdf_files(station_name):
    print('Get Site Specific related Event Pks')
    list = get_event_pks_for_a_site(endpoint_list, station_name)

    # Loop through each separate event_pk
    for event_pk in list['event_pk'].unique():
        create_bottle_netcdf(event_pk)


def compile_netcdf_variable_and_attributes(xarray, variable_log_file_path):
    # Get list of variables and coordinates
    variable_list = list(xarray.coords) + list(xarray.keys())

    # Define which attributes to have a look at
    attribute_list = ['long_name', 'units', 'definition']
    meta_dict = {}

    # Compile a dictionary all the variables and corresponding attributes
    for variable in variable_list:
        meta_dict[variable] = {}
        for attribute in attribute_list:
            if attribute in xarray[variable].attrs:
                meta_dict[variable][attribute] = xarray[variable].attrs[attribute]
            else:
                meta_dict[variable][attribute] = None

    # Convert to a dataframe format
    df_meta = pd.DataFrame(meta_dict).transpose().reset_index().rename(columns={'index': 'Variable'})

    # Look if there's previous file already saved
    print('Add variables to variable list')
    try:
        # Read existing list of variable
        df_previous_meta = pd.read_csv(variable_log_file_path)
        df_previous_meta = df_previous_meta.drop('Unnamed: 0', axis=1)

        # Merge new variables with previous given and keep only the unique ones
        df_merged_meta = df_previous_meta.append(df_meta).drop_duplicates()
    except FileNotFoundError:
        # If previous variable list exist just consider what's given now
        df_merged_meta = df_meta

    # Write variable list
    df_merged_meta.to_csv(variable_log_file_path)

    return df_merged_meta


def compile_variable_names(df, variable_log_file_path):
    print('Add variables to variable list')
    try:
        # Read existing list of variable
        with open(variable_log_file_path, "r") as file:
            variable_list = json.load(file)

        # Append new variables given and keep only the unique ones
        variable_list = list(set(variable_list + df.columns.tolist()))
    except FileNotFoundError:
        # If previous variable list exist just consider what's given now
        variable_list = df.columns.tolist()

    # Write variable list
    with open(variable_log_file_path, 'w') as file:
        json.dump(variable_list, file, indent=2)


# Test Variables
site_name = 'QU39'
event_pk = 416
event_pk = 3983709
event_pk = 516386
event_pk = 504
event_pk = 466112
#event_pk = 3098
create_bottle_netcdf(event_pk)
#get_site_netcdf_files(site_name)


#TODO Remove the lat/long values for each sample type
#TODO Work on order of the samples

#TODO review why there's only a few duplicates, can't be really in the data itself
#TODO min/max values should apply only to data with replicates