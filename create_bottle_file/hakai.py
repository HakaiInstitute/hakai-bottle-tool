import datetime as dt
import json
import re

import numpy as np
import pandas as pd
from hakai_api import Client

from create_bottle_file import transform
from create_bottle_file import erddap_output


def get_hakai_data(endpoint_url, filter_url):
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

    # Remove extra columns in metadata if they have been filtered in the query
    meta = meta[df.columns]

    if any(df):
        # Standardize output based on database
        dt_variables = meta.loc[:, meta.loc['udt_name', :].isin(['date', 'timestamp', 'timestamptz'])].columns.tolist()
        if any(dt_variables):
            for item in dt_variables:
                df[item] = pd.to_datetime(df[item])

        # Integer
        int_variables = meta.loc[:, meta.loc['udt_name', :].isin(['int4', 'int8', 'int16'])].columns.tolist()
        if any(int_variables):
            df[int_variables].astype('int')

        # Float
        float_variables = meta.loc[:, meta.loc['udt_name', :].isin(['float4', 'float8', 'float16',
                                                                    'float32', 'numeric'])].columns.tolist()
        if any(float_variables):
            df[float_variables].astype('float')

        # Bool
        bool_variables = meta.loc[:, meta.loc['udt_name', :].isin(['bool'])].columns.tolist()
        if any(bool_variables):
            df[bool_variables].astype('bool')

        # Text data is kept has object for now

    return df, url, meta


def generate_depth_matching_variable(df, index_variable_list):
    # Find median values for pressure transducer
    sample_depth = df.filter(like='pressure_transducer_depth').median(axis=1)

    # Get Pressure Transducer data first if available
    df.loc[df.index, 'sample_matching_depth'] = sample_depth

    # Fill missing values with line_out_depth values
    df = df.reset_index()  # remove indexes to have access to line_out_depth
    df['sample_matching_depth'] = df['sample_matching_depth'].fillna(df['line_out_depth'])
    df, considered_indexes = transform.set_index_from_list(df, index_variable_list)  # Reapply indexes
    return df


def get_prefix_name_from_hakai_endpoint_url(url):
    out = re.search(r'[a-zA-Z0-9]+\Z', url)
    return out[0]


def process_sample_data(event_pk,
                        endpoint_list,
                        index_variable_list,
                        meta_variable_list,
                        ignored_variable_list,
                        time_variable_list,
                        string_columns_regexp):
    # Get Data for the event_pks
    filter_url = 'event_pk={' + ','.join([str(elem) for elem in event_pk]) + '}&limit=-1'
    df_raw, url, metadata = get_hakai_data(endpoint_list['endpoint'], filter_url)

    # If there's no output from API, just past back None values otherwise apply transformations
    if not df_raw.empty:
        # Work on a copy
        df_data = df_raw.copy()

        # Make sure that dtype object columns have fillna values ''
        df_data[df_data.select_dtypes('object').columns] = df_data[df_data.select_dtypes('object').columns].fillna('')

        # Index data
        df_data, temp_index_list = transform.set_index_from_list(df_data, index_variable_list)

        # Combine variables through a pivot or groupby if no pivot needed
        if 'pivot_variable' in endpoint_list:
            df_data = transform.regroup_data_by_index_and_pivot(df_data,
                                                                temp_index_list,
                                                                endpoint_list['pivot_variable'])
        else:
            df_data = transform.regroup_data_by_index_and_pivot(df_data,
                                                                temp_index_list)

        # Add data type prefix to variable name
        df_data = df_data.add_prefix(get_prefix_name_from_hakai_endpoint_url(endpoint_list['endpoint']) + '_')
        metadata = metadata.add_prefix(get_prefix_name_from_hakai_endpoint_url(endpoint_list['endpoint']) + '*')
        # TODO leverage group from NetCDF files by replacing the separator '_'  by '.'

    else:  # No Sample is available for event_pk and endpoint
        print('No data available from: ' + endpoint_list['endpoint'])
        df_data = None
        metadata = None

    return df_data, metadata


def combine_data_from_hakai_endpoints(event_pk,
                                      format_dict):
    print('Process: ' + str(event_pk))
    for ii in format_dict['endpoint_list']:

        print(format_dict['endpoint_list'][ii]['endpoint'])
        df_temp, metadata = process_sample_data(event_pk, format_dict['endpoint_list'][ii],
                                                format_dict['index_variable_list'],
                                                format_dict['meta_variable_list'],
                                                format_dict['ignored_variable_list'],
                                                format_dict['time_variable_list'],
                                                format_dict['string_columns_regexp'])

        if df_temp is not None:
            if 'df_joined' not in locals():
                df_joined = df_temp
                metadata_joined = metadata
            else:
                # Join new sample type to other samples for the same event_pk
                df_joined = df_joined.merge(df_temp, left_index=True, right_index=True, how='outer')
                metadata_joined = metadata_joined.merge(metadata, left_index=True, right_index=True, how='outer')
                # TODO some collection time aren't always the exact number we could use pd.merge_asof and a time
                #  tolerance to do that match

    # Add aggregated meta variables
    df_joined = transform.create_aggregated_meta_variables(df_joined)

    # Generate Matching Depth Variable and add indexing
    df_joined = generate_depth_matching_variable(df_joined, format_dict['index_variable_list'])

    # Make all missing value being np.nan and remove all empty columns
    df_joined.replace('', np.nan, inplace=True)
    # df_joined = transform.remove_empty_columns(df_joined)

    # Remove indexation
    df_joined = df_joined.reset_index()

    # Replace special characters in column names with underscore ' ' '\'
    df_joined.columns = df_joined.columns.str.replace(' |/', '_', regex=True)

    return df_joined, metadata_joined


def get_matching_ctd_data(df_bottles):
    # Get Matching CTD
    # Get the list of event_pk for a specific site
    # Get Time Range around Sample Collection time
    TIME_RANGE = dt.timedelta(days=1)  # days #TODO review interval to get the CTD data
    station_name = df_bottles['site_id'].unique()
    start_time_range = (df_bottles['collected'].min() - TIME_RANGE).strftime("%Y-%m-%d")
    end_time_range = (df_bottles['collected'].max() + TIME_RANGE).strftime("%Y-%m-%d")

    # Get from Hakai API data, CTD data collected for a specific site over a time range predefined by the user
    filter_url = 'fields=station,ctd_cast_pk,start_dt,bottom_dt,end_dt&station=' + station_name[0] + '&start_dt>' + \
                 start_time_range + '&start_dt<' + end_time_range + '&limit=-1&distinct'
    # TODO handle multiple sites: Not sure if that's necessary','.join(map(str,selected_cast_pks))

    endpoint_url = 'ctd/views/file/cast/data'

    # Find closest Cast PK associated to a profile
    df_ctd_site_specific_drops, url, ctd_metadata = get_hakai_data(endpoint_url, filter_url)

    # If any profile is available merge it to the bottle data
    if len(df_ctd_site_specific_drops) > 0:
        # Extract matching time from CTD data
        df_ctd_site_specific_drops['matching_time'] = df_ctd_site_specific_drops['start_dt']

        # Sort profiles in time
        df_ctd_site_specific_drops = df_ctd_site_specific_drops.sort_values(['matching_time'])

        # Find closest profile
        df_bottles = pd.merge_asof(df_bottles, df_ctd_site_specific_drops[{'matching_time', 'ctd_cast_pk'}],
                                   left_on=['collected'], right_on=['matching_time'], allow_exact_matches=True,
                                   direction='nearest')

        selected_cast_pks = df_bottles['ctd_cast_pk'].unique()

        # Get the corresponding data from the API (ignore CTD with depth flagged: Seabird Flag is -9.99E-29)
        filter_url = 'ctd_cast_pk={' + ','.join(map(str, selected_cast_pks)) + \
                     '}&direction_flag=d&depth!=-9.99E-29&limit=-1'
        df_ctd_profile, url, ctd_metadata = get_hakai_data(endpoint_url, filter_url)
        # FIXME Make the download compatible with multiple cast pks

        # Rename add CTD_ prefix to variables
        df_ctd_profile = df_ctd_profile.add_prefix('CTD_')
        ctd_metadata = ctd_metadata.add_prefix('CTD*')

        # Sort by depth both dataframes before merging them
        df_ctd_profile = df_ctd_profile.sort_values(['CTD_depth'])
        df_bottles = df_bottles.sort_values(['sample_matching_depth'])

        # Make sure that the sample matching depth is same type as CTD.depth
        df_bottles['sample_matching_depth'] = df_bottles['sample_matching_depth'].astype('float64')
        # Match CTD with Samples
        df_with_matched_ctd = pd.merge_asof(df_bottles, df_ctd_profile,
                                            left_on='sample_matching_depth', right_on='CTD_depth',
                                            left_by='ctd_cast_pk', right_by='CTD_ctd_cast_pk', direction='nearest')

        # TODO add range limit in meters that depth values can be with be matched with.
        #  Could be done after the fact with a flag based on percentage of the depth
        # TODO add another step for if CTD from closest drop in time do not reach the maximum sampled depth. Could look
        #  at other profiles completed within the time interval
    else:
        # If no CTD data available just give back the bottle data still.
        df_with_matched_ctd = df_bottles

    return df_with_matched_ctd, ctd_metadata


def create_bottle_netcdf(event_pk, format_dict):
    print('Collect Sample data for event pk: [' + str(event_pk)+']')
    df_bottles, metadata = combine_data_from_hakai_endpoints(event_pk,
                                                             format_dict)

    # Generate a profile_id variable
    # df_bottles = transform.create_id_variable(df_bottles, 'profile_id', format_dict['cdm_variables'])
    df_bottles['bottle_profile_id'] = df_bottles['organization'] + \
                                      '_' + df_bottles['work_area'] + \
                                      '_' + df_bottles['site_id'] + \
                                      '_' + df_bottles['collected'].dt.strftime('%Y%m%d_%H%M%S%Z') + \
                                      '_EventPk' + df_bottles['event_pk'].apply(str)

    # Discard the ignored variables and empty ones
    df_bottles = df_bottles.dropna(axis=1, how='all')
    df_bottles, df_bottle_ignored = transform.remove_variable(df_bottles, format_dict['ignored_variable_list'])

    for event, df_event in df_bottles.groupby(by=['event_pk', 'collected']):
        # Loop through each event pk
        print('Retrieve Corresponding CTD Data')
        df_matched, ctd_metadata = get_matching_ctd_data(df_event.reset_index())

        # Remove discard listed CTD variables
        df_matched, df_ignored = transform.remove_variable(df_matched, format_dict['ctd_variable_list_to_ignore'])

        # Convert time data to a datetime object
        df_matched, converted_columns = transform.convert_columns_to_datetime(df_matched,
                                                                              format_dict['time_variable_list'])

        # Rename variables for ERDDAP time, and depth
        #  Time correspond to the sample collected time
        #  Depth is the matching depth used for the CTD transducer pressure depth > line_out_depth
        df_matched['time'] = df_matched['collected']  # Time correspond to the sample collected time
        df_matched['depth'] = df_matched['sample_matching_depth']

        # Rename some of the variables
        for key in format_dict['rename_variables_dict'].keys():
            df_matched = df_matched.rename(columns=lambda x: re.sub(key, format_dict['rename_variables_dict'][key], x))

        # Sort columns
        df_matched = transform.sort_column_order(df_matched, format_dict['variables_final_order'])

        # Add Index which will be transformed in coordinate in xarray
        df_matched = df_matched.set_index(['depth'])

        # Remove empty columns
        df_matched = df_matched.dropna(how='all', axis=1)

        # Merge metadata from bottles and CTD to fill up the netcdf attributes
        metadata_for_xarray = metadata.merge(ctd_metadata, left_index=True, right_index=True, how='outer')


        # Define the netcdf file name to be created
        netcdf_file_name_out = df_event['bottle_profile_id'].unique()[0] + '.nc'

        # Create netcdf by converting the pandas DataFrame to an xarray
        ds = erddap_output.convert_bottle_data_to_xarray(df_matched, netcdf_file_name_out,
                                                         metadata_for_xarray, format_dict)

        meta_dict = erddap_output.compile_netcdf_variable_and_attributes(ds, 'Hakai_bottle_files_variables.csv')

    return


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


def get_site_netcdf_files(station_name, format_dict):
    print('Get Site Specific related Event Pks')
    pk_list = get_event_pks_for_a_site(format_dict['endpoint_list'], station_name)
    create_bottle_netcdf(pk_list['event_pk'], format_dict)

    ## Loop through each separate event_pk
    #for event_pk in pk_list['event_pk'].unique():
    #    create_bottle_netcdf(event_pk, format_dict)


def get_hakai_variable_order(format_dict):
    print('Retrieve Hakai''s variable order')
    variables_order = []
    for ii in format_dict['endpoint_list']:
        sample_type_name = get_prefix_name_from_hakai_endpoint_url(format_dict['endpoint_list'][ii]['endpoint'])
        df, url, meta = get_hakai_data(format_dict['endpoint_list'][ii]['endpoint'], 'limit=1')
        variables_order.extend(sample_type_name + '_([a-zA-Z0-9_]*_){0,1}' + df.columns)

    return variables_order
