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


def generate_depth_matching_variable(df):
    # Find median values for pressure transducer
    df['sample_matching_depth'] = df.filter(like='pressure_transducer_depth').median(axis=1)

    # Fill values with second column
    df.loc[df['sample_matching_depth'].isnull(), 'sample_matching_depth'] = df.loc[
        df['sample_matching_depth'].isnull()].index.get_level_values('line_out_depth')
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

        # Drop any rows that have empty values within the index_variables
        df_data = df_data.dropna(subset=index_variable_list).set_index(index_variable_list)

        # Combine variables through a pivot or groupby if no pivot needed
        if 'pivot_variable' in endpoint_list:
            df_data = transform.regroup_data_by_index_and_pivot(df_data,
                                                                index_variable_list,
                                                                endpoint_list['pivot_variable'])
        else:
            df_data = transform.regroup_data_by_index_and_pivot(df_data,
                                                                index_variable_list)

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
                df_joined_asof = df_temp

            else:
                # Allow a tolerance between the collected times.
                # df_joined_asof = pd.merge_asof(df_joined_asof.reset_index(), df_temp.reset_index(),
                #                              by='line_out_depth',
                #                              on='collected',
                #                              tolerance=pd.Timedelta('1hour'),
                #                              allow_exact_matches=True)

                # df_joined_asof = df_joined_asof.drop(['level_0', 'index'], axis=1)
                # FIXME asof seems to be working we need to ignore the indexes as long df_joined is always the same
                #  event pk we should be good. However those indexes columns should be merged again after.

                # Join new sample type to other samples for the same event_pk and sampling time
                df_joined = df_joined.merge(df_temp, left_index=True, right_index=True, how='outer')
                metadata_joined = metadata_joined.merge(metadata, left_index=True, right_index=True, how='outer')
                # TODO some collection time aren't always the exact number we could use pd.merge_asof and a time
                #  tolerance to do that match

                # TODO Track mismatched collected time

    # Track the multiple collected time values per sample type
    if len(df_joined.index.unique(level='collected')) > 1:
        df_collected_time = df_joined.filter(like='_action') \
            .rename(columns=lambda x: re.sub('_.*', '', x)).stack().swaplevel(i=5, j=6) \
            .droplevel(level=6).reset_index().drop_duplicates().set_index('collected')

        # Time info to a info_log
        time_info = df_collected_time.reset_index()[
            ['organization', 'work_area', 'site_id', 'event_pk']].drop_duplicates()
        time_info['min_collected'] = df_joined.index.get_level_values('collected').min()
        time_info['max_collected'] = df_joined.index.get_level_values('collected').max()
        time_info['delta_collected'] = time_info['max_collected'] - time_info['min_collected']
        time_info['delta_collected_hours'] = time_info['delta_collected'][0].total_seconds() / 3600
        time_info['n_collected'] = len(df_joined.index.get_level_values('collected').unique())
        time_info['time_list'] = str(
            df_collected_time['level_5'].groupby('collected').apply(lambda x: ','.join(x)).to_dict())

        with open('Collection_time_log.csv', 'a') as f:
            time_info.to_csv(f, mode='a', header=f.tell() == 0, index=False, line_terminator='\n')

    # Add aggregated meta variables
    df_joined = transform.create_aggregated_meta_variables(df_joined)

    # Generate Matching Depth Variable and add indexing
    df_joined = generate_depth_matching_variable(df_joined)

    # Make all missing value being np.nan and remove all empty columns
    df_joined.replace('', np.nan, inplace=True)
    # df_joined = transform.remove_empty_columns(df_joined)

    # Remove indexation
    df_joined = df_joined.reset_index()

    # Replace special characters in column names with underscore ' ' '\'
    df_joined.columns = df_joined.columns.str.replace(' |/', '_', regex=True)

    return df_joined, metadata_joined


def get_matching_ctd_data(df_bottles,
                          df_ctd=None,
                          time_range=dt.timedelta(days=1),
                          ctd_profile_id='hakai_id',
                          bottle_profile_id='collected',
                          ctd_station_id='station',
                          bottle_station_id='site_id',
                          ctd_depth='pressure',
                          bottle_depth='sample_matching_depth',
                          ctd_time='start_dt',
                          bottle_time='collected',
                          ctd_prefix='CTD_',
                          depth_tolerance_range=3,
                          depth_tolerance_ratio=0.15,
                          bin_size=1,
                          hakai_ctd_endpoint='ctd/views/file/cast/data'
                          ):

    """
    Matching Algotrithm use to match CTD Profile to bottle data. The algorithm always match data for the same
    station id on both side (Bottle and CTD). Then then matching will be done by in the following orther:
        1. Bottle data will be matched to the Closest CTD profile in time (before or after) and matched to an exact
        CTD profile depth bin if available.
        If no exact depth bin is available. This bottle will be ignored from this step.
        2. Unmatched bottles will then be matched to the closest profile and closest depth
        bin as long as the difference between the bottle and the matched CTD depth bin is within the tolerance.
        3. Unmatched bottles will then be matched to the closest CTD depth bin available within
        the considered time range as long as the difference in depth between the two remains within the tolerance.
        4. Bottle data will be not matched to any CTD data.
    """
    def _within_depth_tolerance(df, bottle_depth, ctd_depth, depth_tolerance_range, depth_tolerance_ratio):
        dD = (df[ctd_depth] - df[bottle_depth]).abs()
        return (dD < depth_tolerance_range) | ((dD.div(df[bottle_depth]) - 1).abs() < depth_tolerance_ratio)

    # Get Matching CTD
    # Get the list of event_pk for a specific site
    # Get Time Range around Sample Collection time
    if df_ctd is None:
        # Download data from the hakai api
        station_names = df_bottles[bottle_station_id].unique()
        if len(station_names) != 1:
            raise RuntimeError('More than one station is given for matching bottle and CTD data.')

        start_time_range = (df_bottles['collected'].min() - time_range).strftime("%Y-%m-%d")
        end_time_range = (df_bottles['collected'].max() + time_range).strftime("%Y-%m-%d")

        # Get from Hakai API data, CTD data collected for a specific site over a time range predefined by the user
        filter_url = 'station=' + station_names[0] + '&start_dt>' + \
                     start_time_range + '&start_dt<' + end_time_range + '&limit=-1&distinct'
        # Find closest Cast PK associated to a profile
        df_ctd, url, ctd_metadata = get_hakai_data(hakai_ctd_endpoint, filter_url)

    # If any profile is available merge it to the bottle data
    if len(df_ctd) > 0:
        # Add CTD_ prefix to CTD variables
        df_ctd = df_ctd.add_prefix(ctd_prefix)
        ctd_depth = ctd_prefix + ctd_depth
        ctd_time = ctd_prefix + ctd_time
        ctd_profile_id = ctd_prefix + ctd_profile_id
        ctd_station_id = ctd_prefix + ctd_station_id

        # Define time and  depth_bin matching variables and make sure it's the same format.
        df_ctd['matching_depth'] = df_ctd[ctd_depth].div(bin_size).round().astype('int64')
        df_bottles['matching_depth'] = df_bottles[bottle_depth].div(bin_size).round().astype('int64')
        df_bottles['matching_time'] = df_bottles[bottle_time]
        df_ctd['matching_time'] = df_ctd[ctd_time]

        # Find closest profile with the exact same depth
        df_bottles_closest_time_depth = pd.merge_asof(df_bottles.sort_values('matching_time'),
                                                      df_ctd.sort_values(['matching_time']),
                                                      on='matching_time',
                                                      left_by=[bottle_station_id, 'matching_depth'],
                                                      right_by=[ctd_station_id, 'matching_depth'],
                                                      allow_exact_matches=True, direction='nearest')

        # Retrieve bottle data with matching depths and remove those not matching from df_bottles
        in_tolerance = _within_depth_tolerance(df_bottles_closest_time_depth, ctd_depth, bottle_depth,
                                               depth_tolerance_range, depth_tolerance_ratio)
        df_not_matched = df_bottles_closest_time_depth[in_tolerance == False][df_bottles.columns]
        df_bottles_matched = df_bottles_closest_time_depth[in_tolerance]

        # First try to retrieve to the closest profile in time and then closest depth
        if len(df_not_matched) > 0:
            df_bottles_time = pd.merge_asof(df_not_matched.sort_values('matching_time'),
                                            df_ctd.sort_values(['matching_time'])[
                                                [ctd_station_id, ctd_profile_id, 'matching_time']],
                                            on='matching_time',
                                            left_by=[bottle_station_id],
                                            right_by=[ctd_station_id],
                                            allow_exact_matches=True, direction='nearest')
            df_bottles_time = pd.merge_asof(df_bottles_time.sort_values('matching_depth'),
                                            df_ctd.sort_values(['matching_depth']),
                                            on='matching_depth',
                                            by=ctd_profile_id,
                                            suffixes=('','_ctd'),
                                            allow_exact_matches=True, direction='nearest')
            # Verify if matched data is within tolerance
            in_tolerance = _within_depth_tolerance(df_bottles_time, ctd_depth, bottle_depth,
                                                   depth_tolerance_range, depth_tolerance_ratio)
            df_not_matched = df_bottles_time[in_tolerance == False][df_bottles.columns]
            df_bottles_matched = pd.concat([df_bottles_matched, df_bottles_time[in_tolerance]])

        # Then try to match whatever closest depth sample depth within the allowed time range
        if len(df_not_matched) > 0:
            df_bottles_depth = pd.merge_asof(df_not_matched.sort_values('matching_depth'),
                                             df_ctd.sort_values(['matching_depth']),
                                             on='matching_depth',
                                             left_by=[bottle_station_id],
                                             right_by=[ctd_station_id],
                                             suffixes=('', '_ctd'),
                                             allow_exact_matches=True, direction='nearest')
            in_tolerance = _within_depth_tolerance(df_bottles_depth, ctd_depth, bottle_depth,
                                                   depth_tolerance_range, depth_tolerance_ratio)
            df_not_matched = df_bottles_depth[in_tolerance == False][df_bottles.columns]
            df_bottles_matched = pd.concat([df_bottles_matched, df_bottles_depth[in_tolerance]])

        # Finally, keep unmatched bottle data with no possible match
        if len(df_not_matched) > 0:
            df_bottles_matched = pd.concat([df_bottles_matched, df_not_matched])


        # selected_cast_pks = df_bottles['ctd_cast_pk'].unique()
        #
        # # Get the corresponding data from the API (ignore CTD with depth flagged: Seabird Flag is -9.99E-29)
        # filter_url = 'ctd_cast_pk={' + ','.join(map(str, selected_cast_pks)) + \
        #              '}&direction_flag=d&depth!=-9.99E-29&limit=-1'
        # df_ctd_profile, url, ctd_metadata = get_hakai_data(hakai_ctd_endpoint, filter_url)
        #
        # # Rename add CTD_ prefix to variables
        # df_ctd_profile = df_ctd_profile.add_prefix('CTD_')
        # ctd_metadata = ctd_metadata.add_prefix('CTD*')
        #
        # # Sort by depth both dataFrames before merging them
        # df_ctd_profile = df_ctd_profile.sort_values(['CTD_depth'])
        # df_bottles = df_bottles.sort_values(['sample_matching_depth'])
        #
        # # Make sure that the sample matching depth is same type as CTD.depth
        # df_bottles['sample_matching_depth'] = df_bottles['sample_matching_depth'].astype('float64')
        # # Match CTD with Samples
        # df_with_matched_ctd = pd.merge_asof(df_bottles, df_ctd_profile,
        #                                     left_on='sample_matching_depth', right_on='CTD_depth',
        #                                     left_by='ctd_cast_pk', right_by='CTD_ctd_cast_pk', direction='nearest')
        #
        # # TODO add range limit in meters that depth values can be with be matched with.
        # #  Could be done after the fact with a flag based on percentage of the depth
        # # TODO add another step for if CTD from closest drop in time do not reach the maximum sampled depth. Could look
        # #  at other profiles completed within the time interval
    else:
        # If no CTD data available just give back the bottle data still.
        df_bottles_matched = df_bottles

    return df_bottles_matched, ctd_metadata


def create_bottle_netcdf(event_pk, format_dict):
    print('Collect Sample data for event pk: [' + str(event_pk) + ']')
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
    variables_to_ignore = [var for var in df_bottles.columns
                           if re.search('|'.join(format_dict['ignored_variable_list']), var)]
    df_bottles.drop(variables_to_ignore, axis='columns', inplace=True)

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
    create_bottle_netcdf(pk_list['event_pk'].unique().tolist(), format_dict)

    ## Loop through each separate event_pk
    # for event_pk in pk_list['event_pk'].unique():
    #    create_bottle_netcdf(event_pk, format_dict)


def get_hakai_variable_order(format_dict):
    print('Retrieve Hakai''s variable order')
    variables_order = []
    for ii in format_dict['endpoint_list']:
        sample_type_name = get_prefix_name_from_hakai_endpoint_url(format_dict['endpoint_list'][ii]['endpoint'])
        df, url, meta = get_hakai_data(format_dict['endpoint_list'][ii]['endpoint'], 'limit=1')
        variables_order.extend(sample_type_name + '_([a-zA-Z0-9_]*_){0,1}' + df.columns)

    return variables_order
