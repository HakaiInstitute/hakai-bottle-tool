import re
import numpy as np
import pandas as pd
import xarray as xr

from create_bottle_file import transform


def convert_bottle_data_to_xarray(df,
                                  netcdf_file_name,
                                  metadata_for_xarray,
                                  format_dict):

    # Work on a copy of the initial dataframe
    df_formatted = df.copy()

    # Standardize the data types
    # Text data to Strings
    df_formatted = transform.standardize_object_type(df_formatted, df_formatted.select_dtypes('object').columns, '|S',
                                                    np.nan, '')

    # Convert DatetimeTZ objects to datetime64s UTC with no time zone. Xarray can't handle them.
    for time_variable in df_formatted.select_dtypes(['datetimetz']).columns:
        df_formatted[time_variable] = pd.to_datetime(df_formatted[time_variable], utc=True).dt.tz_localize(None)

    print('Convert DataFrame to Xarray')
    # Convert to a xarray
    ds = df_formatted.to_xarray()

    # Add Metadata to xarray
    ds = add_metadata_to_xarray(ds, metadata_for_xarray, df)

    #TODO add documentation to data. Specify all the variable attributes.

    # Save xarray to netcdf
    print('Save to '+netcdf_file_name)
    ds.to_netcdf(netcdf_file_name)
    return ds


def add_metadata_to_xarray(ds, metadata, df):
    # Give standard attributes to time variables

    # Datetime variables
    for time_variable in df.select_dtypes('datetime').columns:
        if time_variable in ds:
            ds[time_variable].encoding['units'] = 'seconds since 1970-01-01T00:00:00'

    # Datetimetz variables
    for time_variable in df.select_dtypes('datetimetz').columns:
        if time_variable in ds:
            ds[time_variable].encoding['units'] = 'seconds since 1970-01-01T00:00:00Z'
            ds[time_variable].attrs['timezone'] = 'UTC'

    # timedelta variables
    for time_variable in df.select_dtypes('timedelta').columns:
        if time_variable in ds:
            ds[time_variable].encoding['units'] = 'seconds'

    # Loop through all the metadata variable listed and find any matching ones in the xarray
    for keys in metadata.columns:
        searchKey = keys.replace('*', '.*?')+'($|_min|_max|_range)'  # Consider values and min/max columns have the metadata
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


def create_combined_variable_empty_netcdf(file_list, variable_order):
    print('Combine NetCDF File to Full Variable List')
    initialize = True

    for file in file_list:
        print(file)
        # Read the netcdf file with xarray
        if initialize:
            # Read initial file
            ds = xr.open_dataset(file)
            # Crop to keep just the first line
            depth_mask = ds.depth == ds.depth[0]
            ds_meta = ds.where(depth_mask, drop=True)

            # Unflag initialize step
            initialize = False
        else:
            ds = xr.open_dataset(file)
            ds_mask = ds.depth == -1
            ds_mask[0] = True
            ds = ds.where(ds_mask, drop=True)
            ds_meta_temp = ds_meta.merge(ds, compat='identical')

            # Just keep the first depth, we assume it's a 1d dataset
            depth_mask = ds_meta_temp.depth == ds_meta_temp.depth[0]
            ds_meta = ds_meta_temp.where(depth_mask, drop=True)

    # Force all datetime encodings to be seconds since 1970,1,1
    for var in ds_meta:
        if ds_meta[var].dtype == 'datetime64[ns]':
            if 'timezone' in ds_meta[var].attrs and ds_meta[var].attrs['timezone'] == 'UTC':
                ds_meta[var].encoding['units'] = 'seconds since 1970-01-01T00:00:00Z'
            else:
                ds_meta[var].encoding['units'] = 'seconds since 1970-01-01T00:00:00'
        elif ds_meta[var].dtype == 'timedelta64[ns]':
            ds_meta[var].encoding['units'] = 'seconds'

    # Sort dataset based on variable input
    ordered_variables = []
    for var in variable_order:
        #var = re.sub('\.', '_([a-zA-Z0-9_]*_){0,1}', var)
        r = re.compile(var)
        for item in list(filter(r.match, list(ds_meta.variables))):
            if item not in ordered_variables:
                ordered_variables.append(item)

    # Add CTD
    r = re.compile('CTD_.*')
    ordered_variables.extend(list(filter(r.match, list(ds_meta.variables))))

    # Items not listed should be at the beginning
    variables_with_unknown_order = []
    for var in ds_meta.variables:
        if var not in ordered_variables:
            variables_with_unknown_order.append(var)

    # Reorder dataset
    final_order = variables_with_unknown_order + ordered_variables
    ds_meta_out = ds_meta[final_order]

    ds_meta_out.to_netcdf('METADATA_NETCDF_FOR_DATASETS.nc')
