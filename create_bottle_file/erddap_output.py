import re
import numpy as np
import pandas as pd
from hakai_api import Client

from create_bottle_file import transform


def convert_bottle_data_to_xarray(df,
                                  netcdf_file_name,
                                  metadata_for_xarray,
                                  format_dict):
    # Standardize the data types
    # Text data to Strings
    df = transform.standardize_object_type(df, format_dict['string_columns_regexp'], '|S', np.nan, '')

    # Date time objects to datetime64s UTC
    df, converted_time_variables = transform.convert_columns_to_datetime(df, format_dict['time_variable_list'])

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
