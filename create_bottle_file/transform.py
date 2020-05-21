import numpy as np
import pandas as pd
import datetime as dt


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
    if df.index.duplicated().any():
        if variable_to_pivot in df.columns:  # Group data by pivot
            # Get pivoted data frames with the different aggregations
            df_min = pd.pivot_table(df, index=index_variable_list, columns=variable_to_pivot, aggfunc='min')
            df_max = pd.pivot_table(df, index=index_variable_list, columns=variable_to_pivot, aggfunc='max')
            df_count = pd.pivot_table(df, index=index_variable_list, columns=variable_to_pivot, aggfunc='count')

            # Combine strings
            df_str = pd.pivot_table(df, index=index_variable_list, columns=variable_to_pivot,
                                    aggfunc=lambda x: ', '.join(np.unique(x)))

            # Get average value (only numbers will be kept)
            df_mean = pd.pivot_table(df, index=index_variable_list, columns=variable_to_pivot, aggfunc='mean')

        else:  # Group data with groupby()
            # Get all the grouped stats
            df_min = df.groupby(by=index_variable_list).min()
            df_max = df.groupby(by=index_variable_list).max()
            df_count = df.groupby(by=index_variable_list).count()

            # Group strings of duplicates
            df_str = df.dropna(axis=1, how='all').fillna(value=''). \
                groupby(by=index_variable_list).transform(lambda x: ', '.join(np.unique(x))).drop_duplicates()

            # Get Mean values, for some reasons datetime variables aren't calculated in mean
            df_mean = df.groupby(by=index_variable_list).mean()

        # Get time variables since aggregation can't use them
        time_variables = set(df_min.dropna(axis='columns').columns.tolist()) - set(df_str.columns.tolist()) - set(
            df_mean.columns.tolist())

        # Get range difference between min and max values for numbers only
        df_range_val = df_max[df_mean.columns]-df_min[df_mean.columns]
        df_range_time = df_max[time_variables] - df_min[time_variables]

        # Flag range data that has no duplicate
        df_range_val[df_count[df_range_val.columns] <= 1] = np.nan
        df_range_time[df_count[time_variables] <= 1] = pd.NaT

        # define Range and replicates columns
        df_range = df_range_val.join(df_range_time)
        df_replicates = df_count[df_range.columns]

        # Merge column names
        df_range.columns = [f'{j}_{i}' for i, j in df_range.columns]
        df_replicates.columns = [f'{j}_{i}' for i, j in df_replicates.columns]

        # Add suffix to columns
        df_range = df_range.add_suffix('_range')
        df_replicates = df_replicates.add_suffix('_nReplicates')

        # Merge Stats and sort them
        df_stats = df_range.join(df_replicates, how='outer')
        df_stats = df_stats[sorted(df_stats.columns, reverse=True)]

        # Start combining aggregated data
        df_out = df_min
        df_out[df_str.columns] = df_str
        df_out[df_mean.columns] = df_mean

        # Merge Column Names into one layer
        df_out.columns = [f'{j}_{i}' for i, j in df_out.columns]

        # Add Stats columns
        df_out = df_out.join(df_stats, how='outer')

    else:
        # No duplicated values exist just move on
        df_out = df

    # if variable_to_pivot in df.columns:
    #     # Find if there's any duplicate numerical values
    #     number_columns = [variable_to_pivot] + df.select_dtypes(['number']).columns.to_list()
    #
    #     # Count how many values are superposed
    #     df_count = pd.pivot_table(df[number_columns], index=index_variable_list, columns=variable_to_pivot,
    #                               aggfunc='count')
    #
    #     # Merge Column Names into count Data Frame to be similar to others
    #     df_count.columns = [f'{j}_{i}' for i, j in df_count.columns]
    #     # TODO leverage groups in NetCDFs by replacing separator  '_' by '.'
    # else:
    #     # Find if there's any duplicate numerical values
    #     number_columns = df.select_dtypes(['number']).columns.to_list()
    #
    #     df_count = df[number_columns].groupby(by=index_variable_list).count()

    # # Get Min, Max and Count values for replicate numbers
    # if any(df_count > 1):
    #
    #     # Combined duplicated data either with a pivot on one column or by grouping
    #     if variable_to_pivot in df.columns:
    #         df_min = pd.pivot_table(df[number_columns], index=index_variable_list, columns=variable_to_pivot,
    #                                 aggfunc='min')
    #         df_max = pd.pivot_table(df[number_columns], index=index_variable_list, columns=variable_to_pivot,
    #                                 aggfunc='max')
    #
    #         # Merge Column Names into count Data Frame to be similar to others
    #         df_min.columns = [f'{j}_{i}' for i, j in df_min.columns]
    #         df_max.columns = [f'{j}_{i}' for i, j in df_max.columns]
    #
    #     else:
    #         df_min = df[number_columns].groupby(by=index_variable_list).min()
    #         df_max = df[number_columns].groupby(by=index_variable_list).max()
    #
    #     # If there's any duplicates with variations in values, get there min/max values too
    #     if any(df_max-df_min != 0):
    #         # Find columns where there's replicates and different values
    #         columns_with_duplicates = df_min.columns[df_min.where(df_max-df_min != 0).any(axis=0)]
    #
    #         # Remove rows with no replicates
    #         df_min[df_count <= 1] = np.nan
    #         df_max[df_count <= 1] = np.nan
    #         df_count[df_count <= 1] = np.nan
    #
    #         # Add suffix to all columns to force it
    #         df_min = df_min.filter(columns_with_duplicates).add_suffix('_min')
    #         df_max = df_max.filter(columns_with_duplicates).add_suffix('_max')
    #         df_count_reduced = df_count.filter(columns_with_duplicates).add_suffix('_nReplicates')
    #
    #         # Merge stats together
    #         df_stats = df_min.join(df_max, how='outer')
    #         df_stats = df_stats.join(df_count_reduced, how='outer')
    #
    #         # Sort columns to have the similar next to each other
    #         df_stats = df_stats[sorted(df_stats.columns, reverse=True)]
    #
    #         # Merge with the data
    #         df_out = df_out.join(df_stats, how='outer')

    return df_out


def create_id_variable(df, id_name, id_variables):
    df[id_name] = df[id_variables[0]].str.cat(df[id_variables[1:]].astype(str), sep="_").astype(str)
    return df


def convert_columns_to_datetime(df, regexp_string):
    regexp_look = '|'.join(regexp_string)
    columns_to_convert = df.filter(regex=regexp_look).columns

    for item in columns_to_convert:
        if item in df.columns:
            if df[item].dtype==object:
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
    # TODO make it more general with a dictionary input
    return df
