from create_bottle_file import hakai
from create_bottle_file import erddap_output
import glob


# List all information needed for reference and creating standard bottle files
# Endpoint list to get data from

def get_hakai_bottle_processing_list(process_list={}):
    process_list['endpoint_list'] = {
        1: {'endpoint': 'eims/views/output/nutrients'},
        2: {'endpoint': 'eims/views/output/microbial', 'pivot_variable': 'microbial_sample_type'},
        3: {'endpoint': 'eims/views/output/hplc'},
        4: {'endpoint': 'eims/views/output/o18'},
        5: {'endpoint': 'eims/views/output/poms', 'pivot_variable': 'acidified'},
        6: {'endpoint': 'eims/views/output/ysi'},
        7: {'endpoint': 'eims/views/output/chlorophyll', 'pivot_variable': 'filter_type'},
    }

    # List of variables to use as index for matching of the different sample data sets
    process_list['index_variable_list'] = ['organization', 'work_area', 'site_id', 'event_pk', 'collected',
                                           'line_out_depth']

    # List of columns to index, associate with Metadata, and ignore
    process_list['meta_variable_list'] = ['organization', 'work_area', 'site_id', 'date', 'survey',
                                          'pressure_transducer_depth', 'line_out_depth']
    process_list['meta_variable_list'] = []

    # List of variables to ignore from the sample data
    process_list['ignored_variable_list'] = ['action', 'rn', 'sampling_bout', 'analyzed', 'preserved', 'technician',
                                             'lab_technician', 'source', 'volume', 'dna_volume_te_wash',
                                             'before_acid', 'after_acid', 'calibration_slope',
                                             'acid_ratio_correction_factor', 'acid_coefficient', 'acetone_volumne_ml',
                                             'fluorometer_serial_no', 'calibration', 'analyzing_lab']

    # Variable list to ignore from the CTD data
    process_list['ctd_variable_list_to_ignore'] = ['device_firmware', 'file_processing_stage', 'shutoff', 'ctd_file_pk',
                                                   'ctd_data_pk', 'v_main', 'v_lith', 'i_pump', 'i_ext01', 'i_ext2345',
                                                   'cast_processing_stage', 'status', 'duration', 'start_depth',
                                                   'bottom_depth', 'target_depth', 'drop_speed', 'vessel', 'operators',
                                                   'pruth_air_pressure_before_cast', 'min_pressure_before_cast',
                                                   'min_depth_before_cast', 'min_pressure_after_cast',
                                                   'min_depth_after_cast', 'estimated_air_pressure',
                                                   'estimated_depth_shift', 'original_start_dt', 'original_bottom_dt',
                                                   'original_start_depth', 'original_bottom_depth', 'spec_cond',
                                                   'spec_cond_flag', 'oxygen_voltage',
                                                   'oxygen_voltage_flag', 'cast_number']
    # add CTD_ to ctd variables
    process_list['ctd_variable_list_to_ignore'] = ['CTD_' + x for x in process_list['ctd_variable_list_to_ignore']]

    process_list['ignored_variable_list'] = ['action$', 'rn$', 'sampling_bout$']

    # regex keys used to identify the time variables each item is then separated by | in the regex query that data gets
    # converted to datetime objects
    process_list['time_variable_list'] = ['date', 'collected', 'preserved', 'analyzed', '_dt', 'time']

    # regex keys used to identify text columns from the different data sets
    process_list['string_columns_regexp'] = ['flag', 'comments', 'hakai_id', 'survey', 'method', 'organization',
                                             'site_id', 'work_area', 'quality_level', 'quality_log', 'row_flag',
                                             'serial_no', 'filename', 'device_sn', 'cruise', 'filter_type', 'units',
                                             'project_specific_id', 'station', 'device_model']

    # List of Expressions to rename from the different variable names
    process_list['rename_variables_dict'] = {'poms_True': 'poms_Acidified', 'poms_False': 'poms_nonAcidified'}

    # Variable order at the end
    process_list['variables_final_order'] = ['bottle_profile_id', 'organization', 'work_area', 'site_id', 'latitude',
                                             'longitude', 'gather_lat', 'gather_long', 'event_pk', 'time', 'depth',
                                             'collected', 'line_out_depth', 'pressure_transducer_depth',
                                             'matching_depth']
    process_list['cdm_variables'] = ['organization', 'work_area', 'site_id', 'collected', 'event_pk']
    return process_list


# Test Variables
site_name = 'QU39'
event_pk = 416
event_pk = 3983709
event_pk = 516386
event_pk = 504
event_pk = 466112
event_pk = 3098
event_pk = 504
event_pk = 30471
event_pk = 65094
event_pk = 185115
event_pk = 296416
event_pk = 348755
event_pk = 343710
event_pk = 68551
event_pk = 363208
# event_pk = 1733

event_pk = 137035
event_pk = 3983709
event_pk = 4600473
event_pk = [3403465]
event_pk = [179830]
path_out = r'E:\temp\bottle'
# hakai.create_bottle_netcdf(event_pk, get_hakai_bottle_processing_list())
# hakai.get_site_netcdf_files(site_name, get_hakai_bottle_processing_list())
# hakai.get_site_netcdf_files('QU5', get_hakai_bottle_processing_list(), path_out=path_out)
hakai.get_site_netcdf_files('QU39', get_hakai_bottle_processing_list(), path_out=path_out)
#hakai.get_site_netcdf_files('DFO2', get_hakai_bottle_processing_list(), path_out=path_out)
#hakai.get_site_netcdf_files('FZH01', get_hakai_bottle_processing_list(), path_out=path_out)
# hakai.get_site_netcdf_files('KC10', get_hakai_bottle_processing_list(), path_out=path_out)
# hakai.get_site_netcdf_files('QU43', get_hakai_bottle_processing_list(), path_out=path_out)
# hakai.get_site_netcdf_files('QCS01', get_hakai_bottle_processing_list(), path_out=path_out)
# hakai.get_site_netcdf_files('ROCKY06', get_hakai_bottle_processing_list(), path_out=path_out)
# local_file_list = glob.glob('*.nc')
# variable_order = hakai.get_hakai_variable_order(get_hakai_bottle_processing_list())
# erddap_output.create_combined_variable_empty_netcdf(local_file_list, variable_order)

# erddap_output.compare_netcdf(local_file_list, 'METADATA_NETCDF_FOR_DATASETS.nc')

hakai.get_site_netcdf_files('ROCKY04', get_hakai_bottle_processing_list())

# hakai.get_site_netcdf_files('QU39', get_hakai_bottle_processing_list(), start_time='2020-01-01')
