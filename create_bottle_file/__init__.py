import datetime as dt
import json
import re

import numpy as np
import pandas as pd
from hakai_api import Client

from create_bottle_file import hakai
from create_bottle_file import transform
from create_bottle_file import erddap_output

#####################################################################################
# Transformation tools to apply on Pandas DataFrames





##################







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
#site_name = 'QU39'
#event_pk = 416
#event_pk = 3983709
#event_pk = 516386
#event_pk = 504
#event_pk = 466112
#event_pk = 3098
#hakai.create_bottle_netcdf(event_pk)
#get_site_netcdf_files(site_name)


#TODO Remove the lat/long values for each sample type
#TODO Work on order of the samples

#TODO review why there's only a few duplicates, can't be really in the data itself
#TODO min/max values should apply only to data with replicates