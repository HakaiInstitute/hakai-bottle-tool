# Define each sample type endpoint and the need transformations needed per endpoint
SAMPLES = {
    "eims/views/output/nutrients": {
        "query_filter": "&(no2_no3_flag!=SVD|no2_no3_flag=null)&(po4_flag!=SVD|po4_flag=null)&(sio2_flag!=SVD|sio2_flag=null)"
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
    "eims/views/output/phytoplankton": {},
}

CTD_DATA = "ctd/views/file/cast/data"
