# Hakai Bottle Tools
The hakai bottle retrieve sample data collected by the Hakai Institute and available from the following endpoints:

```python 
    "eims/views/output/nutrients"
    "eims/views/output/microbial"
    "eims/views/output/hplc"
    "eims/views/output/o18"
    "eims/views/output/poms"
    "eims/views/output/ysi"
    "eims/views/output/chlorophyll"
    "eims/views/output/doc"
```

And the ctd data from the endpoint: `ctd/views/file/cast/data`

# How to
You can install the package locally by running for the following command:
```
pip install git+
```

# Method
Each sample type is first groupby  `site_id`, `event_pk`, `line_out_depth` and `collected` time (± 5 minutes) and aggregated by mean (numerical), comma seperated joined strings for strings, count(see _nReplicates), and difference between min and max for numerical values.

All sample type then is joined together by matching `site_id`, `event_pk`, `line_out_depth` and `collected` time (± 5 minutes).

Once all the sample data available. The corresponding CTD profile data collected over the corresponding time period and station is downloaded and merged to the bottle data by using the following sequence:
1. Bottle data is matched to the **nearest CTD profile within 3 hours of the collected time and matched to an exact binned depth**  if available. If no exact binned depth is available, this bottle will be ignored from this step.
2. Unmatched bottles are then matched to the **nearest profile and depth** within the depth tolerance.
3. Unmatched bottles are then matched to **any CTD collected at taht station within the last day and at the nearest depth** within the tolerance
4. Unmatched bottle data left remained unmatched to any CTD data.

A sample is considered within the depth tolerance if:
<p align="center">
<img src="https://latex.codecogs.com/svg.image?|D_{ctd}-D_{bottle}|&space;<&space;3m&space;\;&space;or&space;\;&space;|\frac{D_{ctd}}{D_{bottle}}-1|<&space;15%" title="|D_{ctd}-D_{bottle}| < 3m \; or \; |\frac{D_{ctd}}{D_{bottle}}-1|< 15%" />
</p>
