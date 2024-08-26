# Hakai Bottle Tools
The hakai bottle tool join together sample and ctd profile data collected by the Hakai Institute and available from the following endpoints within the [Hakai API](https://github.com/HakaiInstitute/hakai-api):

```python
{
    "eims/views/output/nutrients",
    "eims/views/output/microbial",
    "eims/views/output/hplc",
    "eims/views/output/poms",
    "eims/views/output/ysi",
    "eims/views/output/chlorophyll",
    "eims/views/output/phytoplankton",
}
```

The ctd data is retrieved from the API endpoint:

```python
"ctd/views/file/cast/data"
```

## Installation

You can install the package locally by running for the following command:

```shell
pip install git+https://github.com/HakaiInstitute/hakai-bottle-tool.git
```

You however don't need to install necessarily the package and just use the
following jupyter notebook on google colab
[here](https://colab.research.google.com/github/HakaiInstitute/hakai-bottle-tool/blob/master/run_hakai_bottle_tool.ipynb).

## How To

The hakai-bottle-tool can either be run on the [google colab jupyter notebook](https://colab.research.google.com/github/HakaiInstitute/hakai-bottle-tool/blob/master/run_hakai_bottle_tool.ipynb) or, if installed locally, by running the following command:

```console
> python hakai_bottle_tool -station QU39 -time_min 2020-01-01 -time_max 2021-01-01
```

## Method

Each sample type is first groupby  `site_id`, `event_pk`, `line_out_depth`
and `collected` time (± 5 minutes) and aggregated by mean (numerical),
comma seperated joined strings for strings, count(see _nReplicates),
and difference between min and max for numerical values.

All sample type then is joined together by matching `site_id`, `event_pk`,
`line_out_depth` and `collected` time (± 5 minutes).

Once all the sample data available. The corresponding CTD profile data collected
over the corresponding time period and station is downloaded and merged to the
bottle data by using the following sequence:

1. Bottle data is matched to the **nearest CTD profile within 3 hours of
the collected time and matched to an exact binned depth**  if available.
If no exact binned depth is available, this bottle will be ignored from this step.
2. Unmatched bottles are then matched to the **nearest profile and depth**
within the depth tolerance.
3. Unmatched bottles are then matched to **any CTD collected at taht station
within the last day and at the nearest depth** within the tolerance
4. Unmatched bottle data left remained unmatched to any CTD data.

A sample is considered within the depth tolerance if the following condition is respected:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{150}&space;\bg_white&space;\inline&space;|D_{ctd}-D_{bottle}|&space;<&space;3m&space;\;&space;or&space;\;&space;|\frac{D_{ctd}}{D_{bottle}}-1|<&space;15%" title="\bg_white \inline |D_{ctd}-D_{bottle}| < 3m \; or \; |\frac{D_{ctd}}{D_{bottle}}-1|< 15%" />
</p>
<p align="right"><em>
where D<sub>ctd</sub> and D<sub>bottle</sub> corresponds respectively to the CTD and bottle measurements associated depths.
</em></p>
