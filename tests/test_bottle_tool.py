import pytest

from hakai_bottle_tool import hakai_bottle_tool


@pytest.mark.parametrize(
    "station,date_start,date_end,query",
    [
        ("QU39", "2016-10-11", "2017-01-01", "phytoplankton_hakai_id == 'QPHY501'"),
    ],
)
def test_station_match(station, date_start, date_end, query):
    df = hakai_bottle_tool.get_bottle_data(
        station,
        date_start,
        date_end,
    )
    assert not df.empty, "failed to get any data"
    assert not df.query(query).empty, f"No match found for query: {query}"


@pytest.mark.parametrize(
    "station,date_start,date_end,query",
    [
        ("QU39", "2016-10-11", "2017-01-01", "phytoplankton_hakai_id == 'QPHY501'"),
    ],
)
def test_station_match_by_line_out_depth(station, date_start, date_end, query):
    df = hakai_bottle_tool.get_bottle_data(
        station, date_start, date_end, bottle_depth_variable="line_out_depth"
    )
    assert not df.empty, "failed to get any data"
    assert not df.query(query).empty, f"No match found for query: {query}"
    assert (df["matching_depth"] == df["line_out_depth"]).all()
