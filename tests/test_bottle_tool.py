import unittest

from hakai_bottle_tool import hakai_bottle_tool


class TestBottleMatchingTool(unittest.TestCase):
    def test_qu39_nov2016_matching(self):
        df = hakai_bottle_tool.get_bottle_data("QU39", "2016-10-11", "2017-01-01")
        assert not df.empty, "failed to get any data"
        assert not df.query("phytoplankton_hakai_id == 'QPHY501'").empty, "Missing phytoplankton_hakai_id = 'QPHY501'"
