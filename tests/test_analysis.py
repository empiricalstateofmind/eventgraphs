import unittest
from unittest import TestCase

import env
import data
from eventgraphs import EventGraph
from eventgraphs.analysis import *

import pandas as pd
from pandas.testing import assert_frame_equal

class CompletitionTests(TestCase):
    """
    Tests that each function runs (not currently testing output).
    """

    def setUp(self):
        self.EG = EventGraph.from_file("./tests/test.json")

    def tearDown(self):
        self.EG = None

    def test_calculate_iet_distribution(self):
        """"""

        result = calculate_iet_distribution(self.EG)
        self.assertTrue(isinstance(result, pd.Series))

        with self.assertRaises(KeyError):
            result = calculate_iet_distribution(self.EG, by_motif=True)
        
        result = calculate_iet_distribution(self.EG, normalize=True)
        self.assertAlmostEqual(result.sum(), 1.0)

        result = calculate_iet_distribution(self.EG, 
                                            cumulative=True,
                                            bins=[0,10,20,30,40,50])
        self.assertTrue(len(result)==6)

    def test_component_distribution(self):
        """"""

        result = calculate_component_distribution(self.EG)


    def test_component_distribution_over_delta(self):
        """"""
        result = calculate_component_distribution_over_delta(self.EG,
                                                             delta_range=[0,10,20,30,40])


    def test_iet_entropy(self):
        """"""

        result = calculate_iet_entropy(self.EG)

    def test_calculate_activity(self):
        """"""

        result = calculate_activity(self.EG, rescale=False)

    def test_cluster_timeseries(self):
        """"""

        with self.assertRaises(Exception):
            result = calculate_cluster_timeseries(self.EG, [0,10,20,30,40])