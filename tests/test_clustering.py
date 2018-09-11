import unittest
from unittest import TestCase

import env
import data
from eventgraphs import EventGraph
from eventgraphs.clustering import *

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

class CompletitionTests(TestCase):
    """
    Tests that each function runs (not currently testing output).
    """

    def setUp(self):
        self.EG = EventGraph.from_file("./tests/test.json")
        self.components = self.EG.connected_components()

    def tearDown(self):
        self.EG = None

    @unittest.skip("Skipping until we have motifs in the prebuild EGs")
    def test_generate_features(self):
        """"""
        features, scale_features = generate_features(self.components, 
                                                     feature_spec=FEATURE_SPEC)

    def test_generate_distance_matrix(self):
        """"""
        A = pd.DataFrame(np.random.random(size=(20,100)))
        result = generate_distance_matrix(A, metric='euclidean')

    def test_generate_linkage(self):
        """"""
        A = pd.DataFrame(np.random.random(size=(20,100)))
        distance = generate_distance_matrix(A, metric='euclidean')
        result = generate_linkage(distance, kind='ward')