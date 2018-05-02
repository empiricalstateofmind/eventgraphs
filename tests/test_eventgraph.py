import unittest
from unittest import TestCase

import env
import data
from eventgraphs import EventGraph, BadInputError

import pandas as pd
from pandas.testing import assert_frame_equal

DATASETS = [data.directed, 
			data.directed_hyper, 
			data.directed_hyper_single, 
			data.undirected_hyper,
			data.extra_columns,
			data.string_labels]

class IOTests(TestCase):
	"""
	Tests the input and output functionality of EventGraph class.
	"""

	def from_pandas(self, dataset):
		""""""
		df = pd.DataFrame(dataset)
		EG = EventGraph.from_pandas_eventlist(events=df,
											  graph_rules='teg')
		self.assertTrue(True)

	def from_dict(self, dataset):
		""""""
		EG = EventGraph.from_dict_eventlist(events=dataset,
									  		graph_rules='teg')
		self.assertTrue(True)

	def from_save_and_load(self, dataset):
		""""""

		EG = EventGraph.from_dict_eventlist(events=dataset,
									  		graph_rules='teg')
		EG.build()
		EG.save('test.json')
		LG = EventGraph.from_file('test.json')
		# Cannot compare dataframes where columns are in different order
		LG.eg_edges = LG.eg_edges[EG.eg_edges.columns] 

		assert_frame_equal(LG.events, EG.events)
		assert_frame_equal(LG.eg_edges, EG.eg_edges)
		assert_frame_equal(LG.events_meta, EG.events_meta)
		self.assertTrue(hasattr(LG, '_event_pair_processed'))

	def test_from_dict(self):
		""" """
		for dataset in DATASETS:
			self.from_dict(dataset)

	def test_from_pandas(self):
		""" """
		for dataset in DATASETS:
			self.from_pandas(dataset)

	def test_from_save_and_load(self):
		""" """
		for dataset in DATASETS:
			self.from_save_and_load(dataset)

	def test_bad_input(self):
		""""""
		with self.assertRaises(BadInputError):
			EG = EventGraph(events=['abc','cba'],
							graph_rules='teg')
		


class BuildTests(TestCase):
	"""
	"""
	pass


class MatrixMethodTests(TestCase):
	"""
	Tests the methods associated with building matrices from data.
	"""
	pass


class FilterTests(TestCase):
	"""
	"""
	pass


class ComponentsTests(TestCase):
	"""
	Tests functionality of component decomposition.
	"""

	def setUp(self):
		self.EG = EventGraph.from_dict_eventlist(events=data.string_labels,
											graph_rules='teg')
		self.EG.build()

	def tearDown(self):
		self.EG = None

	def test_connected_components(self):
		components = self.EG.connected_components()
		
if __name__ == '__main__':
    unittest.main()
