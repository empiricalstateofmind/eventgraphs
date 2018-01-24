import unittest
from unittest import TestCase

import env
import data
from eventgraphs import EventGraph, BadInputError

class InputOutputTests(TestCase):
	"""
	Tests the input and output functionality of EventGraph class.
	"""

	# Run only if pandas is installed?
	def test_from_pandas(self):
		""""""
		import pandas as pd # Really want to remove the pandas dependency
		df = pd.DataFrame(data.directed)
		EG = EventGraph.from_pandas_eventlist(events=df,
											  graph_rules='teg')
		self.assertTrue(True)

	def test_from_dict(self):
		""""""
		EG = EventGraph.from_dict_eventlist(events=data.directed,
									  		graph_rules='teg')
		self.assertTrue(True)


	def test_from_save_and_load(self):
		""""""

		EG = EventGraph.from_dict_eventlist(events=data.directed,
									  		graph_rules='teg')
		EG._build()
		EG.save('test.json')
		LG = EventGraph.from_file('test.json')
		# Cannot compare dataframes where columns are in different order
		LG.eg_edges = LG.eg_edges[EG.eg_edges.columns] 

		self.assertTrue((LG.events == EG.events).all().all())
		self.assertTrue((LG.eg_edges == EG.eg_edges).all().all())
		self.assertTrue((LG.events_meta == EG.events_meta).all().all())
		self.assertTrue(hasattr(LG, 'event_pair_processed'))

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
		self.EG._build()

	def tearDown(self):
		self.EG = None

	def test_connected_components(self):
		components = self.EG.connected_components()
		
if __name__ == '__main__':
    unittest.main()
