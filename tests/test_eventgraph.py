import unittest
from unittest import TestCase

import env
import data
from eventgraphs import EventGraph

class InputOutputTests(TestCase):
	"""
	Tests the input and output functionality of EventGraph class.
	"""

	def test_from_pandas(self):
		pass

	def test_from_dict(self):
		pass

	def test_from_file(self):
		pass

	def test_bad_input(self):
		pass


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
		self.EG = EventGraph.from_dict_eventlist(events=data.directed_hyper,
											graph_rules='teg')
		self.EG._build()

	def tearDown(self):
		pass

	def test_connected_components(self):
		components = self.EG.connected_components()
		
if __name__ == '__main__':
    unittest.main()
