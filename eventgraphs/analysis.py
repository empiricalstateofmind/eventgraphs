import numpy as np
import pandas as pd
from scipy.sparse import linalg as spla
from scipy.sparse import csgraph as csg

from copy import deepcopy

def calculate_iet_distribution(eventgraph, by_motif=False, normalize=True, cumulative=False, bins=None):
	""" 
	Calculate the inter-event time distribution for an event graph. 
	

	"""

	if by_motif:
		store = {}
		for motif, edges in eventgraph.eg_edges.groupby('motif'):
			iets = edges.delta.value_counts(normalize=normalize).sort_index()
			if cumulative:
				iets = iets.cumsum()
			if bins is not None:
				iets = iets.reindex(bins, method='nearest')
			store[motif] = iets
		iets = store


	else:
		iets = eventgraph.eg_edges.delta.value_counts(normalize=normalize).sort_index()
		if cumulative:
			iets = iets.cumsum()
		if bins is not None:
			iets = iets.reindex(bins, method='nearest')

	return iets


def calculate_motif_distribution(eventgraph, normalize=True):
	"""
	Calculate the motif distributon of an event graph

	"""
	return eventgraph.eg_edges.motif.value_counts(normalize=normalize)


def calculate_component_distribution(eventgraph, normalize=True, cumulative=False, bins=None):
	"""
	"""
	if 'component' not in eventgraph.events_meta.columns:
		eventgraph._generate_eg_matrix()

	component_ixs = csg.connected_components(eventgraph.eg_matrix, 
										  directed=True,
										  connection='weak',
										  return_labels=True)[1]

	component_dist = pd.Series(component_ixs).value_counts().value_counts(normalize=normalize).sort_index()
	if cumulative:
		component_dist = component_dist.cumsum()
	if bins is not None:
		component_dist = component_dist.reindex(bins, method='nearest')

	return component_dist

def calculate_component_distribution_over_delta(eventgraph, delta_range, normalize=True):
	""" 
	Calculates the component size distribution (# events) over a range of dt values.

	dt range must be less than that of the eventgraph.
	"""

	if hasattr(eventgraph,'eg_matrix'):
		eg_matrix = deepcopy(eventgraph.eg_matrix)
	else:
		eg_matrix = deepcopy(eventgraph._generate_eg_matrix())

	largest_component = {}
	component_distributions = {}
	for dt in delta_range[::-1]:
		eg_matrix.data = np.where(eg_matrix.data < dt, eg_matrix.data, 0)
		eg_matrix.eliminate_zeros()
		component_ixs = csg.connected_components(eg_matrix, 
										  directed=True,
										  connection='weak',
										  return_labels=True)[1]
		components = pd.Series(component_ixs).value_counts().value_counts(normalize=normalize).sort_index()
		component_distributions[dt] = components
		largest_component[dt] = components.index.max()
	largest_component = pd.Series(largest_component)

	return component_distributions, largest_component