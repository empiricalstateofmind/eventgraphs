import numpy as np

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