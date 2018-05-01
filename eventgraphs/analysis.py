import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import linalg as spla
from scipy.sparse import csgraph as csg

from copy import deepcopy

def calculate_iet_distribution(eventgraph, by_motif=False, normalize=True, cumulative=False, bins=None):
	""" 
	Calculate the inter-event time distribution for an event graph. 
	
	Input:

	Returns:
		None
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

	Input:

	Returns:
		None
	"""
	return eventgraph.eg_edges.motif.value_counts(normalize=normalize)


def calculate_component_distribution(eventgraph, normalize=True, cumulative=False, bins=None):
	"""


	Input:

	Returns:
		None
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

	Input:

	Returns:
		None
	"""

	if hasattr(eventgraph,'eg_matrix'):
		eg_matrix = deepcopy(eventgraph.eg_matrix)
	else:
		eg_matrix = deepcopy(eventgraph._generate_eg_matrix())

	largest_component = {}
	component_distributions = {}
	for dt in delta_range[::-1]:
		eg_matrix.data = np.where(eg_matrix.data <= dt, eg_matrix.data, 0)
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

def calculate_motif_entropy(eventgraph, normalize=False):
	""" 
	Calculate the motif entropy 

	Input:

	Returns:
		None
	"""
	motifs = calculate_motif_distribution(eventgraph)
	motif_entropy = -sum([p*np.log(p) for p in motifs.values if p>0])
	return motif_entropy

def calculate_iet_entropy(eventgraph):
	"""


	Input:

	Returns:
		None
	"""
	iets = calculate_iet_distribution(eventgraph, cumulative=True)
	max_iet = max(iets.index)
	if max_iet > 0:
		iets.index = [i/max_iet for i in iets.index]
	divisions = 10000
	iets = iets.reindex(np.arange(0,1,divisions), method='nearest')
	iet_entropy = -sum([(1/(divisions-1))*val*np.log2(val) for val in iets.values if val !=0 ])
	return iet_entropy

def calculate_activity(eventgraph, unit=1):
	""" 

	Input:

	Returns:
		None
	"""
	duration = eventgraph.D
	if duration==0: 
		activity = np.inf
	else:
		activity = (len(eventgraph.events)/duration)*unit
	return activity

def calculate_edge_density(G):
	""" 


	Input:

	Returns:
		None
	"""
	N = len(G.nodes())
	if N > 1:
		return len(G.edges())/(N*(N-1))
	else:
		return 0

def calculate_clustering_coefficient(G):
	"""

	Input:

	Returns:
		None
	"""

	N = len(G.nodes())
	if N > 1:
		recip = G.to_undirected(reciprocal=False)
		clustering = nx.cluster.average_clustering(recip)
		return clustering
	else:
		return 0

def calculate_reciprocity_ratio(G):
	""" 

	Input:

	Returns:
		None
	"""
	
	N = len(G.nodes())
	if N > 1:
		recip = G.to_undirected(reciprocal=True)
		recip_ratio = 2*len(recip.edges())/len(G.edges())
		return recip_ratio
	else:
		return 0

def calculate_degree_assortativity(G):
	""" 
	Calculates a 'fake' degree assortivitity. To be ironed out as a concept. 

	Input:

	Returns:
		None
	"""

	N = len(G.nodes())
	if N <= 1:
		return {'assort_ii':0,
				'assort_io':0,
				'assort_oi':0,
				'assort_oo':0,}

	degrees = pd.DataFrame([(G.out_degree(a), G.out_degree(b), G.in_degree(a), G.in_degree(b)) for a,b in G.edges()],
						   columns=['o_source', 'o_target', 'i_source', 'i_target'])
	assorts = {}
	for alpha, beta in [('o','o'), ('o','i'), ('i','i'), ('i','o')]:
		c1 = '{}_source'.format(alpha)
		c2 = '{}_target'.format(beta)

		x = (degrees[c1]-degrees[c2]).mean()
		if x == 0:
			assorts['assort_{}{}'.format(alpha,beta)] = 0.0
		else:
			assorts['assort_{}{}'.format(alpha,beta)] = x/(degrees[c1]-degrees[c2]).abs().max()
			
	return assorts

def calculate_cluster_timeseries(eventgraph, interval_width):
	"""

	Input:

	Returns:
		None	
	"""

	if 'cluster' not in eventgraph.events_meta.columns:
		raise Exception("No clusters present. Please run eventgraph.add_cluster_assignments().")

	timeseries = {}
	for cluster in sorted(eventgraph.events_meta.cluster.unique()):
	    events = eventgraph.events[eventgraph.events_meta.cluster==cluster]
	    timeseries[cluster] = events.groupby(by=events.time//interval_width).size()
	    
	total = eventgraph.events.groupby(by=eventgraph.events.time//interval_width).size()

	return timeseries, total