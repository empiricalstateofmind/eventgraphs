import networkx as nx
import matplotlib
import pandas as pd
import numpy as np
from collections import defaultdict
from IPython.display import Image
import matplotlib.pyplot as plt

def plot_aggregate_graph(eventgraph, edge_colormap=None, display=True, **kwargs):
	""" 
	Plots the aggregate graph of nodes of an eventgraph. 

	Currently doesn't support argument changes.
	"""

	G = eventgraph.create_networkx_aggregate_graph(edge_colormap)

	nx.set_node_attributes(G, 'shape', 'circle')
	nx.set_node_attributes(G, 'label', '')
	nx.set_node_attributes(G, 'fillcolor', 'grey')
	nx.set_node_attributes(G, 'style', 'filled')
	nx.set_node_attributes(G, 'fixedsize', 'true')
	nx.set_node_attributes(G, 'width', 0.3)

	nx.set_edge_attributes(G, 'style', "bold")
	if eventgraph.directed:
		nx.set_edge_attributes(G, 'arrowhead', "open") # normal open halfopen vee
		nx.set_edge_attributes(G, 'arrowsize', 2) # normal open halfopen vee

	A = nx.drawing.nx_pydot.to_pydot(G)

	A.set('layout','fdp')
	A.set('size', 5)
	A.set('ratio', 1)
	A.set('dim', 2)
	A.set('overlap', 'false')
	A.set('mode', 'spring')
	A.set('K', 1)
	A.set('start', 3)

	if display:
		return Image(A.create(format='png'))
	
	return A

def plot_event_graph(eventgraph, event_colormap=None, remove_singles=False, **kwargs):
	""""""

	# This needs changing.
	if 'size' not in kwargs.keys():
		kwargs['size']=10
	if 'ratio' not in kwargs.keys():
		kwargs['ratio']=0.5

	G = eventgraph.create_networkx_event_graph(event_colormap)

	nx.set_node_attributes(G, 'circle', 'shape')

	nx.set_node_attributes(G, 'fontsize', 8)

	nx.set_node_attributes(G, 'style', 'filled')
	nx.set_node_attributes(G, 'fixedsize', 'true')
	nx.set_node_attributes(G, 'width', 0.3)

	nx.set_edge_attributes(G, 'arrowhead', 'open') # normal open halfopen vee
	nx.set_edge_attributes(G, 'style', 'bold')

	if remove_singles:
		ins = nx.in_degree_centrality(G)
		outs= nx.out_degree_centrality(G)
		nodes_to_remove = [x for x in G.nodes() if ins[x] + outs[x] == 0.0]
		G.remove_nodes_from(nodes_to_remove)

	A = nx.drawing.nx_pydot.to_pydot(G)

	A.set('layout','dot')

	A.set('size', kwargs['size'])
	A.set('ratio', kwargs['ratio'])
	A.set('dpi', 500)
	A.set('dim', 2)
	A.set('overlap', 'false')
	A.set('minlen', 2) # dot only

	A.set('mode', 'spring')
	A.set('K', 1)
	A.set('start', 3)


	if display:
		return Image(A.create(format='png'))
	
	return A

def plot_full_barcode_efficiently(eventgraph, delta_ub, top, ax=None):
	""" Prints a barcode. """

	if ax is None:
		ax = plt.gca()
		
	filtered = eventgraph.filter_edges(delta_ub=delta_ub)
	segs = []
	tmin, tmax = 1e99, 0
	components = pd.Series(filtered.connected_components_indices()).value_counts()
	for ix,component in enumerate(components.index[:top]):
		component = filtered.events[filtered.events_meta.component==component]
		for _, event in component.iterrows():
			segs.append(((event.time, ix),(event.time, ix+1)))
			tmax = max(tmax, event.time)
			tmin = min(tmin, event.time)
			
	ln_coll = matplotlib.collections.LineCollection(segs, linewidths=1, colors='k')
	bc = ax.add_collection(ln_coll)
	ax.set_ylim((0, top+1))
	ax.set_xlim((tmin,tmax))   
	return ax

def plot_barcode(eventgraph, delta_ub, top, ax=None):
	""" Prints a fancier barcode. """

	if ax is None:
		ax = plt.gca()
	bc = plot_full_barcode_efficiently(eventgraph, delta_ub, top, ax)

	ax.set_ylim((0,top))
	ax.set_yticks(np.arange(0,top), minor=False)
	ax.set_yticklabels(['C{}'.format(x) for x in np.arange(1,top+1)])
	
	for tick in ax.yaxis.get_majorticklabels():
		tick.set_verticalalignment("bottom")
	
	ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
	ax.xaxis.grid(False)
	ax.tick_params(axis='x', length=5, which='major', bottom=True, top=False)
	
	return ax