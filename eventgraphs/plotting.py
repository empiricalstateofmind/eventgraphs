import networkx as nx
import matplotlib
import pandas as pd
import numpy as np
from collections import defaultdict
from IPython.display import Image

def plot_aggregate_graph(eventgraph, edge_colormap=None, display=True, **kwargs):
	""" 
	Plots the aggregate graph of nodes of an eventgraph. 

	Currently doesn't support argument changes.
	"""

	if eventgraph.directed:
		G = nx.DiGraph()
	else:
		G = nx.Graph()

	if edge_colormap is None:
		edge_colormap = defaultdict(lambda: 'k')

	for _, event in conversation.events.iterrows():
	    if len(event.target) == 0:
	        G.add_node(event.source)
	    else:
	        for target in event.target:
	            G.add_edge(event.source, target, {'type':event.type, 'color':edge_colormap[event.type]})

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

def plot_event_graph(eventgraph):
	""""""

	G = PRUNED.create_networkx_graph(include_motif=True)

	colormap= {'message':'green', 'retweet':'red', 'status':'grey', 'reply':'blue'}
	node_colors = EG.events.type.apply(lambda x: colormap[x])
	#node_colors='grey'
	nx.set_node_attributes(G, node_colors, 'fillcolor')

	nx.set_node_attributes(G, 'circle', 'shape')

	#node_labels = {x:x for x in G.nodes()}
	#nx.set_node_attributes(G, node_labels, 'label')
	nx.set_node_attributes(G, 8, 'fontsize')

	node_urls = {x:"https://twitter.com/statuses/{}".format(EG.events.loc[x].id) for x in G.nodes()}
	node_tooltips = {x:"""Source: {}&#10;Target: {}""".format(EG.events.loc[x].source, EG.events.loc[x].target) for x in G.nodes()}
	nx.set_node_attributes(G, node_urls, 'URL')
	nx.set_node_attributes(G, node_tooltips, 'tooltip')

	#nx.set_node_attributes(G, 'grey', 'fillcolor')
	nx.set_node_attributes(G, 'filled', 'style')
	nx.set_node_attributes(G, 'true', 'fixedsize')
	#nx.set_node_attributes(G, 'width', {key:(val**0.5)/5 for key,val in nx.degree(G).items()})
	nx.set_node_attributes(G, 0.3, 'width')


	edge_motifs = {(e1,e2):"Motif:{}&#10;IET:{}".format(key['type'],key['delta']) for e1,e2,key in G.edges(data=True)}
	nx.set_edge_attributes(G, edge_motifs, 'edgetooltip') #normal open halfopen vee

	nx.set_edge_attributes(G, "open", 'arrowhead') #normal open halfopen vee
	nx.set_edge_attributes(G, "bold", 'style' )
	#nx.set_edge_attributes(G, 'len',1) # fdp, neato Preferred edge length, in inches.
	#nx.set_edge_attributes(G, 'splines', "curved") #normal open halfopen vee

	# Remove boring nodes...
	ins = nx.in_degree_centrality(G)
	outs= nx.out_degree_centrality(G)
	nodes_to_remove = [x for x in G.nodes() if ins[x] + outs[x] == 0.0]
	G.remove_nodes_from(nodes_to_remove)

	A = nx.drawing.nx_pydot.to_pydot(G)

	A.set('layout','dot')
	#A.set('splines','spline')

	A.set('size', 200)
	A.set('ratio', 0.4)
	A.set('dpi', 500)
	A.set('dim', 2)
	A.set('overlap', 'false')
	A.set('minlen', 2) #dot only
	A.set('mindist', 0.5) #circo only
	A.set('mode', 'spring')
	A.set('K', 1)
	A.set('start', 3)

	fp = '/home/mellor/Dropbox/Oxford PDRA/Python/1801 - Generalised Event Graph/'
	#A.write_dot(fp + 'twitter.dot')
	#A.write_png(fp +'twitter.png')

	A.write_pdf(fp +'emirates_180a.pdf')
	#A.write_svg(fp +'twitter_interactive.svg')

	# Can we add a pre or post processing step so that the y axis is time and not heirarchy?

	from IPython.display import Image

	# with open('./figures/test{}.pdf'.format(c), 'wb') as file:
	#     file.write(A.create_pdf())
	#Image(A.create(format='png'))


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

def plot_fancy_barcode(eventgraph, delta_ub, top, ax=None):
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