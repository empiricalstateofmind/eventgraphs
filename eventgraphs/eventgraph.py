import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, identity
from scipy.sparse import linalg as spla
from scipy.sparse import csgraph as csg

from collections import defaultdict
from collections.abc import Iterable
import json
import pickle
import ast

from .prebuilt import PREBUILT
from .motif import Motif

# 1. There are issues with events and node names that are not indexed from 0 
# 2. Current we can easily remove numpy and pandas dependencies (leaving only scipy and 
#    inbuilt)


class NotImplementedError(BaseException):
	"""Returns when a function is not implemented."""
	pass

class BadInputError(BaseException):
	"""Returns when data input is not in the correct format."""
	pass

class EventGraph(object):
	"""
	General event graph class for building event graphs from sequences of events.

	Event graphs can be constructed for arbitrary event sequences, containing directed
	and undirected hyperevents, and containing arbitrary extra data.
	A number of event joining rules are implemented, and custom rules can used.

	The event graph is described by two tables, the table of events (EventGraph.events), 
	and the table of event edges (EventGraph.eg_edges). There also exists an extra table
	(EventGraph.events_meta) which gives further information on events such as component
	or cluster membership.

	Event graphs should be created using one of the following class methods:

	EventGraph.from_pandas_eventlist() (default)
	EventGraph.from_dict_eventlist()
	EventGraph.from_filter()
	EventGraph.from_file()

	Example:
	events = [{'source': 0, 'target':1, 'time':1, 'type':'A'},
			  {'source': 1, 'target':2, 'time':3, 'type':'B'}]
	eg = EventGraph.from_json_eventlist(events, graph_rules='teg')

	References:
		[1] A. Mellor, The Temporal Event Graph, Jouurnal of Complex Networks (2017) 
		[2] A. Mellor, Classifying Conversation in Digital Communication, ICWSM (2018)
		[3] A. Mellor, Generalised Event Graphs and Temporal Motifs, In prepartion (2018) 

	"""

	# TO DO, along with other magic methods where needed.
	def __repr__(self):
		status = 'built' if hasattr(self, 'eg_edges') else 'unbuilt'		
		return "<EventGraph with {} nodes and {} events (status: {})>".format(self.N, 
																			  self.M,
																			  status)

	def __len__(self):
		return self.M

	@classmethod
	def from_pandas_eventlist(cls, events, graph_rules, **kwargs):
		"""
		Loads an event list in the form of a Pandas DataFrame into an unbuilt EventGraph.

		Input:
			events (pd.DataFrame): Table of temporal events
			graph_rules (str or dict): Rule set to build the event graph. Currently implemented
			is ['teg', 'eg', 'pfg']. See prebuilt.py for custom schema.
					
		Returns:
			EventGraph
		"""
		return cls(events=events, graph_rules=graph_rules, **kwargs)

	@classmethod
	def from_dict_eventlist(cls, events, graph_rules, **kwargs):
		"""
		Loads an event list in the form of a list of records into an unbuilt EventGraph.

		Input:
			events (list): List of events of the minimal form {'source': X, 'target': X, 'time': X}.
			graph_rules (str or dict): Rule set to build the event graph. Currently implemented
			is ['teg', 'eg', 'pfg']. See prebuilt.py for custom schema.
					
		Returns:
			EventGraph
		"""

		return cls(events=pd.DataFrame(events), graph_rules=graph_rules, **kwargs)

	@classmethod
	def from_file(cls, filepath):
		"""
		Load a built event graph from file (either stored as .json or .pkl)

		Input:
			filepath: Filepath to saved event graph

		Returns:
			EventGraph

		"""
		if filepath.endswith('.json'):
			with open(filepath, 'r', encoding='utf-8') as file:
				payload = json.load(file)
				for item in ['events', 'events_meta', 'eg_edges']:
					payload[item] = pd.DataFrame.from_dict(payload[item])

		return cls(**payload)
	
	def __init__(self, *args, **kwargs):

		# This massively needs tidying!
		
		self.events = kwargs['events']
		if not isinstance(self.events, pd.DataFrame):
			raise BadInputError("Events must be a DataFrame ({} passed), or passed through classmethods.".format(type(self.events)))

		self.directed = True
		if 'target' not in self.events.columns:
			self.events['target'] = np.empty((len(self.events), 0)).tolist()
			self.directed = False # Efficiency savings to be had if we treat seperately.

		if 'events_meta' in kwargs.keys():
			self.events_meta = kwargs['events_meta']
		else:
			self.events_meta = pd.DataFrame(index=self.events.index)
		
		self.ne_incidence = None
		self.oe_incidence = None
		self.ne_matrix = None
		self.oe_matrix = None
		self.eg_matrix = None

		if 'rules' in kwargs.keys():
			self.event_graph_rules = kwargs['graph_rules']
		else:
			if isinstance(kwargs['graph_rules'], dict):
				# Possibly require further checks for custom rules
				self.event_graph_rules = kwargs['graph_rules']
			elif kwargs['graph_rules'].lower() in ['teg', 'temporal event graph', 'temporal_event_graph']:
				self.event_graph_rules = PREBUILT['temporal_event_graph']
			elif kwargs['graph_rules'].lower() in ['eg', 'event graph', 'event_graph']:
				self.event_graph_rules = PREBUILT['general_event_graph']
			elif kwargs['graph_rules'].lower() in ['pfg', 'path finder graph', 'path_finder_graph']:
				self.event_graph_rules = PREBUILT['path_finder_graph']
			else:
				raise Exception("Incompatible Rules")    
		
		if 'eg_edges' in kwargs.keys():
			self.eg_edges = kwargs['eg_edges']

		# This will now give the index of the event pair (edge) in the event graph
		built = kwargs.get('built', False)
		if built:
			self.event_pair_processed = {row.source:{row.target: ix} for ix, row in self.eg_edges.iterrows()}
		else:
			self.event_pair_processed = kwargs.get('event_pair_processed',
											   defaultdict(lambda: defaultdict(bool)))

		self._generate_node_event_incidence()

		build_on_creation = kwargs.get('build_on_creation', False)
		if build_on_creation:
			self._build()

		return None

	@property
	def M(self):
		return len(self.events)

	@property
	def N(self):
		return len(self.ne_incidence)

	@property
	def D(self):
		# Currently doesn't care about events with duration.
		return self.events.time.max() - self.events.time.min()

	def _generate_node_event_incidence(self):
		"""
		Creates a node-event incidence dictionary used to build the event graph.

		Input:
			None

		Returns:
			None
		"""        
		self.ne_incidence = defaultdict(list)
		for ix, event in self.events.iterrows():

			for group in ['source', 'target']:
				if isinstance(event[group], Iterable) and not isinstance(event[group], str):
					for node in event[group]:
						self.ne_incidence[node].append(ix)
				else:
					self.ne_incidence[event[group]].append(ix)

	def _generate_object_event_incidence(self):
		"""
		Creates an object-event incidence dictionary used to build the event graph.

		Input:
			None
			
		Returns:
			None
		"""   
		self.oe_incidence = defaultdict(list)
		for ix, event in self.events.iterrows():
			if isinstance(event.objects, Iterable) and not isinstance(event.objects, str):
				for obj in event.objects:
					self.ne_incidence[obj].append(ix)
			else:
				self.oe_incidence[event.objects].append(ix)
					
	def _generate_node_event_matrix(self):
		"""
		Creates a node-event matrix using the node-event incidence dictionary.

		Input:
			None
		Returns:
			scipy.sparse.csc_matrix
		"""
		
		if self.ne_incidence is None:
			self._generate_node_event_incidence()

		if not hasattr(self, 'event_map'):
			self.event_map = self.events.reset_index(drop=False)['index']
		if not hasattr(self, 'node_map'):			
			self.node_map = pd.Series(sorted([x for x in self.ne_incidence.keys()]))
		
		inv_event_map = pd.Series(self.event_map.index, index=self.event_map)
		inv_node_map = pd.Series(self.node_map.index, index=self.node_map)
	
		rows = []
		cols = []
		for node, events in self.ne_incidence.items():
			for event in events:
				rows.append(inv_node_map[node])
				cols.append(inv_event_map[event])
		data = np.ones_like(rows)
		self.ne_matrix = csc_matrix((data, (rows, cols)), dtype=bool)
		return self.ne_matrix
	
	def _generate_eg_matrix(self, binary=False):
		"""
	
		Input:
			None
		Returns:
			scipy.sparse.csc_matrix
		"""
		
		if not hasattr(self, 'event_map'):
			self.event_map = self.events.reset_index(drop=False)['index']

		inv_event_map = pd.Series(self.event_map.index, index=self.event_map)

		# Make a sparse EG matrix
		rows = []
		cols = []
		data = []
		for ix, edge in self.eg_edges.iterrows():
			rows.append(inv_event_map[edge.source])
			cols.append(inv_event_map[edge.target])
			data.append(edge.delta)
		if binary:
			data = [1 for d in data]
		self.eg_matrix = csc_matrix((data, (rows, cols)), 
									shape=(self.M,self.M), 
									dtype=int)
		return self.eg_matrix
	
	# Shouldn't be hidden
	def _build(self, verbose=False):
		"""
		Builds the event graph from event sequence.

		Input:
			verbose (bool): If True, prints out progress of build [default=False]
		Returns:
			None
		"""
		eg_edges = []
		edge_indexer = 0
		for count, events in enumerate(self.ne_incidence.values()):
			if verbose and count%50==0: print(count, '/', self.N, end='\r', flush=True)
			for ix, event_one in enumerate(events): 

				for event_two in events[ix+1:]:

					if self.event_pair_processed[event_one][event_two]:
						pass
					else:
						e1 = self.events.loc[event_one]
						e2 = self.events.loc[event_two]
						connected, dt = self.event_graph_rules['event_processor'](e1,e2)

						self.event_pair_processed[event_one][event_two] = edge_indexer
						edge_indexer += 1

						# If we want to enforce a dt
						if dt > self.event_graph_rules['delta_cutoff']:
							break

						if connected:
							eg_edges.append((event_one, event_two, dt))

					# if subsequent event only then break 
					# Can extend our rules so that we can do 'next X events only'.
					if self.event_graph_rules['subsequential_only']:
						break

		self.eg_edges = pd.DataFrame(eg_edges, columns=['source','target', 'delta'])

	def _build_from_objects(self):
		"""
		Builds the event graph using object relations (instead of, of in addition to
		the node relations)

		Input:
			None
		Returns:
			None
		"""
		self._generate_object_event_incidence()

		eg_edges = []
		for count, events in enumerate(self.oe_incidence.values()):
			if verbose and count%50==0: print(count, '/', self.N, end='\r', flush=True)
			for ix, event_one in enumerate(events): 

				for event_two in events[ix+1:]:

					if self.event_pair_processed[event_one][event_two]:
						pass
					else:
						e1 = self.events.loc[event_one]
						e2 = self.events.loc[event_two]
						connected, dt = self.event_graph_rules['event_object_processor'](e1,e2)

						self.event_pair_processed[event_one][event_two] = True

						# If we want to enforce a dt
						if dt > self.event_graph_rules['delta_cutoff']:
							break

						if connected:
							eg_edges.append((event_one, event_two, dt))

					# if subsequent event only then break 
					# Can extend our rules so that we can do 'next X events only'.
					if self.event_graph_rules['subsequential_only']:
						break

		if self.eg_edges is not None:
			new_edges = pd.DataFrame(eg_edges, columns=['source','target', 'delta'])
			self.eg_edges = pd.concat([self.eg_edges, new_edges])
		else:
			self.eg_edges = pd.DataFrame(eg_edges, columns=['source','target', 'delta'])

	def calculate_edge_motifs(self, edge_type=None, condensed=False, verbose=False):
		"""
		Calculates the motifs for edges
		Add a column to use as edge type

		Input:
			edge_type (str):
			condensed (bool):
			verbose (bool): 
		Returns:
			None
		"""
		motifs = {}

		# If edge type isnt a single letter we may want to shorten it.
		if edge_type is None or edge_type not in self.events.columns:
			columns = ['source', 'target', 'time']
		else:
			columns = ['source', 'target', 'time', edge_type]

		if verbose:
			total_edges = len(self.eg_edges)
		for ix, row in self.eg_edges.iterrows():
			if verbose and ix%50==0: print( ix, '/', total_edges, end='\r', flush=True)
			e1 = self.events.loc[row.source][columns]
			e2 = self.events.loc[row.target][columns] 
			motif = Motif(e1,e2,condensed, self.directed)
			motifs[ix] = str(motif) # We'll just work with the string for now

		self.eg_edges['motif'] = pd.Series(motifs)


	def create_networkx_graph(self, include_motif=False):
		"""

		Input:
			include_motif (bool): Include the motif type as an edge property in the graph [default=False]

		Returns:

		"""
		try: 
			import networkx as nx
		except ImportError:
			raise ImportError("Networkx package required to create graphs.")

		G = nx.DiGraph()
		if include_motif:
			edges = [(edge.source,edge.target,{'delta':edge.delta, 'type':str(edge.motif)}) for ix,edge in self.eg_edges.iterrows()]
		else:
			edges = [(edge.source,edge.target,{'delta':edge.delta}) for ix,edge in self.eg_edges.iterrows()]

		G.add_nodes_from(range(self.N)) # This will break if we dont have reindexed events for subgraphs.
		G.add_edges_from(edges)
		return G

	def draw_graph(self):
		"""
		Work in progress

		Input:

		Returns:

		"""
		try:
			import matplotlib.pyplot as plt
		except ImportError:
			raise ImportError("Matplotlib required for graph drawning.")

		G = self.create_networkx_graph()
		plt.figure(figsize=(40,10))
		pos = nx.drawing.nx_pydot.pydot_layout(G, prog='dot')
		nx.draw_networkx(G, pos=pos)
		plt.axis('off')
		return G # Should return axis also
		
	def connected_components_indices(self):
		"""
		Should this be a property? How can we do this?

		Perhaps have two functions (connected component indices (this one))
		and a function that returns a list of EventGraphs (subgraphs)

		Input:

		Returns:

		"""

		self._generate_eg_matrix()
		
		components = csg.connected_components(self.eg_matrix, 
									  directed=True,
									  connection='weak',
									  return_labels=True)[1]

		self.events_meta['component'] = components
		return components

	def connected_components(self):
		"""
		Returns the conneted components of the event graph as a list of
		event graphs

		Input:
			None

		Returns:
			list: A list of EventGraph objects corresponding to the connected components
				  of the original EventGraph
		"""

		if not hasattr(self.events_meta, 'component'):
			self.connected_components_indices()

		cpts = defaultdict(list)
		for ix, comp in self.events_meta.component.items():
		    cpts[comp].append(ix)

		cpts_edges = defaultdict(list)
		for ix, row in self.eg_edges.iterrows():
		    c1 = self.events_meta.component[row.source]
		    c2 = self.events_meta.component[row.target]
		    if c1 == c2:
		        cpts_edges[c1].append(ix)

		# Do we want to do some sorting and enumeration of components?
		# Currently the index does not match the meta table if we sort.

		cpts = sorted([self.filter_events(events, cpts_edges[comp]) for comp, events in cpts.items()], 
					  key=lambda x: len(x),
					  reverse=True)
		return cpts

	def filter_edges(self, delta_lb=None, delta_ub=None, motif_types=None):
		"""
		Filter edges based on edge weight (or motif possibly)

		Input:

		Returns:
			EventGraph

		"""

		# Events stay the same, it's just eg_edges that changes.
		eg_edges = self.eg_edges

		if delta_lb is not None:
			eg_edges = eg_edges[eg_edges.delta > delta_lb]
		if delta_ub is not None:
			eg_edges = eg_edges[eg_edges.delta < delta_ub]
		if motif_types is not None:
			eg_edges = eg_edges[eg_edges.motif.isin(motif_types)]

		event_pair_processed = {row.source:{row.target: ix} for ix, row in eg_edges.iterrows()}

		payload = {'eg_edges':eg_edges,
				   'events':self.events,
				   'events_meta':self.events_meta,
				   'graph_rules': self.event_graph_rules,
				   'event_pair_processed': event_pair_processed
				   }

		return self.__class__(**payload)

	def filter_events(self, event_indices, edge_indices=None):
		"""
		Create a new event graph from a subset of events.

		If it's connected components, our pruning of the eg_edges is simpler, but we want 
		to be general if possible.

		Input:

		Returns:
			EventGraph: 
		"""

		events = self.events.loc[event_indices]
		events_meta = self.events_meta.loc[event_indices]


		# This is slow if edge_indices is not passed through.
		if edge_indices is None:
			event_pair_processed = {row.source:{row.target: ix} for ix, row in self.eg_edges.iterrows() \
									if (row.source in events.index) and (row.target in events.index)}

			edges_to_keep = [val for e1 in event_pair_processed.values() for val in e1.values()]
			eg_edges = self.eg_edges.loc[edges_to_keep] 
		else:
			eg_edges = self.eg_edges.loc[edge_indices]
			event_pair_processed = {row.source:{row.target: ix} for ix, row in eg_edges.iterrows()}

		payload = {'eg_edges': eg_edges,
		   'events': events,
		   'events_meta': events_meta,
		   'graph_rules': self.event_graph_rules,
		   'event_pair_processed': event_pair_processed
		   }

		return self.__class__(**payload)


	def save(self, fp, method='json'):
		"""
		Save to file, either as json or object pickle.

		Input:

		Returns:
			None
		"""

		if method=='json':
			# A temporary fix for the unserialisable Motif class
			# Need to also fix numpy integers! Sigh
			#edges = self.eg_edges.copy()
			#edges.motif = edges.motif.astype(str)
			payload = {'events': ast.literal_eval(self.events.to_json(orient='records')),
					   'events_meta': ast.literal_eval(self.events_meta.to_json(orient='records')),
					   'eg_edges': ast.literal_eval(self.eg_edges.to_json(orient='records')),
					   'graph_rules': "teg", # self.event_graph_rules, # omitted as function is not serialisable.
					   'built': True
					   }

			with open(fp, 'w', encoding='utf-8') as file:
				json.dump(payload, file, sort_keys=True, indent=4)

		else:
			raise Exception