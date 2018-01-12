import numpy as np
import pandas as pd

from collections import defaultdict
from collections.abc import Iterable
import json

from scipy.sparse import csc_matrix, identity
from scipy.sparse import linalg as spla

from .prebuilt import PREBUILT
from .motif import Motif

# Should be optional 
import matplotlib.pyplot as plt
import networkx as nx

class EventGraph(object):
    """
    A class to store temporal events.
    """

    @classmethod
    def from_pandas_eventlist(cls, events, graph_rules):
        """
        Loading in from event list (needs to be built).
        Add in some sanitation.
        """
        return cls(events=events, graph_rules=graph_rules)

    @classmethod
    def from_json_eventlist(cls, filepath, graph_rules):
        """
        """



        return cls(events=events, graph_rules=graph_rules)

    @classmethod
    def from_filter(cls, events, graph_rules):
        """
        Create from a larger event graph.
        """
        pass

    @classmethod
    def load_from_file(cls, filepath):
        """
        Loading in from saved event graph.
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as file:
                payload = json.load(file)
                for item in ['events', 'events_meta', 'eg_edges']:
                    payload[item] = pd.DataFrame.from_dict(payload[item])

        return cls(**payload)
    
    def __init__(self, *args, **kwargs):
        """
        """
        
        self.events = kwargs['events']

        if 'events_meta' in kwargs.keys():
            self.events_meta = kwargs['events_meta']
        else:
            self.events_meta = pd.DataFrame(index=self.events.index)
        
        self.ne_incidence = None
        self.ne_matrix = None
        self.event_pair_processed = event_pair_processed = defaultdict(lambda: defaultdict(bool))

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

        self._generate_node_event_incidence()

        self.M = len(self.events)
        self.N = len(self.ne_incidence)
        self.directed = None # Efficiency savings to be had if we treat seperately.
        
    def _generate_node_event_incidence(self):
        """
        """        
        self.ne_incidence = defaultdict(list)
        for ix, event in self.events.iterrows():

            for group in ['source', 'target']:
                if isinstance(event[group], Iterable) and not isinstance(event[group], str):
                    for node in event[group]:
                        self.ne_incidence[node].append(ix)
                else:
                    self.ne_incidence[event[group]].append(ix)
                    
    def _generate_node_event_matrix(self):
        """
        """
        
        #if ne_incidence doesn't exist then make it.

        # If nodes are not integers (or non-index integers then we need to index them)
        
        rows = []
        cols = []
        for node, events in self.ne_incidence.items():
            for event in events:
                rows.append(node)
                cols.append(event)
        data = np.ones_like(rows)
        self.ne_matrix = csc_matrix((data, (rows, cols)), dtype=bool)
    
    def _generate_eg_matrix(self):
        """
        """
        
        # Make a sparse EG matrix
        rows = []
        cols = []
        data = []
        for ix, edge in self.eg_edges.iterrows():
            rows.append(edge.source)
            cols.append(edge.target)
            data.append(edge.delta)
        self.eg_matrix = csc_matrix((data, (rows, cols)), 
                                    shape=(self.M,self.M), 
                                    dtype=int)
    
    def _build(self, verbose=False):
        """
        
        """
        eg_edges = []
        for count, events in enumerate(self.ne_incidence.values()):
            if verbose and count%50==0: print(count, '/', self.N, end='\r', flush=True)
            for ix, event_one in enumerate(events): 

                for event_two in events[ix+1:]:

                    if self.event_pair_processed[event_one][event_two]:
                        pass
                    else:
                        e1 = self.events.iloc[event_one]
                        e2 = self.events.iloc[event_two]
                        connected, dt = self.event_graph_rules['event_processor'](e1,e2)

                        self.event_pair_processed[event_one][event_two] = True

                        # If we want to enforce a dt
                        if dt > self.event_graph_rules['delta_cutoff']:
                            break

                        if connected:
                            eg_edges.append((event_one, event_two, dt))

                    # if subsequent event only then break 
                    if self.event_graph_rules['subsequential_only']:
                        break

        self.eg_edges = pd.DataFrame(eg_edges, columns=['source','target', 'delta'])

    def calculate_edge_motifs(self, edge_type=None, condensed=False, verbose=False):
        """
        Calculates the motifs for edges
        Add a column to use as edge type
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
            e1 = self.events.iloc[row.source][columns]
            e2 = self.events.iloc[row.target][columns] 
            motif = Motif(e1,e2,condensed)
            motifs[ix] = motif

        self.eg_edges['motif'] = pd.Series(motifs)


    def create_networkx_graph(self, include_motif=False):
        """
        """

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
        """

        G = self.create_networkx_graph()
        plt.figure(figsize=(40,10))
        pos = nx.drawing.nx_pydot.pydot_layout(G, prog='dot')
        nx.draw_networkx(G, pos=pos)
        plt.axis('off')
        return G # Should return axis also
        
    def connected_components(self):
        """
        Should this be a property? How can we do this?
        """
        
        components = csg.connected_components(self.eg_matrix, 
                                      directed=True,
                                      connection='weak',
                                      return_labels=True)[1]

    def save(self, fp, method='json'):
        """
        Save to file, either as json or object pickle.
        """

        if method=='json':
            # A temporary fix for the unserialisable Motif class
            # Need to also fix numpy integers! Sigh
            edges = self.eg_edges.copy()
            edges.motif = edges.motif.astype(str)
            payload = {'events': self.events.to_dict('records'),
                       'events_meta': self.events_meta.to_dict('records'),
                       'eg_edges': edges.to_dict('records'),
                       'rules':self.event_graph_rules}

            with open(fp, 'w', encoding='utf-8') as file:
                json.dump(payload, file, sort_keys=True, indent=4)

        else:
            raise Exception