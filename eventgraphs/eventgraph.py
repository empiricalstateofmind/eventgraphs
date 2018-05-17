"""
Copyright (C) 2018 Andrew Mellor (mellor91@hotmail.co.uk)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ast
import json
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from itertools import product, combinations

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse import csgraph as csg

from .motif import Motif
from .prebuilt import PREBUILT


# 1. _event_pair_processed contains event pairs event if they do not create an edge first time round.
# 	 Need to check that event pairs are being processes for objects even if they didn't make an edge 
#	 when we considered node connectivity.
# 2. Current we can easily remove numpy and pandas dependencies (leaving only scipy and 
#    inbuilt)


class NotImplementedError(BaseException):
    """Returns when a function is not implemented."""
    pass


class BadInputError(BaseException):
    """Returns when data input is not in the correct format."""
    pass


class NoObjectError(BaseException):
    """Returns when events do not have any objects."""
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
    EventGraph.from_file()

    Example:
    events = [{'source': 0, 'target':1, 'time':1, 'type':'A'},
              {'source': 1, 'target':2, 'time':3, 'type':'B'}]
    eg = EventGraph.from_dict_eventlist(events, graph_rules='teg')

    References:
        [1] A. Mellor, The Temporal Event Graph, Jouurnal of Complex Networks (2017)
        [2] A. Mellor, Classifying Conversation in Digital Communication, EPJ Data Science (2018)
        [3] A. Mellor, Generalised Event Graphs and Temporal Motifs, In prepartion (2018)

    """

    # TO DO, along with other magic methods where needed.
    def __repr__(self):
        status = 'built' if hasattr(self, 'eg_edges') else 'unbuilt'
        edges = len(self.eg_edges) if hasattr(self, 'eg_edges') else 0
        return "<EventGraph with {} nodes, {} events, and {} edges (status: {})>".format(self.N,
                                                                                         self.M,
                                                                                         edges,
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
                                       are ['teg', 'eg', 'pfg']. See prebuilt.py for custom schema.

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
                                       are ['teg', 'eg', 'pfg']. See prebuilt.py for custom schema.

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
        else:
            raise Exception("Currently only import from .json supported.")

        return cls(**payload)

    def __init__(self, *args, **kwargs):

        # This massively needs tidying!

        # SANITISATION
        # 1. ENSURE EVENTS ARE IN TIME ORDER
        # 2.

        self.events = kwargs['events']
        if not isinstance(self.events, pd.DataFrame):
            raise BadInputError(
                "Events must be a DataFrame ({} passed), or passed through classmethods.".format(type(self.events)))

        self.directed = kwargs.get('directed', True)
        if 'target' not in self.events.columns:
            self.events['target'] = np.empty((len(self.events), 0)).tolist()
            self.directed = False  # Efficiency savings to be had if we treat seperately.

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
            self._event_pair_processed = {row.source: {row.target: ix} for ix, row in self.eg_edges.iterrows()}
        else:
            self._event_pair_processed = kwargs.get('_event_pair_processed',
                                                    defaultdict(lambda: defaultdict(bool)))

        self.generate_node_event_incidence()

        build_on_creation = kwargs.get('build_on_creation', False)
        if build_on_creation:
            self.build()

        # Indexes edges of the eventgraph as we create them.
        self._edge_indexer = 0

    @property
    def M(self):
        """ Number of events in the event graph."""
        return len(self.events)

    @property
    def N(self):
        """ Number of nodes in the event graph."""
        return len(self.ne_incidence)

    @property
    def D(self):
        """ Duration of the event graph (requires ordered event table). """
        if 'duration' in self.events.columns:
            return self.events.iloc[-1].time + self.events.iloc[-1].duration - self.events.iloc[0].time
        else:
            return self.events.iloc[-1].time - self.events.iloc[0].time

    def generate_node_event_incidence(self):
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
                    self.oe_incidence[obj].append(ix)
            else:
                self.oe_incidence[event.objects].append(ix)

    def generate_node_event_matrix(self):
        """
        Creates a node-event matrix using the node-event incidence dictionary.

        The matrix A_{ij} = 1 if node i is a participant in event j.
        The matrix is of size (N,M).

        Input:
            None
        Returns:
            event_matrix (scipy.sparse.csc_matrix):
        """

        if self.ne_incidence is None:
            self.generate_node_event_incidence()

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

    def generate_eg_matrix(self, binary=False):
        """
        Generate an (MxM) matrix of the event graph, weighted by inter-event times.

        Input:
            None
        Returns:
            event_matrix (scipy.sparse.csc_matrix):
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
                                    shape=(self.M, self.M),
                                    dtype=int)
        return self.eg_matrix

    def build(self, verbose=False):
        """
        Builds the event graph from event sequence.

        Input:
            verbose (bool): If True, prints out progress of build [default=False]
        Returns:
            None
        """
        eg_edges = {}

        for count, events in enumerate(self.ne_incidence.values()):
            if verbose and count % 50 == 0: print(count, '/', self.N, end='\r', flush=True)
            for ix1, event_one in enumerate(events):

                for ix2, event_two in enumerate(events[ix1 + 1:]):

                    if self._event_pair_processed[event_one][event_two]:
                        pass
                    else:
                        e1 = self.events.loc[event_one]
                        e2 = self.events.loc[event_two]
                        connected, dt = self.event_graph_rules['event_processor'](e1, e2)

                        self._event_pair_processed[event_one][event_two] = self._edge_indexer

                        # If we want to enforce a dt
                        if dt > self.event_graph_rules['delta_cutoff']:
                            break

                        if connected:
                            eg_edges[self._edge_indexer] = (event_one, event_two, dt)
                            self._edge_indexer += 1

                    # if subsequent event only then break
                    # Can extend our rules so that we can do 'next X events only'.
                    if self.event_graph_rules['subsequential']:
                        if ix2 + 1 == self.event_graph_rules['subsequential']:
                            break

        if hasattr(self, 'eg_edges'):
            new_edges = pd.DataFrame.from_dict(eg_edges, orient='index')
            new_edges.columns = ['source', 'target', 'delta']
            self.eg_edges = pd.concat([self.eg_edges, new_edges], join='inner')
        else:
            self.eg_edges = pd.DataFrame.from_dict(eg_edges, orient='index')
            self.eg_edges.columns = ['source', 'target', 'delta']

    def _build_from_objects(self, verbose=False):
        """
        Builds the event graph using object relations (instead of, of in addition to
        the node relations)

        Input:
            verbose (bool): [default=False]
        Returns:
            None
        """

        if 'objects' not in self.events.columns:
            raise NoObjectError("Event data must contain 'objects'.")

        self._generate_object_event_incidence()

        eg_edges = {}
        for count, events in enumerate(self.oe_incidence.values()):
            if verbose and count % 50 == 0: print(count, '/', self.N, end='\r', flush=True)
            for ix, event_one in enumerate(events):

                for event_two in events[ix + 1:]:

                    if self._event_pair_processed[event_one][event_two]:
                        pass
                    else:
                        e1 = self.events.loc[event_one]
                        e2 = self.events.loc[event_two]
                        connected, dt = self.event_graph_rules['event_object_processor'](e1, e2)

                        self._event_pair_processed[event_one][event_two] = self._edge_indexer

                        # If we want to enforce a dt
                        if dt > self.event_graph_rules['delta_cutoff']:
                            break

                        if connected:
                            self._edge_indexer += 1
                            eg_edges[self._edge_indexer] = (event_one, event_two, dt)

                    # if subsequent event only then break
                    # Can extend our rules so that we can do 'next X events only'.
                    if self.event_graph_rules['subsequential']:
                        if count + 1 == self.event_graph_rules['subsequential']:
                            break

        if hasattr(self, 'eg_edges'):
            new_edges = pd.DataFrame.from_dict(eg_edges, orient='index')
            new_edges.columns = ['source', 'target', 'delta']
            self.eg_edges = pd.concat([self.eg_edges, new_edges], join='inner')
        else:
            self.eg_edges = pd.DataFrame.from_dict(eg_edges, orient='index')
            self.eg_edges.columns = ['source', 'target', 'delta']

    def randomize_event_times(self, seed=None):
        """
        Shuffles the times for all events. 

        Can only be called before the event graph is built.

        Input:
            seed (int): The seed for the random shuffle [default=None].

        Returns:
            None
        """

        if hasattr(self, 'eg_edges'):
            raise Exception("Event Graph has already been built. To randomize data create a new EventGraph object.")

        self.events.time = self.events.time.sample(frac=1, random_state=seed).values
        self.events = self.events.sort_values(by='time').reset_index(drop=True)


    def calculate_edge_motifs(self, edge_type=None, condensed=False):
        """
        Calculates the two-event motif for all edges of the event graph.

        Currently events with duration are unsupported.

        Input:
            edge_type (str): Column name for which edge types are to be differentiated.
            condensed (bool): If True, condenses the motif to be agnostic to the number
                              of nodes in each event [default=False].
        Returns:
            None
        """

        if edge_type is None or edge_type not in self.events.columns:
            columns = ['source', 'target', 'time']
        else:
            columns = ['source', 'target', 'time', edge_type]

        def find_motif(x, columns, condensed=False, directed=False):
            """ Create a motif from the joined event table. """
            e1 = tuple(x['{}_s'.format(c)] for c in columns)
            e2 = tuple(x['{}_t'.format(c)] for c in columns)
            return str(Motif(e1, e2, condensed, directed))

        temp = pd.merge(pd.merge(self.eg_edges[['source', 'target']],
                                 self.events,
                                 left_on='source',
                                 right_index=True,
                                 suffixes=('_d', ''))[['target_d'] + columns],
                        self.events,
                        left_on='target_d',
                        right_index=True,
                        suffixes=('_s', '_t'))[
            ["{}_{}".format(field, code) for field, code in product(columns, ('s', 't'))]]

        self.eg_edges['motif'] = temp.apply(
            lambda x: find_motif(x, columns, condensed=condensed, directed=self.directed), axis=1)

    def create_networkx_aggregate_graph(self, edge_colormap=None):
        """
        Creates an aggregate static network of node interactions within the event graph.

        Input:
            edge_colormap (dict): Mapping from edge type to a color.

        Returns:
            G (nx.Graph/nx.DiGraph): Aggregrate graph of the event graph.
                                     Directed or undirected dependent on event graph type.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("Networkx package required to create graphs.")

        if edge_colormap is None:
            edge_colormap = defaultdict(lambda: 'black')

        if self.directed:
            G = nx.DiGraph()

            for _, event in self.events.iterrows():
                typed = ('type' in self.events.columns)
                attrs = {'type': event.type, 'color': edge_colormap[event.type]} if typed  else {'color':'black'}
                if isinstance(event.target, Iterable) and (len(event.target) == 0):
                    G.add_node(event.source)
                elif isinstance(event.target, str) or isinstance(event.target, int):
                    G.add_edge(event.source, event.target, **attrs)
                else:
                    for target in event.target:
                        G.add_edge(event.source, target, **attrs)

        else:
            G = nx.Graph()

            for _, event in self.events.iterrows():
                typed = ('type' in self.events.columns)
                attrs = {'type': event.type, 'color': edge_colormap[event.type]} if typed  else {'color':'black'}
                G.add_edges_from(combinations(event.source, 2), **attrs)

        return G

    def create_networkx_event_graph(self, event_colormap=None):
        """
        Creates a networkx graph representation of the event graph.

        Input:
            edge_colormap (dict): Mapping from edge type to a color.

        Returns:
            G (nx.DiGraph): Aggregrate graph of the event graph.
                            Directed or undirected dependent on event graph type.
        """

        try:
            import networkx as nx
        except ImportError:
            raise ImportError("Networkx package required to create graphs.")

        G = nx.DiGraph()

        if event_colormap is None:
            event_colormap = defaultdict(lambda: 'grey')

        typed = ('type' in self.events.columns)

        for ix, event in self.events.iterrows():
            attrs = {'type': event.type, 'fillcolor': event_colormap[event.type]} if typed else {'fillcolor':'grey'}
            G.add_node(ix, **attrs)

        for _, edge in self.eg_edges.iterrows():
            G.add_edge(edge.source, edge.target, **{'delta': edge.delta, 'motif': edge.motif})

        return G

    def connected_components_indices(self):
        """
        Calculates the component that each event belongs to and saves it to self.events_meta.

        Note that component numerical assignment is random and can different after each run.

        Input:
            None
        Returns:
            components (list): A list containing the component allocation for each event.
        """

        self.generate_eg_matrix()

        components = csg.connected_components(self.eg_matrix,
                                              directed=True,
                                              connection='weak',
                                              return_labels=True)[1]

        self.events_meta.loc[:, 'component'] = components
        return components

    def get_component(self, ix):
        """
        Returns a component of the event graph as an EventGraph object.

        Input:
            ix (int): Index of component.

        Returns:
            eventgraph: EventGraph of the component.
        """

        if not hasattr(self.events_meta, 'component'):
            self.connected_components_indices()
           
        event_ids = self.events_meta.component == ix
        events = self.events[event_ids]
        events_meta = self.events_meta[event_ids]

        edge_ids = pd.merge(self.eg_edges, events_meta, left_on='source', right_index=True).index
        eg_edges = self.eg_edges.loc[edge_ids]
        _event_pair_processed = {row.source: {row.target: ix} for ix, row in eg_edges.iterrows()}

        payload = {'eg_edges': eg_edges,
                   'events': events,
                   'events_meta': events_meta,
                   'graph_rules': self.event_graph_rules,
                   '_event_pair_processed': _event_pair_processed,
                   'directed':self.directed
                   }

        return self.__class__(**payload)

    def connected_components(self, use_previous=False, min_size=None, top=None):
        """
        Returns the connected components of the event graph as a list of
        EventGraphs objects.

        Input:
            use_previous (bool): If True, use the previously calculated components (even if they have changes with event/edge manipulation).
            min_size (int): The minimum number of events for a component to be counted (cannot be used with top).
            top (int): The largest 'top' components in the event graph (cannot be used with min_size)

        Returns:
            components (list): A list of EventGraph objects corresponding to the connected components
                                 of the original event graph.
        """

        if (min_size is not None) and (top is not None):
            raise Exception("Please specify only one of 'min_size' or 'top'")

        if not ('component' in self.events_meta.columns) or (not use_previous):
            self.connected_components_indices()

        component_sizes = self.events_meta.component.value_counts()
        if min_size is not None:
            component_sizes = component_sizes[component_sizes >= min_size]
        if top is not None:
            component_sizes = component_sizes.nlargest(top)

        components = {ix: self.get_component(ix) for ix in component_sizes.index}

        return components

    def filter_edges(self, delta_lb=None, delta_ub=None, motif_types=None, inplace=False):
        """
        Filter edges based on edge weight or motif type.

        Input:
            delta_lb (float): Minimum edge weight (keep if delta > delta_lb).
            delta_ub (float): Maximum edge weight (keep if delta <= delta_ub).
            motif_types (list): List of motif types to keep.
            inplace (bool): Chose whether to perform the edge removal inplace or to create a new EventGraph [default=False].
        Returns:
            filtered (EventGraph): If inplace=False returns an EventGraph, else returns None.

        """

        eg_edges = self.eg_edges

        if delta_lb is not None:
            eg_edges = eg_edges[eg_edges.delta > delta_lb]
        if delta_ub is not None:
            eg_edges = eg_edges[eg_edges.delta <= delta_ub]
        if motif_types is not None:
            eg_edges = eg_edges[eg_edges.motif.isin(motif_types)]

        _event_pair_processed = {row.source: {row.target: ix} for ix, row in eg_edges.iterrows()}

        if inplace:
            self.eg_edges = eg_edges
            self._event_pair_processed = _event_pair_processed
            return None
        else:
            filtered = deepcopy(self)
            filtered.eg_edges = eg_edges
            filtered._event_pair_processed = _event_pair_processed
            return filtered

    def filter_events(self, event_indices, edge_indices=None):
        """
        Create a new event graph from a subset of events.

        If it's connected components, our pruning of the eg_edges is simpler, but we want
        to be general if possible.

        Input:
            event_indices (list):
            edge_indices (list):

        Returns:
            filtered (EventGraph):
        """

        events = self.events.loc[event_indices]
        events_meta = self.events_meta.loc[event_indices]

        if edge_indices is None:
            eg_edges = pd.merge(self.eg_edges,
                                events,
                                left_on='source',
                                right_index=True,
                                suffixes=('', '_tmp'))[self.eg_edges.columns]
            eg_edges = pd.merge(eg_edges,
                                events,
                                left_on='target',
                                right_index=True,
                                suffixes=('', '_tmp'))[self.eg_edges.columns]

            _event_pair_processed = {row.source: {row.target: ix} for ix, row in eg_edges.iterrows()}

        else:
            eg_edges = self.eg_edges.loc[edge_indices]
            _event_pair_processed = {row.source: {row.target: ix} for ix, row in eg_edges.iterrows()}

        payload = {'eg_edges': eg_edges,
                   'events': events,
                   'events_meta': events_meta,
                   'graph_rules': self.event_graph_rules,
                   '_event_pair_processed': _event_pair_processed,
                   'directed':self.directed
                   }

        return self.__class__(**payload)

    def add_cluster_assignments(self, cluster_assignments):
        """
        Adds the cluster assignment to the events_meta table.

        Input:
            cluster_assignments (dict/pd.Series):
        Returns:
            None
        """

        if isinstance(cluster_assignments, dict):
            cluster_assignments = pd.Series(cluster_assignments)

        def cluster_or_zero(x, cluster_assignments):
            """ Returns the cluster of an event, or zero if unclustered. """
            if x in cluster_assignments.index:
                return cluster_assignments[x]
            else:
                return 0

        self.events_meta['cluster'] = self.events_meta['component'].apply(
            lambda x: cluster_or_zero(x, cluster_assignments))

    def save(self, fp, method='json'):
        """
        Save to file, either as json or object pickle.

        Note: Currenltly only save to JSON is available (and advised!).

        Input:
            fp (str):
            method (str):
        Returns:
            None
        """

        if method == 'json':
            # A temporary fix for the unserialisable Motif class
            # Need to also fix numpy integers! Sigh
            # edges = self.eg_edges.copy()
            # edges.motif = edges.motif.astype(str)
            payload = {'events': ast.literal_eval(self.events.to_json(orient='records')),
                       'events_meta': ast.literal_eval(self.events_meta.to_json(orient='records')),
                       'eg_edges': ast.literal_eval(self.eg_edges.to_json(orient='records')),
                       'graph_rules': "teg",  # self.event_graph_rules, # omitted as function is not serialisable.
                       'built': True,
                       'directed':self.directed
                       }

            with open(fp, 'w', encoding='utf-8') as file:
                json.dump(payload, file, sort_keys=True, indent=4)

        else:
            raise Exception