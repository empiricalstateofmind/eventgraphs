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

from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csgraph as csg


def calculate_iet_distribution(eventgraph, by_motif=False, normalize=True, cumulative=False, bins=None):
    """
    Calculate the inter-event time distribution for an event graph.

    Input:
        eventgraph (EventGraph):
        by_motif (bool): [default=False]
        normalize (bool): [default=True]
        cumulative (bool): [default=False]
        bins (array): [default=None]

    Returns:
        iets (pd.Series):
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
    Calculate the motif distribution of an event graph

    Input:
        eventgraph (EventGraph):
        normalize (bool): [default=True]

    Returns:
        motifs (pd.Series):
    """

    return eventgraph.eg_edges.motif.value_counts(normalize=normalize)


def calculate_component_distribution(eventgraph, normalize=True, cumulative=False, bins=None):
    """


    Input:
        eventgraph (EventGraph):
        normalize (bool): [default=True]
        cumulative (bool): [default=False]
        bins (array): [default=None]

    Returns:
        component_dist (pd.Series):
    """

    if 'component' not in eventgraph.events_meta.columns:
        eventgraph.generate_eg_matrix()

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
        eventgraph (EventGraph):
        delta_range (array):
        normalize (bool): [default=True]

    Returns:
        component_distributions (dict):
        largest_component (pd.Series):
    """

    if eventgraph.eg_matrix is not None:
        eg_matrix = deepcopy(eventgraph.eg_matrix)
    else:
        eg_matrix = deepcopy(eventgraph.generate_eg_matrix())

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

def calculate_component_durations_over_delta(eventgraph, delta_range, normalize=True):
    """
    Calculates the component duration distribution over a range of dt values.

    dt range must be less than that of the eventgraph.

    Input:
        eventgraph (EventGraph):
        delta_range (array):
        normalize (bool): [default=True]

    Returns:
        duration_distributions (dict):
        largest_durations (pd.Series):
    """

    if eventgraph.eg_matrix is not None:
        eg_matrix = deepcopy(eventgraph.eg_matrix)
    else:
        eg_matrix = deepcopy(eventgraph.generate_eg_matrix())

    largest_duration = {}
    duration_distributions = {}
    for dt in delta_range[::-1]:
        eg_matrix.data = np.where(eg_matrix.data <= dt, eg_matrix.data, 0)
        eg_matrix.eliminate_zeros()
        component_ixs = csg.connected_components(eg_matrix,
                                                 directed=True,
                                                 connection='weak',
                                                 return_labels=True)[1]

        durations = []
        for c in set(component_ixs):
            events = np.where(component_ixs==c)
            lower, upper = events[0][0], events[0][-1]
            first = eventgraph.events.loc[lower]
            last = eventgraph.events.loc[upper]
            duration = last.time - first.time
            durations.append(duration)

        durations = pd.Series(durations).value_counts(normalize=normalize).sort_index()
        duration_distributions[dt] = durations
        largest_duration[dt] = durations.index.max()
    largest_duration = pd.Series(largest_duration)

    return duration_distributions, largest_duration

def calculate_motif_entropy(eventgraph, miller_correct=False,  k=None, normalize=False):
    """
    Calculate the motif entropy

    Input:
        eventgraph (EventGraph):
        miller_correct (bool): Apply the Miller bias correction for finite size samples[default=True]
        k (int): Number of possible motif combinations (should be automated at some point).
        normalize (bool): [default=False]

    Returns:
        motif_entropy (float):
    """

    motifs = calculate_motif_distribution(eventgraph)
    motif_entropy = -sum([p * np.log2(p) for p in motifs.values if p > 0])

    if miller_correct:
        if k is None:
            raise Exception("If 'miller_correct' is True, then 'k', the number of possible motifs, must be provided")
        N = len(eventgraph.eg_edges)
        return motif_entropy + (k-1)/(2*N)
    if normalize:
        return motif_entropy/ np.log2(k)

    return motif_entropy


def calculate_iet_entropy(eventgraph, normalize=True, miller_correct=True, divisions=10):
    """


    Input:
        eventgraph (EventGraph):
        normalize (bool): Normalise the entropy by the maximum entropy possible [default=True]
        miller_correct (bool): Apply the Miller bias correction for finite size samples [default=True]
        divisions (int): How many bins to divide the time-space into [default=10]

    Returns:
        iet_entropy (float):
    """

    iets = eventgraph.eg_edges.delta
    observations = len(iets)
    
    if iets.nunique() == 1:
        bins = divisions
    else:
        bins = np.linspace(iets.min(),iets.max(),divisions)
    binned = pd.cut(iets, bins=bins, include_lowest=True).value_counts(normalize=True).sort_index()

    iet_entropy = -sum([val * np.log2(val) for val in binned.values if val != 0])
        
    if miller_correct:
        iet_entropy = iet_entropy + (divisions-1)/(2*observations)
        if normalize:
            return iet_entropy/(np.log2(divisions)+(divisions-1)/(2*observations))
        else:
            return iet_entropy
        
    if normalize:
        return iet_entropy/np.log2(divisions)
    else:
        return iet_entropy


def calculate_activity(eventgraph, unit=1):
    """

    Input:
        eventgraph (EventGraph):
        unit (int): [default=1]

    Returns:
        activity (float):
    """

    duration = eventgraph.D
    if duration == 0:
        activity = np.inf
    else:
        activity = (len(eventgraph.events) / duration) * unit
    return activity


def calculate_edge_density(G):
    """


    Input:
        G (nx.Graph/nx.DiGraph):
    Return:
        density (float):
    """

    N = len(G.nodes())
    if N > 1:
        return len(G.edges()) / (N * (N - 1))
    else:
        return 0.0


def calculate_clustering_coefficient(G):
    """


    Input:
        G (nx.Graph/nx.DiGraph):
    Return:
        clustering (float):
    """

    N = len(G.nodes())
    if N > 1:
        recip = G.to_undirected(reciprocal=False)
        clustering = nx.cluster.average_clustering(recip)
        return clustering
    else:
        return 0.0


def calculate_reciprocity_ratio(G):
    """


    Input:
        G (nx.Graph/nx.DiGraph):
    Return:
        recip_ratio (float):
    """

    N = len(G.nodes())
    if N > 1:
        recip = G.to_undirected(reciprocal=True)
        recip_ratio = 2 * len(recip.edges()) / len(G.edges())
        return recip_ratio
    else:
        return 0.0


def calculate_degree_assortativity(G):
    """
    Calculates a 'fake' degree assortativity. To be ironed out as a concept.

    Input:
        G (nx.Graph/nx.DiGraph):
    Return:
        assorts (dict):
    """

    N = len(G.nodes())
    if N <= 1:
        return {'assort_ii': 0,
                'assort_io': 0,
                'assort_oi': 0,
                'assort_oo': 0, }

    degrees = pd.DataFrame([(G.out_degree(a), G.out_degree(b), G.in_degree(a), G.in_degree(b)) for a, b in G.edges()],
                           columns=['o_source', 'o_target', 'i_source', 'i_target'])
    assorts = {}
    for alpha, beta in [('o', 'o'), ('o', 'i'), ('i', 'i'), ('i', 'o')]:
        c1 = '{}_source'.format(alpha)
        c2 = '{}_target'.format(beta)

        x = (degrees[c1] - degrees[c2]).mean()
        if x == 0:
            assorts['assort_{}{}'.format(alpha, beta)] = 0.0
        else:
            assorts['assort_{}{}'.format(alpha, beta)] = x / (degrees[c1] - degrees[c2]).abs().max()

    return assorts


def calculate_cluster_timeseries(eventgraph, interval_width):
    """

    Input:
        eventgraph (EventGraph):
        interval_width (int):

    Returns:
        timeseries (dict):
        total (pd.Series):
    """

    if 'cluster' not in eventgraph.events_meta.columns:
        raise Exception("No clusters present. Please run eventgraph.add_cluster_assignments().")

    timeseries = {}
    for cluster in sorted(eventgraph.events_meta.cluster.unique()):
        events = eventgraph.events[eventgraph.events_meta.cluster == cluster]
        timeseries[cluster] = events.groupby(by=events.time // interval_width).size()

    total = eventgraph.events.groupby(by=eventgraph.events.time // interval_width).size()

    return timeseries, total
