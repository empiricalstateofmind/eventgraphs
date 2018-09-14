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

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import Image
from matplotlib import lines as mlines
from matplotlib import collections as mcollect
from scipy.cluster.hierarchy import dendrogram


def plot_aggregate_graph(eventgraph, edge_colormap=None, ax=None, minimum_size=None, **kwargs):
    """
    Plots the aggregate graph of nodes of an eventgraph.

    Currently doesn't support argument changes.

    Input:
        eventgraph (EventGraph):
        edge_colormap (dict): [default=None]
        ax (plt.axis): [default=None]
        minimum_size (int): [default=None]
        kwargs:

    Returns:
        ax (plt.axis):
    """

    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111)

    G = eventgraph.create_networkx_aggregate_graph(edge_colormap)

    if minimum_size is not None:
        for component in list(nx.weakly_connected_components(G)):
            if len(component) < minimum_size:
                for node in component:
                    G.remove_node(node)

    edgecolors = nx.get_edge_attributes(G, 'color')
    edges = list(edgecolors.keys())
    colors = list(edgecolors.values())

    pos = nx.drawing.nx_pydot.pydot_layout(G, prog='neato')

    nx.draw_networkx_nodes(G, 
                        pos=pos, 
                        edgecolors='k',
                        node_color='grey',
                        ax=ax)

    nx.draw_networkx_edges(G, 
                        pos=pos,
                        edge_color=colors,
                        ax=ax);

    ax.axis('off');

    return ax


def plot_event_graph(eventgraph, event_colormap=None, ax=None, time_scaled=False, minimum_size=None,**kwargs):
    """

    Input:
        eventgraph (EventGraph):
        event_colormap (dict): [default=None]
        ax (plt.axis): [default=None]
        time_scaled (bool): [default=False]
        minimum_size (int): [default=None]
        kwargs:

    Returns:
        ax (plt.axis):
    """
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111)

    G = eventgraph.create_networkx_event_graph(event_colormap)

    if minimum_size is not None:
        for component in list(nx.weakly_connected_components(G)):
            if len(component) < minimum_size:
                for node in component:
                    G.remove_node(node)

    nodecolors = nx.get_node_attributes(G, 'fillcolor')
    nodes = list(nodecolors.keys())
    colors = list(nodecolors.values())

    pos = nx.drawing.nx_pydot.pydot_layout(G, prog='dot')

    # Convert position to time
    if time_scaled:
        pos = {key: (val[0],-eventgraph.events.time[key]) for key,val in pos.items()}

    nx.draw_networkx_nodes(G, 
                        pos=pos, 
                        edgecolors='k',
                        node_color=colors,
                        ax=ax)

    nx.draw_networkx_edges(G, pos=pos)

    ax.axis('off');

    return ax

def plot_full_barcode_efficiently(eventgraph, delta_ub, top, ax=None):
    """
    Prints a barcode.

    Input:
        eventgraph (EventGraph):
        delta_ub (int):
        top (int):
        ax (matplotlib.axes._subplots.AxesSubplot): [default=None]

    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot):
    """

    if ax is None:
        ax = plt.gca()

    filtered = eventgraph.filter_edges(delta_ub=delta_ub)
    segs = []
    tmin, tmax = 1e99, 0
    components = pd.Series(filtered.connected_components_indices()).value_counts()
    for ix, component in enumerate(components.index[:top]):
        component = filtered.events[filtered.events_meta.component == component]
        for _, event in component.iterrows():
            segs.append(((event.time, ix), (event.time, ix + 1)))
            tmax = max(tmax, event.time)
            tmin = min(tmin, event.time)

    ln_coll = mcollect.LineCollection(segs, linewidths=1, colors='k')
    bc = ax.add_collection(ln_coll)
    ax.set_ylim((0, top + 1))
    ax.set_xlim((tmin, tmax))
    return ax


def plot_barcode(eventgraph, delta_ub, top, ax=None):
    """
    Prints a barcode.

    Input:
        eventgraph (EventGraph):
        delta_ub (int):
        top (int):
        ax (matplotlib.axes._subplots.AxesSubplot): [default=None]

    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot):
    """

    if ax is None:
        ax = plt.gca()
    bc = plot_full_barcode_efficiently(eventgraph, delta_ub, top, ax)

    ax.set_ylim((0, top))
    ax.set_yticks(np.arange(0, top), minor=False)
    ax.set_yticklabels(['C{}'.format(x) for x in np.arange(1, top + 1)])

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_verticalalignment("bottom")

    ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    ax.xaxis.grid(False)
    ax.tick_params(axis='x', length=5, which='major', bottom=True, top=False)

    return ax


def plot_cluster_timeseries(eventgraph, interval_width, normalized=False, ax=None, plot_unclustered=False, plotting_kwargs=None):
    """


    Input:
        eventgraph (EventGraph):
        interval_width (int):
        normalized (bool): [default=False]
        ax (matplotlib.axes._subplots.AxesSubplot): [default=None]
        plotting_kwargs: [default=None]

    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot):
    """

    from .analysis import calculate_cluster_timeseries

    timeseries, total = calculate_cluster_timeseries(eventgraph, interval_width)

    if ax is None:
        ax = plt.gca()
    if plotting_kwargs is None:
        plotting_kwargs = {'logy': False, 'linestyle': '--', 'marker': 's'}

    for cluster, ts in timeseries.items():
        if cluster==0 and not plot_unclustered:
            continue
        label = cluster if cluster > 0 else 'Unclustered'
        if normalized:
            (ts / total).plot(label=label, ax=ax, **plotting_kwargs)
        else:
            ts.plot(label=label, ax=ax, **plotting_kwargs)
    ax.legend(loc='best')

    return ax


def plot_component_dendrogram(Z, ax=None, dendrogram_kwargs=None):
    """

    Input:
        Z (array):
        ax (matplotlib.axes._subplots.AxesSubplot): [default=None]
        dendrogram_kwargs: [default=None]

    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot):
    """

    if ax is None:
        ax = plt.gca()
    if dendrogram_kwargs is None:
        dendrogram_kwargs = {'leaf_rotation': 90, 
                             'truncate_mode': 'lastp', 
                             'p': 100,
                             'no_labels': True, 
                             'distance_sort': False, 
                             'count_sort': True,
                             'above_threshold_color': 'k', 'color_threshold': 80}

    ax.set_ylabel('Distance')

    dendrogram(Z,
               ax=ax,
               **dendrogram_kwargs
               )

    return ax


def plot_component_embedding(X, clusters=None, ax=None):
    """

    Colors will cycle for greater than 10 clusters (although by then the plot will
    be too confusing anyway!).

    Input:
        X (array):
        clusters (list/pd.Series):
        ax (matplotlib.axes._subplots.AxesSubplot): [default=None]

    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot):
    """

    if ax is None:
        ax = plt.gca()

    if X.shape[1] > 2:
        raise Exception("Only 2-dimensional data can be plotted.")

    if clusters is None:
        scatter = ax.scatter(X[:, 0], X[:, 1], marker='o')

    else:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=['C{}'.format(c - 1) for c in clusters], marker='o', label=clusters)
        handles = [mlines.Line2D([], 
                                 [], 
                                 color='C{}'.format(c - 1), 
                                 marker='o', 
                                 linestyle='', 
                                 label='Cluster {}'.format(c)) for c in sorted(clusters.unique())]
        ax.legend(loc='best', handles=handles)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
