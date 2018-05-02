import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import Image
from matplotlib import lines as mlines
from matplotlib import collections as mcollect
from scipy.cluster.hierarchy import dendrogram


def plot_aggregate_graph(eventgraph, edge_colormap=None, display=True, **kwargs):
    """
    Plots the aggregate graph of nodes of an eventgraph.

    Currently doesn't support argument changes.

    Input:
        eventgraph (EventGraph):
        edge_colormap (dict): [default=None]
        display (bool): [default=True]
        kwargs:

    Returns:
        A ():
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
        nx.set_edge_attributes(G, 'arrowhead', "open")  # normal open halfopen vee
        nx.set_edge_attributes(G, 'arrowsize', 2)  # normal open halfopen vee

    A = nx.drawing.nx_pydot.to_pydot(G)

    A.set('layout', 'fdp')
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


def plot_event_graph(eventgraph, event_colormap=None, remove_singles=False, display=True, **kwargs):
    """

    Input:
        eventgraph (EventGraph):
        event_colormap (dict): [default=None]
        remove_singles (bool): [default=False]
        kwargs:

    Returns:
        A ()L
    """

    # This needs changing.
    if 'size' not in kwargs.keys():
        kwargs['size'] = 10
    if 'ratio' not in kwargs.keys():
        kwargs['ratio'] = 0.5

    G = eventgraph.create_networkx_event_graph(event_colormap)

    nx.set_node_attributes(G, 'circle', 'shape')

    nx.set_node_attributes(G, 'fontsize', 8)

    nx.set_node_attributes(G, 'style', 'filled')
    nx.set_node_attributes(G, 'fixedsize', 'true')
    nx.set_node_attributes(G, 'width', 0.3)

    nx.set_edge_attributes(G, 'arrowhead', 'open')  # normal open halfopen vee
    nx.set_edge_attributes(G, 'style', 'bold')

    if remove_singles:
        ins = nx.in_degree_centrality(G)
        outs = nx.out_degree_centrality(G)
        nodes_to_remove = [x for x in G.nodes() if ins[x] + outs[x] == 0.0]
        G.remove_nodes_from(nodes_to_remove)

    A = nx.drawing.nx_pydot.to_pydot(G)

    A.set('layout', 'dot')

    A.set('size', kwargs['size'])
    A.set('ratio', kwargs['ratio'])
    A.set('dpi', 500)
    A.set('dim', 2)
    A.set('overlap', 'false')
    A.set('minlen', 2)  # dot only

    A.set('mode', 'spring')
    A.set('K', 1)
    A.set('start', 3)

    if display:
        return Image(A.create(format='png'))

    return A


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


def plot_cluster_timeseries(eventgraph, interval_width, normalized=False, ax=None, plotting_kwargs=None):
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
        dendrogram_kwargs = {'leaf_rotation': 90, 'truncate_mode': 'lastp', 'p': 100,
                             'no_labels': True, 'distance_sort': False, 'count_sort': True,
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
        handles = [mlines.Line2D([], [], color='C{}'.format(c - 1), marker='o', linestyle='',
                                 label='Cluster {}'.format(c)) for c in sorted(clusters.unique())]
        ax.legend(loc='best', handles=handles, fontsize=12, frameon=True, fancybox=True)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
