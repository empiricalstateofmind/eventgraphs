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

import pandas as pd
from collections import defaultdict

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .analysis import (calculate_motif_distribution,
                       calculate_motif_entropy,
                       calculate_iet_entropy,
                       calculate_activity,
                       calculate_edge_density,
                       calculate_clustering_coefficient,
                       calculate_reciprocity_ratio,
                       calculate_degree_imbalance
                       )

EVENT_GRAPH_FEATURES = [{'name': 'motifs',
                         'function': calculate_motif_distribution,
                         'kwargs': {},
                         'scale': False},
                        {'name': 'motif_entropy',
                         'function': calculate_motif_entropy,
                         'kwargs': {},
                         'scale': False
                         },
                        {'name': 'iet_entropy',
                         'function': calculate_iet_entropy,
                         'kwargs': {},
                         'scale': False
                         },
                        {'name': 'activity',
                         'function': calculate_activity,
                         'kwargs': {},
                         'scale': False
                         },
                        {'name': 'duration',
                         'function': lambda eventgraph: eventgraph.D,
                         'kwargs': {},
                         'scale': True
                         },
                        {'name': 'num_events',
                         'function': lambda eventgraph: eventgraph.M,
                         'kwargs': {},
                         'scale': True
                         },
                        {'name': 'num_nodes',
                         'function': lambda eventgraph: eventgraph.N,
                         'kwargs': {},
                         'scale': True
                         },
                        ]

AGGREGATE_GRAPH_FEATURES = [{'name': 'edge_density',
                             'function': calculate_edge_density,
                             'kwargs': {},
                             'scale': True},
                            {'name': 'clustering_coefficient',
                             'function': calculate_clustering_coefficient,
                             'kwargs': {},
                             'scale': False},
                            {'name': 'reciprocity_ratio',
                             'function': calculate_reciprocity_ratio,
                             'kwargs': {},
                             'scale': False},
                            {'name': 'degree_imbalance',
                             'function': calculate_degree_imbalance,
                             'kwargs': {},
                             'scale': False}]

FEATURE_SPEC = {'event_graph_features': EVENT_GRAPH_FEATURES,
                'aggregate_graph_features': AGGREGATE_GRAPH_FEATURES}


def generate_features(components, feature_spec=FEATURE_SPEC, verbose=False):
    """
    Generates the feature table for a collection of components.

    Input:
        components (list):
        feature_spec (dict): [default=FEATURE_SPEC]
        verbose (bool): [default=False]

    Returns:
        features (pd.DataFrame):
        scale_features (pd.DataFrame):
    """

    features = defaultdict(dict)
    scale_features = defaultdict(dict)
    for comp_ix, component in components.items():
        if verbose: print(comp_ix, end='\r')

        for feature in feature_spec['event_graph_features']:
            result = feature['function'](component, **feature.get('kwargs', {}))
            if isinstance(result, pd.Series):
                result = dict(result)

            if isinstance(result, dict):
                if feature['scale']:
                    scale_features[comp_ix].update(result)
                else:
                    features[comp_ix].update(result)
            else:
                if feature['scale']:
                    scale_features[comp_ix][feature['name']] = result
                else:
                    features[comp_ix][feature['name']] = result

        if len(feature_spec['aggregate_graph_features']) > 0:
            G = component.create_networkx_aggregate_graph()
        else:
            continue
        for feature in feature_spec['aggregate_graph_features']:
            result = feature['function'](G, **feature.get('kwargs', {}))
            if isinstance(result, pd.Series):
                result = dict(result)

            if isinstance(result, dict):
                if feature['scale']:
                    scale_features[comp_ix].update(result)
                else:
                    features[comp_ix].update(result)
            else:
                if feature['scale']:
                    scale_features[comp_ix][feature['name']] = result
                else:
                    features[comp_ix][feature['name']] = result

    features = pd.DataFrame(features).T.fillna(0)
    scale_features = pd.DataFrame(scale_features).T.fillna(0)
    return features, scale_features


def generate_distance_matrix(features, metric='euclidean', normalize=True):
    """
    Calculate the pairwise distances between components.

    Input:
        features (pd.DataFrame):
        metric (str): [default='euclidean']
        normalize (bool): [default=True]
    Returns:
        distance (array):
    """

    if normalize:
        X = StandardScaler().fit_transform(features.values)
    else:
        X = features.values
    distance = pdist(X, metric)
    return distance


def generate_linkage(distance_matrix, kind='ward'):
    """

    Input:
        distance_matrix (array):
        kind (str): [default='ward']

    Returns:
        linkage (array):
    """

    return linkage(distance_matrix, kind)


def find_clusters(features, kind='ward', criterion='maxclust', max_clusters=4, **kwargs):
    """

    Input:
        features (pd.DataFrame):
        kind (str): [default='ward']
        criterion (str): [default='maxclust']
        max_clusters (int): [default=4]
        kwargs: Key word arguments passed to generate_distance_matrix().

    Returns:
        clusters (pd.Series): A mapping from component to cluster.
        cluster_centers (pd.DataFrame): The average quantity of each feature for each cluster.
    """

    sim = generate_distance_matrix(features, **kwargs)

    Z = linkage(sim, kind)
    clusters = fcluster(Z, max_clusters, criterion='maxclust')

    cluster_centers = pd.DataFrame(
        {cluster: features.loc[clusters == cluster].mean() for cluster in range(1, max_clusters + 1)})

    clusters = pd.Series({f: c for f, c in zip(features.index, clusters)})

    return clusters, cluster_centers


def assign_to_cluster(observation, cluster_centers):
    """ Assign an observation to a cluster.

    Input:
        observation (array):
        cluster_centers (pd.DataFrame):
    Returns:
        index (int): Cluster assignment for the observation.
    """

    return (cluster_centers.subtract(observation, axis=0) ** 2).sum().idxmin()


def reduce_feature_dimensionality(features, ndim=2, method='pca', rescale=False, return_scalers=False, **tsne_kwargs):
    """
    Reduce the dimensionality of the component features using PCA or t-SNE (or both).

    Input:
        features (pd.DataFrame):
        ndim (int): [default=2]
        method (str): [default='pca']
        tsne_kwargs: Key word arguments to be passed to the TSNE class.

    Returns:
        X (array): Array of shape (len(features), ndim) of the reduced data.
    """

    if method.lower() == 'pca':
        pca = PCA(ndim)
        
        if rescale:
        X = StandardScaler().fit_transform(features.values)
        else:
            X = features.values

        if return_scalers:
            return pca.fit_transform(X), pca
        return pca.fit_transform(X)

    if method.lower() == 'tsne':

        if rescale:
        X = StandardScaler().fit_transform(features.values)
        else:
            X = features.values

        if X.shape[1] > 50:
            pca = PCA(50)
            X = pca.fit_transform(X)
        else:
            pca = None
        if tsne_kwargs is None:
            tsne_kwargs = {'perplexity': 40, 'n_iter': 1000, 'verbose': 1}
        tsne = TSNE(n_components=ndim, **tsne_kwargs)

        if return_scalers:
            return tsne.fit_transform(X), (pca, tsne) # Code is currently untidy and verbose.
        return tsne.fit_transform(X)
