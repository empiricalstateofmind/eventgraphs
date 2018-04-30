import pandas as pd
from collections import defaultdict

from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from .analysis import (calculate_motif_distribution, 
					   calculate_motif_entropy,
					   calculate_iet_entropy, 
					   calculate_activity,
					   calculate_edge_density,
					   calculate_clustering_coefficient,
					   calculate_reciprocity_ratio,
					   calculate_degree_assortativity
					   )

EVENT_GRAPH_FEATURES = [{'name':'motifs',
						 'function': calculate_motif_distribution,
						 'kwargs':{},
						 'scale':False},
						{'name':'motif_entropy',
						 'function': calculate_motif_entropy,
						 'kwargs':{},
						 'scale':False
						 },						
						{'name':'iet_entropy',
						 'function': calculate_iet_entropy,
						 'kwargs':{},
						 'scale':False
						 },
						{'name':'activity',
						 'function': calculate_activity,
						 'kwargs':{},
						 'scale':False
						 },
						{'name':'duration',
						 'function': lambda eventgraph: eventgraph.D,
						 'kwargs':{},
						 'scale':True
						 },
						{'name':'num_events',
						 'function': lambda eventgraph: eventgraph.M,
						 'kwargs':{},
						 'scale':True
						 },
						{'name':'num_nodes',
						 'function': lambda eventgraph: eventgraph.N,
						 'kwargs':{},
						 'scale':True
						 },
						]


AGGREGATE_GRAPH_FEATURES = [{'name':'edge_density',
							 'function': calculate_edge_density,
							 'kwargs':{},
							 'scale':True},
							 {'name':'clustering_coefficient',
							 'function': calculate_clustering_coefficient,
							 'kwargs':{},
							 'scale':False},
							 {'name':'reciprocity_ratio',
							 'function': calculate_reciprocity_ratio,
							 'kwargs':{},
							 'scale':False},
							 {'name':'degree_assortativity',
							 'function': calculate_degree_assortativity,
							 'kwargs':{},
							 'scale':False}]

FEATURE_SPEC = {'event_graph_features': EVENT_GRAPH_FEATURES,
				'aggregate_graph_features':AGGREGATE_GRAPH_FEATURES}

def generate_features(components, feature_spec=FEATURE_SPEC, verbose=False):
	""" Generates the feature table for a collection of components. """

	features = defaultdict(dict)
	scale_features = defaultdict(dict)
	for comp_ix, component in components.items():
		if verbose: print(comp_ix, end='\r')

		for feature in feature_spec['event_graph_features']:
			result = feature['function'](component, **feature.get('kwargs',{}))
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
		for feature in feature_spec['aggregate_graph_features']:
			result = feature['function'](G, **feature.get('kwargs',{}))
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

def similarity(features, metric='euclidean', normalize=True):
    """ Normalise features and calculate similarity matrix """
    if normalize:
        X = StandardScaler().fit_transform(features.values)
    else:
        X = features.values
    sim = pdist(X, metric)
    return sim

def find_clusters(similarity, kind='ward', criterion='maxclust', max_clusters=4):
    """"""
    Z = linkage(similarity, kind)
    clusters = fcluster(Z, max_clusters, criterion='maxclust')
    
    cluster_centers = pd.DataFrame({cluster:features.loc[clusters==cluster].mean() for cluster in range(1, max_clusters+1)})
    return clusters, cluster_centers

def assign_to_cluster(observation, cluster_centers):
	""" Assign an observation to a cluster. """

	return (cluster_centers.subtract(observation, axis=0)**2).sum().idxmin()