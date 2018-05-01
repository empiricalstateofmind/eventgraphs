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
	""" 
	Generates the feature table for a collection of components. 

	Input:

	Returns:
		None
	"""

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

def generate_distance_matrix(features, metric='euclidean', normalize=True):
	""" 
	Normalise features and calculate similarity matrix. 

	Input:

	Returns:
		None
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

	Returns:
		None
	"""

	return linkage(distance_matrix, kind)


def find_clusters(features, kind='ward', criterion='maxclust', max_clusters=4, **kwargs):
	"""

	Input:

	Returns:
		None
	"""

	sim = generate_distance_matrix(features, **kwargs)

	Z = linkage(sim, kind)
	clusters = fcluster(Z, max_clusters, criterion='maxclust')

	cluster_centers = pd.DataFrame({cluster:features.loc[clusters==cluster].mean() for cluster in range(1, max_clusters+1)})

	clusters = pd.Series({f:c for f,c in zip(features.index, clusters)})

	return clusters, cluster_centers

def assign_to_cluster(observation, cluster_centers):
	""" Assign an observation to a cluster. 

	Input:

	Returns:
		None
	"""

	return (cluster_centers.subtract(observation, axis=0)**2).sum().idxmin()

def reduce_feature_dimensionality(features, ndim=2, method='pca', tsne_kwargs=None):
	"""
	Reduce the dimensionality of the component features using PCA or t-SNE (or both).

	Input:

	Returns:
		None	
	"""

	if method=='pca':
		pca = PCA(dimensions)
		X = StandardScaler().fit_transform(features.values)
		return pca.fit_transform(X)

	if method=='tsne':
		X = StandardScaler().fit_transform(features.values)
		if X.shape[1] > 50:
			pca = PCA(50)
			X = pca.fit_transform(X)
		if tsne_kwargs is None:
			tsne_kwargs = {'perplexity':40, 'n_iter':1000, 'verbose':1}
		tsne = TSNE(n_components=ndim, **tsne_kwargs)
		return tsne.fit_transform(X)