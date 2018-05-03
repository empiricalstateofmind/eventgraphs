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

import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import pandas as pd
import numpy as np

from collections import defaultdict

def calculate_resolvant(E, truncated=False):
	""""""

	I = sp.eye(E.shape[0])

	if truncated:
		Q = I
		for _ in range(truncated):
			Q = I + E.dot(Q)

	else:
		Q = linalg.inv((I-E).tocsc())

	return Q

def calculate_node_reachability(T, Q):
	""""""

	return T.dot(Q)

def node_event_influence(C, remove_participating=False, NE=None):
	""" How many events a node can influence. """
	
	if remove_participating:
		return pd.Series(np.asarray(((C>0) - NE).sum(axis=1)).flatten())
	else:
		return pd.Series(np.asarray((C>0).sum(axis=1)).flatten())

def event_conductance(C):
	""" How many nodes can pass their state through an event. """

	return pd.Series(np.asarray((C>0).sum(axis=0)).flatten())

def inbound_and_outbound_node_sets(C, CT):
	""" 
	Returns the set of nodes that can reach an event and can be reached by an event,
	and the difference between those sets (outbound / inbound).
	"""

	inbound = defaultdict(set)
	for node, event in zip(*np.nonzero(C)):
	    inbound[event].add(node)

	outbound = defaultdict(set)
	for node, event in zip(*np.nonzero(CT)):
	    outbound[event].add(node)

	difference = {}
	for event, in_nodes in inbound.items():
	    difference[event] = outbound[event] - in_nodes

	return inbound, outbound, difference

def create_io_dataframe(inbound, outbound, difference):
	""" Create a dataframe of the input, output, and difference node set sizes. """

	io = pd.DataFrame([pd.Series({event:len(d) for event,d in inbound.items()}), 
					   pd.Series({event:len(d) for event,d in outbound.items()}),
					   pd.Series({event:len(d) for event,d in difference.items()})], 
					   index=['inbound','outbound','difference']).T

	return io