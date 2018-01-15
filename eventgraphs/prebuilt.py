def basic_event_processor(e1,e2):
	"""
	   
	"""

	if hasattr(e1, 'duration'):
		dt = e2.time - (e1.time + e1.duration)
	else:
		dt = e2.time - e1.time
	return True, dt

temporal_event_graph = {'event_processor':basic_event_processor,
						'subsequential_only':True,
						'delta_cutoff':1e9,
						'join_on_object':False,
						'event_object_processor':None
						}


general_event_graph = {'event_processor':basic_event_processor,
						'subsequential_only':False,
						'delta_cutoff':1e9,
						'join_on_object':False,
						'event_object_processor':None
						}

def path_finding_event_processor(e1,e2):
	"""
	"""

	if hasattr(e1, 'duration'):
		dt = e2.time - (e1.time + e1.duration)
	else:
		dt = e2.time - e1.time

	if len(e2.target) == 0:
		return False, dt

	# Issues with whether we have lists or strings (need to decide if we sanitize earlier)
		
	for node in e1.target:
		if node in e2.source:
			return True, dt

	return False, dt


path_finder_graph = {'event_processor':path_finding_event_processor,
						'subsequential_only':False,
						'delta_cutoff':1e9,
						'join_on_object':False,
						'event_object_processor':None
						}


PREBUILT = {'temporal_event_graph': temporal_event_graph,
			'general_event_graph': general_event_graph,
			'path_finder_graph': path_finder_graph}