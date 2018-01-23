import numpy as np

nevents = 150
nodes = np.arange(0,100)

# Directed events
directed = []
t = 0
for _ in range(nevents):
    u,v = np.random.choice(nodes, replace=False, size=2)
    directed.append({'source':u,
    					  'target':v,
    					  'time':t})
    t += 1

# Directed hyper-events
directed_hyper = []
t = 0
for _ in range(nevents):
    r = np.random.randint(2,6)
    s = np.random.randint(1,r+1)
    u = np.random.choice(nodes, replace=False, size=r)
    directed_hyper.append({'source':u[:s],
						    	'target':u[s:],
						    	'time':t})
    t += 1

# Directed hyper-events (single source)
directed_hyper_single = []
t = 0
for _ in range(nevents):
    r = np.random.randint(2,6)
    u = np.random.choice(nodes, replace=False, size=r)
    directed_hyper_single.append({'source':u[:1],
    								   'target':u[1:],
    								   'time':t})
    t += 1

# Undirected hyper-events
undirected_hyper = []
t = 0
for _ in range(nevents):
    r = np.random.randint(1,6)
    u = np.random.choice(nodes, replace=False, size=r)
    undirected_hyper.append({'source':u[:1],
    							  'target':[],
    							  'time':t})
    t += 1

# Extra columns


# Non-integer labels