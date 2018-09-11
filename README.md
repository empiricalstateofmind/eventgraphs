# Eventgraphs [![PyPI version](https://img.shields.io/pypi/v/eventgraphs.svg)](https://pypi.org/project/eventgraphs/) <img src="https://travis-ci.org/empiricalstateofmind/eventgraphs.svg?branch=master" alt="build:passed">
A Python library for analysing sequences of event-based data and temporal networks.

## Features

1. Build EventGraphs (static representation of a temporal network) using arbitrary joining rules.
2. Calculate inter-event time distributions (including motif-dependent inter-event times).
3. Calculate temporal motif distributions.
4. Network decomposition into temporal components.
5. Calculate event centralities.
6. Dimension-reduction and component clustering.
7. Plot network components and EventGraphs.
8. IO functionality (saving as JSON).

## Installation

For the latest version, installation from Github is recommended. S
The PyPI package is updated periodically.

#### Install from Github (latest version, recommended)

```bash
pip install git+https://github.com/empiricalstateofmind/eventgraphs
```

#### Install from PyPI

```bash
pip install eventgraphs
```

## Getting Started

The best place to get started using EventGraphs is with the tutorial [here](/examples/eventgraphs_tutorial.ipynb).

## References

**Event Graphs: Advances and Applications of Second-Order Time-Unfolded Temporal Network Models**. *Andrew Mellor* (2018) [[ArXiv](https://arxiv.org/abs/1809.03457)]

**Analysing Collective Behaviour in Temporal Networks Using Event Graphs and Temporal Motifs**. *Andrew Mellor* (2018) [[ArXiv](https://arxiv.org/abs/1801.10527)]

**The Temporal Event Graph**. *Andrew Mellor* (2017) [[Journal of Complex Networks](https://academic.oup.com/comnet/advance-article-abstract/doi/10.1093/comnet/cnx048/4360827)]
[[ArXiv](https://arxiv.org/abs/1706.02128)] 

Please consider citing these papers if you use this code for further research.