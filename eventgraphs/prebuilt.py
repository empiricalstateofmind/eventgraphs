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

def basic_event_processor(e1, e2):
    """

    """

    if hasattr(e1, 'duration'):
        dt = e2.time - (e1.time + e1.duration)
    else:
        dt = e2.time - e1.time

    if dt > 0:
        return True, dt
    else:
        return False, dt


temporal_event_graph = {'event_processor': basic_event_processor,
                        'subsequential': True,
                        'delta_cutoff': 1e9,
                        'join_on_object': False,
                        'event_object_processor': None
                        }

general_event_graph = {'event_processor': basic_event_processor,
                       'subsequential': False,
                       'delta_cutoff': 1e9,
                       'join_on_object': False,
                       'event_object_processor': None
                       }


def path_finding_event_processor(e1, e2):
    """
    """

    if hasattr(e1, 'duration'):
        dt = e2.time - (e1.time + e1.duration)
    else:
        dt = e2.time - e1.time

    if len(e2.target) == 0:
        return False, dt

    if isinstance(e1.target, str) or isinstance(e1.target, int):
        targets = [e1.target]
    else:
        targets = e1.target
    if isinstance(e2.source, str) or isinstance(e2.source, int):
        sources = [e2.source]
    else:
        sources = e2.source

    for node in targets:
        if node in sources:
            return True, dt

    return False, dt


path_finder_graph = {'event_processor': path_finding_event_processor,
                     'subsequential': False,
                     'delta_cutoff': 1e9,
                     'join_on_object': False,
                     'event_object_processor': None
                     }

PREBUILT = {'temporal_event_graph': temporal_event_graph,
            'general_event_graph': general_event_graph,
            'path_finder_graph': path_finder_graph}
