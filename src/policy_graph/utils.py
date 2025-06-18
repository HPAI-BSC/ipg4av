from typing import List, Tuple
from policy_graph.intention_introspector import AVIntentionIntrospector
from policy_graph.discretizer import AVPredicate
from typing import List, Tuple


def get_trajectory_metrics(ii: AVIntentionIntrospector, trajectory:List[Tuple[Tuple[AVPredicate], int]]):
    intention_track = []
    prob_track = []
    desire_fulfill_track = {}
    episode_length = len(trajectory)

    for t, (state, action_idx) in enumerate(trajectory):
        curr_node = state
        prob_track.append(ii.pg.nodes[curr_node]['probability'])
        intention_track.append(ii.intention.get(curr_node, {}))
        
        for desire in ii.desires:
            curr_intention = ii.intention.get(curr_node, {})
            if curr_intention.get(desire, 0) > 0.999:
                desire_fulfill_track[t] = desire.name
            #if ii.check_desire(curr_node, desire.clause, desire.actions) is not None and action_idx in desire.actions:
                #desire_fulfill_track[t] = desire.name
    return desire_fulfill_track, episode_length, intention_track, prob_track

    
