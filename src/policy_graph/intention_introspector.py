from typing import Set, List, Dict, Optional, Any
import numpy as np
import networkx.classes.coreviews
from discretizer.utils import get_action_from_id, get_action_id
from policy_graph.discretizer import AVPredicate
from policy_graph.policy_graph import AVPolicyGraph
from policy_graph.desire import AVDesire, DesireCategory
from enum import Enum
from pgeon.intention_introspector import IntentionIntrospector
import pprint
import time


class AVIntentionIntrospector(IntentionIntrospector):
    def __init__(self, desires: Set[AVDesire], pg:AVPolicyGraph):
        self.desires = desires
        self.pg = pg
        self.all_actions_ids= [get_action_id(action) for action in self.pg.discretizer.all_actions()]
        self.intention: Dict[Set[AVPredicate], Dict[AVDesire, float]] = {}
        self.register_all_desires( self.desires)

    
    def find_intentions(self, commitment_threshold: float):
        total_results = {desire.name: self.get_intention_metrics(commitment_threshold, desire) for desire in self.desires}
        return total_results

    
    @staticmethod
    def check_state_condition(node, atom, condition_values):
        """
        Given a state (node) of predicates, checks if its predicates have the values in the condition.
        """
        for elem in node:
            if elem.predicate == atom and elem.value[0] in condition_values:
                return True
        return False

    @staticmethod
    def compute_differences(node1: Set[AVPredicate], node2: Set[AVPredicate]):
        shared = node1.intersection(node2)
        added = node2 - shared
        removed = node1 - shared
        return shared, added, removed


    def check_desire(self, node: Set[AVPredicate], desire_clause: Dict[Enum, Set[Any]], actions_id:List[int]) -> bool:

        # Returns None if desire is not satisfied. Else, returns probability of fulfilling desire
        #   ie: executing the action when in node
        desire_clause_satisfied = True
        for atom, condition_values in desire_clause.items():
            desire_clause_satisfied = desire_clause_satisfied and self.check_state_condition(node, atom, condition_values)
            if not desire_clause_satisfied:
                return None
        action_probabilities = [self.get_action_probability(node, action_id) for action_id in actions_id]
        total_probability = np.sum(action_probabilities)
        return total_probability if total_probability > 0 else None


        #if not all(self.check_state_condition(node, atom, condition_values) 
        #        for atom, condition_values in desire_clause.items()):
        #    return None
        #total_probability = sum(self.get_action_probability(node, action_id) for action_id in actions_id)
        #return total_probability if total_probability > 0 else None

        
    @staticmethod
    def get_prob(unknown_dict: Optional[Dict[str, object]]):
        if unknown_dict is None:
            return 0
        else:
            return unknown_dict.get("probability", 0)

    def get_action_probability(self, node: Set[AVPredicate], action_id: int):
        try:
            destinations: networkx.classes.coreviews.AdjacencyView = self.pg[node]
            return sum([self.get_prob(self.pg.get_edge_data(node, destination, key=action_id))
                        for destination in destinations])
        except KeyError:
            print(f'Warning: State {node} has no sampled successors which were asked for')
            return 0

    
    def update_intention(self, node: Set[AVPredicate], desire: AVDesire, probability: float,
                         ):
        if node not in self.intention:
            self.intention[node] = {}
        current_intention_val = self.intention[node].get(desire, 0)

        self.intention[node][desire] = current_intention_val + probability

    
    def propagate_intention(self, node: Set[AVPredicate], desire: AVDesire, probability,
                            stop_criterion=1e-4):
        self.update_intention(node, desire, probability)

        for coincider in self.pg.predecessors(node):
            
            if self.check_desire(coincider, desire.clause, desire.actions) is None:
                
                successors = list(self.pg.successors(coincider))

                coincider_transitions = {
                    (successor, action_id): self.get_prob(self.pg.get_edge_data(coincider, successor, key=action_id))
                    for successor in successors
                    for action_id in self.all_actions_ids 
                }
            else:
                
                successors = list(self.pg.successors(coincider))

                # If coincider can fulfill desire themselves, do not propagate it through desirable actions branch
                coincider_transitions = {
                    (successor, action_id): self.get_prob(self.pg.get_edge_data(coincider, successor, key=action_id))
                    for successor in successors
                    for action_id in self.all_actions_ids if action_id not in desire.actions
                }

                
            prob_of_transition = sum(prob for (succ, _), prob in coincider_transitions.items() if succ == node)
            new_coincider_intention_value = prob_of_transition * probability

            if new_coincider_intention_value >= stop_criterion:
                try:
                    self.propagate_intention(coincider, desire, new_coincider_intention_value)
                except RecursionError:
                    print("Maximum recursion reach, skipping branch with intention of", new_coincider_intention_value)

   
    def register_desire(self, desire: AVDesire):
        for node in self.pg.nodes:
            p = self.check_desire(node, desire.clause, desire.actions)
            if p is not None:
                self.propagate_intention(node, desire, p)

    def register_all_desires(self, desires: Set[AVDesire]):
        for desire in desires:

            print(f'Registering desire: {desire.name}')

            start_time = time.time()
            self.register_desire(desire)
            end_time = time.time()

            print(f"Execution time: { end_time - start_time} seconds")


    ##############################
    # Intention and AVDesire metrics
    ##############################
    
    
    def get_intention_metrics(self, commitment_threshold:float, desire: AVDesire): 
        """
        Computes intention metrics for a specific desire or for any desire.
        """
        if desire.category is not DesireCategory.ANY:
            intention_full_nodes = [node for node in self.pg.nodes if node in self.intention and desire in self.intention[node] and self.intention[node][desire]>=commitment_threshold]
            node_probabilities = np.array([self.pg.nodes[node]['probability'] for node in intention_full_nodes])
            intention_probability = np.sum(node_probabilities)
            intention_vals = np.array([self.intention[node][desire] for node in intention_full_nodes])
            expected_int_probability = np.dot(intention_vals, node_probabilities)/intention_probability if intention_probability >0 else 0
        else:
            intention_full_nodes = [
                node for node in self.pg.nodes 
                if node in self.intention and 
                any(self.intention[node][d] >= commitment_threshold for d in self.intention[node])] 
            
            if intention_full_nodes:
                node_probabilities = np.array([self.pg.nodes[node]['probability'] for node in intention_full_nodes])
                max_intention_vals = np.array([max(self.intention[node].values()) for node in intention_full_nodes])

                intention_probability = np.sum(node_probabilities)
                expected_int_probability = np.dot(max_intention_vals, node_probabilities)/intention_probability if intention_probability >0 else 0
            else:
                intention_probability = 0
                expected_int_probability = 0

        return intention_probability, expected_int_probability


    def get_desire_metrics(self, desire):
        desire_prob, expected_action_prob = 0,0
        desire_nodes = [(node, self.check_desire(node, desire.clause, desire.actions)) for node in self.pg.nodes if self.check_desire(node,desire.clause, desire.actions) is not None]
        if desire_nodes:
            node_probabilities = np.array([self.pg.nodes[node]['probability'] for node,_ in desire_nodes])
            desire_prob = np.sum(node_probabilities)
            expected_action_prob = np.dot([p for _, p in desire_nodes],node_probabilities)/desire_prob
        return desire_prob, expected_action_prob


    def find_desires(self):
        total_results = {desire.name: self.get_desire_metrics(desire) for desire in self.desires}
        return total_results

    ##################
    # Questions
    ##################

    
    def what_question(self, node:Set[AVPredicate], commitment_threshold:float):
        print(f"What do you intend to do in state {node}?")
        #all desires with an Id (s) over a certain threshold
        if node in self.intention:
            intented_desire = [(d.name, self.intention[node][d]) for d in self.intention[node] if self.intention[node][d] >= commitment_threshold ]
            print(f'Attributed intention of the following desires: {intented_desire}')
        else:
            print("No attributed intention in this state.")

    def how_question(self, desire:AVDesire, state:Set[AVPredicate]):
        print(f'How do you plan to fulfill {desire.name} from state {state}?')
        paths = self.how(desire, state)
        path_staging = []
        curr_state = state
        if paths:
            for path in paths[:-1]:
                
                action_id, new_state, new_intention= path
                a = get_action_from_id(action_id)
                pred_difs = self.compute_differences(set(curr_state), set(new_state))
                equal, added, removed = pred_difs
                path_staging.append((a, {'Added':added, 'Removed':removed}, new_intention))
                
                curr_state = new_state
            
            path_staging.append([get_action_from_id(action_id) for action_id in paths[-1][0]])

            print(desire, " from the node description:")
            pprint.pprint(state)
            pprint.pprint(path_staging)

        else:
            print('From such state there is not path to fulfill the desire.')
        

        
    def question5(self, desire: AVDesire):
        """
        Calculates the probability of performing a desirable action given the state region.
        """
        print(f"How likely are you to perform your desirable action {desire.actions} when you are in the state region {desire.clause}?")
        print(f"Probability: {self.get_desire_metrics(desire)[1]}")
   
    
    def question4(self, desire: AVDesire):
        print(f"How likely are you to find yourself in a state where you can fulfill your desire {desire.name} by performing the action {desire.actions}?")
        print(f"Probability: {self.get_desire_metrics(desire)[0]}")
     
    
    def get_successors(self, state:Set[AVPredicate]):
        # Return a list of tuples (successor, action_id) where each successor is connected by an action 
        successors = list(self.pg.successors(state))
        return [(successor, action_id) for successor in successors
            for action_id in self.pg.get_edge_data(state, successor).keys()]


    
    def how(self, desire:AVDesire, state: Set[AVPredicate]):
        
        if self.check_desire(state, desire.clause, desire.actions) is not None:
            return [[ desire.actions, None, None ]]
 
        successors = list(self.pg.successors(state))
        intention_vals =  [(successor, self.intention[successor][desire]) for successor in successors if successor in self.intention and desire in self.intention[successor] ]
        if not intention_vals:
            return []
        

        max_intention = max(intention_vals, key=lambda x: x[1])[1]

        # List of all successors (1 or more) that have the maximum intention value
        max_successors = [
            (successor, action) 
            for successor, _ in intention_vals
            if self.intention[successor].get(desire, 0) == max_intention
            for action in self.pg.get_edge_data(state, successor).keys()
        ]
        # Get action probabilities for each action and successor
        action_probabilities = {
            (successor, action): self.get_action_probability(state, action)
            for successor, action in max_successors
        }

        # Select the (successor, action) tuple with the highest action probability

        best_successor, best_action = max(action_probabilities, key=action_probabilities.get)
        
        return [[best_action,best_successor, max_intention]] + self.how(desire, best_successor)

    
   
    
    
    
    
    def why_action(self, state: Set[AVPredicate], action_id:int, c_threshold:float, probability_threshold:float=0 ):
        # Credits: pgeon library
        # probability_threshold: minimum probability of intention increase by which we attribute the agent is trying to
        # further an intention. eg: if it has 5% prob of increasing a desire but 95% of decreasing it
        attr_ints = {d: self.intention[state][d] for d in self.intention[state] if self.intention[state][d]>=c_threshold}
        if len(attr_ints) == 0:
            return {}
        else:
            successors = {
                    successor: (self.get_prob(self.pg.get_edge_data(state, successor, key=action_id)), self.intention[successor])
                    for successor in list(self.pg.successors(state))
                } 

            int_increase = {}

            for d, I_val in attr_ints.items():
                int_increase[d] = dict()
                int_increase[d]['expected']=0
                int_increase[d]['prob_increase'] = 0
                int_increase[d]['expected_pos_increase'] = 0
                for successor, (p, ints) in successors.items():
                    int_value = ints.get(d, 0) if ints.get(d) is not None else 0
                    int_increase[d]['expected'] += p*int_value
                    int_increase[d]['prob_increase'] += p if int_value >= I_val else 0
                    int_increase[d]['expected_pos_increase'] += p * int_value if int_value >= I_val else 0
                int_increase[d]['expected'] -= I_val
                int_increase[d]['expected_pos_increase'] = \
                    int_increase[d]['expected_pos_increase']/int_increase[d]['prob_increase']\
                        if int_increase[d]['prob_increase']>0 else 0
                
                int_increase[d]['expected_pos_increase'] -=I_val
                if int_increase[d]['prob_increase'] <=probability_threshold:
                    # Action detracts from intention. If threshold = 0, it always detracts. Else: it has at least
                    # 1-threshold probability of decreasing intention.
                    del (int_increase[d])
        
            return int_increase
        

    def why_question(self, state: Set[AVPredicate], action_id:int, c_threshold:float, probability_threshold:float):
        # Credits: pgeon library
        ints = self.why_action(state, action_id, c_threshold,probability_threshold)
        action = get_action_from_id(action_id)
        if ints == {}:
            print(f"Action {action} does not seem intentional.")
    
        for d, info in ints.items():
            print(f'I want to do {action} for the purpose of furthering {d} as', end=' ')
            if info['expected']>0:
                print(f"if it expected to increase my intention by {info['expected']:.3f}")
            elif info['expected']==0:
                print(f"it will keep my intention of fulfilling it the same.")
            else: 
                print(f"it has a {info['prob_increase']:.3f} probability of an expected intention "
                  f"increase of {info['expected_pos_increase']:.3f}")
