from enum import Enum
from typing import Tuple, Any, List, Set
import tqdm
from pgeon.discretizer import Discretizer
from policy_graph.discretizer import AVPredicate
import pandas as pd
from discretizer.utils import Velocity, BlockProgress,IdleTime
from discretizer.discretizer_d0 import AVDiscretizer
from policy_graph.environment import AVEnvironment
from pgeon.policy_graph import PolicyGraph, PGBasedPolicy, PGBasedPolicyMode, PGBasedPolicyNodeNotFoundMode

class AVPolicyGraph(PolicyGraph):

    def __init__(self,
                 environment: AVEnvironment,
                 discretizer: Discretizer
                 ):
        super().__init__(environment, discretizer)
        

    ######################
    # FITTING
    ######################

    def _run_episode(self,
                     scene,
                     scene_detections,
                     verbose = False
                     ) -> List[Any]:

        """
            Discretizes a trajectory (list of states) and stores unique states and actions.

            Args:
                states: DataFrame containing state information of a scene for each time step.

            Returns:
                List containing tuples of (current state ID, action ID, next state ID).
        """
        trajectory = []
        in_intersection = False
        intersection_info = []

        consecutive_stopped_count = 0

        for i in range(len(scene)-1):
            sample_detections = scene_detections[scene_detections['sample_token'] == scene.iloc[i]['sample_token']]
            disc_state, action_id = self.discretizer._discretize_state_and_action(scene.iloc[i], scene.iloc[i+1], sample_detections)
            
            #check for intersection start
            block_progress = next((predicate for predicate in disc_state if predicate.predicate.__name__ == 'BlockProgress'), None)

            if block_progress == AVPredicate(BlockProgress, BlockProgress.INTERSECTION) and not in_intersection:
                in_intersection = True
                intersection_start = i
                start_intersection_x, start_intersection_y = scene.iloc[i][['x', 'y']] if i > 0 else scene.iloc[i+1][['x', 'y']]
                pre_intersection_x, pre_intersection_y = scene.iloc[i-1][['x', 'y']] if i > 0 else scene.iloc[i][['x', 'y']]

            
            #check for intersection end
            if in_intersection and block_progress != AVPredicate(BlockProgress, BlockProgress.INTERSECTION):

                in_intersection = False

                end_intersection_x, end_intersection_y = scene.iloc[i][['x', 'y']] if i+1<len(scene) else scene.iloc[i-1][['x', 'y']]
                post_intersection_x, post_intersection_y = scene.iloc[i+1][['x', 'y']] if i+1<len(scene) else scene.iloc[i][['x', 'y']]

                intersection_action = AVDiscretizer.determine_intersection_action((pre_intersection_x, pre_intersection_y, start_intersection_x, start_intersection_y), (end_intersection_x, end_intersection_y, post_intersection_x, post_intersection_y))
                intersection_info.append((intersection_start, intersection_action))
            
            
            
            if '2' in self.discretizer.id:
                velocity = next((predicate for predicate in disc_state if predicate.predicate.__name__ == 'Velocity' ), None)
                if velocity == AVPredicate(Velocity, Velocity.STOPPED):
                    consecutive_stopped_count+=1
                    if consecutive_stopped_count>1:
                        state = list(disc_state)
                        idle_predicate_idx = next((i for i, predicate in enumerate(state) if predicate.predicate.__name__ == 'IdleTime'), None)
                        state[idle_predicate_idx] = AVPredicate(IdleTime,[IdleTime(consecutive_stopped_count-1)])
                        disc_state = tuple(state)
                else:
                    consecutive_stopped_count=0




            if '0' in self.discretizer.id:
                # Filter out the BlockProgress predicate from the discretized state
                disc_state = tuple(predicate for predicate in disc_state if predicate.predicate.__name__ != 'BlockProgress')
            trajectory.extend([disc_state, action_id])        


        #add last state
        last_state_to_discretize = scene.iloc[len(scene)-1][self.discretizer.vehicle_state_columns]
        last_state_detections = scene_detections[scene_detections['sample_token'] == scene.iloc[len(scene)-1]['sample_token']]

        disc_last_state = self.discretizer.discretize(last_state_to_discretize, last_state_detections)
        if '0' in self.discretizer.id:
            disc_last_state = tuple(predicate for predicate in disc_last_state if predicate.predicate.__name__ != 'BlockProgress')

        if '2' in self.discretizer.id:
            velocity = next((predicate for predicate in disc_last_state if predicate.predicate.__name__ == 'Velocity' ), None)
            if velocity == AVPredicate(Velocity, Velocity.STOPPED):
                consecutive_stopped_count+=1
                if consecutive_stopped_count>1:
                    state = list(disc_last_state)
                    idle_predicate_idx = next((i for i, predicate in enumerate(state) if predicate.predicate.__name__ == 'IdleTime'), None)
                    state[idle_predicate_idx] = AVPredicate(IdleTime,[IdleTime(consecutive_stopped_count-1)])
                    disc_last_state = tuple(state)  
            else:
                consecutive_stopped_count=0



        trajectory.append(disc_last_state)


        self.discretizer.assign_intersection_actions(trajectory, intersection_info, verbose)                

        return trajectory
    
    
    def fit(self,
            scenes: pd.DataFrame,
            detections: pd.DataFrame,
            update: bool = False,
            verbose = True
            ):

        if not update:
            self.clear()
            self._trajectories_of_last_fit = []
            self._is_fit = False

        scene_groups = scenes.groupby('scene_token')
        progress_bar = tqdm.tqdm(total=len(scene_groups), desc='Fitting PG from scenes...')

        progress_bar.set_description('Fitting PG from scenes...')

        for scene_token, groups in scene_groups:
            scene_detections = detections[detections['scene_token']==scene_token]
            print(f'Scene token: {scene_token}')
            trajectory_result = self._run_episode(groups, scene_detections, verbose)
            self._update_with_trajectory(trajectory_result)
            self._trajectories_of_last_fit.append(trajectory_result)

            progress_bar.update(1)

        self._normalize()
        self._is_fit = True


        return self
    



    ######################
    # EXPLANATIONS
    ######################

    def _is_predicate_in_pg_and_usable(self, predicate) -> bool:
        return self.has_node(predicate) and len(self[predicate]) > 0
        
    def get_nearest_predicate(self, input_predicate: Tuple[Enum], verbose=False):
        """ Returns the nearest predicate on the PG. If already exists, then we return the same predicate. If not,
        then tries to change the predicate to find a similar state (Maximum change: 1 value).
        If we don't find a similar state, then we return None

        Args:
        	input_predicate: Existent or non-existent predicate in the PG
        	verbose: Prints additional information
        Returns: 
        	Nearest predicate

        """
        # Predicate exists in the MDP
        if self.has_node(input_predicate):
            if verbose:
                print('NEAREST PREDICATE of existing predicate:', input_predicate)
            return input_predicate
        else:
            if verbose:
                print('NEAREST PREDICATE of NON existing predicate:', input_predicate)

            nearest_state_generator = self.discretizer.nearest_state(input_predicate)
            new_predicate = input_predicate
            try:
                while not self._is_predicate_in_pg_and_usable(new_predicate):
                    new_predicate = next(nearest_state_generator)

            except StopIteration:
                print("No nearest states available.")
                new_predicate = None

            if verbose:
                print('\tNEAREST PREDICATE in PG:', new_predicate)
            return new_predicate     


    def question3(self, predicate, action, greedy=False, verbose=False):
        """
        Answers the question: Why do you not perform action X in state Y?
        """
        if verbose:
            print('***********************************************')
            print('* Why did not you perform X action in Y state?')
            print('***********************************************')

        if greedy:
            mode = PGBasedPolicyMode.GREEDY
        else:
            mode = PGBasedPolicyMode.STOCHASTIC
        pg_policy = PGBasedPolicy(self, mode)
        best_action = pg_policy.act_upon_discretized_state(predicate)
        result = self.nearby_predicates(predicate, greedy)
        explanations = []
        if verbose:
            print('I would have chosen:', best_action)
            print(f"I would have chosen {action} under the following conditions:")
        for a, v, diff in result:
            if a.value == action:
                if verbose:
                    print(f"Hypothetical state: {v}")
                    for predicate_key,  predicate_value in diff.items():
                        print(f"   Actual: {predicate_key} = {predicate_value[0]} -> Counterfactual: {predicate_key} = {predicate_value[1]}")
                explanations.append(diff)
        if len(explanations) == 0 and verbose:
            print("\tI don't know where I would have ended up")
        return explanations


    


class AVPGBasedPolicy(PGBasedPolicy):
    def __init__(self,
                 policy_graph: AVPolicyGraph,
                 mode: PGBasedPolicyMode,
                 node_not_found_mode: PGBasedPolicyNodeNotFoundMode = PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM
                 ):
        
        
        self.pg = policy_graph
        self.dt=0.5
        self.wheel_base = 2.588 #Ref: https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        self.min_steer_angle = -7.7
        self.max_steer_angle = 6.3
        
        assert mode in [PGBasedPolicyMode.GREEDY, PGBasedPolicyMode.STOCHASTIC], \
            'mode must be a member of the PGBasedPolicyMode enum!'
        self.mode = mode

        
        assert node_not_found_mode in [PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM,
                                       PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES], \
            'node_not_found_mode must be a member of the PGBasedPolicyNodeNotFoundMode enum!'
        self.node_not_found_mode = node_not_found_mode

        self.all_possible_actions = self._get_all_possible_actions()    
    
    
    def _get_action_probability(self, predicate, action_id:int )->float:
        """
        Returns the probability P(a|s) for a given action and state (predicate).

        Args:
            predicate: The state for which the action probability is calculated.
            action_id: The action for which the probability is required.

        Returns:
            The probability of the given action for the given state.
        """
        action_weights = self._get_action_probability_dist(predicate)
        action_dict = dict(action_weights)
        return action_dict.get(action_id, 0.0)

    def act_upon_discretized_state(self, predicate):
        if self.pg.has_node(predicate) and len(self.pg[predicate]) > 0:
            action_prob_dist = self._get_action_probability_dist(predicate)
        else:
            if self.node_not_found_mode == PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM:
                action_prob_dist = [(a, 1 / len(self.all_possible_actions)) for a in self.all_possible_actions]
            elif self.node_not_found_mode == PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES:
                nearest_predicate = self.pg.get_nearest_predicate(predicate)
                if nearest_predicate is not None:  
                    action_prob_dist = self._get_action_probability_dist(nearest_predicate)
                    if action_prob_dist == []: #NOTE: we handle the case in which there is a nearest state, but this state has no 'next_state' (i.e. destination node of a scene)
                        action_prob_dist = [(a, 1 / len(self.all_possible_actions)) for a in self.all_possible_actions]
                else:
                    # Fallback if no nearest predicate is found
                    action_prob_dist = [(a, 1 / len(self.all_possible_actions)) for a in self.all_possible_actions]
            else:
                raise NotImplementedError
        return self._get_action(action_prob_dist) 



    

    def get_transition_prob(self, s_t, s_t1, a_t, eps):
        """
        Args:
            s_t: Current state.
            s_t1: Next state.
            a_t: Action taken.
            eps: Small constant to avoid log of zero.

        Returns:
            P(s',a|s)
        """
        edge_data = self.pg.get_edge_data(s_t, s_t1, default={})
        
        if a_t in edge_data and 'probability' in edge_data[a_t]:
            probability = edge_data[a_t]['probability']
        else:
            probability = eps

        return max(probability, eps)



        
