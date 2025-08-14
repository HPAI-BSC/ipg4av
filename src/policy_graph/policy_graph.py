from enum import Enum
from typing import Tuple, Any, List
import tqdm
from policy_graph.discretizer import AVPredicate
import pandas as pd
import networkx as nx
from discretizer.predicates import Velocity, BlockProgress,IdleTime
from discretizer.discretizer_d0 import AVDiscretizer
from policy_graph.environment import AVEnvironment
from pgeon.policy_graph import PolicyGraph, PGBasedPolicy, PGBasedPolicyMode#, PGBasedPolicyNodeNotFoundMode

class AVPolicyGraph(PolicyGraph):

    def __init__(self,
                 environment: AVEnvironment,
                 discretizer: AVDiscretizer
                 ):
        super().__init__(environment, discretizer)
        

    ######################
    # FITTING
    ######################

    def _run_episode(self,
                     scene:pd.DataFrame,
                     scene_detections:pd.DataFrame,
                     verbose:bool = False
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
        scene_len = len(scene)

        # pre-group scene_detections by sample_token for faster lookup
        detections_groups = scene_detections.groupby('sample_token')


        for i in range(scene_len-1):
            
            curr_row = scene.iloc[i]
            next_row = scene.iloc[i + 1]
            sample_token = curr_row['sample_token']
            
            if sample_token in detections_groups.groups:
                sample_detections = detections_groups.get_group(sample_token)
            else:
                sample_detections = pd.DataFrame(columns=scene_detections.columns)

            #sample_detections = scene_detections[scene_detections['sample_token'] == scene.iloc[i]['sample_token']]
            disc_state, action_id = self.discretizer._discretize_state_and_action(curr_row, next_row, sample_detections)
            
            #check for intersection start
            block_progress = next((predicate for predicate in disc_state if predicate.predicate.__name__ == 'BlockProgress'), None)

            if block_progress == AVPredicate(BlockProgress, BlockProgress.INTERSECTION) and not in_intersection:
                in_intersection = True
                intersection_start = i

                if i > 0:
                    start_intersection_x, start_intersection_y = curr_row[['x', 'y']]
                    pre_intersection_x, pre_intersection_y = scene.iloc[i - 1][['x', 'y']]
                else:
                    start_intersection_x, start_intersection_y = next_row[['x', 'y']]
                    pre_intersection_x, pre_intersection_y = curr_row[['x', 'y']]
            
            #check for intersection end
            if in_intersection and block_progress != AVPredicate(BlockProgress, BlockProgress.INTERSECTION):

                in_intersection = False

                if i + 1 < scene_len:
                    end_intersection_x, end_intersection_y = curr_row[['x', 'y']]
                    post_intersection_x, post_intersection_y = next_row[['x', 'y']]
                else:
                    end_intersection_x, end_intersection_y = scene.iloc[i - 1][['x', 'y']]
                    post_intersection_x, post_intersection_y = curr_row[['x', 'y']]

                intersection_action = AVDiscretizer.determine_intersection_action((pre_intersection_x, pre_intersection_y, start_intersection_x, start_intersection_y), (end_intersection_x, end_intersection_y, post_intersection_x, post_intersection_y))
                intersection_info.append((intersection_start, intersection_action))
            
            
            # handle velocity and stopped count ONLY if discretizer id has '2'
            if '2' in self.discretizer.id:
                velocity = next((predicate for predicate in disc_state if predicate.predicate.__name__ == 'Velocity' ), None)
                if velocity == AVPredicate(Velocity, Velocity.STOPPED):
                    consecutive_stopped_count+=1
                    if consecutive_stopped_count>1:
                        state = list(disc_state)
                        
                        # replace IdleTime predicate with updated count
                        idle_predicate_idx = next((i for i, predicate in enumerate(state) if predicate.predicate.__name__ == 'IdleTime'), None)
                        state[idle_predicate_idx] = AVPredicate(IdleTime,[IdleTime(consecutive_stopped_count-1)])
                        disc_state = tuple(state)
                else:
                    consecutive_stopped_count=0 # reset count if moving



            # if discretizer id has '0', filter out BlockProgress predicates
            if '0' in self.discretizer.id:
                disc_state = tuple(predicate for predicate in disc_state if predicate.predicate.__name__ != 'BlockProgress')
            trajectory.extend([disc_state, action_id])        


        #process last state (no action after)
        last_row = scene.iloc[scene_len - 1]
        last_sample_token = last_row['sample_token']
        if last_sample_token in detections_groups.groups:
            last_state_detections = detections_groups.get_group(last_sample_token)
        else:
            last_state_detections = pd.DataFrame(columns=scene_detections.columns)

        # discretize last state only
        disc_last_state = self.discretizer.discretize(last_row[self.discretizer.vehicle_state_columns], last_state_detections)
        
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

        #assign intersection actions to the trajectory after processing
        self.discretizer.assign_intersection_actions(trajectory, intersection_info, verbose)                

        return trajectory
    
    
    def fit(self,
            scenes: pd.DataFrame,
            detections: pd.DataFrame,
            update: bool = False,
            verbose = True
            ):

        """
        Fit the policy graph using scenes and detections data.

        Args:
            scenes: DataFrame containing scene data with 'scene_token' column.
            detections: DataFrame containing detection data with 'scene_token' column.
            update: Whether to update existing policy graph or clear before.
            verbose: Whether to print logs and progress bar.
        """
        if not update:
            self.clear()
            self._trajectories_of_last_fit = []
            self._is_fit = False

        scene_groups = scenes.groupby('scene_token')
        detections_groups = detections.groupby('scene_token')

        progress_bar = tqdm.tqdm(total=len(scene_groups), desc='Fitting PG from scenes...')

        for scene_token, scene_data in scene_groups: 
            if verbose:
                print(f'Scene token: {scene_token}')
            
            if scene_token in detections_groups.groups:
                scene_detections = detections_groups.get_group(scene_token)
            else:
                scene_detections = pd.DataFrame(columns=detections.columns)

            #scene_detections = detections_groups.get_group(scene_token)
            trajectory_result = self._run_episode(scene_data, scene_detections, verbose)
            self._update_with_trajectory(trajectory_result)
            self._trajectories_of_last_fit.append(trajectory_result)

            progress_bar.update(1)

        self._normalize()
        self._is_fit = True


        return self
    
    

    def remove_isolated_nodes(self, verbose:bool=True):
        """
        Remove isolated (weakly connected component of size 1) nodes from a policy graph
        and clean them from self._trajectories_of_last_fit to avoid saving errors.
        """
        isolated_nodes = {
            next(iter(component))
            for component in nx.weakly_connected_components(self) # generator
            if len(component) == 1
        }
        if not isolated_nodes:
            if verbose:
                print("No isolated nodes found.")
            return 
        
        if verbose:
            print(f"Removing {len(isolated_nodes)} isolated nodes")
        
        # remove nodes from PG
        self.remove_nodes_from(isolated_nodes)

        # remove node information from trajectories
        if hasattr(self, "_trajectories_of_last_fit"):
            cleaned_trajectories = []
            for traj in self._trajectories_of_last_fit:
                cleaned_traj = []
                skip_next_action = False
                for idx, elem in enumerate(traj):
                    if idx % 2 == 0:  # state position
                        if elem not in isolated_nodes:
                            cleaned_traj.append(elem)
                            skip_next_action = False
                        else:
                            skip_next_action = True  
                    else:  # action position
                        if not skip_next_action:
                            cleaned_traj.append(elem)

                if cleaned_traj:
                    cleaned_trajectories.append(cleaned_traj)

            self._trajectories_of_last_fit = cleaned_trajectories

    




    ######################
    # EXPLANATIONS
    ######################

    def _is_predicate_in_pg_and_usable(self, predicate:Tuple[Enum]) -> bool:
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


    def question3(self, predicate:Tuple[Enum], action, greedy:bool=False, verbose:bool=False)-> List[Any]:
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


    


"""

class AVPGBasedPolicy(PGBasedPolicy):
    def __init__(self,
                 policy_graph: AVPolicyGraph,
                 mode: PGBasedPolicyMode,
                 node_not_found_mode: PGBasedPolicyNodeNotFoundMode = PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM
                 ):
        
        
        self.pg = policy_graph
        self.dt=0.5
        self.wheel_base = 2.588 #Reference: https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
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
        '''
        Returns the probability P(a|s) for a given action and state (predicate).

        Args:
            predicate: The state for which the action probability is calculated.
            action_id: The action for which the probability is required.

        Returns:
            The probability of the given action for the given state.
        '''
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

"""

    

    
