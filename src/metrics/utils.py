import numpy as np
import pandas as pd
from typing import Tuple, List
from pgeon.policy_graph import PolicyGraph, PGBasedPolicy
from collections import defaultdict
from policy_graph.environment import AVEnvironment

##########################
# Negative Log-Likelihood
##########################

def compute_scene_nll(pg_agent:PGBasedPolicy, trajectory:List, eps= 0.00001, verbose: bool = False) -> Tuple[float, float]:
        """
        Computes the likelihood of trajectory for a given scene.

        Args:
            pg: policy_graph
            trajectory: discretized state-action trajectory
            verbose: If True, prints detailed information during testing
        Return:
            nll_a: Negative log likelihood of the policy
            nll_w: Negative log likelihood of the world model
        """
        nll_a = 0
        nll_w = 0
        nll_tot = 0

        for t in range(0, len(trajectory) - 2, 2):
            s_t, a_t, s_next = trajectory[t], trajectory[t + 1], trajectory[t + 2]
            #P(a|s)
            if pg_agent.pg.has_node(s_t) and len(pg_agent.pg[s_t]) > 0:
                action_prob_dist = pg_agent._get_action_probability_dist(s_t) 
                p_a_given_s = next((prob for action, prob in action_prob_dist if action == a_t), eps)
            else:
                p_a_given_s = eps

            nll_a -=np.log(p_a_given_s)

        
            #P(s'|s,a)
            p_s_next_given_s_a = pg_agent.get_transition_prob(s_t, s_next, a_t, eps)/p_a_given_s
            nll_w -=np.log(p_s_next_given_s_a)
            
            nll_tot -=np.log(pg_agent.get_transition_prob(s_t, s_next, a_t, eps))
        
        if verbose:       
            print(f"Scene NLL for Policy: {nll_a:.4f}")
            print(f"Scene NLL for World Model: {nll_w:.4f}")
            print(f'Scene NLL for P(s_t1, a| s_t0): {nll_tot:.4f}')
            print()
        return nll_a, nll_w, nll_tot



def compute_global_nll(pg_agent:PGBasedPolicy, scenes: pd.DataFrame, detections:pd.DataFrame, verbose: bool = False, eps =  0.00001) -> pd.DataFrame:
        """
        Computes the likelihood of trajectory for all scenes in a set.

        Args:
            scenes set: DataFrame containing the test set with a 'scene_token' column
            detections: DataFrame containing information about detections for each sample in a scene.
            verbose: If True, prints detailed information during testing
            policy_mode: if 'original', computes the metric of the original agent. If 'random', computes the metrics for a random agent (P(a) is uniformly distributed.). 

        Returns:
            DataFrame containing the cumulative negative log-likelihoods for each scene.
        """
        if verbose:
            print('---------------------------------')
            print('* START \n')
        
        results: List[Tuple[str, float, float]] = []
        
        if pg_agent.pg.environment.city == 'all':
            for city in ['boston-seaport', 'singapore-hollandvillage','singapore-onenorth','singapore-queenstown']:
                city_environment = AVEnvironment(city)
                pg_agent.pg.discretizer.environment = city_environment
                city_test_df = scenes[scenes['location'] == city]
                if city_test_df.empty:
                    continue
                for scene_token, scene in city_test_df.groupby('scene_token'):
                        scene_detections = detections[detections['scene_token']==scene_token]
                        trajectory = pg_agent.pg._run_episode(scene, scene_detections, verbose)
                        scene_nll_action, scene_nll_world,scene_nll_tot = compute_scene_nll(trajectory, eps, verbose)
                        results.append((scene_token, scene_nll_action, scene_nll_world, scene_nll_tot))
        else:
            for scene_token, scene in scenes.groupby('scene_token'):
                    scene_detections = detections[detections['scene_token']==scene_token]
                    trajectory = pg_agent.pg._run_episode(scene,scene_detections, verbose)
                    scene_nll_action, scene_nll_world,scene_nll_tot = compute_scene_nll(trajectory, eps, verbose)
                    results.append((scene_token, scene_nll_action, scene_nll_world,scene_nll_tot))

        results_df = pd.DataFrame(results, columns=['scene_token', 'nll_a', 'nll_w', 'nll_tot'])

        avg_nll_a = results_df['nll_a'].mean()
        std_nll_a = results_df['nll_a'].std()

        avg_nll_w = results_df['nll_w'].mean()
        std_nll_w = results_df['nll_w'].std()        
        
        avg_nll_tot = results_df['nll_tot'].mean()
        std_nll_tot = results_df['nll_tot'].std()

        if verbose:
            print('* END ')
            print('---------------------------------')
            print('* RESULTS')
            
            print(f"Avg NLL for Policy: {avg_nll_a:.4f}")
            print(f"Std NLL for Policy: {std_nll_a:.4f}")
            print(f"Avg NLL for World Model: {avg_nll_w:.4f}")
            print(f"Std NLL for World Model: {std_nll_w:.4f}")
            print(f"Avg NLL for P(s',a|s): {avg_nll_tot:.4f}")
            print(f"Std NLL for P(s',a|s): {std_nll_tot:.4f}")

            

        return avg_nll_a, std_nll_a, avg_nll_w, std_nll_w, avg_nll_tot, std_nll_tot



############
# Entropy #
############

def compute_internal_entropy(policy_graph:PolicyGraph):
        """
        Compute the entropy metrics for the Policy Graph.

        Returns:
        - A dictionary containing the values of H(s), Ha(s), and Hw(s) for each state.
        """
        entropy_metrics = {}
        
        for state in policy_graph.nodes():
            # Compute probabilities for actions given the current state
            action_freq = defaultdict(int)
            total_action_freq = 0
            
            for _, next_state, data in policy_graph.out_edges(state, data=True):
                action = data['action']
                freq = data['frequency'] 
                action_freq[action] += freq
                total_action_freq += freq
            
            Ha = 0
            Hw = 0
            Hs = 0
            
            for action, freq in action_freq.items():
                P_a_given_s = freq / total_action_freq #P(a|s)
                #select the edges from the policy graph that correspond to a specific action.
                action_specific_out_edges = [edge for edge in policy_graph.out_edges(state, data=True) if edge[2]['action'] == action]
                Ha -=P_a_given_s * np.log2(P_a_given_s)
                
                for _, next_state, data in action_specific_out_edges:
                    P_s_a_given_s = data['probability'] #p(s',a|s) 
                    Hs -=P_s_a_given_s*np.log2(P_s_a_given_s)
                    Hw -=  P_s_a_given_s* np.log2(P_s_a_given_s/P_a_given_s) 
                    
            entropy_metrics[state] = {'p_s':policy_graph.nodes[state]['probability'],'Hs': Hs, 'Ha': Ha, 'Hw': Hw}
            
        return entropy_metrics
           
    
    
def compute_external_entropy(policy_graph: PolicyGraph):
    '''
    Compute weighted average of internal entropy metrics across all states in the policy graph.
      
    Returns:
    - A dictionary with E[Hs], E[Ha], E[Hw].
    '''
    entropy_metrics = compute_internal_entropy(policy_graph)
        
    expected_Hs = sum(metrics['p_s'] * metrics['Hs'] for metrics in entropy_metrics.values())
    expected_Ha = sum(metrics['p_s'] * metrics['Ha'] for metrics in entropy_metrics.values())
    expected_Hw = sum(metrics['p_s'] * metrics['Hw'] for metrics in entropy_metrics.values())
        
    return {'Expected_Hs': expected_Hs, 'Expected_Ha': expected_Ha, 'Expected_Hw': expected_Hw}
