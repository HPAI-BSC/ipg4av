from policy_graph.intention_introspector import AVIntentionIntrospector
import pgeon.policy_graph as PG
import pickle
from experiments.desire_config import DESIRE_MAPPING,ANY
import argparse
from policy_graph.pg_utils import setup_env_and_disc, DATA_ROOT
from policy_graph.plot_utils import plot_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pg_id',
                        help='The id of the Policy Graph to load.')
    parser.add_argument('--discretizer', 
                        help='Specify the discretizer of the input data.', choices=['0a', '0b', '1a','1b', '2a', '2b'], default='0a')
    parser.add_argument('--commitment', type=float, default=0.5,
                        help='Commitment threshold.')
    parser.add_argument('--desire_plot', help='Whether to plot desire metrics or not',
                        action='store_false')
    parser.add_argument('--intention_plot', help='Whether to plot intention metrics or not',
                        action='store_false')
    parser.add_argument('--verbose', help='Whether to output log statements or not',
                        action='store_true')
    
    #TODO: allow user to specify which desire to evaluate
        
    args = parser.parse_args()
    pg_id, discretizer_id, commitment_threshold, desire_plot, intention_plot, verbose = args.pg_id, args.discretizer, args.commitment, args.desire_plot, args.intention_plot, args.verbose

    nodes_path = f'{DATA_ROOT}/policy_graphs/{pg_id}_nodes.csv'
    edges_path = f'{DATA_ROOT}/policy_graphs/{pg_id}_edges.csv'
        
    
    # Load PG
    environment,discretizer = setup_env_and_disc(discretizer_id=discretizer_id)
    pg = PG.PolicyGraph.from_nodes_and_edges(nodes_path, edges_path, environment, discretizer)


    # Define desires
    desires = list(DESIRE_MAPPING.values())
    ii = AVIntentionIntrospector(desires, pg)

    # Compute desire metrics
    if verbose:
        print(f'Computing desire metrics... \n')
    desire_metrics = ii.find_desires()

    #Compute intention metrics
    if verbose:
        print(f'Computing intention metrics... \n')
    intention_metrics = ii.find_intentions(commitment_threshold)
    intention_metrics['any']=ii.get_intention_metrics(commitment_threshold,ANY)

    # Save IPG into pkl file
    with open(f'{DATA_ROOT}/intentions/I{pg_id}.pkl', 'wb') as f:
        pickle.dump(ii, f)

    if verbose:
        print(f'IPG successfully stored in {DATA_ROOT}/intentions/I{pg_id}.pkl \n')
    
    if desire_plot:
        plot_metrics(desire_metrics, discretizer_id, metric_type='Desire', output_folder=f'{DATA_ROOT}/intentions/img', fig_size=(90, 30))
        if verbose:
            print(f'Plots successfully stored in {DATA_ROOT}/intentions/img\n')

    if intention_plot:
        plot_metrics(intention_metrics, discretizer_id, metric_type='Intention', output_folder=f'{DATA_ROOT}/intentions/img', fig_size=(90,30))
        if verbose:
            print(f'Plots successfully stored in {DATA_ROOT}/intentions/img \n')

    