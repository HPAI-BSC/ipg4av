import argparse
import pgeon.policy_graph as PG
from metrics.utils import compute_external_entropy
from policy_graph.pg_utils import setup_env_and_disc, DATA_ROOT
import csv
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--pg_folder',
    #                    help='The folder where the Policy Graph is stored.')
    parser.add_argument('--pg_id',
                        help='The id of the Policy Graph to load.') #name of file without extension
    parser.add_argument('--discretizer', 
                        help='Specify the discretizer of the input data.', choices=['0a', '0b', '1a','1b', '2a', '2b'], default='0a')
    
    
    args = parser.parse_args()

    pg_id, discretizer_id = args.pg_id, args.discretizer

    #discretizer_id = pg_id[pg_id.find('D') + 1].split('_')[0]
    
    nodes_path = f'{DATA_ROOT}/policy_graphs/{pg_id}_nodes.csv'
    edges_path = f'{DATA_ROOT}/policy_graphs/{pg_id}_edges.csv'
    environment,discretizer = setup_env_and_disc(discretizer_id=discretizer_id)
    
    # Load PG
    pg = PG.PolicyGraph.from_nodes_and_edges(nodes_path, edges_path, environment, discretizer)

    ##################
    # static metrics
    ##################

    output_path = f'{DATA_ROOT}/policy_graphs/entropy.csv'
    file_exists = os.path.isfile(output_path)
    with open(output_path, 'a',newline='') as f:
        csv_w = csv.writer(f)
        if not file_exists:
            header = ['pg_id', 'Expected_Hs', 'Expected_Ha', 'Expected_Hw']
            csv_w.writerow(header)
        entropy_values = compute_external_entropy(pg)
        new_row = [
            pg_id,
            entropy_values.get('Expected_Hs'),
            entropy_values.get('Expected_Ha'),
            entropy_values.get('Expected_Hw')
            ]
        csv_w.writerow(new_row)   
  
    
    print(f'Successfully evaluated Policy Graph with {len(pg.nodes)} nodes and {len(pg.edges)} edges.')
    
