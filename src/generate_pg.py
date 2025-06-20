import argparse
from pathlib import Path
from policy_graph.environment import AVEnvironment
from discretizer.discretizer_d0 import AVDiscretizer
from discretizer.discretizer_d1 import AVDiscretizerD1
from discretizer.discretizer_d2 import AVDiscretizerD2
import policy_graph.policy_graph as PG

import pandas as pd
import networkx as nx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input data folder of states and actions.', default=".")
    parser.add_argument('--file', help='Input data file name of states and actions.')
    parser.add_argument('--city', help='Specify city to consider when building the PG.', choices=['all', 'b','s1','s2', 's3'], default="all")
    parser.add_argument('--weather', help='Specify whether the Policy Graph should contain bad weather scenes only (with rain), good weather scenes only or all.', default='all', choices=['all','rain','no_rain'])
    parser.add_argument('--tod', help='Flag to specify whether the Policy Graph should contain night scenes or day scenes.',  default='all', choices=['all','day','night'])
    parser.add_argument('--discretizer', help='Specify the discretizer of the input data.', choices=['0a', '0b', '1a','1b', '2a', '2b'], default='0a')
    parser.add_argument('--output', help='Which format to output the Policy Graph',
                        default='csv', choices=['pickle', 'csv', 'gram'])
    parser.add_argument('--verbose', help='Whether to make the Policy Graph code output log statements or not',
                        action='store_true')
     
    args = parser.parse_args()
    data_folder, data_file, city_id, verbose, output_format, discretizer_id, weather, tod = args.input, args.file, args.city, args.verbose, args.output, args.discretizer, args.weather, args.tod
   

    dtype_dict = {
        'modality': 'category',  # for limited set of modalities, 'category' is efficient
        'scene_token': 'str',  
        'steering_angle': 'float64', 
        'rain': 'int',
        'night':'int',
        'location': 'str',
        'timestamp': 'str',  # To enable datetime operations
        'rotation': 'object',  # Quaternion (lists)
        'x': 'float64',
        'y': 'float64',
        'z': 'float64',
        'yaw': 'float64',  
        'velocity': 'float64',
        'acceleration': 'float64',
        'yaw_rate': 'float64'
    }

    #filter by city
    cities = ['boston-seaport', 'singapore-hollandvillage','singapore-onenorth','singapore-queenstown']
    if city_id == 'b': 
        city = cities[0]
    elif city_id == 's1':
        city = cities[1]
    elif city_id == 's2':
        city = cities[2]
    elif city_id == 's3':
        city = cities[3]
    
    nuscenes_folder = Path(data_folder) / 'nuscenes'
    scenes_df = pd.read_csv(nuscenes_folder / data_file, dtype=dtype_dict, parse_dates=['timestamp'])
    cam_data_df = pd.read_csv(nuscenes_folder / 'cam_data.csv')

    if city_id != 'all':
        scenes_df = scenes_df[scenes_df['location'] == city]

    # Filter by weather
    if weather != 'all':
        rain_filter = 1 if weather == 'rain' else 0
        scenes_df = scenes_df[scenes_df['rain'] == rain_filter]

    # Filter by time-of-day
    if tod != 'all':
        night_filter = 1 if tod == 'night' else 0
        scenes_df = scenes_df[scenes_df['night'] == night_filter]

    #set discretizercam_data_df
    discretizer_configs = {
    'a': {'obj_discretizer': 'binary', 'vel_discretizer': 'binary' },
    'b': {'obj_discretizer': 'binary', 'vel_discretizer': 'multiple' }
    #'c': {multiple, multiple}
    }

    default_config = discretizer_configs['a']

    config = default_config
    for key in discretizer_configs:
        if key in discretizer_id:
            config = discretizer_configs[key]
            break

    DiscretizerClass = AVDiscretizer if '0' in discretizer_id else AVDiscretizerD1 if '1' in discretizer_id else AVDiscretizerD2

    
    if city_id!='all':
        env = AVEnvironment(city, nuscenes_folder)
        # Instantiate the discretizer with the chosen configuration
        discretizer = DiscretizerClass(
            env,
            vel_discretization=config['vel_discretizer'],
            obj_discretization=config['obj_discretizer'],
            id=discretizer_id
        ) 
        pg = PG.AVPolicyGraph(env, discretizer)
        pg = pg.fit(scenes_df, cam_data_df,update=False, verbose=verbose)
    else:
        env = AVEnvironment(city_id, nuscenes_folder)
        discretizer = DiscretizerClass(
            env,
            vel_discretization=config['vel_discretizer'],
            obj_discretization=config['obj_discretizer'],
            id=discretizer_id
        ) 
        pg = PG.AVPolicyGraph(env, discretizer)
        for city in cities:
            city_environment = AVEnvironment(city, nuscenes_folder)
            pg.discretizer.environment = city_environment
            #pg.environment = city_environment
            city_df = scenes_df[scenes_df['location'] == city]
            pg = pg.fit(city_df, cam_data_df, update=True, verbose=verbose)

    # Remove uninformative scenes (|WCC|=1)
    weakly_connected_components = list(nx.weakly_connected_components(pg))
    for component in weakly_connected_components:
        if len(component) == 1:
            pg.remove_node(next(iter(component))) 

           
    split = 'mini' if 'mini' in data_file else 'trainval'

    if output_format == 'csv':
        nodes_path = f'{data_folder}/policy_graphs/PG_nuscenes_{split}_C{city_id}_D{discretizer_id}_W{weather}_T{tod}_nodes.{output_format}'
        edges_path = f'{data_folder}/policy_graphs/PG_nuscenes_{split}_C{city_id}_D{discretizer_id}_W{weather}_T{tod}_edges.{output_format}'
        traj_path = f'{data_folder}/policy_graphs/PG_nuscenes_{split}_C{city_id}_D{discretizer_id}_W{weather}_T{tod}_traj.{output_format}' 
        pg.save(output_format, [nodes_path, edges_path, traj_path])
    else:
        pg.save(output_format, f'{data_folder}/policy_graphs/PG_nuscenes_{split}_C{city_id}_D{discretizer_id}.{output_format}')


    if verbose:
        print(f'Successfully generated Policy Graph with {len(pg.nodes)} nodes and {len(pg.edges)} edges.')


