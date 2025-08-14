from policy_graph.pg_utils import load_scenes, create_pg, CITIES, DATA_ROOT
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input', help='Input data folder.', default="./data")
    parser.add_argument('--sensor_file', help='Input sensor data file name.')
    parser.add_argument('--camera_file', help='Input camera data file name.')
    parser.add_argument('--alpha', type=int, default=None, help='Distance within consider detections. Defaults to None if not specified.')
    parser.add_argument('--city_id', help='Specify city to consider when building the PG.', choices=['all', 'b','s1','s2', 's3'], default="all")
    parser.add_argument('--weather', help='Specify whether the PG should contain bad weather scenes only (with rain), good weather scenes only or all.', default='all', choices=['all','rain','no_rain'])
    parser.add_argument('--tod', help='Flag to specify whether the PG should contain night scenes or day scenes.',  default='all', choices=['all','day','night'])
    parser.add_argument('--discretizer', help='Specify the discretizer of the input data.', choices=['0a', '0b', '1a','1b', '2a', '2b'], default='0a')
    #parser.add_argument('--output', help='Which format to output the PG',
    #                    default='csv', choices=['pickle', 'csv'])
    parser.add_argument('--verbose', help='Whether to output log statements or not',
                        action='store_true')
     
    args = parser.parse_args()
    sensor_data_file, camera_data_file, city_id, verbose, discretizer_id, weather, tod = args.sensor_file, args.camera_file ,args.city_id, args.verbose, args.discretizer, args.weather, args.tod

    city = CITIES.get(city_id, 'all')

    scenes_states_df, scenes_observations_df = load_scenes(sensor_data_file, camera_data_file, city, weather )
    pg = create_pg(scenes_states_df, scenes_observations_df, city, discretizer_id, verbose)


    split = 'mini' if 'mini' in sensor_data_file else 'trainval'

    #if output_format == 'csv':
    nodes_path = f'{DATA_ROOT}/policy_graphs/PG_nuscenes_{split}_C{city_id}_D{discretizer_id}_W{weather}_T{tod}_{args.alpha}_nodes.csv'
    edges_path = f'{DATA_ROOT}/policy_graphs/PG_nuscenes_{split}_C{city_id}_D{discretizer_id}_W{weather}_T{tod}_{args.alpha}_edges.csv'
    traj_path = f'{DATA_ROOT}/policy_graphs/PG_nuscenes_{split}_C{city_id}_D{discretizer_id}_W{weather}_T{tod}_{args.alpha}_traj.csv' 
    pg.save('csv', [nodes_path, edges_path, traj_path])
    #else:
    #    pg.save(output_format, f'{DATA_ROOT}/policy_graphs/PG_nuscenes_{split}_C{city_id}_D{discretizer_id}_W{weather}_T{tod}_{args.alpha}.{output_format}')
    
    if verbose:
        print(f'Successfully generated Policy Graph with {len(pg.nodes)} nodes and {len(pg.edges)} edges.')


