from policy_graph.environment import AVEnvironment
from discretizer.discretizer_d0 import AVDiscretizer
from discretizer.discretizer_d1 import AVDiscretizerD1
from discretizer.discretizer_d2 import AVDiscretizerD2
from pgeon.discretizer import Discretizer
from pathlib import Path
import policy_graph.policy_graph as PG
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / 'data' / 'sets' #/ 'nuscenes'

CITIES = {
    'b': 'boston-seaport',
    's1': 'singapore-hollandvillage',
    's2': 'singapore-onenorth',
    's3': 'singapore-queenstown'
}


DTYPE_DICT = {
        'modality': 'category',  
        'scene_token': 'str',  
        'steering_angle': 'float64', 
        'rain': 'int',
        'night':'int',
        'location': 'str',
        'timestamp': 'str',  # To enable datetime operations
        'rotation': 'object',  # Quaternion
        'x': 'float64',
        'y': 'float64',
        'z': 'float64',
        'yaw': 'float64',  
        'velocity': 'float64',
        'acceleration': 'float64',
        'yaw_rate': 'float64'
    }


def get_discretizer_class(discretizer_id:str)-> Discretizer:
    if '0' in discretizer_id:
        return AVDiscretizer
    elif '1' in discretizer_id:
        return AVDiscretizerD1
    else:
        return AVDiscretizerD2
    
def setup_env_and_disc(city:str='all', discretizer_id:str='0a') -> tuple[AVEnvironment, Discretizer]:
    env = AVEnvironment(city, dataroot=DATA_ROOT/'nuscenes')
    DiscretizerClass = get_discretizer_class(discretizer_id)
    return env, DiscretizerClass(env, discretizer_id)


def load_scenes(scenes_states_file:str, scenes_camera_file:str,city:str='all', weather:str='all', tod:str='all')-> pd.DataFrame:

    """
    Load and filter scenes states and detections from respective CSV files, then return detections that correspond to filtered scenes.
    
    Args:
        scenes_states_file: filename of the scene states CSV.
        scenes_camera_file: filename of the scene camera detections CSV.
        city: filter by city or 'all' for no filter.
        weather: filter by weather condition, 'rain', 'no_rain', or 'all'.
        tod: filter by time of day, 'night', 'day', or 'all'.
    
    Returns:
        pd.DataFrame : Filtered scenes detections whose `scene_token` values exist in filtered states dataframe.
    """
    scenes_states_df = pd.read_csv(DATA_ROOT/'nuscenes' / scenes_states_file, dtype=DTYPE_DICT, parse_dates=['timestamp'])

    # filter scenes_states_df by city, weather, tod
    if city != 'all': 
        scenes_states_df = scenes_states_df[scenes_states_df['location']==city]
    
    if weather != 'all':
        rain_flag = 1 if weather == 'rain' else 0
        scenes_states_df = scenes_states_df[scenes_states_df['rain'] == rain_flag]
    
    if tod != 'all':
        night_flag = 1 if tod == 'night' else 0
        scenes_states_df = scenes_states_df[scenes_states_df['night'] == night_flag]
    
    filtered_scene_tokens = set(scenes_states_df['scene_token'])

    detections_filtered_chunks = []
    chunksize = 100000  #NOTE: tune this number based on memory
    for chunk in pd.read_csv(DATA_ROOT / 'nuscenes' / scenes_camera_file, chunksize=chunksize):
        filtered_chunk = chunk[chunk['scene_token'].isin(filtered_scene_tokens)]
        detections_filtered_chunks.append(filtered_chunk)

    filtered_detections_df = pd.concat(detections_filtered_chunks, ignore_index=True)
    
    return scenes_states_df, filtered_detections_df   
    

def load_scene(scene_token: str, scenes_states_file: str, scene_camera_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Returns scene information from sensors and cameras
    """
    states_path = DATA_ROOT / 'nuscenes' / scenes_states_file
    camera_path = DATA_ROOT / 'nuscenes' / scene_camera_file
    
    # Load and filter directly
    scenes_states_df = pd.read_csv(
        states_path,
        dtype=DTYPE_DICT,
        parse_dates=['timestamp']
    ).query("scene_token == @scene_token")

    scenes_camera_df = pd.read_csv(
        camera_path
    ).query("scene_token == @scene_token")
    
    return scenes_states_df, scenes_camera_df





def create_pg(scenes_data:pd.DataFrame, camera_data:pd.DataFrame, city:str='all', discretizer_id:str='0a', verbose:bool=True) -> PG.PolicyGraph:
    env, disc = setup_env_and_disc(city, discretizer_id)
    pg = PG.AVPolicyGraph(env, disc)

    if city!='all':
        pg = pg.fit(scenes_data, camera_data,update=False, verbose=verbose)

    else:

        for city_id, city in CITIES.items():
            city_environment = AVEnvironment(city, dataroot=DATA_ROOT/'nuscenes')
            pg.discretizer.environment = city_environment
            city_scenes_df = scenes_data[scenes_data['location'] == city]
            # Filter camera_data by scenes that belong to current city using scene_token
            city_scene_tokens = set(city_scenes_df['scene_token'])
            city_camera_df = camera_data[camera_data['scene_token'].isin(city_scene_tokens)]

            pg = pg.fit(city_scenes_df, city_camera_df, update=True, verbose=verbose)
            
    # Remove uninformative scenes (|WCC|=1)
    pg.remove_isolated_nodes(verbose)
    #weakly_connected_components = list(nx.weakly_connected_components(pg))
    #for component in weakly_connected_components:
    #    if len(component) == 1:
    #        pg.remove_node(next(iter(component))) 



    return  pg