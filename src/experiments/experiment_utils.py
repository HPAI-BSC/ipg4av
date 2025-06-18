import os
import pickle
import pandas as pd
import logging
from policy_graph.policy_graph import AVPolicyGraph
from discretizer.discretizer_d1 import AVDiscretizerD1
from policy_graph.environment import AVEnvironment
from policy_graph.plot_utils import plot_int_progess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pkl(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, file_path):
    """Save data into a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def setup_env_and_disc(city, data_root, vel_disc='multiple', disc_id='1b'):
    """Initialize AV Environment and Discretizer."""
    env = AVEnvironment(city, dataroot=data_root)
    disc = AVDiscretizerD1(env, vel_discretization=vel_disc, id=disc_id)
    return env, disc


def load_scene_csv(pg_folder_path,csv_data_file, scene_token, city):
    """Load CSV datasets and filter by city and scene."""
    dtype_dict = {
        'modality': 'category',
        'scene_token': 'str',  
        'steering_angle': 'float64',
        'rain': 'int',
        'night': 'int',
        'location': 'str',
        'timestamp': 'str',
        'rotation': 'object',
        'x': 'float64',
        'y': 'float64',
        'z': 'float64',
        'yaw': 'float64',  
        'velocity': 'float64',
        'acceleration': 'float64',
        'yaw_rate': 'float64'
    }
    csv_path = os.path.join(pg_folder_path, 'nuscenes', csv_data_file)
    df = pd.read_csv(csv_path, dtype=dtype_dict, parse_dates=['timestamp'])
    
    cam_data_csv = os.path.join(pg_folder_path, 'nuscenes', 'cam_data.csv')
    cam_data_df = pd.read_csv(cam_data_csv)

    #  Filter for the desired city and scene token 
    df = df[(df['location'] == city) & (df['scene_token'] == scene_token)]
    cam_data_df = cam_data_df[cam_data_df['scene_token'] == scene_token]
    return df, cam_data_df


def scene_int_progress_experiment(pg_folder_path, intention_file, csv_data_file, scene_token, city, vel_disc, disc_id, commitment_threshold=0.5):
    logging.info("Loading global agent intention data...")
    loaded_intention = load_intentions_pkl(intention_file)
    
    data_root = os.path.join(pg_folder_path, 'nuscenes')
    logging.info("Setting up environment and discretizer...")
    env, disc = setup_env_and_disc(city, data_root, vel_disc=vel_disc, disc_id=disc_id)

    logging.info("Loading scene...")
    df, cam_data_df = load_scene_csv(pg_folder_path,csv_data_file, scene_token, city)
    
    logging.info("Initializing policy graph...")
    pg = AVPolicyGraph(env, disc)
    trajectory = pg._run_episode(df, cam_data_df, verbose=False)
    state_action_trajectory = [(trajectory[i], int(trajectory[i + 1])) for i in range(0, len(trajectory) - 1, 2)]
    state_action_trajectory.append((trajectory[-1], None))
    
    logging.info("Plotting scene intention progress...")
    plot_int_progess(comm_threshold=commitment_threshold, ii=loaded_intention, s_a_trajectory=state_action_trajectory)

    logging.info("Experiment completed successfully.")

