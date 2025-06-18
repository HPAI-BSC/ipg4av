import argparse
from pathlib import Path
import pandas as pd
import utils
from table_loader import BaseTableLoader
import time
from joblib import Parallel, delayed
import json

class SceneDataProcessor(BaseTableLoader):
    """
        Processes agent data from the nuScenes dataset, creating a DataFrame with additional columns for velocity,
        acceleration, and heading change rate, based on sensor data filtering. 
        Merges data from CAN bus and Camera as well.

    """
    def __init__(self, dataroot, dataoutput, version, key_frames):
        super().__init__(dataroot, version)

        self.dataroot = dataroot
        self.dataoutput = dataoutput
        self.version = version
        self.key_frames = key_frames
        self.sensor = 'lidar'
        #self.nuscenes = NuScenes(version, dataroot=Path(dataroot), verbose=True)

        start_time = time.time()
        print("======\nLoading NuScenes tables for version {}...".format(self.version))

        self.sample_data = self.__load_table__('sample_data', drop_fields=['next', 'prev', 'width', 'height', 'filename','timestamp','fileformat'])
        print('sample_data table loaded')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        print('calibrated_sensor table loaded')
        self.sensors = self.__load_table__('sensor')
        print('sensor table loaded')        
        self.ego_pose = self.__load_table__('ego_pose')     
        print('ego_pose table loaded')
        #self.__make_reverse_index__(verbose=True)
        print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))




    def process_scene_data(self):
        """Process scene data and merge relevant tables into a final DataFrame."""
        can_data = pd.read_csv(Path(self.dataoutput) / 'can_data.csv')
        #cam_data = pd.read_csv(Path(self.dataoutput) / 'cam_detection.csv')

        # Perform a full outer join to keep all CAN bus and camera data
        #sample = pd.merge(can_data, cam_data, on='sample_token', how='outer')

        sample_data = pd.DataFrame(self.sample_data)
        sample_data = sample_data.loc[sample_data['is_key_frame'] == self.key_frames, ['sample_token', 'ego_pose_token', 'calibrated_sensor_token']]

        # Select specified sensor (i.e. lidar) data
        calibrated_sensors = pd.DataFrame(self.calibrated_sensor).rename(columns={'token': 'calibrated_sensor_token'})
        sensors = pd.DataFrame(self.sensors).rename(columns={'token': 'sensor_token'})
        sensors = sensors[sensors['modality'] == self.sensor].merge(calibrated_sensors, on='sensor_token').drop(columns=['rotation', 'translation', 'channel', 'camera_intrinsic', 'sensor_token'])
        merged_df = sensors.merge(sample_data, on='calibrated_sensor_token').drop(columns=['calibrated_sensor_token'])

        ego_pose = pd.DataFrame(self.ego_pose).rename(columns={'token': 'ego_pose_token'})
        ego_pose[['x', 'y', 'z']] = pd.DataFrame(ego_pose['translation'].tolist(), index=ego_pose.index)
        #merged_df = sample.merge(merged_df, on='sample_token', how='outer').merge(ego_pose, on='ego_pose_token', how='left').drop(columns=['ego_pose_token', 'translation'])
        merged_df = can_data.merge(merged_df, on='sample_token').merge(ego_pose, on='ego_pose_token').drop(columns=['ego_pose_token', 'translation'])
        
        merged_df['yaw'] = merged_df['rotation'].apply(utils.quaternion_yaw).drop(columns=['rotation'])

        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], unit='us')
        merged_df.sort_values(by=['scene_token', 'timestamp'], inplace=True)
        
         # Parallel processing for groupby operations
        grouped = merged_df.groupby('scene_token')
        final_df = pd.concat(Parallel(n_jobs=-1)(
            delayed(lambda group: utils.calculate_dynamics(group))(group) for _, group in grouped
        ), ignore_index=True)

        return final_df



    def run_processing(self, test_size, random_state=42):
        """Run the data processing and save results as CSV files."""
        states = self.process_scene_data()
        if test_size > 0:
            train_df, test_df = utils.train_test_split_by_scene(states, test_size, random_state) 
            train_df.to_csv(Path(self.dataoutput) / f'train_{self.version}.csv', index=False)
            test_df.to_csv(Path(self.dataoutput) / f'test_{self.version}.csv', index=False)
        else:
            states.to_csv(Path(self.dataoutput) / f'full_{self.version}.csv', index=False)
         
        print(f"Dataset successfully saved to {self.dataoutput}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CAN and camera data, process scene data, and save the final dataset.")
    parser.add_argument('--dataroot', required=True, type=str, help='Path to the nuScenes dataset directory.')
    parser.add_argument('--dataoutput', required=True, type=str, help='Path for the output data file directory.')
    parser.add_argument('--version', required=True, type=str, choices=["v1.0-mini", "v1.0-trainval"], help='Version of the nuScenes dataset to process.')
    parser.add_argument('--key_frames', action='store_true', help='Whether to use key frames only.')
    #parser.add_argument('--sensor', required=True, type=str, help='Sensor modality to process.')
    parser.add_argument('--camera', action='store_true', help='Whether to add camera information or not.')
    parser.add_argument('--test_size', required=True, type=float, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_state', required=False, type=int, default=42, help='Random state for train-test split.')

    args = parser.parse_args()
    processor = SceneDataProcessor(args.dataroot, args.dataoutput, args.version, args.key_frames)
    
    
    processor.run_processing(args.test_size, args.random_state)

