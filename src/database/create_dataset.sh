export NUSCENES_DATA_ROOT='{your_main_folder}/data/sets/nuscenes'

split='v1.0-trainval' #'v1.0-mini' 



#1. Process camera sensor data
python3 process_cam_data.py \
	--dataroot ${NUSCENES_DATA_ROOT} \
	--dataoutput ${NUSCENES_DATA_ROOT} \
	--version ${split} --alpha 18 
	#12
	
#2. Process CAN Bus sensor data    
python3 process_can_data.py \
	--dataroot ${NUSCENES_DATA_ROOT} \
	--dataoutput ${NUSCENES_DATA_ROOT} \
	--version ${split}
	
#3. Merge sensors' data into one file
python3 merge_sensor_data.py \
	--dataroot ${NUSCENES_DATA_ROOT} \
	--dataoutput ${NUSCENES_DATA_ROOT} \
	--version  ${split} \
	--key_frame \
	--camera \
	--test_size 0 #0.2
