export NUSCENES_DATA_ROOT='/home/sara/Desktop/ipg4av/data/sets/nuscenes'
#{path to}/data/sets/nuscenes

split='v1.0-trainval' #'v1.0-mini' 



#1. Process camera sensor data
python3 process_cam_data.py \
	--dataroot ${NUSCENES_DATA_ROOT} \
	--dataoutput ${NUSCENES_DATA_ROOT} \
	--version ${split} --alpha 18 
	
#2. Process CAN Bus sensor data    
python3 process_can_data.py \
	--dataroot ${NUSCENES_DATA_ROOT} \
	--dataoutput ${NUSCENES_DATA_ROOT} \
	--version ${split}
	
#3. Merge CAN bus and other sensors' data (except cameras) into one file
python3 merge_sensor_data.py \
	--dataroot ${NUSCENES_DATA_ROOT} \
	--dataoutput ${NUSCENES_DATA_ROOT} \
	--version  ${split} \
	--key_frame \
	--test_size 0 #0.2
