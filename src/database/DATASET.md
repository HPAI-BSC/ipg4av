## Download & Preprocess NuScenes 

### 1. Download Dataset

Download and uncompress the NuScenes dataset. For the __trainval__ version (same for __mini__):

```bash

wget https://www.nuscenes.org/data/v1.0-trainval.tgz

tar -xf v1.0-trainval.tgz -C ./data/sets/nuscenes
```

NOTE: the dowloaded files are suffient to render scenes. For prepocessing, complete the steps below as well.
   
Download CAN bus data and map expansions from the [Download page](https://github.com/nutonomy/nuscenes-devkit?tab=readme-ov-file#can-bus-expansion). After downloading, your folder structure should look like this:

```bash
data/sets/nuscenes
    ├── can_bus/       # CAN bus data
    ├── samples/       # Sensor file blobs (for scene rendering)
    ├── sweeps/        # Sensor file blobs (for scene rendering)
    ├── maps/          # City map data
    └── v1.0-trainval/ # JSON tables with metadata & annotations

``` 
    
    
    
### 2. Preprocess Dataset
To define the necessary inputs for building PGs from vehicle observations, run the preprocessing script:
```bash
bash src/database/create_database.sh
```
Make sure to update the script to point to the correct NuScenes path and dataset split.
This step will generate all the processed data required to create PGs.

