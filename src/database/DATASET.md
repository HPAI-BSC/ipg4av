## NuScenes Setup 

#### 1. Download

-  Go to the [nuScenes Download page](https://www.nuscenes.org/download) and register or log in. 

- Download the latest *Map Expantion* pack.


- Under the "Full Dataset (v1.0)" section, download the __mini__ dataset  (`v1.0-mini.tgz`) or the __trainval__ dataset split.

The files above are sufficient for rendering and visualization. If you aim to perform the preprocessing, download the *CAN bus* data from the same page.

#### 2. Extract
Extract the files and organize them into the repo:

```bash
# Extract the dataset (mini or trainval)
tar -xf /path/to/v1.0-[split].tgz -C ./data/sets/nuscenes

# Extract the map expansion pack
tar -xf /path/to/nuScenes-map-expansion-[version].tgz -C ./data/sets/nuscenes/maps

# Extract CAN bus data (if downloaded)
unzip /path/to/can_bus.zip -d ./data/sets/nuscenes

```

After extraction, the folder `./data` structure should look like this:

```bash
data/sets/nuscenes
    ├── can_bus/       # CAN bus data [optional]
    ├── samples/       # Sensor file blobs
    ├── sweeps/        # Sensor file blobs
    ├── maps/          # City map data
    └── v1.0-[split]/ # JSON tables with metadata & annotations

``` 





