## NuScenes Setup 

#### 1. Download

##### 1. Go to the [nuScenes Download page](https://www.nuscenes.org/download) and register or log in. 

##### 2. Download the latest *Map Expantion* pack.


##### 3. Under the "Full Dataset (v1.0)" section, dowload the metadata of the __trainval__ split: `v1.0-trainval_meta.tgz`

#### 2. Extract



```bash
tar -xf /path/to/v1.0-trainval.tgz -C ./data/sets/nuscenes

tar -xf /path/to/nuScenes-map-expansion-[version].tgz -C ./data/sets/nuscenes/maps

```

Your folder structure should look like this:

```bash
data/sets/nuscenes
    ├── samples/       # Sensor file blobs
    ├── maps/          # City map data
    └── v1.0-trainval/ # JSON tables with metadata & annotations

``` 





