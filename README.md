# ipg4av

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-10.5555/3635637.3663299-f6c628.svg)](https://dl.acm.org/doi/10.5555/3635637.3663299)
<br/>
[![Website](https://img.shields.io/badge/Website-HPAI-8A2BE2.svg)](https://hpai.bsc.es)
[![GitHub](https://img.shields.io/badge/GitHub-HPAI--BSC-%23121011.svg?logo=github&logoColor=white.svg)](https://github.com/HPAI-BSC)
![GitHub followers](https://img.shields.io/github/followers/HPAI-BSC)
<br/>
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HPAI--BSC-ffc107?color=ffc107&logoColor=white.svg)](https://huggingface.co/HPAI-BSC)
[![LinkedIn](https://img.shields.io/badge/Linkedin-HPAI--BSC-blue.svg)](https://www.linkedin.com/company/hpai)
[![BlueSky](https://img.shields.io/badge/Bluesky-HPAI-0285FF?logo=bluesky&logoColor=fff.svg)](https://bsky.app/profile/hpai.bsky.social)
[![LinkTree](https://img.shields.io/badge/Linktree-HPAI-43E55E?style=flat&logo=linktree&logoColor=white.svg)](https://linktr.ee/hpai_bsc)


</div>


**ipg4av** applies Intention-aware Policy Graphs (IPGs) [1] to explain Autonomous Vehicle (AV) behaviour.
It allows you to: 
- generate Policy Graphs (PGs) from real driving data
- infer intentions behind AV behaviour
- produce local and global explanations of AV behaviour

## Repository Structure
```bash
data/sets/
  ├── nuscenes/       # Raw + preprocessed NuScenes data
  ├── policy_graphs/  # Generated PGs
  └── intentions/     # Generated IPGs + explanatory results

src/
  ├── database/       # NuScenes preprocessing scripts
  ├── discretizer/    # Discretization methods for PGs
  ├── experiments/    # Explanation generation scripts
  ├── metrics/        # PG static metric computation (e.g. entropy)
  └── policy_graph/   # Extensions of [pgeon](https://github.com/HPAI-BSC/pgeon) package to build PGs & IPGs for AVs
```
## Getting started

1. **Clone the Repository**

    ```bash
    git clone https://github.com/HPAI-BSC/ipg4av.git
    cd ipg4av
    ```
    
2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download & Preprocess NuScenes Dataset**
    Follow instructions in [DATASET.md](src/database/DATASET.md).


## Reproducing Results
Start by navigating to the source folder:

 ```bash
    cd src
```

### 1. Generate a Policy Graph
Generate a PG of vehicle behaviour considering scene conditions (city, weather, time of day) and a chosen discretisation approach:

```bash
python3 -m policy_graph.generate_pg \
    --sensor_file full_v1.0-mini.csv \
    --camera_file cam_data_v1.0-mini_18.csv \
    --city_id all \
    --weather all \
    --tod all \
    --discretizer 1b \
    --alpha 18 

```

`sensor_file` is the filename of the CSV containing the preprocessed NuScenes sensor data, and `camera_file` is the filename of the CSV containing the preprocessed NuScenes camera data.


| Parameter       | Possible Values         | Description                                          |
| --------------- | --------------------- | ---------------------------------------------------- |
| `--city_id`     | `b`,`s1`, `s2`, `s3`, `all`           | City to consider (e.g. Boston) |
| `--weather`     | `all`, `rain`, `no_rain`      | Weather filter on scenes     |
| `--tod`         | `all`, `day`, `night` | Time-of-day filter on scenes            |
| `--discretizer` | `0a`, `0b`, `1a`, `1b`, `2a`, `2b`            | PG discretization method
| `--alpha`       | `number > 0`                  | Detection distance threshold (metres)        |


### 2. Hypothesise Desires
- The list of hypothesised desires can be found at `experiments/desire_config.py`. 
- Add/edit to explore different driving intention hypotheses.



### 3. Compute Intentions and Global Explanations 

```bash
python3 -m policy_graph.generate_ipg \
    --discretizer 1b \
    --pg_id PG_nuscenes_mini_Call_D1b_Wall_Tall_18 
```
pg_id = PG filename without `_nodes.csv` / `_edges.csv`.



### 4. Explanations

- **Global Explanations**: once intentions are computed, global intention metrics plots are automatically saved to `data/sets/intentions/img`. These plots summarise overall behavioural patterns across all scenes.

- **Local Explanations**: to extract explanations about why the vehicle made a specific decision in a given scene, run `src/experiments/scene_analysis.ipynb`.


## Citation

```
@inproceedings{montese_ipg4av_2025,
author = {Montese, Sara and Gimenez-Abalos, Victor and Cortés, Atia and Cortés, Ulises and Alvarez-Napagao, Sergio},
title = {Explaining Autonomous Vehicles with Intention-aware Policy Graphs},
year = {2025},
TODO: EXTRAAMAS block
}

```

## References
[1]: Gimenez-Abalos, V., Alvarez-Napagao, S., Tormos, A., Cortés, U., & Vázquez-Salceda, J. [Policy Graphs and Intention: answering ‘why’ and ‘how’ from a telic perspective](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p904.pdf). In Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS '25), International Conference for Autonomous Agents and Multiagent Systems (AAMAS '25).  
