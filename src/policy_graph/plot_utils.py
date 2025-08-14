from sklearn.metrics import auc
import numpy as np
from typing import Set, Dict, List, Tuple
from policy_graph.intention_introspector import AVIntentionIntrospector
from policy_graph.desire import AVDesire
from policy_graph.discretizer import AVPredicate
import matplotlib.pyplot as plt
from experiments.desire_config import ANY
import math
import os


def get_trajectory_metrics(ii: AVIntentionIntrospector, trajectory:List[Tuple[Tuple[AVPredicate], int]]):
    intention_track = []
    prob_track = []
    desire_fulfill_track = {}
    episode_length = len(trajectory)

    for t, (state, action_idx) in enumerate(trajectory):
        curr_node = state
        prob_track.append(ii.pg.nodes[curr_node]['probability'])
        intention_track.append(ii.intention.get(curr_node, {}))
        
        for desire in ii.desires:
            curr_intention = ii.intention.get(curr_node, {})
            if curr_intention.get(desire, 0) > 0.999:
                desire_fulfill_track[t] = desire.name
            #if ii.check_desire(curr_node, desire.clause, desire.actions) is not None and action_idx in desire.actions:
                #desire_fulfill_track[t] = desire.name
    return desire_fulfill_track, episode_length, intention_track, prob_track

    


def plot_int_progess(ii: AVIntentionIntrospector, s_a_trajectory:List[Tuple[Tuple[AVPredicate], int]], scene_id:str="", desires:Set[AVDesire] = [], min_int_threshold:float=0.1, output_folder:str=None):#, fill_desires = True ):

    """
    Plot intention progression of a scene over the specified desires.
    """

    desire_fulfill_track, episode_length, intention_track, _ =  get_trajectory_metrics(ii, s_a_trajectory)

    if not desires: 
        desires = ii.desires

    desire_names = [d.name for d in desires]
    desire_color = {d_name: c for d_name, c in zip(desire_names, plt.cm.get_cmap('tab20').colors)} #NOTE: handles max 20 diff. colors / desires

    fig = plt.figure(figsize=(episode_length/2, 10))
    ax = plt.gca()
    for desire in desires:
        d_name = desire.name
        intention_vals = [entry.get(desire, 0) for entry in intention_track] 
        
        #Filter intentions in the scene to not overload the plot  
        if max(intention_vals) > min_int_threshold: #sum(intention_vals) >min_int_threshold:   
            ax.plot(range(episode_length), intention_vals, label=d_name, color=desire_color[d_name],linestyle='dotted', linewidth=5 )
         
    ax.legend(loc='best', facecolor='white', framealpha = 1,  fontsize=24, frameon=True)
    ax.tick_params(axis='x', labelsize=27)  
    ax.tick_params(axis='y', labelsize=27)  
    ax.set_xlabel('Time', fontsize=36)
    ax.set_ylabel('Intention Value', fontsize = 36)
   

    ax.set_ylim(bottom=0, top=1)
    for t, d_name in desire_fulfill_track.items():
        if d_name in desire_names:
            ax.vlines(t, -0.05, 1.05, label=d_name, colors=desire_color[d_name], linestyles='-', linewidth=4)
    plt.title(f'Intention evolution in scene {scene_id}', fontsize=37)
    if output_folder:
        plt.savefig(f'{output_folder}/int_progress_{scene_id}.png', bbox_inches = 'tight', dpi=200)
        



def animate_int_progress(ii: AVIntentionIntrospector, s_a_trajectory:List[Tuple[int, int]], scene_id:str="", desires:Set[AVDesire] = [], min_int_threshold:float=0.1, output_folder:str=None):
    from matplotlib.animation import FuncAnimation

    """
    Create an animated plot of intention progression through time.
    """
    desire_fulfill_track, episode_length, intention_track, _ = get_trajectory_metrics(ii, s_a_trajectory)

    if not desires:
        desires = ii.desires

    desire_names = [d.name for d in desires]
    desire_color = {d_name: c for d_name, c in zip(desire_names, plt.cm.get_cmap('tab20').colors)} #NOTE: handles max 20 diff. colors / desires

    fig, ax = plt.subplots(figsize=(episode_length / 2, 10))
    ax.set_xlim(0, episode_length)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Time', fontsize=23)
    ax.set_ylabel('Intention Value', fontsize=23)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_title(f'Intention evolution in scene {scene_id}', fontsize=25)
    
    # Precompute intention data
    lines = {}
    data = {d.name: [entry.get(d, 0) for entry in intention_track] for d in desires}
    
    for d in desires:
        d_name = d.name
        if max(data[d_name]) > min_int_threshold:#sum(data[d_name]) > min_int_threshold:
            line, = ax.plot([], [], label=d_name, color=desire_color[d_name], linestyle='dotted', linewidth=5)
            lines[d_name] = line


    def init():
        for line in lines.values():
            line.set_data([], [])
        ax.legend(loc='upper left', facecolor='white', framealpha=1, fontsize=16, frameon=True)
        return lines.values()

    def update(frame):
        for d_name, line in lines.items():
            line.set_data(range(frame + 1), data[d_name][:frame + 1])
        return lines.values()

    anim = FuncAnimation(fig, update, frames=episode_length, init_func=init, blit=True, repeat=False)

    if output_folder:
        from matplotlib.animation import PillowWriter
        anim.save(f'{output_folder}/int_progress_{scene_id}.gif', writer=PillowWriter(fps=5), dpi=200) #.mp4

    else:
        plt.show()

             


def roc_curve(ipgs: List[AVIntentionIntrospector], output_folder:str=None, step:float=0.1) -> Dict[str, float]:
    """
    Generate ROC curve for intention metrics and find the best threshold for each discretizer.
    
    Args:
        ipgs: intentional policy graphs with different discretizers
        output_folder: Path to save the generated ROC curve plot. 
        step: Step size of commitment threshold.

    Returns:
        A dictionary of the best thresholds for each discretizer.
    """

    thresholds = np.arange(0, 1, step)
    best_thresholds = {}
    
    plt.figure(figsize=(10, 6))


    for ipg in ipgs:
        
        disc_id = ipg.pg.discretizer.id
        num_thresholds = len(thresholds)
        intention_probs = np.zeros(num_thresholds)
        expected_probs = np.zeros(num_thresholds)
        combined_scores = np.zeros(num_thresholds)
                
        for idx, threshold in enumerate(thresholds):
            intention_probs[idx], expected_probs[idx] = ipg.get_intention_metrics(commitment_threshold=threshold, desire=ANY)
        
        combined_scores = intention_probs + expected_probs
        
        best_idx = np.argmax(combined_scores)
        best_threshold = thresholds[best_idx]
        best_thresholds[disc_id] = best_threshold
        
        roc_auc = auc(intention_probs, expected_probs)

        print(f'Discretizer D{disc_id}: Best Threshold: {best_threshold:.2f},  (AUC = {roc_auc:.2f})')

        plt.plot(intention_probs, expected_probs, label=f'D{disc_id}')           
                
    plt.xlabel('Intention Probability', fontsize=15)
    plt.ylabel('Expected Intention Probability', fontsize = 15)
    plt.title(f'Intention Metrics for ANY Desire', fontsize  = 17)
    plt.legend(fontsize=13)
    plt.grid(True)
    #plt.tight_layout()

    if output_folder:
        plt.savefig(f'{output_folder}/roc_s{step}.png', bbox_inches = 'tight', dpi=100)
    else:
        plt.show()

    return best_thresholds


def annotate_bars(rects, ax, fontsize=8):
    """Annotate bars with their height."""
    for rect in rects:
        height = rect.get_height()
        text = f'{height:.3f}' if height > 0 else '0'
        ax.text(
            rect.get_x() + rect.get_width() / 2, height, text,
            ha='center', va='bottom', fontsize=fontsize
       )


def plot_metrics(metrics_data:Dict[str, Tuple[float, float]], discretizer_id:str, output_folder:str, c:float=0.5, metric_type:str='Desire', fig_size:Tuple[int, int]=(45, 15), y_lim:float=1.15, colors:Tuple[str,str]=['#008080', '#FF7F50']):
    """
    Displays bar plots with desire or intention metrics for each desire.

    Args:
        metrics_data: Dictionary mapping desire names to their respective metric values.
        discretizer_id: ID of the discretizer being visualized.
        metric_type: Type of metric to display ('Desire' or 'Intention').
        fig_size: Size of the figure.
        output_folder: Path to store the generated plots.
        y_lim: Limit of y value for better plot clarity.
    """
    desires = list(metrics_data.keys())
    val1 = np.array([metrics_data[desire][0] for desire in desires])
    val2 = np.array([metrics_data[desire][1] for desire in desires])

    x = np.arange(len(desires))  # Ensure x values are tightly packed
    width = 0.3  # Bar width to make pairs touch

    fig, ax = plt.subplots(figsize=fig_size)

    rects1 = ax.bar(x - width/2, val1, width, color=colors[0], label=f'{metric_type} Probability')
    metric_label = 'Expected Action Probability' if metric_type == 'Desire' else 'Expected Intention Probability'
    rects2 = ax.bar(x + width/2, val2, width, color=colors[1], label=metric_label)

    ax.set_ylabel('Probability', fontsize=35)
    ax.set_ylim(0, y_lim)
    ax.set_title(f'Intention Metrics, C = {c}' if metric_type == 'Intention' else 'Desire Metrics', fontsize=50)

    ax.set_xticks(x)
    ax.set_xticklabels(desires, fontsize=35)
    plt.yticks(fontsize=35)

    # Remove padding before first bar and after last bar
    ax.set_xlim([min(x) - width, max(x) + width])

    annotate_bars(rects1, ax, fontsize=35)
    annotate_bars(rects2, ax, fontsize=35)

    ax.legend(ncol=2, fontsize=35, loc='upper left', facecolor='white')

    save_path = f"{output_folder}/{metric_type}_{discretizer_id}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)


def plot_metrics_per_desire(desires_data, desire, output_folder, metric_type='Desire', y_lim=1, colors=['#008080', '#FF7F50']):
    """
    Displays bar plots for all discretizers for a specific desire or intention.
    
    Args:
        desires_data: Dictionary of discretizer data mapping to metric values for each desire.
        desire: Specific desire to visualize.
        metric_type: Type of metric to display ('Desire' or 'Intention').
        output_folder: Path to save the generated plot.
        y_lim: Max value for y
    """
    num_discretizers = len(desires_data)
    
    # Calculate grid size for subplots
    cols = 3  # Number of columns in the grid
    rows = math.ceil(num_discretizers / cols)  # Calculate rows to fit all subplots

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()  # Flatten the axes array for easy indexing


    for i, (discretizer_id, metrics_data) in enumerate(desires_data.items()):
        ax = axes[i]

        # Skip discretizers without the requested desire
        if desire not in metrics_data:
            print(f"Skipping Discretizer {discretizer_id}: Desire '{desire}' not found.")
            fig.delaxes(ax)  # Remove unused subplot
            continue

        # Extract metrics
        val1, val2 = metrics_data[desire]

        # Bar plot setup
        labels = [desire]
        x = np.arange(len(labels))
        width = 0.3

        # Bar colors and plots
        rects1 = ax.bar(x - width / 2, val1, width, color=colors[0], label=f'{metric_type} Probability')
        metric_label = 'Expected Action Probability' if metric_type == 'Desire' else 'Expected Intention Probability'
        rects2 = ax.bar(x + width / 2, val2, width, color=colors[1], label=metric_label)

        # Axis labels and title
        ax.set_ylabel(f'{metric_type} Metrics', fontsize=10)
        ax.set_title(f'Discretizer D{discretizer_id}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, y_lim)
        ax.legend(ncol=1, loc='upper left')

        # Annotate bar heights
        annotate_bars(rects1, ax, fontsize=8)
        annotate_bars(rects2, ax, fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f'{output_folder}/desire_metrics_{desire.name}.png', dpi=100)

