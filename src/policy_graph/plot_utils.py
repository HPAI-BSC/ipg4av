import random
from sklearn.metrics import auc
import numpy as np
from typing import Set, Dict, List, Tuple
from policy_graph.intention_introspector import AVIntentionIntrospector
from policy_graph.policy_graph import AVPolicyGraph
from policy_graph.desire import AVDesire
from policy_graph.utils import get_trajectory_metrics
import matplotlib.pyplot as plt
import networkx as nx
import math

def plot_int_progess(ii: AVIntentionIntrospector, s_a_trajectory:List[Tuple[int, int]], scene_id:str="", desires:Set[AVDesire] = [], min_int_threshold:float=0.1, save=False):#, fill_desires = True ):

    """
    Plot intention progression of a scene over the specified desires.
    """

    desire_fulfill_track, episode_length, intention_track, _ =  get_trajectory_metrics(ii, s_a_trajectory)

    if not desires: 
        desires = ii.desires

    desire_names = [d.name for d in desires]
    desire_color = {d_name: c for d_name, c in zip(desire_names, plt.cm.get_cmap('tab20').colors)} #NOTE: handles max 20 diff. colors

    fig = plt.figure(figsize=(episode_length/2, 10))
    ax = plt.gca()
    for desire in desires:
        d_name = desire.name
        intention_vals = [entry.get(desire, 0) for entry in intention_track] 
        
        #Filter intetntions in the scene to not overload the plot  
        if sum(intention_vals) >min_int_threshold:   
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
    if save:
        plt.savefig(f'./img/intention_progression_{scene_id}.png', bbox_inches = 'tight', dpi=200)
        



def animate_int_progress(ii: AVIntentionIntrospector, s_a_trajectory:List[Tuple[int, int]], scene_id:str="", desires:Set[AVDesire] = [], min_int_threshold:float=0.1, save=False):
    from matplotlib.animation import FuncAnimation

    """
    Create an animated plot of intention progression through time.
    """
    desire_fulfill_track, episode_length, intention_track, _ = get_trajectory_metrics(ii, s_a_trajectory)

    if not desires:
        desires = ii.desires

    desire_names = [d.name for d in desires]
    desire_color = {d_name: c for d_name, c in zip(desire_names, plt.cm.get_cmap('tab20').colors)}

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
        if sum(data[d_name]) > min_int_threshold:
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

    if save:
        #anim.save(f'./img/intention_progression_{scene_id}.mp4', fps=5, dpi=200)  # or use .gif
        from matplotlib.animation import PillowWriter
        anim.save(f'./img/intention_progression_{scene_id}.gif', writer=PillowWriter(fps=5), dpi=200)

    else:
        plt.show()

             


def roc_curve(discretisers_info: Dict[str, AVPolicyGraph], desires:Set[AVDesire]):
    """
    Generate ROC curve for intention metrics and find the best thresholds for each discretizer.
    
    Args:
        discretisers_info: Dictionary mapping discretizer ID to policy graph.
        desires: list of desires to consider.
        output_folder: Path to save the generated ROC curve plot.

    Returns:
        A dictionary of the best thresholds for each discretizer.
    """
    
    plt.figure(figsize=(10, 6))
    thresholds = np.arange(0, 1, 0.1)
    best_thresholds = {}
    any = AVDesire("any", None, set())
    for discretizer_id, pg in discretisers_info.items():
        
        num_thresholds = len(thresholds)
        intention_probabilities = np.zeros(num_thresholds)
        expected_probabilities = np.zeros(num_thresholds)
        combined_scores = np.zeros(num_thresholds)
        
        ii = AVIntentionIntrospector(desires, pg)

        for i, threshold in enumerate(thresholds):

            intention_prob, expected_prob = ii.get_intention_metrics(commitment_threshold=threshold, desire=any)
            intention_probabilities[i] = intention_prob
            expected_probabilities[i] = expected_prob
            combined_scores[i] = intention_prob + expected_prob

        roc_auc = auc(intention_probabilities, expected_probabilities)
        
        best_index = np.argmax(combined_scores)
        best_threshold = thresholds[best_index]
        best_thresholds[discretizer_id] = best_threshold
        
        print(f'Discretizer D{discretizer_id}: Best Threshold: {best_threshold:.2f},  (AUC = {roc_auc:.2f})')

        plt.plot(intention_probabilities, expected_probabilities, label=f'D{discretizer_id}')           
                
    plt.xlabel('Intention Probability', fontsize=15)
    plt.ylabel('Expected Intention Probability', fontsize = 15)
    plt.title('Intention Metrics for $\mathit{any}$ Desire', fontsize  = 17)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig(f'./img/roc.png', bbox_inches = 'tight', dpi=100)
    
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


def plot_all_metrics(metrics_data, discretizer_id, output_folder, c=0.5, desire_type='',metric_type='Desire', fig_size=(45, 15), y_lim=1.15, colors=['#008080', '#FF7F50']):
    """
    Displays bar plots with desire or intention metrics for each desire.

    Args:
        metrics_data: Dictionary mapping desires to their respective metric values.
        discretizer_id: ID of the discretizer being visualized.
        metric_type: Type of metric to display ('Desire' or 'Intention').
        fig_size: Size of the figure.
        output_folder: Path to store the generated plots.
        y_lim: Limit of y value.
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
    ax.set_title(f'{"Intention" if metric_type == "Intention" else "Desire"} metrics for {desire_type} desires,  C = {c}' if metric_type == 'Intention' else 'Desire Metrics', fontsize=50)

    ax.set_xticks(x)
    ax.set_xticklabels(desires, fontsize=35)
    plt.yticks(fontsize=35)

    # Remove padding before first bar and after last bar
    ax.set_xlim([min(x) - width, max(x) + width])

    annotate_bars(rects1, ax, fontsize=35)
    annotate_bars(rects2, ax, fontsize=35)

    ax.legend(ncol=2, fontsize=35, loc='upper left', facecolor='white')

    plt.savefig(f'{output_folder}/{metric_type}_{desire_type}_{discretizer_id}.png', bbox_inches='tight', dpi=200)


def plot_all_metrics_per_desire(desires_data, desire, output_folder, metric_type='Desire', y_lim=1, colors=['#008080', '#FF7F50']):
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
    plt.savefig(f'{output_folder}/desire_metrics.png', dpi=100)



def plot_pg(pg:AVPolicyGraph, allow_recursion=False, font_size=7, output_folder = '', training_id='', layout = 'circular'):
        """
        Normal plots for the graph
        """
        num_of_decimals = 2

        if layout == 'circular':
            pos = nx.circular_layout(pg)
        elif layout == 'spring':
            pos = nx.spring_layout(pg, scale = 5)
        elif layout == 'random':
            pos = nx.random_layout(pg)
        elif layout == 'spectral':
            pos = nx.spectral_layout(pg)
        elif layout == 'shell':
            pos = nx.shell_layout(pg)
        elif layout == 'fr':
          pos = nx.fruchterman_reingold_layout(pg)

        # Save the color and label of each edge
        edge_labels = {}
        edge_colors = []
        for edge in pg.edges:
            if edge[0] != edge[1] or allow_recursion:
                attributes = pg.get_edge_data(edge[0], edge[1])
                for key in attributes:
                  weight = attributes[key]['probability']
                  edge_labels[(edge[0], edge[1])] = '{} - {}'.format(
                    attributes[key]['action'],
                    round(weight, num_of_decimals)
                  )
                  edge_colors.append('#332FD0')
        nodes = {node: "" for node in pg.nodes() 
                 if pg.in_degree(node) + pg.out_degree(node) > 0}
        
        # Get node colors based on their component
        connected_components = list(nx.strongly_connected_components(pg))
        color_map = {}
        for component in connected_components:
            color = get_random_color()
            for node in component:
                
                color_map[node] = color
        node_colors = [color_map[node] for node in pg.nodes()]
        
        nx.draw(
            pg, pos,
            edge_color=edge_colors,
            width=1,
            linewidths=1,
            node_size=8,
            node_color=node_colors,
            alpha=0.8,
            arrowsize=1.5,
            labels=nodes,
            font_size=font_size,
            edgelist=[edge for edge in list(pg.edges()) if edge[0] != edge[1] or allow_recursion]
        )
 
        if output_folder:
            plt.savefig(f'{output_folder}/{training_id}.png')
        else:
            plt.show()



def get_random_color():
      return "#{:06x}".format(random.randint(0, 0xFFFFFF))
