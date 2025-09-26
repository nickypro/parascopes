# %%
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import plotly.graph_objects as go

from utils_plot import load_rubric_results, process_scores, load_result


def calculate_score_proportions(scores, cumulative=False):
    total = len(scores)
    if total == 0:
        return []

    # Get unique possible scores and sort them
    unique_scores = sorted(set(scores))
    proportions = []

    if cumulative:
        # Calculate proportion >= each score
        for threshold in unique_scores:
            count = sum(1 for score in scores if score >= threshold)
            proportions.append(count / total)
    else:
        # Calculate proportion = each score
        for score in unique_scores:
            count = sum(1 for s in scores if s == score)
            proportions.append(count / total)

    return proportions

def plot_score_proportions(data_dicts, metric, output_image=None):
    plt.figure(figsize=(12, 6))

    # Process each comparison type
    for label, data_dict in data_dicts.items():
        # Load and process data
        data_list = list(data_dict.values())
        scores = process_scores(data_list, metric)

        proportions = calculate_score_proportions(scores)

        # Get unique scores for x-axis
        unique_scores = sorted(set(scores))

        # Plot as lines
        # Plot stacked bars for each score threshold
        # Create a base color for this label using a consistent mapping
        label_index = list(data_dicts.keys()).index(label)
        base_color = plt.cm.Pastel1(label_index / len(data_dicts))

        bottom = 0
        for i, prop in reversed(list(enumerate(proportions))):
            # Darken the base color based on score level
            darkness = 1 - (i/len(proportions))
            color = tuple(c * darkness for c in base_color[:3]) + (base_color[3],)

            bar = plt.bar([label], [prop], bottom=bottom,
                   label=f'{label} (≥{unique_scores[i]})',
                   color=color)
            plt.text(bar[0].get_x() + bar[0].get_width()/2, bottom + prop/2,
                    str(unique_scores[i]),
                    ha='center', va='center')
            bottom += prop

    # plt.xlabel(f'{metric} Score Threshold')
    plt.ylabel('Proportion >= Score')
    plt.title(f'Cumulative Score Distribution for {metric.capitalize()} ({unique_scores[0]} to {unique_scores[-1]})')
    plt.grid(True, alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_image:
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.show()

def get_examples(data_dict, metric, shuffle=True, limit=10):
    examples = defaultdict(list)
    for index, item in data_dict.items():
        score = load_result(item["result"])["scoring"][metric]
        examples[score].append(item)
    if shuffle:
        for score in examples:
            random.shuffle(examples[score])
    if limit is not None:
        for score in examples:
            examples[score] = examples[score][:limit]
    return examples

def plot_score_proportions_interactive(data_dicts, metric, colour_map=None):
    fig = go.Figure()
    df_proportions = pd.DataFrame()

    for label, data_dict in data_dicts.items():
        data_list = list(data_dict.values())
        scores = process_scores(data_list, metric)
        proportions = calculate_score_proportions(scores)
        unique_scores = sorted(set(scores))
        examples = get_examples(data_dict, metric, limit=5) # examples[score]["reference"]

        cumulative_proportions = []
        for i, prop in enumerate(proportions):
            cumulative_proportions.append(sum(proportions[i:]))
        df_proportions = pd.concat([df_proportions, pd.DataFrame({"label": label, "score": unique_scores, "proportion": cumulative_proportions})])

        label_index = list(data_dicts.keys()).index(label)

        bottom = 0
        for i, prop in reversed(list(enumerate(proportions))):
            #base_color = f'rgba({label_index * 50 % 255}, {label_index * 80 %
            #255}, {label_index * 110 % 255}, 0.6)'
            # base_color = f'rgba({label_index * 50 % 255}, {label_index * 80 % 255}, {label_index * 110 % 255}, {(i+1)/len(proportions)})'
            if colour_map is None:
                base_color = f'hsla({label_index * 150 % 360}, 50%, 50%, {(i+1)/len(proportions)})'
            else:
                rgb = colour_map[label]
                base_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {(i+1)/len(proportions)})'

            # Safely access examples
            hover_text = f"<br><b>Score:</b> {unique_scores[i]}"
            for example in examples[unique_scores[i]]:
                hover_text += (
                    f"<br><b>Reference:</b> {example['reference'][:80]}"
                    +f"<br><b>Comparison:</b> {example['comparison'][:80]}<br>"
                )

            fig.add_trace(go.Bar(
                x=[label],
                y=[prop],
                base=bottom,
                name=f'{label} (≥{unique_scores[i]})',
                marker_color=base_color,
                hoverinfo='text',
                hovertext=hover_text,
                text=f'{unique_scores[i]}',  # Add number in middle
                textposition='inside',  # Position text in middle of bar
                textfont=dict(size=10),  # Set consistent font size
            ))
            bottom += prop

    fig.update_layout(
        title=f'Cumulative Score Distribution for {metric.capitalize()} ({unique_scores[0]} to {unique_scores[-1]})',
        # xaxis_title=f'{metric} Score Threshold',
        yaxis_title='Proportion >= Score',
        barmode='stack',
        xaxis=dict(
            tickangle=-30  # Tilt labels diagonally up to the right
        ),
        margin=dict(l=20, r=20, t=40, b=20),  # Reduce margins
        font=dict(size=10),  # Reduce font size
        autosize=True  # Automatically adjust figure size
    )

    # hide legend
    fig.update_layout(showlegend=False)

    # save png to ./figures
    fig.write_image(f"../figures/score_distribution_{metric}.png")

    fig.show(renderer="notebook_connected")
    # Create a nicely formatted table showing score distributions
    print(f"\n{metric.capitalize()} Score Cumulative Distribution Table")
    print("-" * 65)
    print("Model      | Score -1 | Score 0  | Score 1  | Score 2  | Score 3  |")
    print("-" * 65)

    for label in df_proportions['label'].unique():
        model_data = df_proportions[df_proportions['label'] == label]
        proportions = model_data['proportion'].values
        # Handle cases where we don't have all 5 scores
        row = f"{label:<10} |"
        for i in range(5):
            if i < len(proportions):
                row += f" {proportions[i]:8.2%} |"
            else:
                row += f" {'N/A':>8} |"
        print(row)
    print("-" * 65)

def check_references_match(data_dicts):
    references = {}
    for data_type, data_dict in data_dicts.items():
        for index, item in data_dict.items():
            references[int(index)] = item["reference"]
        break

    for data_type, data_dict in data_dicts.items():
        for index, item in data_dict.items():
            if item["reference"] != references[int(index)]:
                print(f"{data_type} {index} {item['reference']} != {references[int(index)]}")

if __name__ == "__main__":

    # Manually list the files.
    data_dicts = load_rubric_results(
        file_path="../data/processed_rubrics/all_data_dicts.json",
        indices_intersection=True,
        check_short_indices=False,
        check_references_match=False,
    )

    data_dicts = {
        "TAE": data_dicts["linear"],
        "Cont.": data_dicts["continued"],
        "blind": data_dicts["baseline"],
        "cheat-1": data_dicts["cheat-1"],
        "cheat-5": data_dicts["cheat-5"],
        "cheat-10": data_dicts["cheat-10"],
        "regenerated": data_dicts["regenerated"],
        "auto-decoded": data_dicts["auto-decoded"],
    }

    
    ones = np.ones(3)
    base = ones * 0.5
    r = np.array([0.0, 0.1, 0.3])
    g = np.array([0.1, 0.3, 0.0])
    b = np.array([0.3, 0.0, 0.1])

    colour_map = {
        "TAE": base + r ,
        "Cont.": base + r,
        "blind": base + b,
        "cheat-1": base + b,
        "cheat-5": base + b,
        "cheat-10": base + b,
        "regenerated": base + g,
        "auto-decoded": base + g,
    }
    # Rename baseline

    # metrics = ["complexity", "coherence", "structure", "subject", "entities", "details", "terminology", "tone"]
    metrics = ["coherence", "subject", "entities", "details"]

    for metric in metrics:
        plot_score_proportions_interactive(data_dicts, metric, colour_map)

# %%
# Generate LaTeX table comparing outlines
import random
# Get 5 random indices
random.seed(42)  # For reproducibility
sample_indices = random.sample(range(len(data_dicts["TAE"])), min(5, len(data_dicts["TAE"])))
sample_indices = [list(data_dicts["TAE"].keys())[i] for i in sample_indices]

for i, idx in enumerate(sample_indices):
    print(f"\\begin{{table}}[h!]")
    print("\\centering")
    print("\\begin{tabular}{p{4.5cm}p{4.5cm}p{4.5cm}}")
    print("\\toprule")
    print("\\textbf{TAE} & \\textbf{Cont.} & \\textbf{Original} \\\\")
    print("\\midrule")
    
    # Get the TAE text
    tae_text = data_dicts["TAE"][str(idx)]['comparison']
    # Clean and escape LaTeX special characters
    tae_text = tae_text.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('\n', '\\newline')
    
    # Get the Cont. text
    cont_text = data_dicts["Cont."][str(idx)]['comparison']
    # Clean and escape LaTeX special characters
    cont_text = cont_text.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('\n', '\\newline')
    
    # Get the Original text
    original_text = data_dicts["TAE"][str(idx)]['reference']
    # Clean and escape LaTeX special characters
    original_text = original_text.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('\n', '\\newline')
    print(f"{tae_text} & {cont_text} & {original_text} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{Comparison of TAE vs Cont. vs Original - Example {i+1}}}")
    print(f"\\label{{tab:text_comparison_{i+1}}}")
    print("\\end{table}")
    print()
# %%
