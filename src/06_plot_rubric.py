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

BASE_DIR = "./hdd_cache"
FIGURES_DIR = f"{BASE_DIR}/figures"

try:
    get_ipython()
    show_interactive = True
except NameError:
    show_interactive = False

# all_data datatype
# {
#     "0": {
#         "all_valid": True,
#         "scores": {
#             "TAE cat": {
#                 "complexity": 0,
#                 "coherence": 0,
#                 "structure": 0,
#                 "subject": 0,
#                 "entities": 0,
#                 "details": 0,
#                 "terminology": 0,
#                 "tone": 0,
#                 "identical": 0,
#             },
#             "TAE sum": {...},
#             ...
#         }
#     },
#     "1": {...},
#     ...
# }


def load_rubric_results(file_path: str):
    """Load rubric results from processed JSON file, filtering to only valid entries"""
    with open(file_path, 'r') as f:
        all_data = json.load(f)
    
    # Filter to only valid entries
    valid_data = {k: v for k, v in all_data.items() if v.get("all_valid", False)}
    
    return valid_data


def get_examples(data_dict, model_name, metric, shuffle=True, limit=10):
    """Get example indices for each score value"""
    examples = defaultdict(list)
    for index, entry in data_dict.items():
        if model_name in entry["scores"]:
            score = entry["scores"][model_name][metric]
            examples[score].append({"index": index, "scores": entry["scores"][model_name]})
    
    if shuffle:
        for score in examples:
            random.shuffle(examples[score])
    
    if limit is not None:
        for score in examples:
            examples[score] = examples[score][:limit]
    
    return examples


def plot_score_proportions_interactive(data_dict, metric, model_names, colour_map=None):
    """Create interactive stacked bar chart of score distributions"""
    fig = go.Figure()
    df_proportions = pd.DataFrame()

    for label in model_names:
        # Get scores for this model across all valid entries
        scores = []
        for entry in data_dict.values():
            if label in entry["scores"]:
                scores.append(entry["scores"][label][metric])
        
        # Clean scores: drop None/NaN and non-numeric, cast floats like 1.0 to ints
        cleaned_scores = []
        for s in scores:
            if isinstance(s, (int, float)):
                # Filter out NaN
                if isinstance(s, float) and np.isnan(s):
                    continue
                cleaned_scores.append(int(s))

        if not cleaned_scores:
            continue

        # Calculate proportions for each unique score
        unique_scores = sorted(set(cleaned_scores))
        proportions = []
        for score in unique_scores:
            count = sum(1 for s in cleaned_scores if s == score)
            proportions.append(count / len(cleaned_scores))

        # Calculate cumulative proportions (>= each score)
        cumulative_proportions = []
        for i in range(len(proportions)):
            cumulative_proportions.append(sum(proportions[i:]))
        
        df_proportions = pd.concat([
            df_proportions, 
            pd.DataFrame({
                "label": label, 
                "score": unique_scores, 
                "proportion": cumulative_proportions
            })
        ])

        # Get examples for hover text
        examples = get_examples(data_dict, label, metric, limit=5)

        label_index = model_names.index(label)

        # Add stacked bars (reversed order for visual effect)
        bottom = 0
        for i, prop in reversed(list(enumerate(proportions))):
            if colour_map is None:
                base_color = f'hsla({label_index * 150 % 360}, 50%, 50%, {(i+1)/len(proportions)})'
            else:
                rgb = colour_map[label]
                base_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {(i+1)/len(proportions)})'

            # Build hover text with examples
            hover_text = f"<br><b>Score:</b> {unique_scores[i]}"
            for example in examples[unique_scores[i]]:
                hover_text += (
                    f"<br><b>Index:</b> {example['index']}"
                    f"<br><b>Complexity:</b> {example['scores'].get('complexity', 'N/A')}<br>"
                )

            fig.add_trace(go.Bar(
                x=[label],
                y=[prop],
                base=bottom,
                name=f'{label} (≥{unique_scores[i]})',
                marker_color=base_color,
                hoverinfo='text',
                hovertext=hover_text,
                text=f'{unique_scores[i]}',
                textposition='inside',
                textfont=dict(size=10),
            ))
            bottom += prop

    fig.update_layout(
        title=f'Cumulative Score Distribution for {metric.capitalize()}',
        yaxis_title='Proportion >= Score',
        barmode='stack',
        xaxis=dict(tickangle=-30),
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=10),
        autosize=True,
        showlegend=False
    )

    # Save figure
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.write_image(f"{FIGURES_DIR}/rubric_score_distribution_{metric}.png")

    if show_interactive:
        fig.show(renderer="notebook_connected")
    
    # Print table
    print(f"\n{metric.capitalize()} Score Cumulative Distribution Table")
    print("-" * 65)
    print("Model      | Score -1 | Score 0  | Score 1  | Score 2  | Score 3  |")
    print("-" * 65)

    for label in df_proportions['label'].unique():
        model_data = df_proportions[df_proportions['label'] == label]
        proportions = model_data['proportion'].values
        row = f"{label:<10} |"
        for i in range(5):
            if i < len(proportions):
                row += f" {proportions[i]:8.2%} |"
            else:
                row += f" {'N/A':>8} |"
        print(row)
    print("-" * 65)

def print_mean_scores(data_dict, metrics, model_names, sorted=True):
    print("\nMean scores:")
    
    # Print header
    header = f"{'Model':<15}"
    for metric in metrics:
        header += f" | {metric.capitalize():<15}"
    print(header)
    print("-" * (15 + len(metrics) * 18))
    
    model_scores = []

    # Print each model's scores
    for model_name in model_names:
        row = f"{model_name:<15}"
        total_score = 0
        for metric in metrics:
            scores = []
            for entry in data_dict.values():
                if model_name in entry["scores"]:
                    score = entry["scores"][model_name][metric]
                    # Filter out None values
                    if score is None or score < 0:
                        continue
                    scores.append(score)
            mean_score = np.mean(scores) if scores else float('nan')
            total_score += mean_score
            std_err = np.std(scores) / np.sqrt(len(scores)) if scores else float('nan')
            row += f" | {mean_score:<6.3f} ± {std_err:<6.3f}"
        model_scores.append((total_score, row))

    if sorted:
        model_scores.sort(key=lambda x: x[0], reverse=True)
    for _, row in model_scores:
        print(row)


if __name__ == "__main__":

    # Load the processed rubric data (only valid entries)
    data_dict = load_rubric_results(
        file_path="./hdd_cache/processed_rubrics/llama-3b/all_data.json"
    )

    print(f"Loaded {len(data_dict)} valid entries")

    # Define the model names to compare
    model_names = [
        "TAE cat",
        "TAE sum",
        "TAE no diff",
        "TAE attn",
        "TAE mlp",
        "auto-decoded",
    ]

    # Define color map
    ones = np.ones(3)
    base = ones * 0.5
    r = np.array([0.0, 0.1, 0.3])
    g = np.array([0.1, 0.3, 0.0])
    b = np.array([0.3, 0.0, 0.1])

    colour_map = {
        "TAE cat": base + r,
        "TAE sum": base + r,
        "TAE no diff": base + r,
        "TAE attn": base + b,
        "TAE mlp": base + b,
        "auto-decoded": base + g,
    }

    # Plot selected metrics
    metrics = ["coherence", "subject", "entities", "details"]

    for metric in metrics:
        plot_score_proportions_interactive(data_dict, metric, model_names, colour_map)

    print_mean_scores(data_dict, metrics, model_names)


# %%
# # Generate LaTeX table comparing outlines
# random.seed(42)  # For reproducibility

# # Get 5 random indices from valid entries
# valid_indices = list(data_dict.keys())
# sample_indices = random.sample(valid_indices, min(5, len(valid_indices)))

# for i, idx in enumerate(sample_indices):
#     print(f"\\begin{{table}}[h!]")
#     print("\\centering")
#     print("\\begin{tabular}{p{4.5cm}p{4.5cm}}")
#     print("\\toprule")
#     print("\\textbf{TAE cat} & \\textbf{auto-decoded} \\\\")
#     print("\\midrule")
    
#     # Get the TAE cat scores
#     tae_scores = data_dict[idx]["scores"].get("TAE cat", {})
#     tae_text = f"Coherence: {tae_scores.get('coherence', 'N/A')}, Subject: {tae_scores.get('subject', 'N/A')}"
    
#     # Get the auto-decoded scores
#     auto_scores = data_dict[idx]["scores"].get("auto-decoded", {})
#     auto_text = f"Coherence: {auto_scores.get('coherence', 'N/A')}, Subject: {auto_scores.get('subject', 'N/A')}"
    
#     # Escape LaTeX special characters
#     for char, escaped in [('&', '\\&'), ('%', '\\%'), ('$', '\\$'), 
#                           ('#', '\\#'), ('_', '\\_'), ('{', '\\{'), ('}', '\\}')]:
#         tae_text = tae_text.replace(char, escaped)
#         auto_text = auto_text.replace(char, escaped)
    
#     print(f"{tae_text} & {auto_text} \\\\")
#     print("\\bottomrule")
#     print("\\end{tabular}")
#     print(f"\\caption{{Comparison of TAE cat vs auto-decoded - Example {i+1} (Index: {idx})}}")
#     print(f"\\label{{tab:score_comparison_{i+1}}}")
#     print("\\end{table}")
#     print()
# %%
