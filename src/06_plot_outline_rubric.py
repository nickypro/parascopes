# %%
import seaborn as sns
sns.set_theme()
import pandas as pd

df1 = pd.read_csv("../data/outline_rubrics/regen.csv")
df2 = pd.read_csv("../data/outline_rubrics/tae-outline-rsd.csv")

# Index(['example_id', 'dataset_idx', 'model', 'completion', 'outline_Gemma',
#        'reconstructed_text', 'embedding_id', 'index',
#        'reference_outline_llama_70B', 'rubric_json', 'score_complexity',
#        'score_coherence', 'score_hierarchy', 'score_coverage',
#        'score_ordering', 'score_subject', 'score_entities', 'score_details',
#        'score_conciseness', 'score_identical', 'score_sum', 'score_mean'],
#       dtype='object')

# Get scoring attributes (columns that start with 'score_' but exclude summary scores)
score_columns = [col for col in df1.columns if col.startswith('score_') and col not in ['score_sum', 'score_mean']]

print("Available scoring attributes:", score_columns)

# Function to calculate counts and cumulative percentages for a score column
def analyze_scores(df, score_col):
    scores = df[score_col].dropna()  # Remove any NaN values
    
    # Count occurrences of each score
    score_counts = scores.value_counts().sort_index()
    
    # Calculate cumulative percentages
    total_count = len(scores)
    cumulative_counts = {}
    cumulative_percentages = {}
    
    for score in sorted(score_counts.index):
        # Count how many scores are >= this score
        cumulative_counts[score] = sum(scores >= score)
        cumulative_percentages[score] = f"{(cumulative_counts[score] / total_count * 100):.1f}%"
    
    return score_counts, cumulative_percentages

# Analyze each dataset
print("\n" + "="*50)
print("DATASET 1 (regen.csv) Analysis:")
print("="*50)

for score_col in score_columns:
    print(f"\n{score_col}:")
    counts, cum_pct = analyze_scores(df1, score_col)
    
    print("Counts:")
    for score, count in counts.items():
        print(f"  {score}: {count}")
    
    print("Cumulative %:")
    for score, pct in cum_pct.items():
        print(f"  {score}: {pct}")

print("\n" + "="*50)
print("DATASET 2 (tae-ouline-rsd.csv) Analysis:")
print("="*50)

for score_col in score_columns:
    print(f"\n{score_col}:")
    counts, cum_pct = analyze_scores(df2, score_col)
    
    print("Counts:")
    for score, count in counts.items():
        print(f"  {score}: {count}")
    
    print("Cumulative %:")
    for score, pct in cum_pct.items():
        print(f"  {score}: {pct}")

#%%


def create_cumulative_table(df, score_columns, dataset_name, column_name_map=None):
    """Create a formatted table showing cumulative percentages for each score threshold"""
    
    column_name_map = column_name_map or {}
    
    # Get all unique scores across all columns to determine table width
    all_scores = set()
    for col in score_columns:
        all_scores.update(df[col].unique())
    all_scores = sorted(all_scores)
    
    print(f"\n% {'-' * (15 + len(all_scores) * 10)}")
    
    # Header
    header = f"% {'Score':<13} |"
    for score in all_scores:
        header += f" Score {score:>2} |"
    print(header)
    print(f"% {'-' * (15 + len(all_scores) * 10)}")
    
    # Data rows
    for score_col in score_columns:
        display_name = column_name_map.get(score_col, score_col)
        # Truncate long names to fit table format
        display_name = display_name[:12] if len(display_name) > 12 else display_name
        
        row = f"% {display_name:<13} |"
        
        total_count = len(df[score_col])
        for score in all_scores:
            if score in df[score_col].values:
                # Count how many scores are >= this score
                cumulative_count = sum(df[score_col] >= score)
                cumulative_pct = (cumulative_count / total_count * 100)
                row += f" {cumulative_pct:>7.2f}% |"
            else:
                row += f" {'N/A':>7} |"
        
        print(row)
    
    print(f"% {'-' * (15 + len(all_scores) * 10)}")

# Create tables for both datasets
column_name_map = {
    "score_coverage": "Coverage",
    "score_ordering": "Ordering", 
    "score_subject": "Subject",
    "score_entities": "Entities",
    "score_details": "Details",
}

print("\n" + "="*60)
print("CUMULATIVE PERCENTAGE TABLES")
print("="*60)

create_cumulative_table(df1, score_columns, "Ground Truth Baseline", column_name_map)
create_cumulative_table(df2, score_columns, "TAE Outline RSD", column_name_map)


# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

# Create stacked bar chart comparing both datasets
def create_stacked_comparison_plot(
    df1,
    df2,
    score_columns,
    column_name_map=None,
    dataset_labels=("Ground Truth Baseline", "TAE Outline RSD"),
    annotate_thresholds=True,
):
    """Create a stacked bar chart comparing score distributions between two datasets

    Args:
        df1: Left dataset (baseline)
        df2: Right dataset (comparison)
        score_columns: List of score column names to plot
        column_name_map: Optional mapping {column_name: display_name}. If a key is not present, the original name is used.
        dataset_labels: Tuple of labels (left_label, right_label) for legend
        annotate_thresholds: If True, show the score threshold number inside each stacked segment
    """
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Base colors for each dataset
    color1 = np.array([1.0, 0.6, 0.7])  # Red base for df1
    color2 = np.array([0.5, 0.8, 0.7])  # Blue base for df2
    
    # Optionally filter columns to only those in the provided map
    column_name_map = column_name_map or {}
    if len(column_name_map) > 0:
        score_columns = [c for c in score_columns if c in column_name_map]

    # Early exit if nothing to plot
    if len(score_columns) == 0:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis('off')
        ax.text(0.5, 0.5, 'No columns to plot', ha='center', va='center')
        plt.show()
        return

    # Width of bars and positions
    bar_width = 0.35
    x_positions = np.arange(len(score_columns))
    
    # For each metric (score column)
    for i, metric in enumerate(score_columns):
        # Get score distributions for both datasets
        df1_scores = df1[metric].value_counts().sort_index()
        df2_scores = df2[metric].value_counts().sort_index()
        
        # Convert to proportions
        df1_total = len(df1[metric].dropna())
        df2_total = len(df2[metric].dropna())
        
        df1_props = df1_scores / df1_total
        df2_props = df2_scores / df2_total
        
        # Ensure we have values for all possible scores (-1 to 3)
        all_scores = list(range(-1, 4))
        df1_by_score = {score: df1_props.get(score, 0) for score in all_scores}
        df2_by_score = {score: df2_props.get(score, 0) for score in all_scores}
        
        # Create stacked bars
        bottom1 = 0
        bottom2 = 0
        
        # Iterate from highest to lowest score so the stack represents cumulative "at least this score"
        base_hsv1 = mcolors.rgb_to_hsv(color1)
        base_hsv2 = mcolors.rgb_to_hsv(color2)
        for j, score in enumerate(reversed(all_scores)):
            # Color variants: keep hue, increase saturation, increase brightness per level
            t = (j / (len(all_scores) - 1)) if (len(all_scores) - 1) > 0 else 0
            brightness = 0.5 + 0.5 * t
            saturation = 0.9 - 0.7 * t
            alpha = 0.75 - 0.4 * t

            hsv1 = base_hsv1.copy()
            hsv1[1] = saturation
            hsv1[2] = brightness
            color1_variant = mcolors.hsv_to_rgb(hsv1)

            hsv2 = base_hsv2.copy()
            hsv2[1] = saturation
            hsv2[2] = brightness
            color2_variant = mcolors.hsv_to_rgb(hsv2)
            
            # Heights for this segment
            h1 = df1_by_score[score]
            h2 = df2_by_score[score]

            # Left bar (df1) with color1 variants
            left_bar = ax.bar(
                x_positions[i] - bar_width/2,
                h1,
                bar_width,
                bottom=bottom1,
                color=color1_variant,
                alpha=alpha,
            )
            if annotate_thresholds and h1 > 0:
                ax.text(
                    left_bar[0].get_x() + left_bar[0].get_width() / 2,
                    bottom1 + h1 - max(0.005, 0.02 * h1),
                    f"{score}",
                    ha="center",
                    va="top",
                    fontsize=9,
                )
            bottom1 += h1
            
            # Right bar (df2) with color2 variants
            right_bar = ax.bar(
                x_positions[i] + bar_width/2,
                h2,
                bar_width,
                bottom=bottom2,
                color=color2_variant,
                alpha=alpha,
            )
            if annotate_thresholds and h2 > 0:
                ax.text(
                    right_bar[0].get_x() + right_bar[0].get_width() / 2,
                    bottom2 + h2 - max(0.005, 0.02 * h2),
                    f"{score}",
                    ha="center",
                    va="top",
                    fontsize=9,
                )
            bottom2 += h2
    
    # Customize the plot
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Proportion')
    ax.set_title('Score Distribution Comparison')
    ax.set_xticks(x_positions)
    ax.set_ylim(-0.05, 1.05)
    # Add horizontal grid lines for y ticks
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3, linewidth=2)

    display_labels = [column_name_map.get(col, col) for col in score_columns]
    ax.set_xticklabels(display_labels)

    # Custom legend showing dataset names
    legend_handles = [
        Patch(facecolor=color1, alpha=0.6, label=dataset_labels[0]),
        Patch(facecolor=color2, alpha=0.6, label=dataset_labels[1]),
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("../figures/outline_rubric_comparison.png", dpi=300)
    plt.show()




# Create the comparison plot
column_name_map = {
#     "score_complexity": "Complexity",
#     "score_coherence": "Coherence",
#     "score_hierarchy": "Hierarchy",
    "score_coverage": "Coverage of Key Points",
    "score_ordering": "Ordering / Flow",
    "score_subject": "Subject Match",
    "score_entities": "Entities Match",
#     "score_conciseness": "Conciseness",
    "score_details": "Details Match",
#     "score_identical": "Identical",
#     "score_sum": "Sum",
#     "score_mean": "Mean",
}

# Set global font sizes for better readability
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 16

create_stacked_comparison_plot(df2, df1, score_columns, column_name_map, ("TAE Outline RSD", "Ground Truth Baseline"))

# %%

df2.keys()
# %%
# Display random samples of outline_generated and decoded_predicted
import random

# Get 10 random indices
random.seed(42)  # For reproducibility
sample_indices = random.sample(range(len(df2)), min(10, len(df2)))

for i, idx in enumerate(sample_indices):
    print(f"--- [{idx}] ---")
    print("outline_generated:")
    print(df2.iloc[idx]['outline_generated'])
    print("\ndecoded_predicted:")
    decoded_text = df2.iloc[idx]['decoded_predicted']
    # Replace ' - ' with '\n - ' and ' digit.' with '\n digit.'
    formatted_text = decoded_text.replace(' - ', '\n - ')
    for digit in range(10):
        formatted_text = formatted_text.replace(f' {digit}.', f'\n {digit}.')
    print(formatted_text)
    print()

# %%
# Generate LaTeX table comparing outlines
import random
# Get 5 random indices
random.seed(42)  # For reproducibility
sample_indices = random.sample(range(len(df2)), min(5, len(df2)))

for i, idx in enumerate(sample_indices):
    print(f"\\begin{{table}}[h!]")
    print("\\centering")
    print("\\begin{tabular}{p{7cm}p{7cm}}")
    print("\\toprule")
    print("\\textbf{Original Outline Generate} & \\textbf{TAE Outline Residual Stream Decoder} \\\\")
    print("\\midrule")
    
    # Get the outline_generated text
    outline_generated = df2.iloc[idx]['outline_generated']
    # Clean and escape LaTeX special characters
    outline_generated = outline_generated.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('\n', '\\newline')
    
    # Get the decoded_predicted text
    decoded_text = df2.iloc[idx]['decoded_predicted']
    # Format the text with line breaks
    formatted_text = decoded_text.replace(' - ', '\\newline - ')
    for digit in range(10):
        formatted_text = formatted_text.replace(f' {digit}.', f'\\newline {digit}.')
    # Clean and escape LaTeX special characters
    formatted_text = formatted_text.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}')
    
    print(f"{outline_generated} & {formatted_text} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{Comparison of Original Outline Generate vs TAE Outline Residual Stream Decoder - Example {i+1}}}")
    print(f"\\label{{tab:outline_comparison_{i+1}}}")
    print("\\end{table}")
    print()


# %%
