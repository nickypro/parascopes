# %%
from utils_plot import load_rubric_results
from tqdm import tqdm
import sacrebleu
import numpy as np

data_dicts = load_rubric_results(indices_intersection=False)

print(data_dicts.keys())
data_dicts = {
    "TAE": data_dicts["linear"],
    "Cont.": data_dicts["continued"],
    "baseline": data_dicts["baseline"],
    "cheat-1": data_dicts["cheat-1"],
    "cheat-5": data_dicts["cheat-5"],
    "cheat-10": data_dicts["cheat-10"],
    "regenerated": data_dicts["regenerated"],
    "auto-decoded": data_dicts["auto-decoded"],
}

# %%

# dict_keys(['mlp', 'linear', 'continued', 'baseline', 'cheat-1', 'cheat-5', 'cheat-10', 'regenerated', 'auto-decoded'])


references = [['the quick brown fox jumps']]  # Reference text
candidates = ['the fast brown fox leaps']     # Generated text

score = sacrebleu.sentence_bleu(candidates[0], references[0])
print(f"Standardized BLEU: {score.score:.2f}")

bleu_scores = {}

for data_type, data_dict in data_dicts.items():
    bleu_scores[data_type] = []
    for index, datum in tqdm(data_dict.items()):
        ref_text = datum['reference']
        gen_text = datum['comparison']
        score = sacrebleu.sentence_bleu(gen_text, [ref_text])
        bleu_scores[data_type].append(score.score)
    print(f"{data_type}: {np.mean(bleu_scores[data_type]):.2f} ± {np.std(bleu_scores[data_type]):.2f}")



# %%
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(
    target="The capital of France is Paris",
    prediction="Paris serves as France's capital"
)
print(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.2f}")
print(f"ROUGE-2 Precision: {scores['rouge2'].precision:.2f}")
print(f"ROUGE-L Precision: {scores['rougeL'].precision:.2f}")
print(f"ROUGE-L Recall: {scores['rougeL'].recall:.2f}")
print(f"ROUGE-L F1: {scores['rougeL'].fmeasure:.2f}")

rouge_scores = {}

for data_type, data_dict in data_dicts.items():
    rouge_scores[data_type] = {
        "rouge1_f1": [],      # Multi-doc summary
        "rouge2_precision": [], # Extractive summary
        "rougeL_precision": [], # Legal/medical
        "rougeL_recall": [],    # Abstractive - part 1
        "rougeL_f1": []         # Abstractive - part 2
    }
    for index, datum in tqdm(data_dict.items()):
        ref_text = datum['reference']
        gen_text = datum['comparison']
        scores = scorer.score(ref_text, gen_text)
        rouge_scores[data_type]["rouge1_f1"].append(scores['rouge1'].fmeasure)
        rouge_scores[data_type]["rouge2_precision"].append(scores['rouge2'].precision)
        rouge_scores[data_type]["rougeL_precision"].append(scores['rougeL'].precision)
        rouge_scores[data_type]["rougeL_recall"].append(scores['rougeL'].recall)
        rouge_scores[data_type]["rougeL_f1"].append(scores['rougeL'].fmeasure)
    print(f"{data_type}:")
    print(f"  Multi-doc (R1-F1): {np.mean(rouge_scores[data_type]['rouge1_f1']):.2f} ± {np.std(rouge_scores[data_type]['rouge1_f1']):.2f}")
    print(f"  Extractive (R2-P): {np.mean(rouge_scores[data_type]['rouge2_precision']):.2f} ± {np.std(rouge_scores[data_type]['rouge2_precision']):.2f}")
    print(f"  Legal/Medical (RL-P): {np.mean(rouge_scores[data_type]['rougeL_precision']):.2f} ± {np.std(rouge_scores[data_type]['rougeL_precision']):.2f}")
    print(f"  Abstractive (RL-R+F1): {np.mean(rouge_scores[data_type]['rougeL_recall']):.2f}r/{np.mean(rouge_scores[data_type]['rougeL_f1']):.2f}f1 ± {np.std(rouge_scores[data_type]['rougeL_recall']):.2f}r/{np.std(rouge_scores[data_type]['rougeL_f1']):.2f}f1")

# %%

for data_type, bleu_score in bleu_scores.items():
    print(f"{data_type}: {np.mean(bleu_score):.2f} ± {np.std(bleu_score):.2f}")

for data_type, rouge_score in rouge_scores.items():
    # print(f"{data_type}: {np.mean(rouge_score['rouge1']):.2f} ± {np.std(rouge_score['rouge1']):.2f}, {np.mean(rouge_score['rouge2']):.2f} ± {np.std(rouge_score['rouge2']):.2f}, {np.mean(rouge_score['rougeL']):.2f} ± {np.std(rouge_score['rougeL']):.2f}")

# %%

from evaluate import load as load_bleurt

bleurt = load_bleurt("bleurt", "BLEURT-20", module_type="metric")

results = bleurt.compute(
    predictions=["Generated text", "The cat sat on the mat", "The Roman Collusem of Italy", "eiffel tower"],
    references=["Reference text", "The dog sat on the mat", "The Italian Roman Colluseum", "Eiffel Tower"]
)
print(f"Score: {results['scores']}")
# %%

import numpy as np
bleurt_scores = {}

try: 
    bleurt_scores = json.load(open("../data/cached_results/bleurt_scores.json", "r"))
except:
    for data_type, data_dict in data_dicts.items():

        bleurt_scores[data_type] = []
        batch_size = 64
        # Process in batches of 32
        data_items = list(data_dict.items())
        for i in tqdm(range(0, len(data_items), batch_size)):
            batch = data_items[i:i + batch_size]
            ref_texts = [datum['reference'] for _, datum in batch]
            gen_texts = [datum['comparison'] for _, datum in batch]
            scores = bleurt.compute(predictions=gen_texts, references=ref_texts)
            bleurt_scores[data_type].extend(scores['scores'])
        print(f"{data_type}: {np.mean(bleurt_scores[data_type]):.2f} ± {np.std(bleurt_scores[data_type]):.2f}")

# %%


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create violin plot for BLEURT scores
plt.figure(figsize=(10, 6))
plt.ylim(0, 1)

# Convert bleurt scores to DataFrame
df_plot = pd.DataFrame({
    "BLEURT Score": [score for scores in bleurt_scores.values() for score in scores],
    "Comparison Type": [label for label, scores in bleurt_scores.items() for _ in scores]
})

ones = np.ones(3)
base = ones * 0.6
r = np.array([0.0, 0.2, 0.4])
g = np.array([0.2, 0.4, 0.0])
b = np.array([0.4, 0.0, 0.2])

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
# Rename baseline to blind in the dataframe
df_plot['Comparison Type'] = df_plot['Comparison Type'].replace('baseline', 'blind')

# Create custom color palette
palette = {}
for i, label in enumerate(df_plot["Comparison Type"].unique()):
    if colour_map is None:
        hue = i * 150 % 360
        r = 0.6 + 0.3 * np.cos(np.radians(hue))
        g = 0.6 + 0.3 * np.cos(np.radians(hue - 120))
        b = 0.6 + 0.3 * np.cos(np.radians(hue + 120))
        palette[label] = (r, g, b)
    else:  
        rgb = colour_map[label]
        palette[label] = (rgb[0], rgb[1], rgb[2])

# Create violin plot
sns.violinplot(data=df_plot, x="Comparison Type", y="BLEURT Score",
               hue="Comparison Type", palette=palette, scale="width", legend=False)
plt.title("BLEURT Score Distribution")
plt.tight_layout()
plt.savefig("bleurt-scores.png", dpi=300)
print("\nViolin plot saved to: bleurt-scores.png")
plt.show()

# Print summary statistics
print(df_plot.groupby("Comparison Type")["BLEURT Score"].agg(["mean", "std"]))

# check min and max for each comparison type
for data_type, scores in bleurt_scores.items():
    print(f"{data_type}: {min(scores)} to {max(scores)}")

# %%
# import json
# json.dump(bleurt_scores, open("bleurt_scores.json", "w"))
# %%
