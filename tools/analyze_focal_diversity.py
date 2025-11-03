import os

import yaml
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from protego import BASE_PATH

if __name__ == "__main__":
    ######################### Configuration #########################
    recall_res_base_path = f"{BASE_PATH}/results/eval/len3ensemble"
    baseline_recall_dict = {
        'ir50_adaface_casia': 0.1670, 
        'ir50_magface_ms1mv2': 0.2356, 
        'transfaces_arcface_ms1mv2': 0.4843
    }
    focal_diversity_base_path = f"{BASE_PATH}/results/focal_diversity"
    focal_diversity_fname = "focal_diversities_ens3_performance_only.yaml"
    plot_save_path = focal_diversity_base_path
    #################################################################
    metrics = ['1a', '2a', '2b']
    recall_res = {}
    for exp_name in os.listdir(recall_res_base_path):
        exp_path = os.path.join(recall_res_base_path, exp_name)
        names = [n for n in os.listdir(exp_path) if not n.startswith(('.', '_'))]
        cfgs = os.path.join(exp_path, names[0], 'cfgs.yaml')
        with open(cfgs, 'r') as f:
            cfgs = yaml.safe_load(f)
        train_frs = tuple(cfgs['train_fr_names'])
        eval_frs = cfgs['eval_fr_names']
        eval_frs = list(eval_frs) + ["ir50_adaface_casia"]
        for eval_fr in eval_frs:
            if eval_fr not in recall_res:
                recall_res[eval_fr] = {}
            recall_res[eval_fr][train_frs] = {}
            for name in names:
                res_path = os.path.join(exp_path, name, f'eval_res_{eval_fr}.yaml')
                with open(res_path, 'r') as f:
                    res = yaml.safe_load(f)
                for metric in metrics:
                    if metric not in recall_res[eval_fr][train_frs]:
                        recall_res[eval_fr][train_frs][metric] = []
                    recall_res[eval_fr][train_frs][metric].append(res[metric])
            for metric in metrics:
                recall_res[eval_fr][train_frs][metric] = np.mean(recall_res[eval_fr][train_frs][metric])
    with open(os.path.join(focal_diversity_base_path, focal_diversity_fname), 'r') as f:
        focal_diversities = yaml.safe_load(f)
    for eval_fr, res in recall_res.items():
        for metric in metrics:
            focal_divs = []
            recs = []
            ensembles = []
            for train_frs, recall in res.items():
                try:
                    focal_div = focal_diversities[str(sorted(list(train_frs)))]
                except KeyError:
                    print(f"Focal diversity not found for ensemble: {train_frs}")
                    continue
                rec = recall[metric]
                focal_divs.append(focal_div)
                recs.append(rec)
                ensembles.append(train_frs)
            # Regression line and correlation
            x = np.array(focal_divs)
            y = np.array(recs)
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            # coefficient of determination R^2
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            # correlations
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_rho, spearman_p = stats.spearmanr(x, y)
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.scatter(focal_divs, recs)
            highest2_recall_idxs = np.argsort(recs)[-2:]
            lowest2_recall_idxs = np.argsort(recs)[:2]
            high_color = 'red'
            low_color = 'orange'
            plt.scatter([focal_divs[i] for i in highest2_recall_idxs], [recs[i] for i in highest2_recall_idxs], color=high_color)
            plt.scatter([focal_divs[i] for i in lowest2_recall_idxs], [recs[i] for i in lowest2_recall_idxs], color=low_color)
            sorted_idxs = np.argsort(x)
            plt.plot(x[sorted_idxs], y_pred[sorted_idxs], color='blue', linestyle='-', label='Regression Line')
            plt.axhline(y=baseline_recall_dict.get(eval_fr, 0), color='red', linestyle='--', label='Recall of Default Mask')
            plt.legend()
            fig = plt.gcf()
            fig.subplots_adjust(top=0.75)
            dot = "●"
            highest_names = [ensembles[i] for i in highest2_recall_idxs]
            lowest_names = [ensembles[i] for i in lowest2_recall_idxs]
            high_block = [f"{dot} High-2 ({high_color.capitalize()})"] + [f"-{n}" for n in highest_names]
            low_block = [f"{dot} Low-2 ({low_color.capitalize()})"] + [f"-{n}" for n in lowest_names]
            fig.text(
                0.01, 0.98, "\n".join(high_block),
                ha='left', va='top', fontsize=8, color=high_color,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor=high_color)
            )
            fig.text(
                0.45, 0.98, "\n".join(low_block),
                ha='left', va='top', fontsize=8, color=low_color,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor=low_color)
            )
            fig.text(
                0.01, 0.85,
                f"R²: {r2:.4f}\n"
                f"Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})\n"
                f"Spearman ρ: {spearman_rho:.4f} (p={spearman_p:.4e})",
                ha='left', va='top', fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='black')
            )
            plt.xlabel('Focal Diversity', fontsize=14)
            plt.ylabel(f'Recall ({metric})', fontsize=14)
            plt.ylim(0, 1)
            plt.grid(axis='y')
            plt.title(f'Eval FR: {eval_fr}', fontsize=16)
            plt.savefig(os.path.join(plot_save_path, f'{focal_diversity_fname.split(".")[0]}_{eval_fr}_{metric}.png'))
            plt.close()


