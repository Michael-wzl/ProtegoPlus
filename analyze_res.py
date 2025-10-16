import os
import copy

import yaml
import matplotlib.pyplot as plt
import numpy as np

from protego import BASE_PATH

#RES_BASE_PATH = os.path.join(BASE_PATH, "experiments")
RES_BASE_PATH = os.path.join(BASE_PATH, "results", "eval")

def ana_res(eval_name: str, compression: bool, use_lpips: bool):
    name_mapping = {
    '1a': 'Clean Query & Protected DB', 
    '2b': 'Protected Query & Clean DB', 
    '2a': 'Protected Query & Protected DB', 
    '1b': 'FR Baseline Performance'
    }
    res_path = os.path.join(RES_BASE_PATH, eval_name)
    user_names = [name for name in os.listdir(res_path) if not name.startswith(('.', '_')) and os.path.isdir(os.path.join(res_path, name))]
    cfgs = yaml.safe_load(open(os.path.join(res_path, user_names[0], 'cfgs.yaml'), 'r'))
    test_frs = cfgs['eval_fr_names']
    compression_methods = cfgs['eval_compression_methods']
    for fr_name in test_frs:
        scenario_keys = ['1a', '1b', '2a', '2b']
        metrics_keys = ['ssim', 'psnr', 'l0']
        if use_lpips:
            metrics_keys.append('lpips')
        res_recorder = {k: [] for k in scenario_keys + metrics_keys}
        compression_recorders = None
        if compression:
            compression_recorders = {
                method: {k: [] for k in scenario_keys} 
                for method in compression_methods
            }
        res_filename = f"eval_res_{fr_name}.yaml"
        for user_name in user_names:
            user_path = os.path.join(res_path, user_name)
            with open(os.path.join(user_path, res_filename), 'r') as f:
                user_res = yaml.safe_load(f)
            for key in scenario_keys:
                if key in user_res:
                    res_recorder[key].append(user_res[key])
            for key in metrics_keys:
                if key in user_res:
                    res_recorder[key].append(user_res[key])
            if compression:
                comp_file = os.path.join(user_path, f"compression_res_{fr_name}.yaml")
                if os.path.exists(comp_file):
                    with open(comp_file, 'r') as f:
                        compression_res = yaml.safe_load(f)
                    for method, res_dict in compression_res.items():
                        for k, v in res_dict.items():
                            if k in compression_recorders[method]:
                                compression_recorders[method][k].append(v)
        print(f"Results for {fr_name}:")
        for key, values in res_recorder.items():
            if key in metrics_keys:
                if values: 
                    print(f"{key}: {np.mean(values):.4f}")
            else:
                if values:
                    print(f"{name_mapping[key]}: {np.mean(values):.4f}")
        if compression and compression_recorders:
            print("Compression Results:")
            for method, recorder in compression_recorders.items():
                print(f"Method: {method}")
                for key, values in recorder.items():
                    if values:
                        print(f"{name_mapping.get(key, key)}: {np.mean(values):.4f}")
        bar_keys = ['1a', '2b', '2a']
        bar_labels = [name_mapping[k] for k in bar_keys]
        bar_values = [np.mean(res_recorder[k]) if res_recorder[k] else 0 for k in bar_keys]
        bar_vars = [np.var(res_recorder[k]) if res_recorder[k] else 0 for k in bar_keys]
        bar_mins = [np.min(res_recorder[k]) if res_recorder[k] else 0 for k in bar_keys]
        bar_maxs = [np.max(res_recorder[k]) if res_recorder[k] else 0 for k in bar_keys]
        
        plt.figure(figsize=(10, 6))
        plt.bar(bar_labels, bar_values, yerr=bar_vars, capsize=6, alpha=0.9)
        for i, (value, var_val, min_val, max_val) in enumerate(zip(bar_values, bar_vars, bar_mins, bar_maxs)):
            plt.text(i, value + 0.01,
                     f"Mean: {value:.2f}\nVar: {var_val:.4f}\nMin: {min_val:.2f} Max: {max_val:.2f}",
                     ha='center', va='bottom', fontsize=9)
        plt.title(f"Evaluation Results for {fr_name.upper()}")
        plt.xlabel("Scenario")
        plt.ylabel("Average Value")
        plt.xticks(rotation=15, ha="right")
        plt.ylim(0, 1)
        baseline_mean = np.mean(res_recorder['1b']) if res_recorder['1b'] else 0
        baseline_var = np.var(res_recorder['1b']) if res_recorder['1b'] else 0
        baseline_min = np.min(res_recorder['1b']) if res_recorder['1b'] else 0
        baseline_max = np.max(res_recorder['1b']) if res_recorder['1b'] else 0
        plt.axhline(y=baseline_mean, color='r', linestyle='--', label='FR Baseline Performance')
        plt.text(len(bar_labels) - 0.1, baseline_mean + 0.01, f"Baseline Mean: {baseline_mean:.2f}\nVar: {baseline_var:.4f}\nMin: {baseline_min:.2f} Max: {baseline_max:.2f}", color='r', ha='right', va='bottom', fontsize=9)
        metric_stats_parts = []
        for mk in metrics_keys:
            if res_recorder[mk]:
                mean_v = np.mean(res_recorder[mk])
                var_v = np.var(res_recorder[mk])
                min_v = np.min(res_recorder[mk])
                max_v = np.max(res_recorder[mk])
                metric_stats_parts.append(
                    f"{mk.upper()}: {mean_v:.4f} (Var: {var_v:.6f}, Min: {min_v:.4f}, Max: {max_v:.4f})"
                )
        plt.figtext(0.5, 0.01, ", ".join(metric_stats_parts), ha='center', fontsize=10)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.legend()
        plt.savefig(os.path.join(res_path, f"{fr_name}_evaluation_results.png"))
        plt.close()
        if compression and compression_recorders:
            method_num = len(compression_methods)
            plt.figure(figsize=(method_num*5, 6))
            for idx, (method, recorder) in enumerate(compression_recorders.items()):
                plt.subplot(1, method_num, idx + 1)
                bar_values = [np.mean(recorder[k]) if recorder[k] else 0 for k in bar_keys]
                bar_vars = [np.var(recorder[k]) if recorder[k] else 0 for k in bar_keys]
                bar_mins = [np.min(recorder[k]) if recorder[k] else 0 for k in bar_keys]
                bar_maxs = [np.max(recorder[k]) if recorder[k] else 0 for k in bar_keys]
                plt.bar(bar_labels, bar_values, yerr=bar_vars, capsize=6, alpha=0.9)
                for i, (value, var_val, min_val, max_val) in enumerate(zip(bar_values, bar_vars, bar_mins, bar_maxs)):
                    plt.text(i, value + 0.01, f"Mean: {value:.2f}\nVar: {var_val:.4f}\nMin: {min_val:.2f} Max: {max_val:.2f}", ha='center', va='bottom', fontsize=9)
                plt.title(f"Compression Method: {method}")
                plt.xlabel("Scenario")
                plt.ylabel("Average Value")
                plt.xticks(rotation=15, ha="right")
                plt.ylim(0, 1)
                baseline_mean = np.mean(recorder['1b']) if recorder['1b'] else 0
                baseline_var = np.var(recorder['1b']) if recorder['1b'] else 0
                baseline_min = np.min(recorder['1b']) if recorder['1b'] else 0
                baseline_max = np.max(recorder['1b']) if recorder['1b'] else 0
                plt.axhline(y=baseline_mean, color='r', linestyle='--', label='FR Baseline Performance')
                plt.text(len(bar_labels) - 0.1, baseline_mean + 0.01, f"Baseline Mean: {baseline_mean:.2f}\nVar: {baseline_var:.4f}\nMin: {baseline_min:.2f} Max: {baseline_max:.2f}", color='r', ha='right', va='bottom', fontsize=9)
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(res_path, f"{fr_name}_compression_results.png"))
            plt.close()

if __name__ == "__main__":
    ######################### Configuration #########################
    eval_name = "facevit_default_eval" # The name of the subfolder under results/eval/ where the evaluation results are saved. Figures from this program will also be saved in this folder.
    compression = False # If you have compression results, set this to True. Otherwise, set it to False.
    need_lpips = True
    #################################################################
    ana_res(eval_name, compression, need_lpips)


