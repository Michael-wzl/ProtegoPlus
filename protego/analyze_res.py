import os
import copy

import yaml
import matplotlib.pyplot as plt
import numpy as np

from utils import BASE_PATH

RES_BASE_PATH = os.path.join(BASE_PATH, "results", "eval")

def ana_res(eval_name: str, compression: bool):
    name_mapping = {
    '1a': 'Clean Query & Protected DB', 
    '2b': 'Protected Query & Clean DB', 
    '2a': 'Protected Query & Protected DB', 
    '1b': 'FR Baseline Performance'
    }
    res_path = os.path.join(RES_BASE_PATH, eval_name)
    user_names = [name for name in os.listdir(res_path) if not name.startswith(('.', '_'))]
    test_frs = yaml.safe_load(open(os.path.join(res_path, user_names[0], 'frpair0_mask0_config.yaml'), 'r'))['eval_fr_names']
    compression_methods = yaml.safe_load(open(os.path.join(res_path, user_names[0], 'frpair0_mask0_config.yaml'), 'r'))['eval_compression_methods']
    for fr_name in test_frs:
        res_recorder = {
            '1a': [],
            '1b': [],
            '2a': [],
            '2b': []
        }
        compression_recorders = {method: copy.deepcopy(res_recorder) for method in compression_methods}
        res_recorder.update({'ssim': [],'psnr': [],'l0': []})
        res_filename = f"frpair0_mask0_testfr{fr_name}_results.yaml"
        for user_name in user_names:
            user_path = os.path.join(res_path, user_name)
            with open(os.path.join(user_path, res_filename), 'r') as f:
                user_res = yaml.safe_load(f)
            for key in res_recorder.keys():
                res_recorder[key].append(user_res[key])
            if compression:
                with open(os.path.join(user_path, f"frpair0_mask0_testfr{fr_name}_compression_results.yaml"), 'r') as f:
                    compression_res = yaml.safe_load(f)
                    for method, res_dict in compression_res.items():
                        for k, v in res_dict.items():
                            compression_recorders[method][k].append(v)
        print(f"Results for {fr_name}:")
        for key, values in res_recorder.items():
            if key in ['ssim', 'psnr', 'l0']:
                print(f"{key}: {np.mean(values):.4f}")
            else:
                print(f"{name_mapping[key]}: {np.mean(values):.4f}")
        if compression:
            print("Compression Results:")
            for method, recorder in compression_recorders.items():
                print(f"Method: {method}")
                for key, values in recorder.items():
                    print(f"{name_mapping[key]}: {np.mean(values):.4f}")
        # Plotting the results
        bar_keys = ['1a', '2b', '2a']
        bar_labels = [name_mapping[k] for k in bar_keys]
        bar_values = [np.mean(res_recorder[k]) for k in bar_keys]
        
        plt.figure(figsize=(10, 6))
        plt.bar(bar_labels, bar_values)
        for i, value in enumerate(bar_values):
            plt.text(i, value + 0.01, f"{value:.2f}", ha='center', va='bottom')
        plt.title(f"Evaluation Results for {fr_name.upper()}")
        plt.xlabel("Scenario")
        plt.ylabel("Average Value")
        plt.xticks(rotation=15, ha="right")
        plt.ylim(0, 1)
        plt.axhline(y=np.mean(res_recorder['1b']), color='r', linestyle='--', label='FR Baseline Performance')
        plt.text(0, np.mean(res_recorder['1b']) + 0.01, f"{np.mean(res_recorder['1b']):.2f}", color='r', ha='center', va='bottom')
        plt.figtext(0.5, 0.01, f"SSIM: {np.mean(res_recorder['ssim']):.4f}, PSNR: {np.mean(res_recorder['psnr']):.4f}, L0: {np.mean(res_recorder['l0']):.4f}", ha='center', fontsize=10)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.legend()
        plt.savefig(os.path.join(res_path, f"{fr_name}_evaluation_results.png"))
        plt.close()
        if compression:
            method_num = len(compression_methods)
            plt.figure(figsize=(method_num*5, 6))
            for idx, (method, recorder) in enumerate(compression_recorders.items()):
                plt.subplot(1, method_num, idx + 1)
                bar_values = [np.mean(recorder[k]) for k in bar_keys]
                plt.bar(bar_labels, bar_values)
                for i, value in enumerate(bar_values):
                    plt.text(i, value + 0.01, f"{value:.2f}", ha='center', va='bottom')
                plt.title(f"Compression Method: {method}")
                plt.xlabel("Scenario")
                plt.ylabel("Average Value")
                plt.xticks(rotation=15, ha="right")
                plt.ylim(0, 1)
                plt.axhline(y=np.mean(recorder['1b']), color='r', linestyle='--', label='FR Baseline Performance')
                plt.text(0, np.mean(recorder['1b']) + 0.01, f"{np.mean(recorder['1b']):.2f}", color='r', ha='center', va='bottom')
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(res_path, f"{fr_name}_compression_results.png"))
            plt.close()

if __name__ == "__main__":
    ######################### Configuration #########################
    eval_name = "eval_default" # The name of the subfolder under results/eval/ where the evaluation results are saved. Figures from this program will also be saved in this folder.
    compression = False # If you have compression results, set this to True. Otherwise, set it to False.
    #################################################################
    ana_res(eval_name, compression)


