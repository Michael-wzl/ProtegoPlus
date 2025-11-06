# Under Development

Please do not use or distribute this code without permission from the author.

## Focal Diversity Analysis Tool

The evaluation results of size 3 and size 4 ensembles are stored in `results/eval/len3ensemble` and `results/eval/len4ensemble` respectively. Customize the focal diversity computation in `protego/focal_diversity.py`'s `get_focal_diversity` function by adding a new definition name. Use the following command to compute focal diversity and automatically analyze the results:

```bash
python3 -m tools.compute_focal_diversity --exp_name your_experiment_name --device cpu
```

You can find the analysis results in `results/focal_diversity/your_experiment_name`.
