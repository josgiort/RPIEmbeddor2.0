
## Usage
1. Generate embeddings: `python scripts/embedding_generation/[script].py --[args]`
2. Train: `python src/[train_script].py --[args]`
3. Baselines: `python baselines/[script].py --[args]`
4. Inference: `python src/inference.py`

Experiments were run on [HPC cluster] with SLURM; example job script: `[path]`.

## Training

Training scripts for the experiments reported in the thesis are available under:

```text
src/
```

These include experiments on eCLIP2 and RNAInterAct using RNA-disjoint, protein-disjoint, and random data splits, as well as frozen-embedding and one-hot ablations.
