# Setup

You will need mamba (or conda, same cli interface). Afterwards you can create a fresh environment with all dependencies as described below

```bash
mamba create -n hybrid_contraction_tree_optimizer cotengra quimb autoray cytoolz loky networkx opt_einsum optuna tqdm pandas rich pygraphviz networkx cython matplotlib quimb nevergrad
mamba activate hybrid_contraction_tree_optimizer
pip install cotengrust
pip install kahypar
pip install julia
```

Alternatively you can try install the exact versions into an existing environment:

```bash
pip install -r requirements.txt
```

Afterwards you can run the different experiments like this

```bash
python experiments/hybrid_experiments.py
```

Our algorithm is stored in [hybrid_hypercut_greedy.py](experiments/hybrid_hypercut_greedy.py).
