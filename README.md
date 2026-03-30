# Model-Based-Inference-of-Synaptic-Plasticity-Rules
PyTorch implementation of "Model-Based Inference of Synaptic Plasticity Rules" (Mehta et al., NeurIPS 2024) with vectorised einsum optimisation. Infers synaptic learning rules from neural activity and behavioural data across three experiments including real Drosophila data. 

PyTorch reproduction of [Mehta et al. (NeurIPS 2024)](https://papers.neurips.cc/paper_files/paper/2024/file/571082ea18d30060177dfcaf662ff0e5-Paper-Conference.pdf) for inferring synaptic plasticity rules from neural and behavioural data.

**Original paper:** [GitHub (JAX)](https://github.com/yashsmehta/MetaLearnPlasticity) | [Paper PDF](https://papers.neurips.cc/paper_files/paper/2024/file/571082ea18d30060177dfcaf662ff0e5-Paper-Conference.pdf)

**Authors:** Irmak Erkol, Umut Alperen Cengiz, Ertugrul Taparci

## Overview

The method parameterises synaptic plasticity rules as either a truncated Taylor series (81 coefficients) or a multilayer perceptron, then optimises the parameters via backpropagation through time to match observed neural activity or behavioural data. We reimplement the entire pipeline in PyTorch and reproduce three experiments of increasing difficulty:

1. **Experiment 1 (Oja's Rule):** Recovering a known plasticity rule from simulated neural activity trajectories
2. **Experiment 2 (Behavioural Plasticity):** Inferring a reward-based covariance rule from binary accept/reject choices
3. **Experiment 3 (Fruit Fly):** Applying the method to real behavioural data from 18 *Drosophila melanogaster*

We also reproduce the paper's appendix analyses: hyperparameter sensitivity (L1, moving average window, input firing mean), scalability across trajectory lengths and network sizes, generalisability across 46 plasticity rules, and held-out validation on fly data.


## Project Structure
```
Model-Based-Inference-of-Synaptic-Plasticity-Rules/
├── src/
│   ├── __init__.py
│   ├── network.py              # Forward simulation (vectorised einsum version) (IRMAK ERKOL)
│   └── plasticity_rules.py     # Taylor, MLP, and fly plasticity parameterisations (IRMAK ERKOL)
├── data/                       # Fly1.mat through Fly18.mat (from Zenodo) 
├── figures/                    # Generated figures 
├── run_behavior.py             # Experiment 2: Behavioural plasticity (IRMAK ERKOL)
├── run_fly.py                  # Experiment 3: Real fly data (IRMAK ERKOL)
├── run_fig4b.py                # Figure 4B: Fly behaviour rasters (IRMAK ERKOL)
├── run_fig5_6_7.py             # Figures 5-7: Hyperparameter sweeps (IRMAK ERKOL)
├── run_fig8_9.py               # Figures 8-9: Held-out validation (IRMAK ERKOL)
├── run_table2.py               # Table 2: Scalability analysis (IRMAK ERKOL)
├── run_tables1_3.py            # Tables 1&3: 46 plasticity rules comparison (IRMAK ERKOL)
├── process_fly_data.py         # Raw Zenodo data to .mat converter (IRMAK ERKOL)
├── process_fly_data_v2.py      # V2 with variable-length trial support (IRMAK ERKOL)
├── run_all_collab.ipynb        # Google Colab notebook to run everything (IRMAK ERKOL)
├── requirements.txt
├── LICENSE                     # MIT License
└── README.md
```
```

## Installation
```bash
git clone https://github.com/YOURREPO.git
cd plasticity_pytorch
pip install torch numpy scipy matplotlib
```

No additional dependencies required. Works with PyTorch 2.0+ and CUDA 12.

## Usage

### Quick start (run all experiments sequentially)
```bash
mkdir -p figures results
python run_oja.py
python run_behavior.py
python run_fly.py
```

### Reproducing appendix figures and tables
```bash
python run_fig4b.py
python run_fig5_6_7.py
python run_table2.py
python run_fig8_9.py
python run_tables1_3.py
```

All scripts auto-detect CUDA and use GPU when available.

## Fly Data

Real fly behavioural data is from [Rajagopalan et al. (2023)](https://doi.org/10.5281/zenodo.7449214). Download and place `.mat` files in the `data/` directory:
```bash
mkdir data
# Download Fly1.mat through Fly18.mat from Zenodo DOI: 10.5281/zenodo.7449214
```

## Implementation Details

Several implementation details not specified in the paper proved critical for correct results

- Hidden layer activation: `tanh` (not sigmoid as implied by Equation 1)
- Output layer: fixed weights of `5.0/n_hidden`, not plastic
- Plasticity learning rate: `1/n_input` applied to weight updates
- Input noise variance: 0.015 (not 0.05)
- Taylor coefficient initialisation: scale 1e-5
- MLP architecture: leaky ReLU hidden + tanh output, all params init scale 0.01
- Expected reward initialised to 0 (not 0.5)

## Citation

If you use this code, please cite the original paper:
```bibtex
@inproceedings{mehta2024model,
  title={Model-Based Inference of Synaptic Plasticity Rules},
  author={Mehta, Yash and Tyulmankov, Danil and Rajagopalan, Adithya and Turner, Glenn and Fitzgerald, James and Funke, Jan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

## License

This is an academic reproduction for educational purposes. The original method and data belong to their respective authors.
