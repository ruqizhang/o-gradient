# Sampling in Constrained Domains with Orthogonal-Space Variational Gradient Descent

This repository contains code for the paper
[Sampling in Constrained Domains with Orthogonal-Space Variational Gradient Descent](https://arxiv.org/pdf/2210.06447.pdf), accepted in _Conference on Neural Information Processing Systems (NeurIPS), 2022_.

```bibtex
@article{zhang2022sampling,
  title={Sampling in Constrained Domains with Orthogonal-Space Variational Gradient Descent},
  author={Zhang, Ruqi and Liu, Qiang and Tong, Xin T.},
  journal={Conference on Neural Information Processing Systems},
  year={2022}
}
```

# Usage
## Synthetic Distribution
Please run
```
python main_synthetic.py
```
## Income Classification with Fairness Constraint 
Please run
```
python main_faireness.py
```
## Loan Classification with Logic Rules 
Please run
```
python main_logic.py
```
## Prior-Agnostic Bayesian Neural Networks
Please run
```
python main_bnn_langevin.py
```
```
python main_bnn_svgd.py
```