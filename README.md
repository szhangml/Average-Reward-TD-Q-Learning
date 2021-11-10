# Average-Reward-TD-Q-Learning

This repository contains the source code to reproduce all the numerical experiments as described in the paper ["Finite Sample Analysis of Average-Reward TD Learning and Q-Learning"](https://openreview.net/pdf?id=1Rxp-demAH0).

Here's a BibTeX entry that you can use to cite it in a publication:
```bibtex
@inproceedings{
zhang2021finite,
title={Finite Sample Analysis of Average-Reward {TD} Learning and \$Q\$-Learning},
author={Sheng Zhang and Zhe Zhang and Siva Theja Maguluri},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=1Rxp-demAH0}
}
```

# Requirements
* Python (>= 3.7)
* Numpy (>= 1.19.1)

# Usage

## Different TD fixed Points
Show the average-reward TD(<img src="https://render.githubusercontent.com/render/math?math=\lambda">) with linear function approximation algorithm converges to different TD fixed points starting from different initial points.
```
python different_TD_fixed_points.py
```

## Rate of Convergence
Show the rate of convergence of the average-reward TD(<img src="https://render.githubusercontent.com/render/math?math=\lambda">) with linear function approximation using diminishing step sizes for <img src="https://render.githubusercontent.com/render/math?math=\lambda \in \{0, 0.2, 0.4, 0.8\}">.
```
python rate_of_convergence.py
```

# Maintainer
* [Sheng Zhang](https://github.com/xiaojianzhang) - shengzhang@gatech.edu
