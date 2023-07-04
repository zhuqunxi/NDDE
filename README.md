# PyTorch Implementation of Neural Delay Differential Equations (NDDEs)
The source code of NDDEs will be fully released once our extended work is published. If you have any question about this software, please contact Qunxi Zhu (qxzhu16@fudan.edu.cn).

## Examples
Examples can be found in the [`Examples`](./Examples) directory.

We encourage those who are interested in using the NDDEs framework to run the code [`Examples/Mackey_Glass/MGlass.py`](./Examples/Mackey_Glass/MGlass.py) for understanding how to use `NDDEs` to fit a typical 1-D delay system, i.e., Mackey Glass system.

Please run the following code in [`Examples/Mackey_Glass/MGlass.py`], the model checkpoints and the reconstruction figures are saved in the files  'model' and 'figures'.
```
python MGlass.py
```
One can use the following Colab link for this example.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O_YoTMWy4HGN9v984EvV8ijUXj_xCjTg?usp=sharing)
## References

If you found this library useful in your research, please consider citing the following papers.


Zhu, Q., Guo, Y., & Lin, W. (2023). "Neural Delay Differential Equations." *International Conference on Learning Representations.* 2021. [[arxiv]](https://arxiv.org/abs/2102.10801)

```
@article{zhu2021neural,
  title={Neural delay differential equations},
  author={Zhu, Qunxi and Guo, Yao and Lin, Wei},
  journal={arXiv preprint arXiv:2102.10801},
  year={2021}
}
```
An extended version of NDDEs:

Zhu, Q., Guo, Y., & Lin, W. (2023). "Neural Delay Differential Equations: System Reconstruction and Image Classification." *arXiv preprint arXiv:2304.05310.* 2023. [[arxiv]](https://arxiv.org/abs/2304.05310)

```
@article{zhu2023neural,
  title={Neural Delay Differential Equations: System Reconstruction and Image Classification},
  author={Zhu, Qunxi and Guo, Yao and Lin, Wei},
  journal={arXiv preprint arXiv:2304.05310},
  year={2023}
}

```

Time delay system reconstruction.

Zhu, Q., Li X., & Lin, W. (2023). "Leveraging neural differential equations and adaptive delayed feedback to detect unstable periodic orbits based on irregularly-sampled time series.", *CHAOS*, Fast track, Editor’s Pick, 33(3),031101. [[url]](https://doi.org/10.1063/5.0143839)
```
@article{zhu2023leveraging,
  title={Leveraging neural differential equations and adaptive delayed feedback to detect unstable periodic orbits based on irregularly sampled time series},
  author={Zhu, Qunxi and Li, Xin and Lin, Wei},
  journal={Chaos: An Interdisciplinary Journal of Nonlinear Science},
  volume={33},
  number={3},
  year={2023},
  publisher={AIP Publishing}
}
```

