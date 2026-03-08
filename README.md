#   Probabilistic Shape

This repository contains simulation code accompanying research on probabilistic methods for shape derivatives and shape optimization. The code is organized by publication. Each directory contains the scripts and experiments used to generate the results for the corresponding paper.

The overall objective is to develop a framework of mesh-free probabilistic shape optimization that applies in high dimensions.

##  Papers and Code

### Paper 1 - A Probabilistic Approach to Shape Derivatives

Code: derivative_taylor_test/

This directory contains the code used for the simulations of the paper **A Probabilistic Approach to Shape Derivatives**

The paper is concerned with the theoretical foundation of the probabilistic approach. Namely the representation of the boundary sensitivity of the underlying state and of the shape functional derivative via probabilistic methods. 
The code then performs a Taylor Test of a mesh-free evaluation of the probabilistic shape derivative for several testing directions. This includes oblique perturbations which do posses a tangential component. For comparison, especially for the test directions that are not in normal direction (and thus would not appear in a respective gradient scheme) we include several mesh-based finite-element methods. Nevertheless we emphasize that the evaluation of the probabilistic shape derivative does not involve any mesh.

###  Paper 2 - A First Step Towards Mesh-Free Probabilistic Shape Optimization

Code: probabilistic_optimization/

This directory contains the code used for the simulations of the paper **A First Step Towards Mesh-Free Probabilistic Shape Optimization**

The code consists of an optimization loop which employs the probabilistic representation to approximate the optimal shape of a Poisson tracking problem. Here the underlying domain is characterized by a boundary mesh. Specifically, the implementation is in 2D, although many methods are written to apply in 3D as well. 

##  Reproducibility

All numerical experiments used in the papers can be reproduced using the scripts provided in the corresponding directories. There is no fixed random seed in neither of the simulations, thus to reproduce the results of paper 1, we refer to Table 1: Shape Derivative Value Comparison; especially the respective standard deviations.

Each directory contains a `main.py` script that runs the experiments for the corresponding paper.

##  Citation

If you use this code in your research, please cite the corresponding paper.

### A Probabilistic Approach to Shape Derivatives
```bibtex
@misc{Schlegel2024ProbabilisticShape,
    title={A Probabilistic Approach to Shape Derivatives}, 
    author={Luka Schlegel and Volker Schulz and Frank T. Seifried and Maximilian W{\"u}rschmidt},
    howpublished = {arXiv:2409.15967},
    year = {2024},
    keywords = {},
    pubstate = {published},
    tppubtype = {misc}
}
```
### A First Step Towards Mesh-Free Probabilistic Shape Optimization
```bibtex
@misc{schmidt2026ProbabilisticShapeOpt,
    title={A First Step Towards Mesh-Free Probabilistic Shape Optimization}, 
    author={Stephan Schmidt and Maximilian W{\"u}rschmidt},
    year={2026},
    howpublished={arxiv:2603.01141}, 
    pubstate = {published},
    tppubtype = {misc}
}
```


