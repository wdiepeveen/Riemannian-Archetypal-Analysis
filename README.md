# Riemannian Archetypal Analysis

    [1] W. Diepeveen, D. Needell.  
    Riemannian Archetypal Analysis: Interpretable non-linear data analysis on deformed star distributions.
    arXiv preprint arXiv:2605.24113. 2026 May 22.

Setup
-----

The recommended (and tested) setup is based on Python 3.13. Install the following dependencies with anaconda:

    # Create conda environment
    conda create --name raa python=3.13
    conda activate raa

    # Clone source code and install
    git clone https://github.com/wdiepeveen/Riemannian-Archetypal-Analysis.git
    cd "Riemannian-Archetypal-Analysis"
    pip install -r requirements.txt


Reproducing the experiments in [1]
----------------------------------

To produce the results in [1]. 
* For the 2D toy example experiments run:
  *  `river_cross_starflow.ipynb`
* For the 3D toy example experiments run:
  *  `tree_multi_ellipsoid_star.ipynb`
* For mnist experiments run:
  *  `single_digit_mnist_multi_ellipsoid_starflow.ipynb`
  *  `mnist_multi_ellipsoid_starflow.ipynb`
