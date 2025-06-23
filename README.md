# ProjectWork
Representing images with known physical interventions in CLIP models

ImageNet classes were pulled from:
https://www.kaggle.com/datasets/skyap79/imagenet-classes

Regarding Reproduceability we will give an overview of the contents of our GitHub:

The Paper:
* Contains Algorithm/Pseudocode for the main Breakdown Algorithm.
* Has relevant information regarding window size and thresholding.
* Proof for the singular value of a symmetric operator (justifiying SVD usage) - Appendix A
* Proof for the choices of semantic similarity measures - See methods section 4.2

The Code:
* All code is tested on Python version 3.12.9

The following are installed packages we use (missing dependencies will be installed automatically)
* numpy - 2.1.0  
* pandas - 2.2.3  
* scipy - 1.15.2  
* scikit-learn - 1.6.1  
* matplotlib - 3.10.0  
* umap-learn - 0.5.7 - Projection technique
* torch - 2.5.1  - Dependency for CLIP
  → Install with: Follow official instructions for your OS and hardware: https://pytorch.org/get-started/locally/  
* torchvision - 0.20.1  
* torchaudio - 2.5.1  
* CLIP - See CLIP's GitHub for installation: https://github.com/openai/CLIP  
* statsmodels - 0.14.4 - For quantile regression plots

Important scripts:
* loadData.py -
* se_correlation.py -
* labelfitting.py -
* geometry_metrics.py -
* clip_pipeline.py -
* CIFAR_test.py -

Important notebooks:
* semantic_direction.ipynb 
* intervention_visual.ipynb
* Exploratory_plotting.ipynb - Contains an interactive look into how data can be loaded and visualized: Plots generated for our results section.
* Breakdown_analysis.ipynb - Analysis comparing projection techniques and breakdown analysis - used for section 5.3
