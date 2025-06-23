# ProjectWork
Representing images with known physical interventions in CLIP models

ImageNet classes were pulled from:
https://www.kaggle.com/datasets/skyap79/imagenet-classes

Regarding Reproduceability, we will give an overview of the contents of our GitHub:

## The Paper:
* Contains Algorithm/Pseudocode for the main Breakdown Algorithm.
* Has relevant information regarding window size and thresholding.
* Proof for the singular value of a symmetric operator (justifiying SVD usage) - Appendix A
* Proof for the choices of semantic similarity measures - See methods section 4.2

## The Code:
* All code is tested on Python version 3.12.9

The following are primay, installed packages we use (missing dependencies will be installed automatically)
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

## The Codebase:

Important scripts:
* loadData.py - 
* se_correlation.py - Conducts the correlation experiment for section 5.4 -> produces entropy_analysis.csv for use with R.
* labelfitting.py - Uses our synthesized datasets to do labelfitting. Creates relevant JSON files to be used with se_correlation.py
* geometry_metrics.py - Implements Algorithm 1, Spectral Entropy and Similarity Matrix copmutation.
* clip_pipeline.py - Loads, embeds and creates JSON files for our datasets.
* CIFAR_test.py - Script to download CIFAR (if missing) and conducting correlation test for section 5.4
* thumbnails.py -
* stats.R - Contains all statistics for the report, see section 4.5 and 5.4 for more details and results.

Important notebooks:
* semantic_direction.ipynb 
* intervention_visual.ipynb
* Exploratory_plotting.ipynb - Contains an interactive look into how data can be loaded and visualized: Plots generated for our results section.
* Breakdown_analysis.ipynb - Analysis comparing projection techniques and breakdown analysis - used for section 5.3

Documents:
* entropy_analysis.csv - contains raw data from se_correlation.py
* cifartest.csv - contains raw data from the CIFAR_test.py script.
* GitBible - Document Albert made that specifies how to work with the repository.

UGV Rover code:

## Folders:
