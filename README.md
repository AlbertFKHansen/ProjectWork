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
* loadData.py - Module for easily loading the datasets.
* se_correlation.py - Conducts the correlation experiment for section 5.4 -> produces entropy_analysis.csv for use with R.
* labelfitting.py - Uses our synthesized datasets to do labelfitting. Creates relevant JSON files to be used with se_correlation.py
* geometry_metrics.py - Implements Algorithm 1, Spectral Entropy and Similarity Matrix copmutation.
* clip_pipeline.py - Loads, embeds and creates JSON files for our datasets.
* CIFAR_test.py - Script to download CIFAR (if missing) and conducting correlation test for section 5.4
* thumbnails.py - Generated the Thumbnail directory and its contents.
* stats.R - Contains all statistics for the report, see section 4.5 and 5.4 for more details and results.
* make_compacts.sh - Script to create combined grids of images for object interventions.
* plot_gifs.sh - Script to create GIFs of the object intervention plots created by intervention_visual.ipynb.

Important notebooks:
* semantic_direction.ipynb - Used for the semantic analysis and visuals in section 5.5
* intervention_visual.ipynb - Used for plotting the intervention UMNAP and PCA visuals of raw images.
* Exploratory_plotting.ipynb - Contains an interactive look into how data can be loaded and visualized: Plots generated for our results section.
* Breakdown_analysis.ipynb - Analysis comparing projection techniques and breakdown analysis - used for section 5.3

Documents:
* entropy_analysis.csv - contains raw data from se_correlation.py
* cifartest.csv - contains raw data from the CIFAR_test.py script.
* GitBible - Document Albert made that specifies how to work with the repository.

UGV Rover code:
* dslt_ctrl.py - Contains the code for controlling the DSLR camera.
* phue_ctrl.py - Contains the code for controlling the Philips Hue Hub.
* main.py - The CLI used to collect the data.

## Directories:

Data/                       <- Contains all datasets used in the paper.
├── coil100/
│   ├── rot/
│   ├── coil100.py          <- script for organizing the coil 100 dataset.
│   ├── GT_labels.json      <- Ground Truths for every object in the dataset.
│   └── labels.json         <- Combination of Dataset GT and ImageNet classes. 
├── combined/               <- Contains the combined GT and labels for coil100 and Dataset. used in report.
│   ├── GT_labels.json
│   └── labels.json
├── Dataset/                <- Our Dataset; TRIIL-17
│   ├── rot/
│   ├── temp/
│   ├── GT_labels.json
│   └── labels.json
└── processed/              <- The processed version of TRILL-17
    ├── rot/
    ├── temp/
    ├── GT_labels.json
    └── labels.json

deprecated/                 <- Old or unused code, kept for reference.

Embeddings/                 <- Contains all JSON files of the dataset embeddings for our datasets.
├── coil100.json
├── Dataset.json
├── labels.json
└── processed.json

Figures/                    <- Visual diagrams (e.g. architectures, frameworks)
rover/                      <- Contains code for the UGV Rover used to collect the data.
Thumbnails/                 <- Contains thumbnails from the datasets.
