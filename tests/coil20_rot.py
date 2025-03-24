# for models and fast tensor/matrix operations
import numpy as np
import torch
import clip

# for image processing and plotting
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

# for dimensionality reduction methods
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

# for user progress bar
from tqdm import tqdm

def plot_features(features, title, location):
    '''
    Takes a 1D array of features from 72 images of same object from coil20 dataset
    and plots the features using TSNE, UMAP, PCA, KernelPCA with rbf, poly and sigmoid kernels

    Parameters:
    features (1D array): Features of 72 images of same object
    title (str): Title of the plot
    location (str): Location to save the plot

    Will produce a pdf file with the title in the location. e.g
    plot_features(features, "baseline/1", "tests/coil20_rot/")
    will save the plot as tests/coil20_rot/baseline/1.pdf
    '''

    features = features.reshape(72,features.shape[0]//72)
    img_tsne = TSNE(n_components=2).fit_transform(features)
    img_umap = UMAP(n_components=2).fit_transform(features)
    img_pca = PCA(n_components=2).fit_transform(features)
    img_rbf = KernelPCA(n_components=2, kernel='rbf').fit_transform(features)
    img_poly = KernelPCA(n_components=2, kernel='poly').fit_transform(features)
    img_sigmoid = KernelPCA(n_components=2, kernel='sigmoid').fit_transform(features)

    cmap = plt.colormaps["plasma"]
    norm = Normalize(vmin=0, vmax=71)

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    fig.suptitle(title)

    # Plot t-SNE
    axes[0,0].plot(img_tsne[:, 0], img_tsne[:, 1], 'k-', alpha=0.3)
    axes[0,0].scatter(img_tsne[:, 0], img_tsne[:, 1], c=np.arange(72), cmap=cmap)
    axes[0,0].set_title('TSNE')

    # Plot UMAP
    axes[0,1].plot(img_umap[:, 0], img_umap[:, 1], 'k-', alpha=0.3)
    axes[0,1].scatter(img_umap[:, 0], img_umap[:, 1], c=np.arange(72), cmap=cmap)
    axes[0,1].set_title('UMAP')

    # Plot PCA
    axes[1,0].plot(img_pca[:, 0], img_pca[:, 1], 'k-', alpha=0.3)
    axes[1,0].scatter(img_pca[:, 0], img_pca[:, 1], c=np.arange(72), cmap=cmap)
    axes[1,0].set_title('PCA')

    # Plot Kernel PCA
    axes[1,1].plot(img_rbf[:, 0], img_rbf[:, 1], 'k-', alpha=0.3)
    axes[1,1].scatter(img_rbf[:, 0], img_rbf[:, 1], c=np.arange(72), cmap=cmap)
    axes[1,1].set_title('rbf-Kernel PCA') # radial basis function kernel

    # Plot Kernel PCA
    axes[2,0].plot(img_poly[:, 0], img_poly[:, 1], 'k-', alpha=0.3)
    axes[2,0].scatter(img_poly[:, 0], img_poly[:, 1], c=np.arange(72), cmap=cmap)
    axes[2,0].set_title('poly-Kernel PCA') # polynomial kernel

    # Plot Kernel PCA
    axes[2,1].plot(img_sigmoid[:, 0], img_sigmoid[:, 1], 'k-', alpha=0.3)
    axes[2,1].scatter(img_sigmoid[:, 0], img_sigmoid[:, 1], c=np.arange(72), cmap=cmap)
    axes[2,1].set_title('sigmoid-Kernel PCA') # sigmoid kernel


    # Adjust layout to make room for colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.88)  # Make room for colorbar on right

    # Add colorbar on the right side
    cbar_ax = fig.add_axes([0.9, 0.05, 0.02, 0.9])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label="Picture number")

    # saving plots as pdf (vector graphics are better for scaling)
    plt.savefig(location + title + ".pdf")
    plt.close()


def load_coil20():
    '''
    Loads the coil20 dataset from the Data folder
    '''
    coil20 = list()
    for i in range(20):
        coil20.append(list())
        for j in range(72):
            try:
                img = Image.open(f'Data/coil-20/{i+1}/obj{i+1}__{j}.png')
                coil20[i].append(img)
            except:
                print(f"Error: ulimit might be too small. Or the file obj{i+1}__{j}.png does not exist")
                quit()
    return coil20



if __name__ == '__main__':
    # importing images
    # if you can't.. change ulimit to a bigger number, by: ulimit -n $BIG_NUMBER
    coil20 = load_coil20()

    # baseline of images without model
    print("Generating baseline plots")
    for i in tqdm(range(20)):
        features = np.array([])
        for j in range(72):
            img = np.asarray(coil20[i][j])
            features = np.append(features, img.flatten())
        
        plot_features(features, f"baseline/{i+1}","tests/coil20_rot/")



    # selecting device to run models on
    if torch.cuda.is_available():
        device = "cuda"
        print("\nUsing " + torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("Using CPU")


    available_model_names = clip.available_models()

    for model_name in available_model_names:
        # loading model and downloading if not already downloaded
        print(f"Loading the {model_name} model")
        model, preprocess = clip.load(model_name, device=device)

        print(f"Generating dim-reduction plots for {model_name}")
        for i in tqdm(range(20)):
            features = np.array([])
            for j in range(72):
                image = preprocess(coil20[i][j]).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                features = np.append(features, image_features.cpu().numpy())
            
            plot_features(features, f"{model_name}/{i+1}","tests/coil20_rot/")