# for models and fast tensor/matrix operations
import numpy as np
import torch
import clip

# for image processing and plotting
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial.distance import chebyshev
from scipy.spatial.distance import cdist



# for user progress bar
from tqdm import tqdm

# function found in our own script
from coil20_rot import load_coil20


def plot_analogy_similarity(features,label_imbeddings, title, location):
    '''
    Takes a 1D array of features from 72 images of an object
    and plots the similarity of the images with a directional label using cosine similarity

    Parameters:
    features (1D array): Features of 72 images of same object
    title (str): Title of the plot
    location (str): Location to save the plot

    Will produce a pdf file with the title in the location. e.g
    plot_features(features, "model/1", "tests/coil20_rot_analogy/")
    will save the plot as tests/coil20_rot_analogy/model/1.pdf
    '''

    features = features.reshape(72,features.shape[0]//72)
    label_imbeddings = label_imbeddings.reshape(3,label_imbeddings.shape[0]//3)

    similarity = cosine_similarity(label_imbeddings,features)

    fig, ax = plt.subplots()
    ax.plot(similarity[0], label="front")
    ax.plot(similarity[1], label="behind")
    ax.plot(similarity[2], label="side")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Image number")
    ax.set_ylabel("Cosine similarity")
    fig.savefig(f"{location}{title}.pdf")
    plt.close(fig)






coil20 = load_coil20()


# object names for the coil20 dataset
object_names = {1: "a rubber duck",
                2: "a figure",
                3: "a toy car",
                4: "Maneki-neko",
                5: "a carton",
                6: "a toy car",
                7: "a figure",
                8: "a baby powder bottle",
                9: "a carton",
                10: "vaseline",
                11: "a figure",
                12: "a mug",
                13: "a piggy bank",
                14: "a socket",
                15: "a container",
                16: "a bottle",
                17: "a mug",
                18: "a cup",
                19: "a toy car",
                20: "a tub of cream cheese"}



# selecting device to run models on
if torch.cuda.is_available():
    device = "cuda"
    print("Using " + torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("Using CPU")


available_model_names = clip.available_models()

for model_name in available_model_names:
    # loading model and downloading if not already downloaded
    print(f"Loading the {model_name} model")
    model, preprocess = clip.load(model_name, device=device)

    # getting the imbeddings for the directions
    directions = ["from the front","from the behind","from the side"]
    lable_imbeddings = np.array([])
    for direction in directions:
        text = f"{direction}"
        text_tokens = clip.tokenize(text).to(device)
        text_features = model.encode_text(text_tokens)
        lable_imbeddings = np.append(lable_imbeddings, text_features.cpu().detach().numpy())

    print(f"Generating directional analogy plots for {model_name}")
    for i in tqdm(range(20)):
        features = np.array([])
        for j in range(72):
            image = preprocess(coil20[i][j]).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)

            features = np.append(features, image_features.cpu().numpy())
        
        
            
        
        plot_analogy_similarity(features,lable_imbeddings, f"{model_name}/{i+1}","tests/coil20_rot_analogy/")