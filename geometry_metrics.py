import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns

from scipy.stats import entropy
import umap

from typing import List, Tuple

###                     ###
##   Numerical Metrics   ##
###                     ###

def calculate_angles(points: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Compute tightest angle between line segments connecting a point,
    and segment lengths. Works for any dimension.

    Parameters:
    ----------
    embeddings : np.ndarray
        A 2D array of shape (n_samples, embedding_dim) representing the embeddings.

    Returns
    -------
    angles : List[float]
        List of angles (in degrees) connected to each point
    lengths : List[float]
        List of edge lengths from previous to current point
    """
    num_points = len(points)
    padded_points = np.vstack((points[-1], points, points[0]))
    angles = []
    lengths = []

    for i in range(1, num_points + 1):
        prev = padded_points[i - 1]
        curr = padded_points[i]
        next_ = padded_points[i + 1]

        vec_prev = prev - curr
        vec_next = next_ - curr

        norm_prev = np.linalg.norm(vec_prev)
        norm_next = np.linalg.norm(vec_next)

        lengths.append(norm_prev)

        #
        #if norm_prev < 1e-8 or norm_next < 1e-8:
        #    angles.append(np.nan)
        #    continue

        cos_theta = np.dot(vec_prev, vec_next) / (norm_prev * norm_next)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle_deg = np.degrees(np.arccos(cos_theta))
        smallest_angle = min(angle_deg, 360 - angle_deg)
        angles.append(smallest_angle)

    return angles, lengths

def CV(input: list) -> float:
    """
    Computes the Coefficient of Variance for a list of numbers

    Parameters:
    ----------
    input : list
        Array-like list

    Returns:
    -------
    cvar : float
            Coefficient of Variance.
    """
    cvar = np.std(input) / np.mean(input)
    return cvar

def sliding_window(embeddings: np.ndarray, window: int) -> Tuple[List[float], List[float]]:
    """
    Computes Coefficient of Variance for angles and lengths using calculate_angles
    in a sliding window.

    Parameters:
    ----------
    embeddings : np.ndarray
        A 2D array of shape (n_samples, embedding_dim) representing the embeddings.
    window : int
        window size s.t. each calculate_angles call is of size window*2+1

    Returns:
    -------
    angleCVs, lengthCVs : list
        lists containing Coefficients of Variance
    """
    n = len(embeddings)
    angleCVs = []
    lengthCVs = []

    for i in range(n):
        indices = [(i + j) % n for j in range(-window, window + 1)]
        bin_points = np.array([embeddings[j] for j in indices])

        angles, lengths = calculate_angles(bin_points)

        angleCVs.append(CV(angles))
        lengthCVs.append(CV(lengths))

    return angleCVs, lengthCVs

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Cosine Similarity for i,j embedding as matrix.

    Parameters:
    ----------
    embeddings : np.ndarray
        A 2D array of shape (n_samples, embedding_dim) representing the embeddings.

    Returns:
    -------
    cosine_sim_matrix : np.ndarray
        Square matrix of shape (embedding_dim, embedding_dim),
        containing the i,j embeddings cosine similarity
    """
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cosine_sim_matrix = np.dot(normed, normed.T)
    return cosine_sim_matrix

def compute_spectral_entropy(similarity_matrix: np.ndarray) -> float:
    """
    Computes the spectral entropy of a cosine similarity matrix using Singular Value Decomposition (SVD).
    
    Parameters:
    ----------
    similarity_matrix : np.ndarray
        A square matrix (n_samples, n_samples) representing pairwise cosine similarities between embeddings.
    
    Returns:
    -------
    spectral_entropy : float
        The spectral entropy of the cosine similarity matrix.
        Higher number corresponds to a lesser symmetric object under rotation
    """
    _, singular_values, _ = np.linalg.svd(similarity_matrix, full_matrices=False)
    
    #Normalize singular values to valid distribution
    s_norm = singular_values / singular_values.sum()
    
    #entropy computes regular shannon entropy
    spectral_entropy = entropy(s_norm)
    return spectral_entropy


###                     ###
##       Plotting        ##
###                     ###

def plot_similarity_heatmap(similarity_matrix: np.ndarray, step: int = 3, cmap: str = "viridis") -> None:
    """
    Plots a heatmap of a cosine similarity matrix with labeled axes.

    Parameters:
    ----------
    similarity_matrix : np.ndarray
        A 2D square matrix (n x n) containing cosine similarity values.
    
    step : int, optional (default=3)
        Interval for tick marks on the axes for readability.
    
    cmap : str, optional (default='viridis')
        Colormap used for the heatmap visualization.

    Returns:
    -------
    None
        Displays the heatmap using matplotlib.
    """
    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(similarity_matrix, cmap=cmap, square=True, cbar=True, 
                     xticklabels=step, yticklabels=step)

    tick_positions = np.arange(0, similarity_matrix.shape[0], step)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions, rotation=90, fontsize=8)
    ax.set_yticklabels(tick_positions, rotation=0, fontsize=8)

    plt.title("Cosine Similarity Heatmap")
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")
    plt.tight_layout()
    plt.show()

def plot_symmetry_coefficients(angle_cvs: List[float], length_cvs: List[float]) -> None:
    """
    Plots the coefficients of variation for angles and lengths.
    Should be used in conjunction with sliding_window.

    Parameters:
    ----------
    angle_cvs : List[float]
        Output from sliding_window.
    
    length_cvs : List[float]
        Output from sliding_window.

    Returns:
    -------
    None
        Displays a plot comparing CVs according to their configutation in sliding_window.
    """
    x = np.arange(len(angle_cvs))

    plt.figure(figsize=(5, 5))
    plt.plot(x, angle_cvs, label='Angle CV', color='blue', linewidth=2)
    plt.plot(x, length_cvs, label='Length CV', color='orange', linewidth=2)

    plt.fill_between(x, angle_cvs, alpha=0.2, color='blue')
    plt.fill_between(x, length_cvs, alpha=0.2, color='orange')

    plt.title("Object Symmetry Coefficients")
    plt.xlabel("Image Index")
    plt.ylabel("Coefficient of Variation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

