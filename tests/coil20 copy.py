import clip.model
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
import clip
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

"""
THIS FILE DOES NOT WORK AT THE MOMENT..

I'm trying to extract the layers from the model and then plot them.
But its taking to long triht now, because classes are rediculously slow in python
"""

























class LayerExtractor:
    def __init__(self, model):
        """
        Initialize the layer extractor with a PyTorch model.
        
        Args:
            model (nn.Module): The PyTorch model to extract layer values from
        """
        self.model = model
        self.hooks = []
        self.layer_outputs = OrderedDict()
        
        # Register hooks for all modules
        self._register_hooks(self.model)

    def _register_hooks(self, module, prefix=''):
        """
        Recursively register forward hooks on all modules.
        
        Args:
            module (nn.Module): The module to register hooks on
            prefix (str): Prefix for the layer name
        """
        for name, submodule in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name
            
            # If it has children, recurse into them
            if list(submodule.named_children()):
                self._register_hooks(submodule, layer_name)
            
            # Register hook on the current module
            hook = submodule.register_forward_hook(
                lambda mod, inp, out, layer_name=layer_name: self._hook_fn(mod, out, layer_name)
            )
            self.hooks.append(hook)
    
    def _hook_fn(self, module, output, layer_name):
        """
        Hook function to save the module output.
        
        Args:
            module (nn.Module): The module
            output: The output tensor(s) from the module
            layer_name (str): Name of the layer
        """
        # Handle different output formats
        if isinstance(output, torch.Tensor):
            self.layer_outputs[layer_name] = output.detach().cpu()
        elif isinstance(output, tuple):
            self.layer_outputs[layer_name] = [t.detach().cpu() if isinstance(t, torch.Tensor) else t for t in output]
    
    def extract(self, input_tensor):
        """
        Extract values of each layer from the input tensor.
        
        Args:
            input_tensor (torch.Tensor): Input to the model
            
        Returns:
            list: List of lists containing layer outputs
        """
        # Clear previous outputs
        self.layer_outputs.clear()
        
        # Run the model
        with torch.no_grad():
            self.model.encode_image(input_tensor)
        
        # Convert OrderedDict to list of lists
        layer_values = []
        for name, output in self.layer_outputs.items():
            if isinstance(output, torch.Tensor):
                layer_values.append(output.tolist())
            elif isinstance(output, list):
                layer_values.append([t.tolist() if isinstance(t, torch.Tensor) else t for t in output])
        
        return layer_values





# importing images 
# if you can't.. in terminal write: ulimit -n [number]
coil20 = list()
for i in range(20):
    coil20.append(list())
    for j in range(72):
        img = Image.open(f'Data/coil-20/{i+1}/obj{i+1}__{j}.png')
        coil20[i].append(img)

# selecting device to run models on
if torch.cuda.is_available():
    device = "cuda"
    print("Using " + torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("Using CPU")


available_model_names = clip.available_models()

for model_name in available_model_names:
    # loading model and installing if not already installed
    print(f"Loading the {model_name} model")
    model, preprocess = clip.load(model_name, device=device)
    # Use the layer extractor
    extractor = LayerExtractor(model)

    print(f"Generating TSNE and PCA plot for {model_name}")
    for i in tqdm(range(20)):
        features = [[] for _ in range(len(extractor.hooks))]
        for j in tqdm(range(72)):
            input = preprocess(coil20[i][j]).unsqueeze(0).to(device)
            # Extract layer values as list of lists
            layer_values = extractor.extract(input)

            for v, layer in enumerate(layer_values):
                features[v].append(layer)
        
        for u, layer_features in enumerate(features):
            layer_features = np.array(layer_features)
            layer_features = layer_features.reshape(72,layer_features.shape[0]//72)
            img_tsne = TSNE(n_components=2).fit_transform(layer_features)
            img_pca = PCA(n_components=2).fit_transform(layer_features)

            cmap = plt.colormaps["plasma"]
            norm = Normalize(vmin=0, vmax=71)

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"Layer {u+1}")

            # Plot t-SNE
            ax1.plot(img_tsne[:, 0], img_tsne[:, 1], 'k-', alpha=0.3)
            ax1.scatter(img_tsne[:, 0], img_tsne[:, 1], c=np.arange(72), cmap=cmap)
            ax1.set_title('TSNE')

            # Plot PCA
            ax2.plot(img_pca[:, 0], img_pca[:, 1], 'k-', alpha=0.3)
            ax2.scatter(img_pca[:, 0], img_pca[:, 1], c=np.arange(72), cmap=cmap)
            ax2.set_title('PCA')

            # Adjust layout to make room for colorbar
            plt.tight_layout()
            fig.subplots_adjust(right=0.88)  # Make room for colorbar on right

            # Add colorbar on the right side with custom width
            cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label="Picture number")

            plt.savefig(f"tests/coil20/{model_name}/layer_{u+1}/{i+1}.png")
            plt.close()