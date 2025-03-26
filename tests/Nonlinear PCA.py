import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import clip

from PIL import Image

np.random.seed(42)
torch.manual_seed(42)

coil20 = []
for i in range(20):  # 20 objects
    for j in range(72):  # 72 rotations per object
        try:
            img = Image.open(f'Data/coil-20/{i+1}/obj{i+1}__{j}.png')
            img_array = np.array(img).astype(np.float32)  # Convert to float32
            img_array /= 255.0  # Normalize to [0,1]
            coil20.append(img_array)  # Append to a single list
        except Exception as e:
            print(f"Error loading obj{i+1}__{j}.png: {e}")
            quit()

coildata = np.array(coil20)
data = coildata[:, np.newaxis, :, :]
data_tensor = torch.from_numpy(data)

dataset = TensorDataset(data_tensor, data_tensor)
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class ConvAutoencoder(nn.Module):
    def __init__(self, encoding_dim=2):
        super(ConvAutoencoder, self).__init__()

        # Encoder with MaxPooling
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),  # 128 → 126
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),  # 126 → 124
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 124 → 62
            nn.Conv2d(32, 64, kernel_size=3),  # 62 → 60
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),  # 60 → 58
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 58 → 29
        )

        # Calculate final encoder dimensions: 128*29*29 = 107,648
        self.fc1 = nn.Linear(128 * 29 * 29, encoding_dim)
        self.fc2 = nn.Linear(encoding_dim, 128 * 29 * 29)

        # Decoder with UpSampling
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),           # 29 → 58
            nn.ConvTranspose2d(128, 64, 3),        # 58 → 60
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3),         # 60 → 62
            nn.ReLU(),
            nn.Upsample(scale_factor=2),           # 62 → 124
            nn.ConvTranspose2d(32, 16, 3),         # 124 → 126
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3),          # 126 → 128
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Bottleneck
        encoded = self.fc1(x)
        x = self.fc2(encoded)
        x = x.view(batch_size, 128, 29, 29)

        # Decoder
        decoded = self.decoder(x)
        return decoded, encoded


class NNAutoencoder(nn.Module):
    def __init__(self, input_dim=1024, encoding_dim=2):
        super(NNAutoencoder, self).__init__()

        # Encoder: Reduce dimensionality
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim)  # Encoded representation
        )

        # Decoder: Expand back to input size
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def save_model_output(model, input_tensor, filename="output.png"):
    """
    Passes an image through the autoencoder, reconstructs it, and saves the output.

    Parameters:
        model (torch.nn.Module): The trained autoencoder model.
        input_tensor (torch.Tensor): A single image tensor of shape (1, 1, 128, 128).
        filename (str): Name of the file to save.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Disable gradient calculations
    print(input_tensor.shape)
    with torch.no_grad():
        reconstructed, _ = model(input_tensor)  # Add batch dimension
    print(reconstructed.shape)
    # Convert output tensor to NumPy array
    for i, image in enumerate(reconstructed):
        image = image.detach().numpy()
        image = (image * 255).astype(np.uint8)[0]

        # Save using PIL
        img = Image.fromarray(image)
        img.save(f'NLPCA/{i}_filename.png')


def base_test():
    encoding_dim = 2
    model = ConvAutoencoder(encoding_dim=encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the autoencoder
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_inputs, _ in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(batch_inputs)
            loss = criterion(outputs, batch_inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    encoded_points = []
    for d in data_tensor:
        d = d.unsqueeze(0)
        encoded = model(d)[1]
        encoded_points.append(encoded.view(-1).detach().numpy())

    encoded_points = np.array(encoded_points)

    num_points = encoded_points.shape[0]
    indices = np.arange(num_points)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(encoded_points[:, 0], encoded_points[:, 1], c=indices, cmap="hsv", alpha=0.7, edgecolors="black")
    cbar = plt.colorbar(sc)
    cbar.set_label("Image Index")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("2D Latent Space Representation")
    plt.grid(True)
    plt.show()

    save_model_output(model, data_tensor)


def clip_test():
    from coil20_rot import load_coil20

    def plot_features(features, title, location):
        dataset = TensorDataset(features, features)
        batch_size = 8
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        encoding_dim = 2
        input_dim = features.shape[2]  # Ensure input_dim is dynamically assigned
        model = NNAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Train the autoencoder
        num_epochs = 50
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_inputs, _ in dataloader:
                optimizer.zero_grad()
                outputs, _ = model(batch_inputs)
                loss = criterion(outputs, batch_inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_inputs.size(0)

            epoch_loss = running_loss / len(dataset)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Extract encoded features
        encoded_points = []
        with torch.no_grad():
            for d in features:
                d = d.unsqueeze(0)  # Add batch dimension
                encoded = model(d)[1]
                encoded_points.append(encoded.view(-1).cpu().numpy())

        encoded_points = np.array(encoded_points)

        # Plot the encoded space
        plt.figure(figsize=(8, 6))
        plt.plot(encoded_points[:, 0], encoded_points[:, 1], 'k-', alpha=0.3)
        sc = plt.scatter(encoded_points[:, 0], encoded_points[:, 1], c=np.arange(len(encoded_points)),
                         cmap="hsv", alpha=0.7, edgecolors="black")
        cbar = plt.colorbar(sc)
        cbar.set_label("Image Index")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.title(title)
        plt.grid(True)
        plt.show()

        plt.savefig(location + title + ".pdf")
        plt.close()


    if torch.cuda.is_available():
        device = "cuda"
        print("\nUsing " + torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("Using CPU")

    available_model_names = clip.available_models()

    coil20 = load_coil20()
    print(available_model_names[5:])
    for model_name in available_model_names[5:]:
        # loading model and downloading if not already downloaded
        print(f"Loading the {model_name} model")
        model, preprocess = clip.load(model_name, device=device)

        print(f"Generating dim-reduction plots for {model_name}")
        for i in tqdm(range(20)):
            features = []
            for j in range(72):
                image = preprocess(coil20[i][j]).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                features.append(image_features.cpu().numpy())
            features = np.array(features)
            print(features.shape)
            features = torch.from_numpy(features).to(device).to(torch.float32)
            plot_features(features, f"{model_name}/{i + 1}", "tests/coil20_rot_NLPCA/")


if __name__ == '__main__':
    while True:
        test = input('[b] Base test image' + '\n' + '[C] Clip test image' + '\n' + 'Input: ').lower()
        if test == 'b':
            base_test()
            break
        elif test == 'c':
            clip_test()
            break
        else:
            raise 'No valid input'
