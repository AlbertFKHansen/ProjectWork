import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from PIL import Image

np.random.seed(42)
torch.manual_seed(42)

coil20 = list()
for i in range(1):
    coil20.append(list())
    for j in range(72):
        try:
            img = Image.open(f'../Data/coil-20/{i+1}/obj{i+1}__{j}.png')
            img_array = np.array(img).astype(np.float32)
            img_array /= 255.0
            coil20[i].append(img_array)
        except Exception as e:
            print(f"Error loading obj{i+1}__{j}.png: {e}")
            quit()

data = np.array(coil20[0])
data = data[:, np.newaxis, :, :]
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
    with torch.no_grad():
        reconstructed, _ = model(input_tensor.unsqueeze(0))  # Add batch dimension

    # Convert output tensor to NumPy array
    reconstructed_image = reconstructed.squeeze(0).squeeze(0).cpu().numpy()  # Shape: (128, 128)

    # Convert to 8-bit image format
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

    # Save using PIL
    img = Image.fromarray(reconstructed_image)
    img.save(filename)
    print(f"Saved reconstructed image as {filename}")


def base_test():
    encoding_dim = 2
    model = ConvAutoencoder(encoding_dim=encoding_dim)
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

    encoded_points = []
    for d in data_tensor:
        d = d.unsqueeze(0)
        encoded = model(d)[1]
        encoded_points.append(encoded.view(-1).detach().numpy())

    encoded_points = np.array(encoded_points)

    num_points = encoded_points.shape[0]
    hues = np.linspace(0, 1, num_points)  # Linearly spaced hues (looping in color)
    colors = [mcolors.hsv_to_rgb((h, 1, 1)) for h in hues]  # Convert hues to RGB

    plt.figure(figsize=(8, 6))
    plt.scatter(encoded_points[:, 0], encoded_points[:, 1], c=colors, alpha=0.7, edgecolors="black")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("2D Latent Space Representation")
    plt.grid(True)
    plt.show()

    save_model_output(model, data_tensor)


def clip_test():
    pass


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


