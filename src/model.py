import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        self.model = nn.Sequential(

            # Convolutional layers
            
            # First 3x244x244 -> 16x244x244
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # Reduced dimensions by half i.e. 16x112x112
            
            # Second  16x112x112 -> 32x112x112
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # Reduced dimensions by half i.e. 32x56x56
            
            # Third  32x56x56 -> 64x56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # Reduced dimensions by half i.e. 64x28x28
            
            # Fourth  64x28x28 -> 128x28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # Reduced dimensions by half i.e. 128x14x14
            
            # Fifth  128x14x14 -> 256x14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # Reduced dimensions by half i.e. 256x7x7
            
                        
            nn.Flatten(),  # Flatten 256x7x7 to 256*7*7

            # Fully Connected Layers
            
            nn.Linear(256 * 7 * 7, 512), # nn.Linear(in_features, out_features)  
            nn.Dropout(p=dropout),

            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
