import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18_64(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        resnet18 = models.resnet18(pretrained=False)
        
        # Modify the first layer to accept 64x64 images
        resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Keep the rest of the layers unchanged
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        
        # Add a new fully connected layer for the new classification task
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x, repr=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        if repr:
            return output, x
        return output

    def get_embedding_dim(self):
        return self.linear.in_features