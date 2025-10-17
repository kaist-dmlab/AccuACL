import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34_64(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        resnet34 = models.resnet34(pretrained=False)
        
        # Modify the first layer to accept 64x64 images
        resnet34.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Keep the rest of the layers unchanged
        self.features = nn.Sequential(*list(resnet34.children())[:-1])
        
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