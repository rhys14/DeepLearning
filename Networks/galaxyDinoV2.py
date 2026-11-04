import torch 
import torch.nn as nn
from transformers import AutoModelForImageClassification

class GalaxyDINOv2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.model = AutoModelForImageClassification.from_pretrained(
            "facebook/dinov2-small",
            num_labels=num_classes,
        )
        
        for param in self.model.dinov2.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits
