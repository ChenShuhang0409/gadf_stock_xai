import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class PaperCNN(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2, dropout: float = 0.3):
        super(PaperCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def get_model(config: dict) -> nn.Module:
    model_name = config['model'].get('name', 'simple_cnn')
    in_channels = config['model']['in_channels']
    num_classes = config['model']['num_classes']
    dropout = config['model'].get('dropout', 0.3)
    
    if model_name == 'simple_cnn':
        model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    elif model_name == 'paper_cnn':
        model = PaperCNN(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Supported models: 'simple_cnn', 'paper_cnn'"
        )
    
    return model
