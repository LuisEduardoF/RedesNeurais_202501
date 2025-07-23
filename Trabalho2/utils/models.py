"""
Definição da arquitetura da rede neural customizada
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Tuple


class CustomCNN(nn.Module):
    """
    Rede neural convolucional customizada para classificação de imagens de raio-x
    """
    def __init__(self, num_classes: int = 3):
        """
        Inicializa a rede neural customizada
        
        Args:
            num_classes: Número de classes para classificação
        """
        super(CustomCNN, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Camadas densas (fully connected)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Para entrada de 224x224
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe para frente na rede
        
        Args:
            x: Batch de imagens (B, C, H, W)
            
        Returns:
            Logits para as classes (B, num_classes)
        """
        # Camadas convolucionais
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Reshape para camadas densas
        x = x.view(x.size(0), -1)
        
        # Camadas densas
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class PretrainedCNN(nn.Module):
    """
    Classe base para redes neurais pré-treinadas
    """
    def __init__(self, model_name: str = "densenet121", num_classes: int = 3, freeze_conv: bool = False):
        """
        Inicializa o modelo pré-treinado
        
        Args:
            model_name: Nome do modelo pré-treinado
            num_classes: Número de classes para classificação
            freeze_conv: Se True, congela os parâmetros da parte convolucional
        """
        super(PretrainedCNN, self).__init__()
        
        # Carrega o modelo pré-treinado
        if model_name == "densenet121":
            self.model = models.densenet121(weights="IMAGENET1K_V1")
        elif model_name == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V1")
        elif model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Modelo '{model_name}' não suportado")
        
        # Congela os parâmetros da parte convolucional, se necessário
        if freeze_conv:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Substitui a camada de classificação
        self._replace_classifier(model_name, num_classes)
        
        self.model_name = model_name
    
    def _replace_classifier(self, model_name, num_classes):
        """
        Substitui a camada de classificação do modelo
        
        Args:
            model_name: Nome do modelo pré-treinado
            num_classes: Número de classes para classificação
        """
        if model_name == "densenet121":
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
        elif model_name == "resnet50":
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif model_name == "mobilenet_v2":
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe para frente na rede
        
        Args:
            x: Batch de imagens (B, C, H, W)
            
        Returns:
            Logits para as classes (B, num_classes)
        """
        return self.model(x)


class PretrainedFrozenCNN(PretrainedCNN):
    """
    Rede neural pré-treinada com parte convolucional congelada
    """
    def __init__(self, model_name: str = "densenet121", num_classes: int = 3):
        """
        Inicializa o modelo pré-treinado com parte convolucional congelada
        
        Args:
            model_name: Nome do modelo pré-treinado
            num_classes: Número de classes para classificação
        """
        super(PretrainedFrozenCNN, self).__init__(model_name, num_classes, freeze_conv=True)


class PretrainedFineTuningCNN(PretrainedCNN):
    """
    Rede neural pré-treinada para fine-tuning completo
    """
    def __init__(self, model_name: str = "densenet121", num_classes: int = 3):
        """
        Inicializa o modelo pré-treinado para fine-tuning completo
        
        Args:
            model_name: Nome do modelo pré-treinado
            num_classes: Número de classes para classificação
        """
        super(PretrainedFineTuningCNN, self).__init__(model_name, num_classes, freeze_conv=False)


# Mantém as funções originais para compatibilidade com código existente
def create_pretrained_model_frozen(model_name: str = "densenet121", num_classes: int = 3) -> nn.Module:
    """
    Cria uma rede neural pré-treinada com parte convolucional congelada
    
    Args:
        model_name: Nome do modelo pré-treinado
        num_classes: Número de classes para classificação
        
    Returns:
        Modelo pré-treinado com parte convolucional congelada
    """
    return PretrainedFrozenCNN(model_name, num_classes)


def create_pretrained_model_fine_tuning(model_name: str = "densenet121", num_classes: int = 3) -> nn.Module:
    """
    Cria uma rede neural pré-treinada para fine-tuning completo
    
    Args:
        model_name: Nome do modelo pré-treinado
        num_classes: Número de classes para classificação
        
    Returns:
        Modelo pré-treinado para fine-tuning completo
    """
    return PretrainedFineTuningCNN(model_name, num_classes)
