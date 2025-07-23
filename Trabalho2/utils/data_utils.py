"""
Utilitários para download e preparação dos dados do dataset Chest X-ray Image
"""

import os
import json
from pathlib import Path
import torch
import kagglehub
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.v2 as transforms
import numpy as np
import cv2
import random
from typing import Tuple, Dict, List, Union, Optional

# Convert tensor to numpy for OpenCV processing
def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array for OpenCV processing"""
    # Convert from (C, H, W) to (H, W, C)
    img_np = img.permute(1, 2, 0).numpy()
    
    # Convert to uint8 if float
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        img_np = (img_np * 255).astype(np.uint8)
        
    return img_np

# Convert numpy back to tensor
def numpy_to_tensor(img_np: np.ndarray, original_dtype: torch.dtype) -> torch.Tensor:
    """Convert NumPy array back to PyTorch tensor"""
    # Convert from (H, W, C) to (C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    
    # Normalize to 0-1 if original was float
    if original_dtype == torch.float32 or original_dtype == torch.float64:
        img_tensor = img_tensor.float() / 255.0
        
    return img_tensor

def apply_clahe(img: torch.Tensor, clip_limit: float = 2.0, 
                tile_grid_size: Tuple[int, int] = (8, 8)) -> torch.Tensor:
    """
    Aplica Contrast Limited Adaptive Histogram Equalization (CLAHE) em imagens
    
    Args:
        img: Imagem como tensor do PyTorch (C, H, W)
        clip_limit: Limite de contraste para CLAHE
        tile_grid_size: Tamanho da grade para CLAHE
        
    Returns:
        Imagem com CLAHE aplicado
    """
    original_dtype = img.dtype
    img_np = tensor_to_numpy(img)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE based on number of channels
    if img_np.shape[2] == 1:  # Grayscale
        img_np[:, :, 0] = clahe.apply(img_np[:, :, 0])
    else:  # RGB
        # Convert to LAB to apply CLAHE only on L channel
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return numpy_to_tensor(img_np, original_dtype)

def download_dataset(target_dir: str = "dataset") -> str:
    """
    Faz o download do dataset Chest X-ray Image do Kaggle
    
    Args:
        target_dir: Diretório onde os dados serão salvos
        
    Returns:
        Caminho para o diretório com os dados
    """
    print("Baixando dataset...")
    # Usando a biblioteca kagglehub para baixar o dataset

    # Only set credentials if not already in environment
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        # Try to find kaggle.json in standard location
        kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
        
        # If not in standard location, check project directory
        if not kaggle_json_path.exists():
            project_kaggle_json = Path('/home/luise/RedesNeurais_202501/kaggle.json')
            if project_kaggle_json.exists():
                kaggle_json_path = project_kaggle_json
        
        if kaggle_json_path.exists():
            print(f"Loading Kaggle credentials from {kaggle_json_path}")
            with open(kaggle_json_path) as f:
                credentials = json.load(f)
                os.environ['KAGGLE_USERNAME'] = credentials['username']
                os.environ['KAGGLE_KEY'] = credentials['key']
        else:
            print("Warning: kaggle.json not found. Make sure it exists in ~/.kaggle/ or project directory.")
    
    dataset_path = kagglehub.dataset_download("alsaniipe/chest-x-ray-image")
    
    print(f"Dataset baixado em: {dataset_path}")
    return dataset_path


def load_img(path: str) -> torch.Tensor:
    """
    Carrega uma imagem e converte para o formato adequado
    
    Args:
        path: Caminho para a imagem
        
    Returns:
        Tensor do PyTorch representando a imagem
    """
    # Le a imagem em diversos formatos e garante que a imagem tenha 3 canais
    img = Image.open(path).convert('RGB')
    # converte para um tensor do pytorch
    img = transforms.functional.to_image(img)
    # garante que seja uma imagem de 8 bits reescalando os valores adequadamente
    img = transforms.functional.to_dtype(img, dtype=torch.uint8, scale=True)
    return img


def create_datasets(dataset_path: str, img_size: int = 224, batch_size: int = 32, 
                    val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria os dataloaders de treino, validação e teste
    
    Args:
        dataset_path: Caminho para o diretório com os dados
        img_size: Tamanho das imagens após o redimensionamento
        batch_size: Tamanho do batch
        val_split: Proporção do conjunto de treino a ser usada para validação
        
    Returns:
        Dataloaders de treino, validação e teste
    """
    # Pre-processing 
    preprocess_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.uint8, scale=True),
        # Use lambda functions to wrap our OpenCV-based functions
        transforms.Lambda(lambda img: apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)))
    ])
    
    # Data augmentation 
    augmentation_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        # Use our custom random grayscale function
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1)
    ])
    
    # Normalization 
    normalization_transform = transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Compose the complete transformations for training
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        preprocess_transform,
        augmentation_transform,
        normalization_transform
    ])
    
    # Transformations for validation and test (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        preprocess_transform,
        normalization_transform
    ])
    
    # Criação dos datasets usando ImageFolder
    train_data_path = os.path.join(dataset_path, 'train')
    test_data_path = os.path.join(dataset_path, 'test')
    
    # Dataset de treino com data augmentation
    full_train_dataset = datasets.ImageFolder(
        root=train_data_path,
        transform=train_transform,
        loader=load_img
    )
    
    # Divisão do conjunto de treino em treino e validação
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Para o conjunto de validação, precisamos aplicar as transformações de validação
    val_dataset.dataset = datasets.ImageFolder(
        root=train_data_path,
        transform=eval_transform,
        loader=load_img
    )
    
    # Dataset de teste
    test_dataset = datasets.ImageFolder(
        root=test_data_path,
        transform=eval_transform,
        loader=load_img
    )
    
    # Criação dos dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_distribution(dataset: Dataset) -> Dict[str, int]:
    """
    Obtém a distribuição das classes no dataset
    
    Args:
        dataset: Dataset a ser analisado
        
    Returns:
        Dicionário com a contagem de amostras por classe
    """
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
        # Caso seja um subset criado com random_split
        classes = dataset.dataset.classes
        class_to_idx = dataset.dataset.class_to_idx
    elif hasattr(dataset, 'classes'):
        # Caso seja um dataset direto
        classes = dataset.classes
        class_to_idx = dataset.class_to_idx
    else:
        raise ValueError("O dataset fornecido não possui atributo 'classes'")
    
    # Inicializa contador para cada classe
    counts = {cls: 0 for cls in classes}
    
    # Conta as ocorrências de cada classe
    for _, label in dataset:
        counts[classes[label]] += 1
    
    return counts


def visualize_batch(dataloader: DataLoader, num_images: int = 8) -> Tuple[torch.Tensor, List[str]]:
    """
    Obtém um batch de imagens e seus rótulos para visualização
    
    Args:
        dataloader: DataLoader para obter o batch
        num_images: Número de imagens a serem retornadas
        
    Returns:
        Tensores das imagens e lista com os rótulos correspondentes
    """
    # Obtém um batch
    images, labels = next(iter(dataloader))
    
    # Limita o número de imagens
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Obtém os nomes das classes
    classes = dataloader.dataset.dataset.classes if hasattr(dataloader.dataset, 'dataset') else dataloader.dataset.classes
    
    # Converte os índices das classes para os nomes
    label_names = [classes[label.item()] for label in labels]
    
    return images, label_names
    images, labels = next(iter(dataloader))
    
    # Limita o número de imagens
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Obtém os nomes das classes
    classes = dataloader.dataset.dataset.classes if hasattr(dataloader.dataset, 'dataset') else dataloader.dataset.classes
    
    # Converte os índices das classes para os nomes
    label_names = [classes[label.item()] for label in labels]
    
    return images, label_names
