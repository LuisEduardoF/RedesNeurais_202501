"""
Funções de visualização para análise dos resultados
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from typing import Dict, List, Tuple, Any
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
import cv2


def plot_training_history(history: Dict[str, List[float]]) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plota o histórico de treinamento
    
    Args:
        history: Dicionário com o histórico de treinamento
        
    Returns:
        Figura e eixos do matplotlib
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot de perda
    ax[0].plot(history['train_loss'], label='Treino')
    ax[0].plot(history['val_loss'], label='Validação')
    ax[0].set_title('Perda')
    ax[0].set_xlabel('Época')
    ax[0].set_ylabel('Perda')
    ax[0].legend()
    
    # Plot de acurácia
    ax[1].plot(history['train_acc'], label='Treino')
    ax[1].plot(history['val_acc'], label='Validação')
    ax[1].set_title('Acurácia')
    ax[1].set_xlabel('Época')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()
    
    fig.tight_layout()
    
    return fig, ax


def plot_confusion_matrix(cm: np.ndarray, classes: List[str]) -> plt.Figure:
    """
    Plota a matriz de confusão
    
    Args:
        cm: Matriz de confusão
        classes: Lista com os nomes das classes
        
    Returns:
        Figura do matplotlib
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    
    return plt.gcf()


def plot_failed_examples(images: List[torch.Tensor], 
                         true_labels: List[int], 
                         pred_labels: List[int], 
                         classes: List[str],
                         rows: int = 2,
                         cols: int = 5) -> plt.Figure:
    """
    Plota exemplos de falhas de classificação
    
    Args:
        images: Lista de imagens
        true_labels: Lista de rótulos verdadeiros
        pred_labels: Lista de rótulos preditos
        classes: Lista com os nomes das classes
        rows: Número de linhas no grid
        cols: Número de colunas no grid
        
    Returns:
        Figura do matplotlib
    """
    fig = plt.figure(figsize=(15, 8))
    
    # Número de imagens a serem exibidas
    num_images = min(len(images), rows * cols)
    
    # Desnormalização das imagens
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    for i in range(num_images):
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # Desnormaliza a imagem
        img = denorm(images[i])
        
        # Converte para formato adequado para visualização
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f"True: {classes[true_labels[i]]}\nPred: {classes[pred_labels[i]]}")
        ax.axis('off')
    
    fig.tight_layout()
    
    return fig


def apply_gradcam(model: nn.Module, 
                  image: torch.Tensor, 
                  target_layer: nn.Module, 
                  device: torch.device,
                  class_idx: int = None) -> np.ndarray:
    """
    Aplica o Grad-CAM em uma imagem
    
    Args:
        model: Modelo para explicabilidade
        image: Imagem a ser explicada (já normalizada como tensor)
        target_layer: Camada alvo para o Grad-CAM
        device: Dispositivo onde o modelo está
        class_idx: Índice da classe para explicar (se None, usa a classe predita)
        
    Returns:
        Imagem com o mapa de calor sobreposto
    """
    # Implementação básica do Grad-CAM
    # Para uma implementação mais robusta, recomenda-se usar a biblioteca pytorch-grad-cam
    
    # Registra hooks para salvar gradientes e ativações
    gradients = []
    activations = []
    
    def save_gradient(grad):
        gradients.append(grad)
    
    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)
    
    # Registra os hooks
    handle = target_layer.register_forward_hook(forward_hook)
    
    # Coloca a imagem no dispositivo e adiciona dimensão de batch
    image = image.to(device).unsqueeze(0)
    image.requires_grad = True
    
    # Forward pass
    output = model(image)
    
    # Limpa gradientes existentes
    model.zero_grad()
    
    # Backpropagation
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    
    one_hot = torch.zeros_like(output)
    one_hot[0][class_idx] = 1
    output.backward(gradient=one_hot)
    
    # Remove o hook
    handle.remove()
    
    # Calcula os pesos
    gradients = gradients[0].cpu().data.numpy()[0]  # [C, H, W]
    activations = activations[0].cpu().data.numpy()[0]  # [C, H, W]
    
    # Média global dos gradientes para cada mapa de características
    weights = np.mean(gradients, axis=(1, 2))  # [C]
    
    # Combinação ponderada dos mapas de ativação
    cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
    for i, w in enumerate(weights):
        cam += w * activations[i]
    
    # Aplicação da função ReLU
    cam = np.maximum(cam, 0)
    
    # Normalização
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    # Redimensionamento para o tamanho da imagem original
    original_img = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    h, w, _ = original_img.shape
    cam = cv2.resize(cam, (w, h))
    
    # Cria o mapa de calor colorido
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Desnormalização da imagem original
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    original_img = denorm(image.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    original_img = np.clip(original_img, 0, 1)
    original_img = (original_img * 255).astype(np.uint8)
    
    # Sobreposição do mapa de calor na imagem original
    result = heatmap * 0.4 + original_img * 0.6
    
    return result.astype(np.uint8)


def create_metrics_table(metrics_custom: Dict[str, Any], 
                        metrics_frozen: Dict[str, Any], 
                        metrics_fine_tuning: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Cria uma tabela com as métricas dos três modelos
    
    Args:
        metrics_custom: Métricas do modelo customizado
        metrics_frozen: Métricas do modelo com parte convolucional congelada
        metrics_fine_tuning: Métricas do modelo com fine-tuning completo
        
    Returns:
        Dicionário com as métricas para criar a tabela
    """
    table_data = {
        'Modelo': ['Custom', 'Frozen', 'Fine-tuning'],
        'Acurácia': [
            metrics_custom['accuracy'],
            metrics_frozen['accuracy'],
            metrics_fine_tuning['accuracy']
        ],
        'Precisão': [
            metrics_custom['precision'],
            metrics_frozen['precision'],
            metrics_fine_tuning['precision']
        ],
        'Revocação': [
            metrics_custom['recall'],
            metrics_frozen['recall'],
            metrics_fine_tuning['recall']
        ],
        'F1-Score': [
            metrics_custom['f1_score'],
            metrics_frozen['f1_score'],
            metrics_fine_tuning['f1_score']
        ]
    }
    
    return table_data
