"""
Funções para treinamento e avaliação dos modelos
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import time
import copy


def train_model(model: nn.Module, 
                dataloaders: Dict[str, DataLoader], 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                device: torch.device,
                num_epochs: int = 25) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Treina o modelo com a função de perda e o otimizador especificados
    
    Args:
        model: Modelo a ser treinado
        dataloaders: Dicionário com os dataloaders 'train' e 'val'
        criterion: Função de perda
        optimizer: Otimizador
        device: Dispositivo onde o treino será realizado (CPU ou GPU)
        num_epochs: Número de épocas de treino
        
    Returns:
        Modelo treinado e histórico de métricas
    """
    since = time.time()
    
    # Inicializa histórico de métricas
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Copia dos melhores pesos do modelo
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Loop de treinamento
    for epoch in range(num_epochs):
        print(f'Época {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Cada época tem uma fase de treino e uma de validação
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modo de treinamento
            else:
                model.eval()   # Modo de avaliação
            
            running_loss = 0.0
            running_corrects = 0
            
            # Itera sobre os dados
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zera os gradientes
                optimizer.zero_grad()
                
                # Forward
                # Habilita o cálculo de gradientes apenas no modo de treinamento
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + otimização apenas no modo de treinamento
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Estatísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Métricas da época
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Salva histórico de métricas
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Copia do modelo se melhor acurácia de validação
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    # Tempo de treinamento
    time_elapsed = time.time() - since
    print(f'Treinamento completo em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Melhor acurácia de validação: {best_acc:.4f}')
    
    # Carrega os melhores pesos
    model.load_state_dict(best_model_wts)
    
    return model, history


def evaluate_model(model: nn.Module, 
                   dataloader: DataLoader, 
                   device: torch.device, 
                   classes: List[str]) -> Dict[str, Any]:
    """
    Avalia o modelo no conjunto de dados especificado
    
    Args:
        model: Modelo a ser avaliado
        dataloader: DataLoader com os dados para avaliação
        device: Dispositivo onde a avaliação será realizada (CPU ou GPU)
        classes: Lista com os nomes das classes
        
    Returns:
        Dicionário com as métricas de avaliação
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Avaliando"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Converte para numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calcula as métricas
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calcula métricas por classe
    class_prec = precision_score(all_labels, all_preds, average=None)
    class_rec = recall_score(all_labels, all_preds, average=None)
    class_f1 = f1_score(all_labels, all_preds, average=None)
    
    # Organiza as métricas por classe
    class_metrics = {}
    for i, cls in enumerate(classes):
        class_metrics[cls] = {
            'precision': class_prec[i],
            'recall': class_rec[i],
            'f1_score': class_f1[i]
        }
    
    # Retorna todas as métricas
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'class_metrics': class_metrics,
        'predictions': all_preds,
        'true_labels': all_labels
    }


def get_failed_examples(model: nn.Module, 
                        dataloader: DataLoader, 
                        device: torch.device,
                        num_examples: int = 10) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """
    Obtém exemplos de falhas de classificação para análise qualitativa
    
    Args:
        model: Modelo para fazer as predições
        dataloader: DataLoader com os dados para avaliação
        device: Dispositivo onde a avaliação será realizada (CPU ou GPU)
        num_examples: Número máximo de exemplos a serem retornados
        
    Returns:
        Lista de imagens, rótulos verdadeiros e predições
    """
    model.eval()
    
    failed_images = []
    failed_labels = []
    failed_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Encontra as amostras incorretamente classificadas
            incorrect_mask = preds != labels
            
            if incorrect_mask.sum() > 0:
                incorrect_inputs = inputs[incorrect_mask].cpu()
                incorrect_labels = labels[incorrect_mask].cpu()
                incorrect_preds = preds[incorrect_mask].cpu()
                
                for i in range(len(incorrect_inputs)):
                    if len(failed_images) < num_examples:
                        failed_images.append(incorrect_inputs[i])
                        failed_labels.append(incorrect_labels[i].item())
                        failed_preds.append(incorrect_preds[i].item())
                    else:
                        break
                
                if len(failed_images) >= num_examples:
                    break
    
    return failed_images, failed_labels, failed_preds
