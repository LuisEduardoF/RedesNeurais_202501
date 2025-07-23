"""
Script para execução com o Google Colab
"""

# @title Instalação das Dependências
!pip install kagglehub scikit-learn matplotlib seaborn tqdm opencv-python

# @title Verificação da GPU disponível
import torch
print(f"PyTorch versão: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")

# @title Baixar o Repositório
# Clone o repositório do GitHub (se aplicável)
# !git clone https://github.com/LuisEduardoF/RedesNeurais_202501.git
# %cd RedesNeurais_202501/Trabalho2

# @title Configuração do Dataset
import os
import kagglehub

# Configuração do Kaggle (caso necessário)
# Faça upload do arquivo kaggle.json para o Colab
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# Baixa o dataset do Kaggle
dataset_path = kagglehub.load('alsaniipe/chest-x-ray-image')
print(f"Dataset baixado em: {dataset_path}")

# Mostra a estrutura do dataset
!ls -la {dataset_path}
!ls -la {dataset_path}/train
!ls -la {dataset_path}/test

# @title Importar Módulos Necessários
import sys
sys.path.append('.')  # Adiciona o diretório atual ao PYTHONPATH

from utils.data_utils import create_datasets, get_class_distribution, visualize_batch
from utils.models import CustomCNN, create_pretrained_model_frozen, create_pretrained_model_fine_tuning
from utils.training import train_model, evaluate_model, get_failed_examples
from utils.visualization import plot_training_history, plot_confusion_matrix, plot_failed_examples, apply_gradcam, create_metrics_table

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# @title Definição dos Hiperparâmetros
img_size = 224  # Tamanho das imagens após o redimensionamento
batch_size = 32  # Tamanho do batch
val_split = 0.2  # Proporção do conjunto de treino a ser usada para validação
epochs = 20  # Número de épocas de treinamento
lr = 0.001  # Taxa de aprendizado
pretrained_model = "densenet121"  # Modelo pré-treinado a ser usado
use_gradcam = True  # Usar Grad-CAM para visualizar as regiões de interesse

# @title Preparação dos Dados
# Criação dos dataloaders
train_loader, val_loader, test_loader = create_datasets(
    dataset_path,
    img_size=img_size,
    batch_size=batch_size,
    val_split=val_split
)

# Dicionário com os dataloaders
dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}

# Obtém a distribuição das classes
class_distribution = get_class_distribution(train_loader.dataset)
print("Distribuição das classes no conjunto de treino:")
for cls, count in class_distribution.items():
    print(f"  {cls}: {count}")

# Visualiza um batch de imagens
images, labels = visualize_batch(train_loader)
plt.figure(figsize=(15, 8))
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    img = images[i].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalização para visualização
    plt.imshow(img)
    plt.title(labels[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Obtém os nomes das classes
classes = train_loader.dataset.dataset.classes if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.classes
num_classes = len(classes)
print(f"Classes: {classes}")

# @title Treinamento do Modelo Customizado
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Função de perda
criterion = nn.CrossEntropyLoss()

# Modelo Customizado
print("\n" + "="*50)
print("Treinando Modelo Customizado")
print("="*50)

model_custom = CustomCNN(num_classes=num_classes).to(device)
optimizer_custom = optim.Adam(model_custom.parameters(), lr=lr)

model_custom, history_custom = train_model(
    model_custom,
    {'train': train_loader, 'val': val_loader},
    criterion,
    optimizer_custom,
    device,
    num_epochs=epochs
)

# Avalia o modelo customizado
metrics_custom = evaluate_model(model_custom, test_loader, device, classes)

# Plota o histórico de treinamento
fig_custom, _ = plot_training_history(history_custom)
plt.show()

# Plota a matriz de confusão
fig_cm_custom = plot_confusion_matrix(metrics_custom['confusion_matrix'], classes)
plt.show()

# Analisa falhas do modelo customizado
failed_images_custom, failed_labels_custom, failed_preds_custom = get_failed_examples(
    model_custom, test_loader, device
)

if failed_images_custom:
    fig_failed_custom = plot_failed_examples(
        failed_images_custom, failed_labels_custom, failed_preds_custom, classes
    )
    plt.show()

# @title Treinamento do Modelo Pré-treinado com Parte Convolucional Congelada
# Modelo Pré-treinado com Parte Convolucional Congelada
print("\n" + "="*50)
print(f"Treinando Modelo Pré-treinado ({pretrained_model}) com Parte Convolucional Congelada")
print("="*50)

model_frozen = create_pretrained_model_frozen(pretrained_model, num_classes).to(device)
optimizer_frozen = optim.Adam(
    filter(lambda p: p.requires_grad, model_frozen.parameters()),
    lr=lr
)

model_frozen, history_frozen = train_model(
    model_frozen,
    {'train': train_loader, 'val': val_loader},
    criterion,
    optimizer_frozen,
    device,
    num_epochs=epochs
)

# Avalia o modelo com parte convolucional congelada
metrics_frozen = evaluate_model(model_frozen, test_loader, device, classes)

# Plota o histórico de treinamento
fig_frozen, _ = plot_training_history(history_frozen)
plt.show()

# Plota a matriz de confusão
fig_cm_frozen = plot_confusion_matrix(metrics_frozen['confusion_matrix'], classes)
plt.show()

# Analisa falhas do modelo com parte convolucional congelada
failed_images_frozen, failed_labels_frozen, failed_preds_frozen = get_failed_examples(
    model_frozen, test_loader, device
)

if failed_images_frozen:
    fig_failed_frozen = plot_failed_examples(
        failed_images_frozen, failed_labels_frozen, failed_preds_frozen, classes
    )
    plt.show()

# @title Treinamento do Modelo Pré-treinado com Fine-tuning Completo
# Modelo Pré-treinado com Fine-tuning Completo
print("\n" + "="*50)
print(f"Treinando Modelo Pré-treinado ({pretrained_model}) com Fine-tuning Completo")
print("="*50)

model_fine_tuning = create_pretrained_model_fine_tuning(pretrained_model, num_classes).to(device)
optimizer_fine_tuning = optim.Adam(model_fine_tuning.parameters(), lr=lr/10)

model_fine_tuning, history_fine_tuning = train_model(
    model_fine_tuning,
    {'train': train_loader, 'val': val_loader},
    criterion,
    optimizer_fine_tuning,
    device,
    num_epochs=epochs
)

# Avalia o modelo com fine-tuning completo
metrics_fine_tuning = evaluate_model(model_fine_tuning, test_loader, device, classes)

# Plota o histórico de treinamento
fig_fine_tuning, _ = plot_training_history(history_fine_tuning)
plt.show()

# Plota a matriz de confusão
fig_cm_fine_tuning = plot_confusion_matrix(metrics_fine_tuning['confusion_matrix'], classes)
plt.show()

# Analisa falhas do modelo com fine-tuning completo
failed_images_fine_tuning, failed_labels_fine_tuning, failed_preds_fine_tuning = get_failed_examples(
    model_fine_tuning, test_loader, device
)

if failed_images_fine_tuning:
    fig_failed_fine_tuning = plot_failed_examples(
        failed_images_fine_tuning, failed_labels_fine_tuning, failed_preds_fine_tuning, classes
    )
    plt.show()

# @title Aplicação do Grad-CAM (Ponto Extra)
if use_gradcam:
    print("\n" + "="*50)
    print("Aplicando Grad-CAM")
    print("="*50)
    
    # Identifica a última camada convolucional dos modelos
    if pretrained_model == "densenet121":
        target_layer = model_fine_tuning.features.denseblock4.denselayer16.conv2
    elif pretrained_model == "resnet50":
        target_layer = model_fine_tuning.layer4[-1].conv3
    elif pretrained_model == "mobilenet_v2":
        target_layer = model_fine_tuning.features[-1]
    
    # Aplica Grad-CAM em algumas imagens de teste
    images, labels = next(iter(test_loader))
    
    for i in range(min(5, len(images))):
        image = images[i]
        label = labels[i].item()
        
        # Aplica Grad-CAM
        gradcam_result = apply_gradcam(model_fine_tuning, image, target_layer, device, label)
        
        # Exibe a imagem com o mapa de calor
        plt.figure(figsize=(10, 5))
        plt.imshow(gradcam_result)
        plt.title(f"Classe: {classes[label]}")
        plt.axis('off')
        plt.show()

# @title Tabela de Comparação de Métricas
# Criação da tabela de métricas comparativas
table_data = create_metrics_table(metrics_custom, metrics_frozen, metrics_fine_tuning)

# Exibe a tabela de métricas
print("# Tabela de Métricas Comparativas\n")
print("| Modelo | Acurácia | Precisão | Revocação | F1-Score |")
print("|--------|----------|----------|-----------|----------|")

for i in range(len(table_data['Modelo'])):
    print(f"| {table_data['Modelo'][i]} | {table_data['Acurácia'][i]:.4f} | {table_data['Precisão'][i]:.4f} | {table_data['Revocação'][i]:.4f} | {table_data['F1-Score'][i]:.4f} |")

# Métricas por classe para cada modelo
print("\n# Métricas por Classe\n")

# Para cada modelo
for model_name, metrics in zip(
    ["Modelo Customizado", "Modelo com Parte Convolucional Congelada", "Modelo com Fine-tuning Completo"],
    [metrics_custom, metrics_frozen, metrics_fine_tuning]
):
    print(f"## {model_name}\n")
    print("| Classe | Precisão | Revocação | F1-Score |")
    print("|--------|----------|-----------|----------|")
    
    for cls, cls_metrics in metrics['class_metrics'].items():
        print(f"| {cls} | {cls_metrics['precision']:.4f} | {cls_metrics['recall']:.4f} | {cls_metrics['f1_score']:.4f} |")
    
    print("\n")

# @title Salvar Modelos
# Cria diretório para salvar os modelos
import os
os.makedirs('models', exist_ok=True)

# Salva os modelos treinados
torch.save(model_custom.state_dict(), 'models/model_custom.pt')
torch.save(model_frozen.state_dict(), 'models/model_frozen.pt')
torch.save(model_fine_tuning.state_dict(), 'models/model_fine_tuning.pt')

print("Modelos salvos em 'models/'")

# @title Conclusão
print("\n" + "="*50)
print("Treinamento completo!")
print("="*50)
