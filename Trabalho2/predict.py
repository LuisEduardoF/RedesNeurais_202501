"""
Script para executar inferência com os modelos treinados
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Adiciona o diretório atual ao path para importar módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.models import CustomCNN, create_pretrained_model_frozen, create_pretrained_model_fine_tuning
from utils.data_utils import load_img
from utils.visualization import apply_gradcam


def parse_arguments():
    """
    Analisa os argumentos da linha de comando
    
    Returns:
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(description="Inferência com modelos treinados de CNN para classificação de raio-X")
    
    parser.add_argument("--image_path", type=str, required=True,
                        help="Caminho para a imagem de entrada")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Caminho para o arquivo do modelo treinado (.pt)")
    parser.add_argument("--model_type", type=str, choices=["custom", "frozen", "fine_tuning"], required=True,
                        help="Tipo de modelo (custom, frozen, fine_tuning)")
    parser.add_argument("--pretrained_model", type=str, default="densenet121",
                        help="Modelo pré-treinado usado (densenet121, resnet50 ou mobilenet_v2)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Tamanho das imagens após o redimensionamento")
    parser.add_argument("--use_gradcam", action="store_true",
                        help="Usa Grad-CAM para visualizar as regiões de interesse")
    
    return parser.parse_args()


def main():
    """
    Função principal para inferência com os modelos treinados
    """
    # Parse dos argumentos
    args = parse_arguments()
    
    # Verifica disponibilidade de GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Classes do dataset
    classes = ["COVID19", "NORMAL", "PNEUMONIA"]
    num_classes = len(classes)
    
    # Carrega o modelo adequado
    if args.model_type == "custom":
        model = CustomCNN(num_classes=num_classes)
    elif args.model_type == "frozen":
        model = create_pretrained_model_frozen(args.pretrained_model, num_classes)
    else:  # fine_tuning
        model = create_pretrained_model_fine_tuning(args.pretrained_model, num_classes)
    
    # Carrega os pesos do modelo
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Transformações para a imagem de entrada
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Carrega e processa a imagem
    image = load_img(args.image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inferência
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
    
    # Mostra os resultados
    print(f"\nImagem: {args.image_path}")
    print(f"Predição: {classes[predicted_class]}")
    print(f"Confiança: {probabilities[predicted_class]:.4f}")
    
    # Mostra todas as probabilidades
    print("\nProbabilidades:")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {probabilities[i]:.4f}")
    
    # Visualiza a imagem com a predição
    plt.figure(figsize=(10, 5))
    
    # Imagem original
    plt.subplot(1, 2 if args.use_gradcam else 1, 1)
    img = transforms.ToPILImage()(image)
    plt.imshow(img)
    plt.title(f"Predição: {classes[predicted_class]}\nConfiança: {probabilities[predicted_class]:.4f}")
    plt.axis('off')
    
    # Aplica Grad-CAM se solicitado
    if args.use_gradcam:
        print("\nAplicando Grad-CAM...")
        
        # Identifica a última camada convolucional do modelo
        if args.model_type == "custom":
            target_layer = model.conv4
        else:
            if args.pretrained_model == "densenet121":
                target_layer = model.features.denseblock4.denselayer16.conv2
            elif args.pretrained_model == "resnet50":
                target_layer = model.layer4[-1].conv3
            elif args.pretrained_model == "mobilenet_v2":
                target_layer = model.features[-1]
        
        # Aplica Grad-CAM
        image_tensor = transform(image).to(device)
        gradcam_result = apply_gradcam(model, image_tensor, target_layer, device, predicted_class)
        
        # Exibe a imagem com o mapa de calor
        plt.subplot(1, 2, 2)
        plt.imshow(gradcam_result)
        plt.title(f"Grad-CAM - {classes[predicted_class]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
