"""
Script principal para treinamento e avaliação dos modelos
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# Adiciona o diretório atual ao path para importar módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import download_dataset, create_datasets, get_class_distribution, visualize_batch
from utils.models import CustomCNN, PretrainedFrozenCNN, PretrainedFineTuningCNN
from utils.training import train_model, evaluate_model, get_failed_examples
from utils.visualization import (
    plot_training_history, plot_confusion_matrix, plot_failed_examples, 
    apply_gradcam, create_metrics_table
)


def parse_arguments():
    """
    Analisa os argumentos da linha de comando
    
    Returns:
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(description="Treinamento de Redes Neurais Convolucionais para Classificação de Raio-X")
    
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Diretório onde os dados serão salvos")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Tamanho do batch para treino e avaliação")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Tamanho das imagens após o redimensionamento")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Número de épocas de treinamento")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Taxa de aprendizado")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Proporção do conjunto de treino a ser usada para validação")
    parser.add_argument("--pretrained_model", type=str, default="densenet121",
                        help="Modelo pré-treinado a ser usado (densenet121, resnet50 ou mobilenet_v2)")
    parser.add_argument("--use_gradcam", action="store_true",
                        help="Usa Grad-CAM para visualizar as regiões de interesse")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Diretório para salvar os resultados")
    
    return parser.parse_args()


def main():
    """
    Função principal para treinamento e avaliação dos modelos
    """
    # Parse dos argumentos
    args = parse_arguments()
    
    # Cria diretório para salvar os resultados
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Nome do diretório para os resultados desta execução (com timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Verifica disponibilidade de GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Download do dataset
    dataset_path = download_dataset(args.data_dir) + "/Data/"
    
    # Criação dos dataloaders
    train_loader, val_loader, test_loader = create_datasets(
        dataset_path ,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split
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
    plt.savefig(os.path.join(run_dir, "sample_images.png"))
    
    # Obtém os nomes das classes
    classes = train_loader.dataset.dataset.classes if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.classes
    num_classes = len(classes)
    print(f"Classes: {classes}")
    
    # Função de perda
    criterion = nn.CrossEntropyLoss()
    
    # Dicionário para salvar os resultados
    results = {}
    
    # Define as configurações para cada modelo
    model_configs = [
        {
            "name": "Custom",
            "model_fn": CustomCNN,
            "model_args": {"num_classes": num_classes},
            "optimizer_fn": optim.Adam,
            "lr_factor": 1.0,
            "description": "Modelo Customizado"
        },
        {
            "name": "Frozen",
            "model_fn": PretrainedFrozenCNN,
            "model_args": {"model_name": args.pretrained_model, "num_classes": num_classes},
            "optimizer_fn": optim.Adam,
            "lr_factor": 1.0,
            "description": f"Modelo Pré-treinado ({args.pretrained_model}) com Parte Convolucional Congelada",
            "filter_params": True
        },
        {
            "name": "FineTuning",
            "model_fn": PretrainedFineTuningCNN,
            "model_args": {"model_name": args.pretrained_model, "num_classes": num_classes},
            "optimizer_fn": optim.Adam,
            "lr_factor": 0.1,
            "description": f"Modelo Pré-treinado ({args.pretrained_model}) com Fine-tuning Completo"
        }
    ]
    
    # Métricas para cada modelo
    all_metrics = {}
    
    # Treina cada modelo
    for config in model_configs:
        print("\n" + "="*50)
        print(f"Treinando {config['description']}")
        print("="*50)
        
        # Cria o modelo
        model = config["model_fn"](**config["model_args"]).to(device)
        
        # Configura o otimizador
        if config.get("filter_params", False):
            # Para modelos com camadas congeladas, otimizamos apenas os parâmetros com requires_grad=True
            optimizer = config["optimizer_fn"](
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * config["lr_factor"]
            )
        else:
            optimizer = config["optimizer_fn"](
                model.parameters(),
                lr=args.lr * config["lr_factor"]
            )
        
        # Treina o modelo usando treino e validação para early stopping
        model, history = train_model(
            model,
            {'train': train_loader, 'val': val_loader},
            criterion,
            optimizer,
            device,
            num_epochs=args.epochs
        )
        
        # Avalia o modelo apenas no conjunto de validação durante o desenvolvimento
        val_metrics = evaluate_model(model, val_loader, device, classes)
        
        # Armazena as métricas para comparação posterior
        all_metrics[config["name"]] = {
            'val': val_metrics
        }
        
        # Plota o histórico de treinamento
        fig_history, _ = plot_training_history(history)
        fig_history.savefig(os.path.join(run_dir, f"history_{config['name'].lower()}.png"))
        
        # Plota a matriz de confusão apenas para o conjunto de validação
        fig_cm_val = plot_confusion_matrix(val_metrics['confusion_matrix'], classes)
        fig_cm_val.savefig(os.path.join(run_dir, f"cm_val_{config['name'].lower()}.png"))
        
        # Salva o modelo
        torch.save(model.state_dict(), os.path.join(run_dir, f"model_{config['name'].lower()}.pt"))
        
        # Analisa falhas do modelo apenas para o conjunto de validação
        failed_val_images, failed_val_labels, failed_val_preds = get_failed_examples(
            model, val_loader, device
        )
        
        if failed_val_images:
            fig_failed_val = plot_failed_examples(
                failed_val_images, failed_val_labels, failed_val_preds, classes
            )
            fig_failed_val.savefig(os.path.join(run_dir, f"failed_examples_val_{config['name'].lower()}.png"))
    
    # Aplicação do Grad-CAM (ponto extra)
    if args.use_gradcam:
        print("\n" + "="*50)
        print("Aplicando Grad-CAM")
        print("="*50)
        
        # Usa o último modelo (fine-tuning) para Grad-CAM
        model = model  # O último modelo treinado no loop (fine-tuning)
        
        # Identifica a última camada convolucional dos modelos
        if args.pretrained_model == "densenet121":
            target_layer = model.model.features.denseblock4.denselayer16.conv2
        elif args.pretrained_model == "resnet50":
            target_layer = model.model.layer4[-1].conv3
        elif args.pretrained_model == "mobilenet_v2":
            target_layer = model.model.features[-1]
        
        # Aplica Grad-CAM em algumas imagens de teste
        images, labels = next(iter(test_loader))
        
        for i in range(min(5, len(images))):
            image = images[i]
            label = labels[i].item()
            
            # Aplica Grad-CAM
            gradcam_result = apply_gradcam(model, image, target_layer, device, label)
            
            # Salva a imagem com o mapa de calor
            plt.figure(figsize=(10, 5))
            plt.imshow(gradcam_result)
            plt.title(f"Classe: {classes[label]}")
            plt.axis('off')
            plt.savefig(os.path.join(run_dir, f"gradcam_sample_{i}.png"))
    
    # Encontrar o melhor modelo baseado na acurácia de validação
    best_model_name = None
    best_val_accuracy = 0.0
    
    for model_name, metrics in all_metrics.items():
        val_accuracy = metrics['val']['accuracy']
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_name = model_name
    
    print(f"\nMelhor modelo baseado na acurácia de validação: {best_model_name} com {best_val_accuracy:.4f}")
    
    # Carregar o melhor modelo para avaliação no conjunto de teste
    best_model_path = os.path.join(run_dir, f"model_{best_model_name.lower()}.pt")
    
    # Identifica qual configuração de modelo é a melhor
    best_config = None
    for config in model_configs:
        if config["name"] == best_model_name:
            best_config = config
            break
    
    # Recria o melhor modelo e carrega os pesos
    best_model = best_config["model_fn"](**best_config["model_args"]).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    
    # Avaliar o melhor modelo no conjunto de teste (só fazemos isso uma vez)
    print("\n" + "="*50)
    print(f"Avaliando o melhor modelo ({best_model_name}) no conjunto de teste")
    print("="*50)
    
    test_metrics = evaluate_model(best_model, test_loader, device, classes)
    all_metrics[best_model_name]['test'] = test_metrics
    
    # Plota a matriz de confusão para o conjunto de teste
    fig_cm_test = plot_confusion_matrix(test_metrics['confusion_matrix'], classes)
    fig_cm_test.savefig(os.path.join(run_dir, f"cm_test_{best_model_name.lower()}.png"))
    
    # Analisa falhas do modelo no conjunto de teste
    failed_test_images, failed_test_labels, failed_test_preds = get_failed_examples(
        best_model, test_loader, device
    )
    
    if failed_test_images:
        fig_failed_test = plot_failed_examples(
            failed_test_images, failed_test_labels, failed_test_preds, classes
        )
        fig_failed_test.savefig(os.path.join(run_dir, f"failed_examples_test_{best_model_name.lower()}.png"))
    
    # Criação da tabela de métricas comparativas apenas para conjuntos de validação
    val_table_data = create_metrics_table(
        all_metrics["Custom"]["val"], 
        all_metrics["Frozen"]["val"], 
        all_metrics["FineTuning"]["val"]
    )
    
    # Salva os resultados em formato de tabela para o relatório
    with open(os.path.join(run_dir, "metrics_table.txt"), 'w') as f:
        # Tabela de métricas para o conjunto de validação
        f.write("# Tabela de Métricas Comparativas - Conjunto de Validação\n\n")
        f.write("| Modelo | Acurácia | Precisão | Revocação | F1-Score |\n")
        f.write("|--------|----------|----------|-----------|----------|\n")
        
        for i in range(len(val_table_data['Modelo'])):
            f.write(f"| {val_table_data['Modelo'][i]} | {val_table_data['Acurácia'][i]:.4f} | {val_table_data['Precisão'][i]:.4f} | {val_table_data['Revocação'][i]:.4f} | {val_table_data['F1-Score'][i]:.4f} |\n")
        
        f.write("\n\n")
        
        # Tabela de métricas para o conjunto de teste (apenas para o melhor modelo)
        f.write(f"# Métricas do Melhor Modelo ({best_model_name}) no Conjunto de Teste\n\n")
        f.write("| Acurácia | Precisão | Revocação | F1-Score |\n")
        f.write("|----------|----------|-----------|----------|\n")
        f.write(f"| {test_metrics['accuracy']:.4f} | {test_metrics['precision']:.4f} | {test_metrics['recall']:.4f} | {test_metrics['f1_score']:.4f} |\n")
    
    # Salva os resultados para cada classe
    with open(os.path.join(run_dir, "class_metrics.txt"), 'w') as f:
        f.write("# Métricas por Classe\n\n")
        
        # Para o conjunto de validação com todos os modelos
        f.write("# Conjunto de Validação\n\n")
        
        # Para cada modelo
        for model_name, description in zip(
            ["Custom", "Frozen", "FineTuning"],
            ["Modelo Customizado", "Modelo com Parte Convolucional Congelada", "Modelo com Fine-tuning Completo"]
        ):
            metrics = all_metrics[model_name]['val']
            f.write(f"## {description}\n\n")
            f.write("| Classe | Precisão | Revocação | F1-Score |\n")
            f.write("|--------|----------|-----------|----------|\n")
            
            for cls, cls_metrics in metrics['class_metrics'].items():
                f.write(f"| {cls} | {cls_metrics['precision']:.4f} | {cls_metrics['recall']:.4f} | {cls_metrics['f1_score']:.4f} |\n")
            
            f.write("\n")
        
        # Para o conjunto de teste apenas com o melhor modelo
        f.write(f"# Conjunto de Teste (Melhor Modelo: {best_model_name})\n\n")
        metrics = all_metrics[best_model_name]['test']
        f.write("| Classe | Precisão | Revocação | F1-Score |\n")
        f.write("|--------|----------|-----------|----------|\n")
        
        for cls, cls_metrics in metrics['class_metrics'].items():
            f.write(f"| {cls} | {cls_metrics['precision']:.4f} | {cls_metrics['recall']:.4f} | {cls_metrics['f1_score']:.4f} |\n")
        
        f.write("\n")
    
    print("\n" + "="*50)
    print(f"Treinamento completo! Resultados salvos em: {run_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
