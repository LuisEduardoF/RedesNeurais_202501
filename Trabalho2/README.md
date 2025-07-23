# Treinamento de Redes Convolucionais - Trabalho 2

Este repositório contém a implementação do segundo trabalho da disciplina de Redes Neurais, que consiste no treinamento de redes neurais convolucionais para classificação de imagens de raio-x de tórax em três categorias: NORMAL, COVID-19 e PNEUMONIA.

## Estrutura do Projeto

```
Trabalho2/
├── utils/
│   ├── data_utils.py     # Funções para manipulação de dados
│   ├── models.py         # Definições das arquiteturas de redes neurais
│   ├── training.py       # Funções para treinamento e avaliação dos modelos
│   └── visualization.py  # Funções para visualização dos resultados
├── main.py               # Script principal para execução local
├── colab_notebook.py     # Script para execução no Google Colab
└── README.md             # Este arquivo
```

## Requisitos

- Python 3.7+
- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib
- seaborn
- tqdm
- opencv-python
- kagglehub (para download do dataset)

## Dataset

O trabalho utiliza o dataset [Chest X-ray Image (COVID19, PNEUMONIA, and NORMAL)](https://www.kaggle.com/datasets/alsaniipe/chest-x-ray-image) disponível no Kaggle. O download do dataset é feito automaticamente pelo script usando a biblioteca `kagglehub`.

## Execução

### Execução Local

Para executar o treinamento localmente:

```bash
python main.py --data_dir "dataset" --batch_size 32 --img_size 224 --epochs 20 --lr 0.001 --val_split 0.2 --pretrained_model "densenet121" --use_gradcam --output_dir "results"
```

Argumentos disponíveis:

- `--data_dir`: Diretório onde os dados serão salvos
- `--batch_size`: Tamanho do batch para treino e avaliação
- `--img_size`: Tamanho das imagens após o redimensionamento
- `--epochs`: Número de épocas de treinamento
- `--lr`: Taxa de aprendizado
- `--val_split`: Proporção do conjunto de treino a ser usada para validação
- `--pretrained_model`: Modelo pré-treinado a ser usado (densenet121, resnet50 ou mobilenet_v2)
- `--use_gradcam`: Usa Grad-CAM para visualizar as regiões de interesse
- `--output_dir`: Diretório para salvar os resultados

### Execução no Google Colab

Para executar no Google Colab, faça o upload do arquivo `colab_notebook.py` e execute-o célula por célula ou converta-o para um notebook Jupyter.

## Modelos Implementados

O trabalho implementa três modelos:

1. **Modelo Customizado**: Uma CNN desenvolvida manualmente.
2. **Modelo Pré-treinado com Parte Convolucional Congelada**: Um modelo pré-treinado (DenseNet121, ResNet50 ou MobileNetV2) com as camadas convolucionais congeladas.
3. **Modelo Pré-treinado com Fine-tuning Completo**: O mesmo modelo pré-treinado, mas com todas as camadas treináveis.

## Resultados

Os resultados são salvos no diretório especificado em `--output_dir` (padrão: "results"). Para cada execução, são salvos:

- Modelos treinados (`.pt`)
- Histórico de treinamento (gráficos de perda e acurácia)
- Matrizes de confusão
- Exemplos de falhas de classificação
- Imagens com Grad-CAM (se `--use_gradcam` for especificado)
- Tabelas com métricas comparativas (acurácia, precisão, revocação e F1-score)

## Ponto Extra

O trabalho inclui a implementação do método de explainability Grad-CAM para visualizar as regiões das imagens que mais influenciaram a decisão da rede neural.
