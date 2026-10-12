# 🎬 Reconhecimento e Detecção Facial do Elenco de Friends com YOLOv11

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v11s-blueviolet?logo=yolo)](https://docs.ultralytics.com/)
[![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-purple?logo=roboflow)](https://roboflow.com/)
[![Google Colab](https://img.shields.io/badge/Colab-GPU%20T4-orange?logo=googlecolab)](https://colab.research.google.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Projeto de Visão Computacional focado em **detecção e classificação simultânea (rede única)** das faces dos seis atores principais da série *Friends*: **Jennifer Aniston, Courteney Cox, Lisa Kudrow, Matt LeBlanc, Matthew Perry e David Schwimmer**.

Diferente do pipeline clássico de dois estágios (*Haar Cascade + Classificador CNN*), que sofre com rostos inclinados, sombras e cenas coletivas, esta solução utiliza **YOLOv11 (Small)** via Transfer Learning, permitindo identificar com precisão múltiplos atores simultaneamente em fotos individuais e em grupo.

---

## 📌 Sumário
- [Visão Geral](#-visão-geral)
- [Classes Reconhecidas](#-classes-reconhecidas)
- [Pipeline de Desenvolvimento](#-pipeline-de-desenvolvimento)
- [Desempenho e Métricas](#-desempenho-e-métricas)
- [Instalação e Uso](#-instalação-e-uso)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Licença](#-licença)

---

## 🎯 Visão Geral

- **Superação de Limitações Clássicas:** Substitui detectores baseados em Haar Cascade por uma arquitetura moderna capaz de processar variações de pose, expressões faciais e semi-perfil.
- **Detecção em Grupo (*Multi-Face*):** Identifica e classifica as coordenadas de todos os atores na mesma imagem sem necessidade de recortes manuais ou normalizações intermediárias.
- **Treinamento Otimizado:** Fine-tuning de 80 épocas com otimizador `AdamW`, aumento de dados (*Data Augmentation*) no Roboflow e resolução de entrada calibrada.

---

## 🏷️ Classes Reconhecidas

| ID | Classe | Personagem Correspondente |
|:--:|:-------|:--------------------------|
| `0` | `jennifer_aniston` | Rachel Green |
| `1` | `courteney_cox` | Monica Geller |
| `2` | `lisa_kudrow` | Phoebe Buffay |
| `3` | `matt_leblanc` | Joey Tribbiani |
| `4` | `mathew_perry` | Chandler Bing |
| `5` | `david_schwimmer` | Ross Geller |

---

## 🔄 Pipeline de Desenvolvimento

```text
  [ Dataset no Roboflow ]
          │ (Auto-orient, Resize 640x640 Letterbox, Flip, Brilho)
          ▼
  [ Treinamento YOLOv11s no Google Colab (GPU T4) ]
          │ (80 épocas, AdamW, patience=15)
          ▼
  [ Pesos Otimizados: best.pt ]
          │
          ├──▶ Validação (mAP@0.5: ~93.7%, Matriz de Confusão)
          └──▶ Inferência Multi-Face (fotos individuais e de grupo)
```
📊 Desempenho e Métricas
O modelo foi validado sobre o conjunto de testes gerando os seguintes resultados:

mAP@0.5 Global: 93.7% em todas as classes.

Recall Geral: ~99% no limiar operacional.

Resultados por Classe (mAP@0.5):

Jennifer Aniston: 97.7%

Lisa Kudrow: 96.5%

David Schwimmer: 94.6%

Matt LeBlanc: 93.2%

Matthew Perry: 93.1%

Courteney Cox: 87.3%

⚙️ Instalação e Uso
1. Clonar o Repositóri

```
Bash
git clone [https://github.com/SEU_USUARIO/friends-face-recognition-yolo.git](https://github.com/SEU_USUARIO/friends-face-recognition-yolo.git)
cd friends-face-recognition-yolo
pip install -r requirements.txt

```
3. Inferência em Imagem
```
Python
from ultralytics import YOLO

```
# Carrega os pesos treinados
```
model = YOLO("weights/best.pt")
```
# Executa a predição na foto do grupo
```
results = model.predict(
    source="assets/friends.jpeg",
    conf=0.40,
    iou=0.45,
    imgsz=1024,
    show=True
)
```
📂 Estrutura do Repositório
```
notebooks/: Notebook completo com download do dataset, treino e rotinas de inferência no Google Colab.

src/: Scripts Python modulares para teste e predição.

weights/: Orientações para download dos pesos treinados best.pt.

assets/: Imagens de evidência, matriz de confusão e curvas de precisão/recall.
```

📄 Licença
Projeto desenvolvido sob a licença MIT.
