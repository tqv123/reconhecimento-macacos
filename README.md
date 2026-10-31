# Reconhecimento de expressões

Projeto em Python para detectar expressões faciais e gestos usando OpenCV e MediaPipe.

## Funcionalidades

- Detecção de expressões faciais como sorriso, olhos fechados, boca aberta, biquinho e descrença.
- Detecção de gestos com as mãos, incluindo o gesto de "hang loose".
- Exibição em tempo real da emoção detectada ao lado do vídeo da câmera.

## Requisitos

- Python 3.14
- Webcam

## Instalação

1. Abra o terminal na pasta do projeto.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução

No Windows, use o launcher do Python para rodar a versão específica do projeto:

```bash
py -3.14 reconhecimento.py
```

Se o Python 3.14 não estiver no PATH do sistema, confirme que ele está instalado e tente novamente no terminal:

```bash
py -3.14 --version
```

## Controles

- `Q`: fecha a aplicação
- `T`: alterna entre tela cheia e janela normal

## Observações

Os modelos do MediaPipe e as imagens usadas pelo projeto já estão incluídos na estrutura do repositório.
