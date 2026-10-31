# Reconhecimento de expressões

Projeto em Python para detectar expressões faciais e gestos em tempo real com OpenCV e MediaPipe.

## Estrutura do projeto

- `config.py`: constantes, limiares e mapeamento de humor para imagem.
- `deteccao.py`: lógica puramente detectiva e de classificação.
- `visualizacao.py`: renderização da janela e ajustes de imagem.
- `main.py`: fluxo principal do aplicativo.
- `reconhecimento.py`: ponto de entrada compatível com o comando antigo.
- `macacos/`: imagens dos macacos usadas como resposta visual.
- `modelos/`: arquivos `.task` do MediaPipe.
- `tests/`: testes automatizados com `pytest`.

> Observação: os nomes reais dos arquivos no repositório incluem espaços, por isso a configuração usa os nomes exatos dos ficheiros existentes. O arquivo `macaco parado.jfif` foi deixado fora da configuração, porque não representa um humor do sistema e não é usado pela classificação.

## Funcionalidades

- Detecção de expressões faciais: sorriso, zen, boca aberta, biquinho e desconfiança.
- Detecção de gestos com as mãos, com destaque para o gesto de "hang loose".
- Exibição de imagem correspondente ao humor ao lado do vídeo da câmera.
- Modo de calibração por tecla `C` para visualizar blendshapes brutos.

## Requisitos

- Python 3.14
- Webcam
- Dependências listadas em `requirements.txt`

## Instalação

1. Abra o terminal na pasta do projeto.
2. Instale as dependências:

```bash
py -3.14 -m pip install -r requirements.txt
```

## Execução

No Windows, rode o projeto com:

```bash
py -3.14 reconhecimento.py
```

Para gravar os scores faciais durante a execução, use:

```bash
py -3.14 reconhecimento.py --log-face
```

O CSV é criado em `logs/` com timestamp, classificação facial e scores de blendshape por quadro. Os dados são numéricos; nenhuma imagem ou vídeo é gravado. A pasta `logs/` fica fora do Git.

Se o Python 3.14 ainda não estiver no PATH, confirme a instalação com:

```bash
py -3.14 --version
```

## Controles

- `Q`: fecha a aplicação
- `T`: alterna entre tela cheia e janela normal
- `C`: ativa/desativa o modo de calibração e mostra os blendshapes brutos
- `--log-face`: grava os blendshapes em CSV local para investigar falsos positivos

## Modo de calibração

Ao pressionar `C`, a janela mostra os valores brutos dos blendshapes do rosto usados na classificação. Isso ajuda a ajustar limiares quando a iluminação ou a condição da webcam mudam.

## Testes

```bash
py -3.14 -m pytest
```

## Observações

- O arquivo `mario dedo do meio.png` não é usado em nenhuma parte do código e foi mantido fora da lógica do programa.
- Os modelos do MediaPipe e as imagens usadas já estão incluídos na pasta do repositório.
