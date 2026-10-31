from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
PASTA_IMAGENS = ROOT / "macacos"
PASTA_MODELOS = ROOT / "modelos"
NOME_JANELA = "Reconhecimento de expressoes"
DEFAULT_WINDOW_SIZE = (1280, 720)

ARQUIVOS_IMAGENS = {
    "boca_aberta": "macaco boca aberta.jfif",
    "dahora": "macaco dahora.jfif",
    "desconfiado": "macaco desconfiado.jfif",
    "biquinho": "macaco do biquinho.jfif",
    "maos_levantadas": "macaco maos levantadas.jfif",
    "zen": "macaco olhos fechados.jfif",
    "sorridente": "macaco sorridente.jfif",
}

ARQUIVO_MODELO_ROSTO = "face_landmarker.task"
ARQUIVO_MODELO_POSE = "pose_landmarker_lite.task"
ARQUIVO_MODELO_MAOS = "hand_landmarker.task"

JANELA_SUAVIZACAO = 15
EMA_ALPHA = 0.35
LIMIAR_ZEN = 0.55
LIMIAR_BOCA_ABERTA = 0.50
LIMIAR_BIQUINHO = 0.70
LIMIAR_SORRISO = 0.45
LIMIAR_DESCONFIADO = 0.30
LIMIAR_HANG_LOOSE = 0.70
LIMIAR_FRENTE = 0.35

OMBRO_E, OMBRO_D, COTOVELO_E, COTOVELO_D, PUNHO_E, PUNHO_D = 11, 12, 13, 14, 15, 16

__all__ = [
    "ROOT",
    "PASTA_IMAGENS",
    "PASTA_MODELOS",
    "NOME_JANELA",
    "DEFAULT_WINDOW_SIZE",
    "ARQUIVOS_IMAGENS",
    "ARQUIVO_MODELO_ROSTO",
    "ARQUIVO_MODELO_POSE",
    "ARQUIVO_MODELO_MAOS",
    "JANELA_SUAVIZACAO",
    "EMA_ALPHA",
    "LIMIAR_ZEN",
    "LIMIAR_BOCA_ABERTA",
    "LIMIAR_BIQUINHO",
    "LIMIAR_SORRISO",
    "LIMIAR_DESCONFIADO",
    "LIMIAR_HANG_LOOSE",
    "LIMIAR_FRENTE",
    "OMBRO_E",
    "OMBRO_D",
    "COTOVELO_E",
    "COTOVELO_D",
    "PUNHO_E",
    "PUNHO_D",
]
