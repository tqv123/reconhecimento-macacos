from __future__ import annotations

import tkinter as tk
from typing import Any

import cv2
import numpy as np

from config import DEFAULT_WINDOW_SIZE, OMBRO_D, OMBRO_E, COTOVELO_D, COTOVELO_E, PUNHO_D, PUNHO_E


def obter_tamanho_tela() -> tuple[int, int]:
    try:
        root = tk.Tk()
        root.withdraw()
        largura = root.winfo_screenwidth()
        altura = root.winfo_screenheight()
        root.update()
        root.destroy()
        if largura and altura:
            return largura, altura
    except Exception:
        pass
    return DEFAULT_WINDOW_SIZE


def ajustar_imagem(imagem: np.ndarray, largura: int, altura: int) -> np.ndarray:
    if imagem is None:
        return np.zeros((altura, largura, 3), dtype=np.uint8)
    escala = min(largura / imagem.shape[1], altura / imagem.shape[0])
    nova_largura = max(1, int(imagem.shape[1] * escala))
    nova_altura = max(1, int(imagem.shape[0] * escala))
    interpolacao = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    redimensionada = cv2.resize(
        imagem,
        (nova_largura, nova_altura),
        interpolation=interpolacao,
    )
    x = (largura - nova_largura) // 2
    y = (altura - nova_altura) // 2
    return cv2.copyMakeBorder(
        redimensionada,
        y,
        altura - nova_altura - y,
        x,
        largura - nova_largura - x,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def montar_tela(quadro: np.ndarray, imagem_expressao: np.ndarray | None, largura: int, altura: int) -> np.ndarray:
    largura_painel = largura // 2
    painel_expressao = (
        ajustar_imagem(imagem_expressao, largura - largura_painel, altura)
        if imagem_expressao is not None
        else np.zeros((altura, largura - largura_painel, 3), dtype=np.uint8)
    )
    return np.hstack((ajustar_imagem(quadro, largura_painel, altura), painel_expressao))


def desenhar_esqueleto(quadro: np.ndarray, resultados_pose: Any) -> None:
    if not getattr(resultados_pose, "pose_landmarks", None):
        return
    overlay = quadro.copy()
    bracos = (
        ((OMBRO_E, COTOVELO_E, PUNHO_E), (0, 220, 0)),
        ((OMBRO_D, COTOVELO_D, PUNHO_D), (180, 0, 180)),
    )
    for pontos in resultados_pose.pose_landmarks:
        for (ombro, cotovelo, punho), cor in bracos:
            indices = (ombro, cotovelo, punho)
            if all(pontos[indice].visibility > 0.5 for indice in indices):
                coordenadas = [
                    (
                        int(pontos[indice].x * quadro.shape[1]),
                        int(pontos[indice].y * quadro.shape[0]),
                    )
                    for indice in indices
                ]
                cv2.line(overlay, coordenadas[0], coordenadas[1], cor, 2, cv2.LINE_AA)
                cv2.line(overlay, coordenadas[1], coordenadas[2], cor, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, quadro, 0.2, 0, quadro)


def desenhar_valores_blendshapes(quadro: np.ndarray, blendshapes: list[Any]) -> None:
    if not blendshapes:
        return
    textos = [
        f"{item.category_name}: {item.score:.3f}"
        for item in blendshapes
        if getattr(item, "category_name", None) is not None
        and getattr(item, "score", None) is not None
    ]
    if not textos:
        return
    for i, texto in enumerate(textos[:12]):
        cv2.putText(
            quadro,
            texto,
            (20, 40 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


__all__ = [
    "obter_tamanho_tela",
    "ajustar_imagem",
    "montar_tela",
    "desenhar_esqueleto",
    "desenhar_valores_blendshapes",
]
