from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from config import (
    EMA_ALPHA,
    LIMIAR_BIQUINHO,
    LIMIAR_BOCA_ABERTA,
    LIMIAR_DESCONFIADO,
    LIMIAR_HANG_LOOSE,
    LIMIAR_SORRISO,
    LIMIAR_ZEN,
    OMBRO_D,
    OMBRO_E,
    PUNHO_D,
    PUNHO_E,
)


@dataclass(frozen=True)
class ResultadoClassificacao:
    humor: str | None
    confianca: float = 0.0


def distancia(a: Any, b: Any) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def obter_blendshapes(resultados_rosto: Any) -> dict[str, float]:
    blendshapes = getattr(resultados_rosto, "face_blendshapes", None)
    if not blendshapes:
        return {}
    itens = blendshapes[0]
    return {
        ponto.category_name: ponto.score
        for ponto in itens
        if getattr(ponto, "category_name", None) is not None
        and getattr(ponto, "score", None) is not None
    }


def _pontuacao(blendshapes: dict[str, float], nome: str) -> float:
    return float(blendshapes.get(nome, 0.0))


def classificar_corpo(resultados_pose: Any) -> ResultadoClassificacao:
    if not getattr(resultados_pose, "pose_landmarks", None):
        return ResultadoClassificacao(None, 0.0)
    pontos = resultados_pose.pose_landmarks[0]
    mao_esquerda_levantada = (
        pontos[PUNHO_E].visibility > 0.5
        and pontos[OMBRO_E].visibility > 0.5
        and pontos[PUNHO_E].y < pontos[OMBRO_E].y
    )
    mao_direita_levantada = (
        pontos[PUNHO_D].visibility > 0.5
        and pontos[OMBRO_D].visibility > 0.5
        and pontos[PUNHO_D].y < pontos[OMBRO_D].y
    )
    if mao_esquerda_levantada and mao_direita_levantada:
        return ResultadoClassificacao("maos_levantadas", 1.0)
    return ResultadoClassificacao(None, 0.0)


def detectar_hangloose(resultados_maos: Any) -> ResultadoClassificacao:
    if not getattr(resultados_maos, "hand_landmarks", None):
        return ResultadoClassificacao(None, 0.0)
    melhor = 0.0
    for pontos in resultados_maos.hand_landmarks:
        if not pontos or len(pontos) < 21:
            continue
        wrist = pontos[0]
        base = pontos[9]
        tamanho_mao = distancia(wrist, base)
        if tamanho_mao <= 1e-6:
            continue
        polegar_estendido = (
            distancia(pontos[0], pontos[4]) > distancia(pontos[0], pontos[2]) * 1.15
        )
        dedos_centrais_dobrados = all(
            distancia(pontos[0], pontos[ponta])
            < distancia(pontos[0], pontos[articulacao])
            for ponta, articulacao in ((8, 6), (12, 10), (16, 14))
        )
        mindinho_estendido = (
            distancia(pontos[0], pontos[20])
            > distancia(pontos[0], pontos[18]) * 1.05
        )
        score = 0.0
        if polegar_estendido:
            score += 0.35
        if mindinho_estendido:
            score += 0.35
        if dedos_centrais_dobrados:
            score += 0.30
        score /= 1.0
        melhor = max(melhor, score)
    if melhor >= LIMIAR_HANG_LOOSE:
        return ResultadoClassificacao("dahora", melhor)
    return ResultadoClassificacao(None, melhor)


def classificar_rosto(resultados_rosto: Any) -> ResultadoClassificacao:
    blendshapes = obter_blendshapes(resultados_rosto)
    if not blendshapes:
        return ResultadoClassificacao(None, 0.0)

    boca_aberta = _pontuacao(blendshapes, "jawOpen")
    if boca_aberta >= LIMIAR_BOCA_ABERTA:
        return ResultadoClassificacao("boca_aberta", boca_aberta)

    candidatos = {
        "zen": (_pontuacao(blendshapes, "eyeBlinkLeft") + _pontuacao(blendshapes, "eyeBlinkRight")) / 2,
        "boca_aberta": boca_aberta,
        "biquinho": max(
            _pontuacao(blendshapes, "mouthPucker"),
            _pontuacao(blendshapes, "mouthFunnel"),
        ),
        "sorridente": (
            _pontuacao(blendshapes, "mouthSmileLeft")
            + _pontuacao(blendshapes, "mouthSmileRight")
        ) / 2,
        "desconfiado": max(
            abs(
                _pontuacao(blendshapes, "browOuterUpLeft")
                - _pontuacao(blendshapes, "browOuterUpRight")
            ),
            abs(
                _pontuacao(blendshapes, "eyeBlinkLeft")
                - _pontuacao(blendshapes, "eyeBlinkRight")
            ),
        ),
    }
    limiares = {
        "zen": LIMIAR_ZEN,
        "boca_aberta": LIMIAR_BOCA_ABERTA,
        "biquinho": LIMIAR_BIQUINHO,
        "sorridente": LIMIAR_SORRISO,
        "desconfiado": LIMIAR_DESCONFIADO,
    }
    nome, confianca = max(candidatos.items(), key=lambda item: item[1])
    if confianca < limiares.get(nome, 0.0):
        return ResultadoClassificacao(None, confianca)
    return ResultadoClassificacao(nome, confianca)


def classificar(resultados_rosto: Any, resultados_pose: Any, resultados_maos: Any) -> str | None:
    rostos = getattr(resultados_rosto, "face_landmarks", [])
    if len(rostos) >= 2:
        return "duas_pessoas"

    hang = detectar_hangloose(resultados_maos)
    if hang.humor == "dahora":
        return "dahora"

    corpo = classificar_corpo(resultados_pose)
    if corpo.humor is not None:
        return corpo.humor

    rosto = classificar_rosto(resultados_rosto)
    return rosto.humor


def registrar_voto(historico: deque[str], humor: str | None) -> deque[str]:
    if humor is not None:
        historico.append(humor)
    return historico


def aplicar_ema(ema_humores: dict[str, float], humor: str | None) -> dict[str, float]:
    if humor is None:
        return ema_humores
    if humor not in ema_humores:
        ema_humores[humor] = 0.0
    for nome in list(ema_humores.keys()):
        if nome != humor:
            ema_humores[nome] *= 1.0 - EMA_ALPHA
    ema_humores[humor] = (1.0 - EMA_ALPHA) * ema_humores[humor] + EMA_ALPHA
    return ema_humores


def escolher_humor_ema(ema_humores: dict[str, float]) -> str | None:
    if not ema_humores:
        return None
    return max(ema_humores.items(), key=lambda item: item[1])[0]


__all__ = [
    "ResultadoClassificacao",
    "classificar",
    "classificar_corpo",
    "classificar_rosto",
    "detectar_hangloose",
    "registrar_voto",
    "aplicar_ema",
    "escolher_humor_ema",
]
