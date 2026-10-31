from __future__ import annotations

from collections import deque

from deteccao import classificar, classificar_rosto, detectar_hangloose, registrar_voto


class FakePoint:
    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


class FakeFaceBlendshape:
    def __init__(self, category_name: str, score: float):
        self.category_name = category_name
        self.score = score


class FakeFaceResults:
    def __init__(self, face_landmarks=None, face_blendshapes=None):
        self.face_landmarks = face_landmarks or []
        self.face_blendshapes = face_blendshapes or []


class FakePoseResults:
    def __init__(self, pose_landmarks=None):
        self.pose_landmarks = pose_landmarks or []


class FakeHandResults:
    def __init__(self, hand_landmarks=None):
        self.hand_landmarks = hand_landmarks or []


def criar_mao_hang_loose():
    pontos = [FakePoint(0, 0, 1.0) for _ in range(21)]
    pontos[0] = FakePoint(0.0, 0.0, 1.0)
    pontos[2] = FakePoint(0.15, 0.16, 1.0)
    pontos[4] = FakePoint(0.55, 0.10, 1.0)
    pontos[6] = FakePoint(0.18, 0.18, 1.0)
    pontos[8] = FakePoint(0.18, 0.20, 1.0)
    pontos[9] = FakePoint(0.00, 0.35, 1.0)
    pontos[10] = FakePoint(0.18, 0.33, 1.0)
    pontos[12] = FakePoint(0.18, 0.36, 1.0)
    pontos[14] = FakePoint(0.18, 0.38, 1.0)
    pontos[16] = FakePoint(0.18, 0.40, 1.0)
    pontos[18] = FakePoint(0.20, 0.47, 1.0)
    pontos[20] = FakePoint(0.55, 0.70, 1.0)
    return [pontos]


def test_hang_loose_retorna_dahora():
    resultados_maos = FakeHandResults(hand_landmarks=criar_mao_hang_loose())
    assert detectar_hangloose(resultados_maos).humor == "dahora"


def test_olhos_fechados_retorna_zen():
    resultados_rosto = FakeFaceResults(
        face_blendshapes=[
            [
                FakeFaceBlendshape("eyeBlinkLeft", 0.9),
                FakeFaceBlendshape("eyeBlinkRight", 0.8),
                FakeFaceBlendshape("jawOpen", 0.1),
                FakeFaceBlendshape("mouthPucker", 0.0),
                FakeFaceBlendshape("mouthSmileLeft", 0.1),
                FakeFaceBlendshape("mouthSmileRight", 0.1),
                FakeFaceBlendshape("browOuterUpLeft", 0.1),
                FakeFaceBlendshape("browOuterUpRight", 0.2),
            ]
        ]
    )
    assert classificar_rosto(resultados_rosto).humor == "zen"


def test_boca_neutra_nao_retorna_biquinho():
    resultados_rosto = FakeFaceResults(
        face_blendshapes=[
            [
                FakeFaceBlendshape("mouthPucker", 0.42),
                FakeFaceBlendshape("mouthFunnel", 0.44),
            ]
        ]
    )
    assert classificar_rosto(resultados_rosto).humor is None


def test_biquinho_evidente_retorna_biquinho():
    resultados_rosto = FakeFaceResults(
        face_blendshapes=[
            [
                FakeFaceBlendshape("mouthPucker", 0.82),
                FakeFaceBlendshape("mouthFunnel", 0.20),
            ]
        ]
    )
    assert classificar_rosto(resultados_rosto).humor == "biquinho"


def test_boca_aberta_tem_prioridade_sobre_sorriso():
    resultados_rosto = FakeFaceResults(
        face_blendshapes=[
            [
                FakeFaceBlendshape("jawOpen", 0.70),
                FakeFaceBlendshape("mouthSmileLeft", 0.95),
                FakeFaceBlendshape("mouthSmileRight", 0.90),
            ]
        ]
    )
    assert classificar_rosto(resultados_rosto).humor == "boca_aberta"


def test_dois_rostos_retorna_duas_pessoas():
    resultados_rosto = FakeFaceResults(face_landmarks=[object(), object()])
    resultados_pose = FakePoseResults()
    resultados_maos = FakeHandResults()
    assert classificar(resultados_rosto, resultados_pose, resultados_maos) == "duas_pessoas"


def test_none_nao_altera_votacao():
    historico = deque(["zen", "zen"])
    resultado = registrar_voto(historico, None)
    assert list(resultado) == ["zen", "zen"]
