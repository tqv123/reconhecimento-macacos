import ctypes
import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
PASTA_IMAGENS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "macacos")
PASTA_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")
NOME_JANELA = "Reconhecimento de expressoes"
ARQUIVOS_IMAGENS = {
    "boca_aberta": "macaco boca aberta.jfif",
    "dahora": "macaco dahora.jfif",
    "desconfiado": "macaco desconfiado.jfif",
    "biquinho": "macaco do biquinho.jfif",
    "maos_levantadas": "macaco maos levantadas.jfif",
    "olhos_fechados": "macaco olhos fechados.jfif",
    "sorridente": "macaco sorridente.jfif",
    "sinal_l": "fazer o sinal de l com a mao.jpg",
}
ARQUIVO_MODELO_ROSTO = "face_landmarker.task"
ARQUIVO_MODELO_POSE = "pose_landmarker_lite.task"
ARQUIVO_MODELO_MAOS = "hand_landmarker.task"

# Quantidade de quadros usados para estabilizar o resultado (evita oscilação)
JANELA_SUAVIZACAO = 15

# Limiares dos blendshapes do MediaPipe (valores entre 0 e 1).
LIMIAR_OLHOS_FECHADOS = 0.55
LIMIAR_BOCA_ABERTA = 0.50
LIMIAR_BIQUINHO = 0.45
LIMIAR_SORRISO = 0.45
LIMIAR_DESCONFIADO = 0.30

# Índices do MediaPipe Pose
OMBRO_E, OMBRO_D, COTOVELO_E, COTOVELO_D, PUNHO_E, PUNHO_D = 11, 12, 13, 14, 15, 16

mp_visao = mp.tasks.vision


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def carregar_imagens():
    imagens = {}
    for humor, arquivo in ARQUIVOS_IMAGENS.items():
        caminho = os.path.join(PASTA_IMAGENS, arquivo)
        img = cv2.imread(caminho)
        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada: {caminho}")
        imagens[humor] = img
    return imagens


def carregar_caminhos_modelos():
    caminhos = {
        "rosto": os.path.join(PASTA_MODELOS, ARQUIVO_MODELO_ROSTO),
        "pose": os.path.join(PASTA_MODELOS, ARQUIVO_MODELO_POSE),
        "maos": os.path.join(PASTA_MODELOS, ARQUIVO_MODELO_MAOS),
    }
    for caminho in caminhos.values():
        if not os.path.isfile(caminho):
            raise FileNotFoundError(f"Modelo do MediaPipe não encontrado: {caminho}")
    return caminhos


# ---------------------------------------------------------------------------
# Classificação do humor
# ---------------------------------------------------------------------------
def distancia(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def detectar_hangloose(resultados_maos):
    """Detecta hang loose: polegar e mindinho estendidos, demais dedos fechados."""
    for pontos in resultados_maos.hand_landmarks:
        polegar_estendido = (
            distancia(pontos[0], pontos[4])
            > distancia(pontos[0], pontos[2]) * 1.15
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
        if polegar_estendido and mindinho_estendido and dedos_centrais_dobrados:
            return True
    return False


def detectar_sinal_l(resultados_maos):
    """Detecta L com o polegar e o indicador estendidos."""
    for pontos in resultados_maos.hand_landmarks:
        polegar_estendido = (
            distancia(pontos[0], pontos[4])
            > distancia(pontos[0], pontos[2]) * 1.15
        )
        indicador_estendido = (
            distancia(pontos[0], pontos[8])
            > distancia(pontos[0], pontos[6]) * 1.15
        )
        demais_dedos_dobrados = all(
            distancia(pontos[0], pontos[ponta])
            < distancia(pontos[0], pontos[articulacao])
            for ponta, articulacao in ((12, 10), (16, 14), (20, 18))
        )
        if polegar_estendido and indicador_estendido and demais_dedos_dobrados:
            return True
    return False


def classificar_corpo(resultados_pose):
    """Detecta quando os dois braços estão levantados."""
    if resultados_pose.pose_landmarks:
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
            return "maos_levantadas"
    return None


def classificar_rosto(resultados_rosto):
    """Classifica a expressão pelos blendshapes do Face Landmarker."""
    if not resultados_rosto.face_blendshapes:
        return None

    pontos = resultados_rosto.face_blendshapes[0]
    expressoes = {
        ponto.category_name: ponto.score
        for ponto in pontos
        if ponto.category_name is not None and ponto.score is not None
    }

    def pontuacao(nome):
        return expressoes.get(nome, 0.0)

    olhos_fechados = (
        pontuacao("eyeBlinkLeft") + pontuacao("eyeBlinkRight")
    ) / 2
    sorriso = (pontuacao("mouthSmileLeft") + pontuacao("mouthSmileRight")) / 2
    assimetria_sobrancelhas = abs(
        pontuacao("browOuterUpLeft") - pontuacao("browOuterUpRight")
    )
    assimetria_olhos = abs(
        pontuacao("eyeBlinkLeft") - pontuacao("eyeBlinkRight")
    )

    if olhos_fechados > LIMIAR_OLHOS_FECHADOS:
        return "olhos_fechados"
    if pontuacao("jawOpen") > LIMIAR_BOCA_ABERTA:
        return "boca_aberta"
    if (
        pontuacao("mouthPucker") > LIMIAR_BIQUINHO
        or pontuacao("mouthFunnel") > LIMIAR_BIQUINHO
    ):
        return "biquinho"
    if sorriso > LIMIAR_SORRISO:
        return "sorridente"
    if (
        assimetria_sobrancelhas > LIMIAR_DESCONFIADO
        or assimetria_olhos > LIMIAR_DESCONFIADO
    ):
        return "desconfiado"
    return None


def classificar(resultados_rosto, resultados_pose, resultados_maos):
    if len(resultados_rosto.face_landmarks) >= 2:
        return "duas_pessoas"
    if detectar_sinal_l(resultados_maos):
        return "sinal_l"
    if detectar_hangloose(resultados_maos):
        return "dahora"
    return classificar_corpo(resultados_pose) or classificar_rosto(resultados_rosto)


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------
def desenhar_esqueleto(quadro, resultados_pose):
    if resultados_pose.pose_landmarks:
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
                    cv2.line(
                        overlay,
                        coordenadas[0],
                        coordenadas[1],
                        cor,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.line(
                        overlay,
                        coordenadas[1],
                        coordenadas[2],
                        cor,
                        2,
                        cv2.LINE_AA,
                    )
        cv2.addWeighted(overlay, 0.8, quadro, 0.2, 0, quadro)


def obter_tamanho_tela():
    user32 = ctypes.windll.user32
    largura = user32.GetSystemMetrics(0)
    altura = user32.GetSystemMetrics(1)
    if largura <= 0 or altura <= 0:
        raise RuntimeError("Não foi possível determinar a resolução da tela.")
    return largura, altura


def ajustar_imagem(imagem, largura, altura):
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


def montar_tela(quadro, imagem_expressao, largura, altura):
    largura_painel = largura // 2
    painel_expressao = (
        ajustar_imagem(imagem_expressao, largura - largura_painel, altura)
        if imagem_expressao is not None
        else np.zeros((altura, largura - largura_painel, 3), dtype=np.uint8)
    )
    return np.hstack(
        (
            ajustar_imagem(quadro, largura_painel, altura),
            painel_expressao,
        )
    )


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    imagens = carregar_imagens()
    modelos = carregar_caminhos_modelos()

    cap = cv2.VideoCapture(0)
    historico = deque(maxlen=JANELA_SUAVIZACAO)

    try:
        if not cap.isOpened():
            raise RuntimeError("Não foi possível acessar a câmera.")

        opcoes_rosto = mp_visao.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=modelos["rosto"],
            ),
            running_mode=mp_visao.RunningMode.VIDEO,
            num_faces=2,
            output_face_blendshapes=True,
        )
        opcoes_pose = mp_visao.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=modelos["pose"],
            ),
            running_mode=mp_visao.RunningMode.VIDEO,
            num_poses=1,
        )
        opcoes_maos = mp_visao.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=modelos["maos"],
            ),
            running_mode=mp_visao.RunningMode.VIDEO,
            num_hands=1,
        )

        with (
            mp_visao.FaceLandmarker.create_from_options(opcoes_rosto) as rosto,
            mp_visao.PoseLandmarker.create_from_options(opcoes_pose) as pose,
            mp_visao.HandLandmarker.create_from_options(opcoes_maos) as maos,
        ):
            largura_tela, altura_tela = obter_tamanho_tela()
            tela_cheia = True
            cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                NOME_JANELA,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            ultimo_timestamp_ms = 0
            while cap.isOpened():
                ok, quadro = cap.read()
                if not ok:
                    raise RuntimeError("Não foi possível ler um quadro da câmera.")

                quadro = cv2.flip(quadro, 1)
                rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
                imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000,
                    ultimo_timestamp_ms + 1,
                )
                ultimo_timestamp_ms = timestamp_ms
                resultados_rosto = rosto.detect_for_video(imagem, timestamp_ms)
                resultados_pose = pose.detect_for_video(imagem, timestamp_ms)
                resultados_maos = maos.detect_for_video(imagem, timestamp_ms)

                desenhar_esqueleto(quadro, resultados_pose)

                historico.append(
                    classificar(resultados_rosto, resultados_pose, resultados_maos)
                )
                humor = Counter(historico).most_common(1)[0][0]

                imagem_expressao = imagens.get(humor) if humor is not None else None
                if humor is not None:
                    cv2.putText(
                        quadro,
                        f"Expressao: {humor.replace('_', ' ')}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        3,
                    )

                if tela_cheia:
                    largura_tela_atual, altura_tela_atual = largura_tela, altura_tela
                else:
                    largura_tela_atual = min(1280, int(largura_tela * 0.9))
                    altura_tela_atual = min(720, int(altura_tela * 0.9))

                tela = montar_tela(
                    quadro,
                    imagem_expressao,
                    largura_tela_atual,
                    altura_tela_atual,
                )
                cv2.imshow(NOME_JANELA, tela)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("t"), ord("T")):
                    tela_cheia = not tela_cheia
                    modo_janela = (
                        cv2.WINDOW_FULLSCREEN if tela_cheia else cv2.WINDOW_NORMAL
                    )
                    cv2.setWindowProperty(
                        NOME_JANELA,
                        cv2.WND_PROP_FULLSCREEN,
                        modo_janela,
                    )
                    if not tela_cheia:
                        largura_tela_atual = min(1280, int(largura_tela * 0.9))
                        altura_tela_atual = min(720, int(altura_tela * 0.9))
                        cv2.resizeWindow(
                            NOME_JANELA,
                            largura_tela_atual,
                            altura_tela_atual,
                        )
                        cv2.moveWindow(
                            NOME_JANELA,
                            (largura_tela - largura_tela_atual) // 2,
                            (altura_tela - altura_tela_atual) // 2,
                        )
                elif tecla == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
