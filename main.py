from __future__ import annotations

import argparse
import csv
import logging
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

from config import (
    ARQUIVOS_IMAGENS,
    DEFAULT_WINDOW_SIZE,
    NOME_JANELA,
    PASTA_IMAGENS,
    PASTA_MODELOS,
    ROOT,
)
from deteccao import aplicar_ema, classificar, classificar_rosto, escolher_humor_ema, obter_blendshapes
from visualizacao import desenhar_esqueleto, desenhar_valores_blendshapes, montar_tela, obter_tamanho_tela


LOGGER = logging.getLogger(__name__)


class FaceCsvLogger:
    def __init__(self, path: Path):
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = None

    def registrar(self, frame: int, humor: str | None, blendshapes: dict[str, float]) -> None:
        if not blendshapes:
            return
        if self._writer is None:
            fields = ["frame", "timestamp_local", "humor_rosto", *sorted(blendshapes)]
            self._writer = csv.DictWriter(self._file, fieldnames=fields)
            self._writer.writeheader()
        self._writer.writerow(
            {
                "frame": frame,
                "timestamp_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "humor_rosto": humor or "",
                **blendshapes,
            }
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def carregar_imagens() -> dict[str, np.ndarray]:
    imagens: dict[str, np.ndarray] = {}
    for humor, arquivo in ARQUIVOS_IMAGENS.items():
        caminho = PASTA_IMAGENS / arquivo
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            raise FileNotFoundError(f"Imagem não encontrada: {caminho}")
        imagens[humor] = cv2.resize(imagem, (420, 320), interpolation=cv2.INTER_LINEAR)
    return imagens


def carregar_caminhos_modelos(model_dir: Path) -> dict[str, str]:
    caminhos = {
        "rosto": model_dir / "face_landmarker.task",
        "pose": model_dir / "pose_landmarker_lite.task",
        "maos": model_dir / "hand_landmarker.task",
    }
    for nome, caminho in caminhos.items():
        if not caminho.is_file():
            raise FileNotFoundError(f"Modelo do MediaPipe não encontrado: {caminho}")
    return {chave: str(valor) for chave, valor in caminhos.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconhecimento de expressões com MediaPipe e OpenCV.")
    parser.add_argument("--camera-index", type=int, default=0, help="Índice da câmera USB ou integrada.")
    parser.add_argument("--model-dir", type=Path, default=PASTA_MODELOS, help="Pasta com os modelos .task do MediaPipe.")
    parser.add_argument("--debug", action="store_true", help="Mostra blendshapes brutos na tela.")
    parser.add_argument("--log-face", action="store_true", help="Grava os scores faciais em CSV local, sem salvar imagens.")
    parser.add_argument("--window-width", type=int, default=DEFAULT_WINDOW_SIZE[0], help="Largura da janela principal.")
    parser.add_argument("--window-height", type=int, default=DEFAULT_WINDOW_SIZE[1], help="Altura da janela principal.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.debug or args.log_face else logging.WARNING, format="%(levelname)s:%(message)s")

    imagens = carregar_imagens()
    modelos = carregar_caminhos_modelos(args.model_dir)
    ema_humores: dict[str, float] = {}
    ultimo_humor_valido: str | None = None
    calibration_mode = False
    tela_cheia = True
    largura_tela, altura_tela = obter_tamanho_tela()

    face_logger = None
    if args.log_face:
        pasta_logs = ROOT / "logs"
        pasta_logs.mkdir(exist_ok=True)
        caminho_log = pasta_logs / f"face_blendshapes_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        face_logger = FaceCsvLogger(caminho_log)
        LOGGER.info("Gravando scores faciais em %s", caminho_log)

    cap = cv2.VideoCapture(args.camera_index)

    def encerrar() -> None:
        cap.release()
        cv2.destroyAllWindows()

    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    try:
        if not cap.isOpened():
            raise RuntimeError("Não foi possível acessar a câmera.")

        mp_visao = mp.tasks.vision
        opcoes_rosto = mp_visao.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=modelos["rosto"]),
            running_mode=mp_visao.RunningMode.VIDEO,
            num_faces=2,
            output_face_blendshapes=True,
        )
        opcoes_pose = mp_visao.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=modelos["pose"]),
            running_mode=mp_visao.RunningMode.VIDEO,
            num_poses=1,
        )
        opcoes_maos = mp_visao.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=modelos["maos"]),
            running_mode=mp_visao.RunningMode.VIDEO,
            num_hands=1,
        )

        with (
            mp_visao.FaceLandmarker.create_from_options(opcoes_rosto) as rosto,
            mp_visao.PoseLandmarker.create_from_options(opcoes_pose) as pose,
            mp_visao.HandLandmarker.create_from_options(opcoes_maos) as maos,
        ):
            cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(NOME_JANELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            ultimo_timestamp_ms = 0
            frame_index = 0

            while cap.isOpened():
                ok, quadro = cap.read()
                if not ok:
                    raise RuntimeError("Não foi possível ler um quadro da câmera.")

                quadro = cv2.flip(quadro, 1)
                rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
                imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(time.monotonic_ns() // 1_000_000, ultimo_timestamp_ms + 1)
                ultimo_timestamp_ms = timestamp_ms

                resultados_rosto = rosto.detect_for_video(imagem, timestamp_ms)
                resultados_pose = pose.detect_for_video(imagem, timestamp_ms)
                resultados_maos = maos.detect_for_video(imagem, timestamp_ms)

                if face_logger is not None:
                    blendshapes = obter_blendshapes(resultados_rosto)
                    if blendshapes:
                        face_logger.registrar(
                            frame_index,
                            classificar_rosto(resultados_rosto).humor,
                            blendshapes,
                        )
                frame_index += 1

                desenhar_esqueleto(quadro, resultados_pose)
                humor = classificar(resultados_rosto, resultados_pose, resultados_maos)
                if humor == "duas_pessoas" and ultimo_humor_valido is not None:
                    humor = ultimo_humor_valido
                if humor is not None:
                    ultimo_humor_valido = humor
                    ema_humores = aplicar_ema(ema_humores, humor)
                    humor_exibido = escolher_humor_ema(ema_humores) or humor
                else:
                    humor_exibido = ultimo_humor_valido

                imagem_expressao = imagens.get(humor_exibido) if humor_exibido is not None else None
                if humor_exibido is not None:
                    cv2.putText(
                        quadro,
                        f"Expressao: {humor_exibido.replace('_', ' ')}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        3,
                    )

                if calibration_mode and getattr(resultados_rosto, "face_blendshapes", None):
                    blendshapes = resultados_rosto.face_blendshapes[0]
                    desenhar_valores_blendshapes(quadro, blendshapes)

                if tela_cheia:
                    largura_tela_atual, altura_tela_atual = largura_tela, altura_tela
                else:
                    largura_tela_atual = min(args.window_width, int(largura_tela * 0.9))
                    altura_tela_atual = min(args.window_height, int(altura_tela * 0.9))

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
                    cv2.setWindowProperty(
                        NOME_JANELA,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if tela_cheia else cv2.WINDOW_NORMAL,
                    )
                    if not tela_cheia:
                        cv2.resizeWindow(NOME_JANELA, min(args.window_width, 1280), min(args.window_height, 720))
                        cv2.moveWindow(NOME_JANELA, (largura_tela - min(args.window_width, 1280)) // 2, (altura_tela - min(args.window_height, 720)) // 2)
                elif tecla in (ord("c"), ord("C")):
                    calibration_mode = not calibration_mode
                elif tecla == ord("q"):
                    break
    except KeyboardInterrupt:
        LOGGER.info("Encerrando pelo Ctrl+C.")
    finally:
        encerrar()
        if face_logger is not None:
            face_logger.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
