import csv

from main import FaceCsvLogger


def test_face_csv_logger_grava_scores_e_classificacao(tmp_path):
    caminho = tmp_path / "face.csv"
    logger = FaceCsvLogger(caminho)
    logger.registrar(12, "biquinho", {"mouthPucker": 0.82, "mouthFunnel": 0.20})
    logger.close()

    with caminho.open(newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    assert len(linhas) == 1
    assert linhas[0]["frame"] == "12"
    assert linhas[0]["humor_rosto"] == "biquinho"
    assert linhas[0]["mouthPucker"] == "0.82"
    assert linhas[0]["mouthFunnel"] == "0.2"
    assert linhas[0]["timestamp_local"]