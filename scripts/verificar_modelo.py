"""Comprueba que un modelo exportado carga y predice igual en otro entorno (por ejemplo, Python 3.12 para el despliegue).

Uso (desde la raíz del repositorio):
    python scripts/verificar_modelo.py models/model.joblib ENTRADA.parquet               # guarda ENTRADA.esperado.npy (referencia)
    python scripts/verificar_modelo.py models/model.joblib ENTRADA.parquet ESPERADO.npy  # compara; sale con código 1 si difieren

`ENTRADA.parquet` son filas con las columnas que espera el modelo (las de `add_features`), por ejemplo pedidos del entrenamiento;
nunca la prueba. No importa nada de `src/`: el modelo solo necesita scikit-learn y LightGBM, como en la app desplegada.
"""
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main(argv: list) -> int:
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    model_path, input_path = Path(argv[0]), Path(argv[1])
    print(f"Python {platform.python_version()} · modelo {model_path.name}")
    model = joblib.load(model_path)
    predictions = np.asarray(model.predict(pd.read_parquet(input_path)), dtype=float)
    if len(argv) == 2:
        reference = input_path.with_suffix(".esperado.npy")
        np.save(reference, predictions)
        print(f"referencia guardada: {reference} ({len(predictions)} plazos)")
        return 0
    expected = np.load(argv[2])
    same = predictions.shape == expected.shape and np.allclose(predictions, expected, rtol=0, atol=1e-9)
    print(f"{len(predictions)} plazos · iguales a la referencia: {same} · diferencia máxima: {np.max(np.abs(predictions - expected)):.2e}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
