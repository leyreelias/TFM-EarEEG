"""Ejecuta los scripts del proyecto en orden, sin borrar salidas por su cuenta.

Uso desde la raiz: python .\ejecutar_pipeline.py
Ver pasos:          python .\ejecutar_pipeline.py --list
Comprobar entornos: python .\ejecutar_pipeline.py --check
Continuar:          python .\ejecutar_pipeline.py --from-step 04_features/01_features.py

Los scripts conservan su comportamiento: varios reconstruyen sus carpetas de
salida. Los scripts de exploracion de probar/ no forman parte de esta lista.
"""
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import json
import os
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent

# Lista explicita: 03_ica debe ir antes de 03_check_ica; no ordenar por nombre.
# Editar esta lista si se anaden o retiran scripts de la pipeline.
STEPS = [
    "01_audit_data_quality/01_check_timestamps_markers.py",
    "01_audit_data_quality/02_recortadatos.py",
    "01_audit_data_quality/03_review_quality.py",
    "02_preprocessing/01_session_channel_window_quality.py",
    "02_preprocessing/02_filter.py",
    "02_preprocessing/03_ica.py",
    "02_preprocessing/03_check_ica.py",
    "02_preprocessing/04_rereference.py",
    "02_preprocessing/05_data_inspection.py",
    "03_epoching/00_umbral_NaN.py",
    "03_epoching/01_epoching.py",
    "03_epoching/02_ label_epoch.py",
    "03_epoching/check_labels.py",
    "04_features/01_features.py",
    "04_features/02_check_features.py",
    "04_features/03_normalize_features.py",
    "04_features/04_check_normalization.py",
    "04_features/05_calibration_eo_ec.py",
    "05_ML/01_1back_vs_3back.py",
    "05_ML/04_check_behavior.py",
    "05_ML/05_3back_vs_rest.py",
    "05_ML/07_bedford_eeg_association.py",
    "05_ML/08_rest_vs_openmatb.py",
    "05_ML/09_bedford_vs_nback.py",
]
ICA_STEP = "02_preprocessing/03_ica.py"


def interpreter(step):
    environment = ".venv_ica" if step == ICA_STEP else ".venv"
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return ROOT / environment / executable


def select_steps(start=None, end=None):
    def index(value, default):
        if value is None:
            return default
        value = value.replace("\\", "/").removeprefix("./")
        if value not in STEPS:
            raise ValueError(f"Paso desconocido: {value}. Consulta --list.")
        return STEPS.index(value)

    first, last = index(start, 0), index(end, len(STEPS) - 1)
    if first > last:
        raise ValueError("El paso inicial debe preceder al paso final.")
    return STEPS[first:last + 1]


def preflight(steps):
    for step in steps:
        if not (ROOT / step).is_file():
            raise FileNotFoundError(f"No existe el script: {step}")
        if not interpreter(step).is_file():
            raise FileNotFoundError(f"No existe Python: {interpreter(step)}")
        # Verificar sintaxis sin importar ni ejecutar el procesamiento.
        compile((ROOT / step).read_text(encoding="utf-8-sig"), step, "exec")
    if not any((ROOT / "data").glob("session_*/participant_*/psychopymarkers_data.csv")):
        raise FileNotFoundError("No hay marcadores originales en data/.")

    versions = {}
    for python in dict.fromkeys(interpreter(step) for step in steps):
        modules = (["numpy", "pandas", "mne", "eeg_positions", "mne_icalabel"]
                   if python == interpreter(ICA_STEP) else
                   ["numpy", "pandas", "scipy", "matplotlib", "mne", "sklearn", "joblib"])
        code = (
            "import importlib, json, sys; "
            f"mods = {{name: importlib.import_module(name) for name in {modules!r}}}; "
            "print(json.dumps({'python': sys.version, 'packages': "
            "{name: getattr(mod, '__version__', 'unknown') for name, mod in mods.items()}}))"
        )
        result = subprocess.run([str(python), "-c", code], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError(f"Faltan dependencias o hay un error en {python}:\n{result.stderr}")
        versions[str(python)] = json.loads(result.stdout.strip().splitlines()[-1])
    return versions


def run_step(step, log):
    command = [str(interpreter(step)), "-u", str(ROOT / step)]
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace")
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()
    except KeyboardInterrupt:
        # La consola tambien entrega Ctrl+C al hijo; esperar su salida.
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()
        raise
    finally:
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Mostrar el orden sin ejecutar scripts")
    parser.add_argument("--check", action="store_true", help="Comprobar scripts, entornos y dependencias sin procesar datos")
    parser.add_argument("--from-step", help="Ruta del primer script que se ejecutara (incluido)")
    parser.add_argument("--to-step", help="Ruta del ultimo script que se ejecutara (incluido)")
    args = parser.parse_args()
    steps = select_steps(args.from_step, args.to_step)
    print(f"Proyecto: {ROOT}\nPasos seleccionados: {len(steps)}", flush=True)
    for number, step in enumerate(steps, 1):
        print(f"{number:02d}. [{interpreter(step).parents[1].name}] {step}", flush=True)
    if args.list:
        return 0
    print("\nComprobando scripts y entornos...", flush=True)
    versions = preflight(steps)
    if args.check:
        print("Comprobacion correcta. No se ha ejecutado ningun script de procesamiento.")
        return 0

    run_dir = ROOT / "results" / "pipeline_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    manifest = dict(started=datetime.now().isoformat(), status="running", environments=versions,
                    steps=[dict(script=step, python=str(interpreter(step)), status="pending",
                                sha256=hashlib.sha256((ROOT / step).read_bytes()).hexdigest()) for step in steps])

    def save_manifest():
        (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    save_manifest()
    print(f"\nRegistro completo: {run_dir / 'console.log'}", flush=True)
    print("Los scripts reconstruiran o sobrescribiran sus salidas habituales. Ctrl+C para detener.\n", flush=True)
    try:
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            for number, entry in enumerate(manifest["steps"], 1):
                step = entry["script"]
                message = f"\n[{number}/{len(steps)}] {step}\n"
                print(message, flush=True)
                log.write(message)
                entry.update(status="running", started=datetime.now().isoformat())
                save_manifest()
                start = time.monotonic()
                code = run_step(step, log)
                entry.update(returncode=code, elapsed_seconds=round(time.monotonic() - start, 2),
                             status="completed" if code == 0 else "failed")
                save_manifest()
                if code:
                    raise RuntimeError(f"Ha fallado {step} (codigo {code}). No se ejecutaran los pasos restantes.\n"
                                       f'Para reintentar desde ahi: python .\\ejecutar_pipeline.py --from-step "{step}"')
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=str(error), finished=datetime.now().isoformat())
        for entry in manifest["steps"]:
            if entry["status"] == "running":
                entry["status"] = manifest["status"]
        save_manifest()
        raise
    manifest.update(status="completed", finished=datetime.now().isoformat())
    save_manifest()
    print(f"\nPIPELINE TERMINADA. Registro: {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nEjecucion interrumpida. Consulta el registro de la pipeline.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
