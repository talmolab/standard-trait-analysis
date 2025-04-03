import argparse
from pathlib import Path
from datetime import datetime
import subprocess
from omegaconf import OmegaConf, DictConfig


def load_and_resolve_config(base_config_path: str) -> DictConfig:
    # Register resolver only once, safely
    try:
        OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))
    except Exception:
        pass  # Resolver already registered

    base_cfg = OmegaConf.load(base_config_path)
    resolved_dict = OmegaConf.to_container(base_cfg, resolve=True)
    return OmegaConf.create(resolved_dict)


def save_config(cfg, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    print(f"[INFO] Saved resolved config to {config_path}")
    return config_path


def run_pipeline(cfg: DictConfig, config_path: Path, dry_run: bool = False):
    steps = cfg.get("steps", [])
    notebook_dir = Path("pipeline/notebooks")

    for step in steps:
        notebook_path = notebook_dir / f"{step}.py"

        command = [
            "marimo",
            "run",
            notebook_path.as_posix(),
            "--",
            "--config_path",
            config_path.as_posix(),
        ]

        print(f"\n[INFO] ----------------------------")
        print(f"[INFO] Step: {step}")
        print(f"[INFO] Notebook: {notebook_path}")
        print(f"[INFO] Command: {' '.join(command)}")
        print(f"[INFO] Config Path: {config_path}")
        print(f"[INFO] ----------------------------\n")

        if dry_run:
            continue

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[ERROR] Step '{step}' failed.")
            print("STDERR:")
            print(result.stderr)
            print("STDOUT:")
            print(result.stdout)
            break
        else:
            print(f"[✓] Step '{step}' completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to base config YAML (with interpolations)",
    )
    args = parser.parse_args()

    # Step 1: Resolve config
    cfg = load_and_resolve_config(args.config)

    # Step 2: Get output dir from resolved config
    output_dir = Path(cfg.output_dir)
    config_path = save_config(cfg, output_dir)

    # Step 3: Run the pipeline
    run_pipeline(cfg, config_path)


if __name__ == "__main__":
    main()
