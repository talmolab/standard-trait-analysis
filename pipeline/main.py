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


def run_pipeline(
    cfg: DictConfig,
    config_path: Path,
    edit_mode: bool = False,
    export_html: bool = False,
    notebook_dir: str = "pipeline/notebooks",
    dry_run: bool = False,
) -> None:
    """Run the pipeline steps defined in the config.

    Args:
        cfg (DictConfig): The OmegaConf config object.
        config_path (Path): Path to the resolved config file.
        edit_mode (bool): If True, open notebooks in edit mode.
        export_html (bool): If True, export executed notebooks to HTML.
        notebook_dir (str): Directory containing the pipeline notebooks.
        dry_run (bool): If True, print commands but don't execute them.
    """
    steps = cfg.get("steps", [])
    if not steps:
        print("[ERROR] No steps defined in the config.")
        return

    notebook_dir = Path(notebook_dir)
    if not notebook_dir.exists():
        print(f"[ERROR] Notebook directory '{notebook_dir}' does not exist.")
        return

    for step in steps:
        notebook_path = notebook_dir / f"{step}.py"
        step_output_dir = config_path.parent / step
        step_output_dir.mkdir(parents=True, exist_ok=True)

        if not notebook_path.exists():
            print(f"[ERROR] Notebook '{notebook_path}' does not exist.")
            continue

        marimo_command = "edit" if edit_mode else "run"

        command = [
            "marimo",
            marimo_command,
            notebook_path.as_posix(),
            "--",
            "--config_path",
            config_path.as_posix(),
        ]

        print(f"\n[INFO] ----------------------------")
        print(f"[INFO] Step: {step}")
        print(f"[INFO] Mode: {'edit' if edit_mode else 'run'}")
        print(f"[INFO] Notebook: {notebook_path}")
        print(f"[INFO] Command: {' '.join(command)}")
        print(f"[INFO] Config Path: {config_path}")
        print(f"[INFO] ----------------------------\n")

        if dry_run:
            continue

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] Step '{step}' failed.")
            print("STDERR:")
            print(result.stderr)
            print("STDOUT:")
            print(result.stdout)
            break
        else:
            print(f"[✓] Step '{step}' completed.")

        if export_html:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            export_path = step_output_dir / f"{step}_{timestamp}.html"
            export_command = [
                "marimo",
                "export",
                "html",
                notebook_path.as_posix(),
                "-o",
                export_path.as_posix(),
                "--include-outputs",
            ]
            print(f"[INFO] Exporting HTML to: {export_path}")
            subprocess.run(export_command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="pipeline/configs/base.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--notebook-dir",
        default="pipeline/notebooks",
        help="Directory containing the pipeline notebooks.",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Open notebooks in edit mode instead of running them.",
    )
    parser.add_argument(
        "--export-html",
        action="store_true",
        help="Export executed notebooks to HTML (applies to all modes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and config info but don't run the pipeline.",
    )
    args = parser.parse_args()

    cfg = load_and_resolve_config(args.config)
    output_dir = Path(cfg.output_dir)
    config_path = save_config(cfg, output_dir)

    run_pipeline(
        cfg,
        config_path,
        edit_mode=args.edit,
        export_html=args.export_html,
        notebook_dir=args.notebook_dir,
        dry_run=args.dry_run,
    )

    print(f"[INFO] Pipeline completed.")


if __name__ == "__main__":
    main()
