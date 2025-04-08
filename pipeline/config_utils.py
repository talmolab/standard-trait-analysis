from omegaconf import OmegaConf
from pathlib import Path
import argparse
import sys


def load_config(config_arg="--config_path", default_path="pipeline/configs/base.yaml"):
    """Load an OmegaConf config file from command line or default path.

    Args:
        config_arg: The command line argument for the config path
        default_path: The default path to the config file

    Returns:
        cfg: The loaded OmegaConf object
        args: The parsed argparse.Namespace
        parser: The argument parser (optional reuse)
    """
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(config_arg, type=str, default=default_path)
        args = parser.parse_args(sys.argv[1:])
        print(f"✅ Parsed args: {args}")
    except Exception as e:
        print(f"❌ Error parsing args: {e}")
        raise

    try:
        cfg = OmegaConf.load(args.config_path)
        print(f"✅ Loaded config from {args.config_path}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        raise

    return cfg, args, parser
