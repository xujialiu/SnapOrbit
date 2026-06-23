import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def parse_args_and_config() -> DictConfig:
    """Parse CLI args and load config with overrides."""
    parser = argparse.ArgumentParser(
        description="MultiScale ViT Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config configs/classification/default.yaml
  python train.py --config configs/classification/default.yaml model.lora.rank=8
  python train.py --config configs/classification/default.yaml --eval paths.ckpt_path=./path/to/your/checkpoint.pth
        """,
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config"
    )
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument(
        "overrides", nargs="*", help="Override config values (e.g., model.lora.rank=8)"
    )

    args = parser.parse_args()

    # Load YAML config
    config = OmegaConf.load(args.config)

    # Apply CLI overrides
    if args.overrides:
        cli_config = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(config, cli_config)

    # Handle --eval flag
    if args.eval:
        config.eval = True

    return config


def save_config(config: DictConfig, output_dir: str, filename: str = "config.yaml"):
    """Save config to output directory for reproducibility."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_file = output_path / filename
    OmegaConf.save(config, config_file)
    print(f"Config saved to: {config_file}")
