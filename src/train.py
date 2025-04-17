from src.models.simple_transformer import SimpleTransformer

MODELS = {
    "simple_transformer": SimpleTransformer,
}


def main(config):
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config file")
    args = parser.parse_args()

    main(args.config)