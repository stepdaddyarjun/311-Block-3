import pandas as pd
import yaml


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_data():
    config = load_config()
    path = config["data"]["raw_data_path"]
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())