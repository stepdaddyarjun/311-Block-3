import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def preprocess_data(df):

    X = df.drop("target", axis=1)
    y = df["target"]

    config = load_config()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test