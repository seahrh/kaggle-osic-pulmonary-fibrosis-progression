import argparse
import logging
import sys
import os
import pandas as pd
import tensorflow as tf
from google.cloud.logging.handlers import ContainerEngineHandler
from sklearn.model_selection import GroupKFold
from tensorflow import keras
from tensorflow.python.lib.io import file_io
from typing import List

formatter = logging.Formatter("%(message)s")
handler = ContainerEngineHandler(stream=sys.stderr)
handler.setFormatter(formatter)
handler.setLevel("INFO")
log = logging.getLogger()
log.addHandler(handler)
log.setLevel("INFO")


MODEL = "efficientnetb4"
TARGET = ["fvc_last_3", "fvc_last_2", "fvc_last_1"]
CONF = {
    "efficientnetb0": {"resolution": 224, "output_size": 1280},
    "efficientnetb1": {"resolution": 240, "output_size": 0},
    "efficientnetb2": {"resolution": 260, "output_size": 1408},
    "efficientnetb3": {"resolution": 300, "output_size": 1536},
    "efficientnetb4": {"resolution": 380, "output_size": 1792},
    "efficientnetb5": {"resolution": 456, "output_size": 2048},
    "efficientnetb6": {"resolution": 528, "output_size": 2304},
    "efficientnetb7": {"resolution": 600, "output_size": 2560},
}
INPUT_SHAPE = (CONF[MODEL]["resolution"], CONF[MODEL]["resolution"], 3)


def _parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir", dest="job_dir", required=True, help="path to job directory"
    )
    parser.add_argument(
        "--bucket", dest="bucket", required=True, help="GCS bucket name",
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        required=True,
        help="path to training dataset directory",
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        default=0.01,
        type=float,
        help="Training probability of dropout",
    )
    parser.add_argument(
        "--epochs", dest="epochs", default=1, type=int, help="Training number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=32,
        type=int,
        help="Training batch size",
    )
    parser.add_argument(
        "--folds",
        dest="folds",
        default=10,
        type=int,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--rlr_patience",
        dest="rlr_patience",
        default=1,
        type=int,
        help="ReduceLrOnPlateau patience",
    )
    parser.add_argument(
        "--es_patience",
        dest="es_patience",
        default=2,
        type=int,
        help="EarlyStopping patience",
    )
    parser.add_argument("--lr", dest="lr", default="1e-3", help="Learning rate")
    parser.add_argument(
        "--pooling", dest="pooling", default="max", help="2D pooling: max, avg",
    )
    parser.add_argument(
        "--hl_sizes",
        dest="hl_sizes",
        required=True,
        help="comma delimited list of hidden layer sizes",
    )
    args, unknown_args = parser.parse_known_args(argv)
    return args, unknown_args


def _split(data, folds):
    spl = GroupKFold(n_splits=folds)
    x = data["img"]
    y = data[TARGET]
    groups = data["pid"]
    train = val = None
    i = 0
    for train_indices, test_indices in spl.split(x, y, groups):
        if i != 0:
            break
        train = data.iloc[train_indices]
        val = data.iloc[test_indices]
        i += 1
    return train, val


def _data_gen(dataframe, directory, batch_size, shuffle=False):
    target_size = (INPUT_SHAPE[0], INPUT_SHAPE[1])
    color_mode = "rgb"
    class_mode = "multi_output"
    idg = keras.preprocessing.image.ImageDataGenerator()
    return idg.flow_from_dataframe(
        dataframe=dataframe,
        x_col="img",
        y_col=TARGET,
        directory=directory,
        target_size=target_size,
        color_mode=color_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode=class_mode,
    )


def _model(dropout: float, lr: float, pooling: str, hidden_layer_sizes: List[int]):
    pretrained = keras.applications.EfficientNetB4(
        include_top=False, input_shape=INPUT_SHAPE, pooling=pooling, weights="imagenet"
    )
    pretrained.trainable = False
    kernel_initializer = keras.initializers.he_normal()
    kernel_regularizer = keras.regularizers.l2(0.01)
    model = keras.models.Sequential()
    model.add(pretrained)
    for i in range(len(hidden_layer_sizes)):
        model.add(keras.layers.BatchNormalization())
        model.add(
            keras.layers.Dense(
                hidden_layer_sizes[i],
                activation="relu",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f"Dense{i + 1}",
            )
        )
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(len(TARGET), name="output"))
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss = keras.losses.MeanSquaredLogarithmicError()
    rmse = keras.metrics.RootMeanSquaredError()
    model.compile(loss=loss, optimizer=optimizer, metrics=[rmse])
    return model


class ModelCheckpointInGcs(keras.callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath,
        gcs_dir: str,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
        **kwargs,
    ):
        super().__init__(
            filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            options=options,
            **kwargs,
        )
        self._gcs_dir = gcs_dir

    def _save_model(self, epoch, logs):
        super()._save_model(epoch, logs)
        filepath = self._get_file_path(epoch, logs)
        if os.path.isfile(filepath):
            with file_io.FileIO(filepath, mode="rb") as inp:
                with file_io.FileIO(
                    os.path.join(self._gcs_dir, filepath), mode="wb+"
                ) as out:
                    out.write(inp.read())


def _callbacks(job_dir: str, filepath: str, rlr_patience: int, es_patience: int):
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=rlr_patience, factor=0.5, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=es_patience, verbose=1
        ),
        ModelCheckpointInGcs(
            filepath=filepath,
            gcs_dir=job_dir,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=job_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
        ),
    ]


def _download_training_data(bucket: str, directory: str) -> None:
    os.system(f"gsutil -m cp -r gs://{bucket}/{directory} .")


def _main(argv=None):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    log.info(f"gpus={gpus}")
    if len(gpus) == 0:
        raise RuntimeError("Expecting at least one gpu but found none.")
    args, unknown_args = _parse(argv)
    log.info(f"args={args}\nunknown_args={unknown_args}")
    lr = float(args.lr)
    hl_sizes = [int(s) for s in args.hl_sizes.split(",")]
    _download_training_data(args.bucket, args.data_dir)
    data = pd.read_parquet(f"{args.data_dir}/train.parquet")
    train, val = _split(data, args.folds)
    train_gen = _data_gen(train, args.data_dir, args.batch_size, shuffle=True)
    val_gen = _data_gen(val, args.data_dir, args.batch_size, shuffle=False)
    model = _model(
        dropout=args.dropout, lr=lr, pooling=args.pooling, hidden_layer_sizes=hl_sizes
    )
    model.summary()
    filepath = "best_model.h5"
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=_callbacks(
            args.job_dir,
            filepath,
            rlr_patience=args.rlr_patience,
            es_patience=args.es_patience,
        ),
    )
    df = pd.DataFrame(history.history)
    df["epoch"] = history.epoch
    path = f"{args.job_dir}/history.csv"
    df.to_csv(path, index=False)
    log.info(f"Done! job_dir={args.job_dir}")


if __name__ == "__main__":
    _main()
