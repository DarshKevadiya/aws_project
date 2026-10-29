from pathlib import Path

import tensorflow as tf
from sklearn.metrics import confusion_matrix

MobileNetV2 = tf.keras.applications.MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
FINE_TUNE_EPOCHS = 8
FINE_TUNE_LAYERS = 30
VALIDATION_SPLIT = 0.2
SEED = 42


def make_early_stopping() -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )


def find_dataset_dir() -> Path:
    candidates = (
        PROJECT_DIR / "dataset",
        PROJECT_DIR / "Garbage classification" / "Garbage classification",
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
        if any(path.is_dir() for path in candidate.iterdir()):
            return candidate

    expected = PROJECT_DIR / "dataset"
    raise FileNotFoundError(
        f"Dataset not found. Put class folders under {expected}, "
        "or use dataset/train/<class> and dataset/val/<class>."
    )


def make_data_generators(
    dataset_dir: Path,
) -> tuple[tf.keras.utils.PyDataset, tf.keras.utils.PyDataset]:
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    augmentation = {
        "preprocessing_function": preprocess_input,
        "rotation_range": 20,
        "zoom_range": 0.15,
        "horizontal_flip": True,
    }

    if train_dir.is_dir() and val_dir.is_dir():
        train_datagen = ImageDataGenerator(**augmentation)
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        train_data = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            seed=SEED,
        )
        val_data = val_datagen.flow_from_directory(
            str(val_dir),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )
        if train_data.class_indices != val_data.class_indices:
            raise ValueError(
                "The train and val folders must contain the same class subfolders."
            )
    else:
        class_dirs = [
            path
            for path in dataset_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        if len(class_dirs) < 2:
            raise ValueError(
                f"Expected at least two class folders in {dataset_dir}."
            )

        train_datagen = ImageDataGenerator(
            **augmentation,
            validation_split=VALIDATION_SPLIT,
        )
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=VALIDATION_SPLIT,
        )
        train_data = train_datagen.flow_from_directory(
            str(dataset_dir),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            seed=SEED,
        )
        val_data = val_datagen.flow_from_directory(
            str(dataset_dir),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=SEED,
        )

    if train_data.samples == 0 or val_data.samples == 0:
        raise ValueError("Both training and validation sets must contain images.")
    return train_data, val_data


def main() -> None:
    dataset_dir = find_dataset_dir()
    train_data, val_data = make_data_generators(dataset_dir)
    class_names = list(train_data.class_indices)
    print(f"Dataset: {dataset_dir}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    features = base_model(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(
        len(class_names),
        activation="softmax",
    )(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[make_early_stopping()],
    )

    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[make_early_stopping()],
    )

    val_data.reset()
    evaluation = model.evaluate(val_data, verbose=0, return_dict=True)
    print(f"Validation accuracy: {evaluation['accuracy']:.2%}")

    val_data.reset()
    predictions = model.predict(val_data, verbose=1).argmax(axis=1)
    matrix = confusion_matrix(
        val_data.classes,
        predictions,
        labels=list(range(len(class_names))),
    )
    cell_width = max(7, max(len(name) for name in class_names)) + 2
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(f"{'':>{cell_width}}" + "".join(
        f"{name:>{cell_width}}" for name in class_names
    ))
    for name, row in zip(class_names, matrix):
        print(f"{name:>{cell_width}}" + "".join(
            f"{count:>{cell_width}}" for count in row
        ))

    output_dir = PROJECT_DIR / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "garbage_model.keras")
    print(f"Saved trained model to {output_dir / 'garbage_model.keras'}")


if __name__ == "__main__":
    main()
