# Base Libraries
from skimage import io
from utility_scripts import utils
import os
import time
import click
import numpy as np
# DL Base Libraries
import tensorflow as tf
from EMA import (
    EMA_finalize,
    EMA_init,
)
# Importing builders and preprocessors
from py_scripts import create_tf_and_keras_modes as tfk


@click.command()
@click.option(
    "--url",
    default="https://i.pinimg.com/originals/56/ea/2b/56ea2bb991a7446776ac2f2f27fdc397.jpg",
    help="Image URL for predictions",
)
@click.option(
    "--load-directory",
    default="models_lib/tf_models",
    help="Directory to load models",
)
@click.option("--model-name", default="MobileNetV3L", help="Name of the model")
@click.option(
    "--xla-enabled",
    is_flag=True,
    default=False,
    help="Enable XLA optimization",
)
@click.option(
    "--num-images", default=512, help="Number of images for benchmarking"
)
@click.option(
    "--results-save-dir",
    default="benchmark_results",
    help="Directory to save benchmark results",
)
@click.option(
    "--num-iterations",
    default=10,
    help="Number of iterations for benchmarking",
)
@click.option("--num-warmup-runs", default=50, help="Number of warm-up runs")
@click.option(
    "--gpu-id", required=True, help="GPU t use, get ID from nvidia-smi"
)
@click.option(
    "--batch-sizes",
    default="8,16,32,64,128,256,512",
    help="Batch sizes for benchmarking, separated by commas",
)
def main(
    url,
    load_directory,
    model_name,
    xla_enabled,
    num_images,
    results_save_dir,
    num_iterations,
    num_warmup_runs,
    gpu_id,
    batch_sizes,
):
    EMA_init()
    result_suffix = GlobalVars.result_suffix
    gpu_id = int(gpu_id)
    print(f"\n=============== Using GPU: {gpu_id} ===============\n")

    # Enable GPU Growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Converting Batch_Sizes to Int
    batch_sizes = [int(i) for i in batch_sizes.split(",")]

    # Save Dirs Creation
    if not os.path.exists(os.path.join(results_save_dir, model_name)):
        print("Results Save Dir Doesn't Exist Creating !!")
        os.makedirs(os.path.join(results_save_dir, model_name), exist_ok=True)

    # Loading and Compiling the models
    tf_model, keras_model = load_tf_and_keras_models(
        load_directory=load_directory, model_name=model_name, img_url=url
    )

    print(
        f"\n=============== Benchmarking TensorFlow {model_name} Model ===============\n"
    )
    tf.keras.backend.clear_session()
    benchmark_model(
        url=url,
        model_name=model_name,
        model=tf_model,
        xla_enabled=xla_enabled,
        results_save_dir=results_save_dir,
        save_file_name=f"{'tfxla' if xla_enabled else 'tf'}_{num_iterations}_it_{result_suffix}.csv",
        num_images=num_images,
        batch_sizes=batch_sizes,
        num_warmup_runs=num_warmup_runs,
        num_model_runs=num_iterations,
        gpu_id=gpu_id,
        framework_name = f"{'tfxla' if xla_enabled else 'tf'}"
    )
    print(
        f"\n=============== Benchmarking Keras {model_name} Model ===============\n"
    )
    tf.keras.backend.clear_session()
    print(f"Benchmarking Keras {model_name} Model")
    benchmark_model(
        url=url,
        model_name=model_name,
        model=keras_model,
        xla_enabled=xla_enabled,
        results_save_dir=results_save_dir,
        save_file_name=f"{'kerasxla' if xla_enabled else 'keras'}_{num_iterations}_it_{result_suffix}.csv",
        num_images=num_images,
        batch_sizes=batch_sizes,
        num_warmup_runs=num_warmup_runs,
        num_model_runs=num_iterations,
        gpu_id=gpu_id,
        framework_name = f"{'kerasxla' if xla_enabled else 'keras'}"

    )

    print(
        f"\n=============== TF & Keras {model_name} Benchmarked  && results saved to {results_save_dir} ===============\n\n"
    )

    # Cleanup.
    EMA_finalize()

def load_tf_and_keras_models(load_directory, model_name, img_url):
    preprocessor = tfk.builders_preprocs().get(model_name)[1]

    load_path = os.path.join(load_directory, model_name)
    tfst_time = time.time()
    tf_model = tf.keras.models.load_model(load_path)
    tfend_time = time.time()
    tfload_time = tfend_time - tfst_time

    krsst_time = time.time()
    keras_model = tf.keras.models.load_model(f"{load_path}.keras")
    krsend_time = time.time()
    krsload_time = krsend_time - krsst_time

    # Make one prediction via each model
    img = io.imread(img_url)
    tfpred_time = utils.make_preds(
        tf_model, "TF_" + model_name, img, preprocessor
    )
    tf_total_time = tfload_time + tfpred_time

    krspred_time = utils.make_preds(
        keras_model, "Keras_" + model_name, img, preprocessor
    )
    krs_total_time = krsload_time + krspred_time

    print(
        f"\n=============== TF Model Compile Time {tf_total_time} secs && Keras Model Compile Time {krs_total_time} secs\n"
    )
    print(
        f"\n=============== TF & Keras formats of {model_name} Model Loading Completed ===============\n\n"
    )

    return tf_model, keras_model


def benchmark_model(
    framework_name,
    url,
    model_name,
    model,
    trt=False,
    torch=False,
    xla_enabled=False,
    results_save_dir="benchmark_results",
    save_file_name="result.csv",
    num_images=512,
    batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    num_warmup_runs=50,
    num_model_runs=10,
    gpu_id=1,
):
    save_path = os.path.join(results_save_dir, model_name, save_file_name)

    preprocessor = tfk.builders_preprocs().get(model_name)[1]

    img = io.imread(url)
    input_data, data_load_time = utils.batch_sigle_img(
        img,
        target_size=(224, 224),
        num_images=num_images,
        preprocessor=preprocessor,
    )
    input_data = input_data.astype(np.float32)

    if xla_enabled:
        model = tf.function(model)
        tf.config.optimizer.set_jit(True)

    _ = utils.batch_model_performances(
        framework_name=framework_name,
        model=model,
        input_data=input_data,
        batch_sizes=batch_sizes,
        csv_path=save_path,
        num_warmup_runs=num_warmup_runs,
        num_model_runs=num_model_runs,
        trt=trt,
        torch=torch,
        gpu_id=gpu_id,
    )


class GlobalVars:
    result_suffix = ""


if __name__ == "__main__":


    nb = int(input("\n\nIf using a jnb enter 1 else 0\n"))
    uni = int(input("\n\nIf using uni gpu enter 1 else 0\n"))
    # If everything is run from jupyter nb then results file generated will have a suffix of uni
    GlobalVars.result_suffix = "uni" if uni else "work"
    GlobalVars.result_suffix += "_nb" if nb else "_py"

    print(
        f"==================== Suffix used with result files will be {GlobalVars.result_suffix}!! ============================="
    )
    main()


0