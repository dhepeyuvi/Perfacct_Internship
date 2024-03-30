# Base Libraries
from utility_scripts import utils
from skimage import io
import os
import click
import time
from skimage import io
import numpy as np
# Utility py files
import py_scripts.create_tf_and_keras_modes as tfk
import py_scripts.benchmark_tf_keras_normal_xla_models as btfk

# DL Base Libraries
import tensorflow as tf
from tensorflow.keras.backend import clear_session

# Model Converting & Runtime libraries
import tf2onnx
import torch
from onnx2torch import convert
from EMA import (
    EMA_finalize,
    EMA_init,
)

@click.command()
@click.option(
    "--url",
    default="https://i.pinimg.com/originals/56/ea/2b/56ea2bb991a7446776ac2f2f27fdc397.jpg",
    help="Image URL for predictions",
)
@click.option(
    "--load-onnx-directory",
    default="models_lib/onnx_models",
    help="Directory to load onnx models from",
)
@click.option(
    "--save-torch-directory",
    default="models_lib/torch_models",
    help="Directory to save full torch models",
)
@click.option("--model-name", default="MobileNetV3L", help="Name of the model")
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
@click.option("--gpu-id", default=0, help="GPU t use, get ID from nvidia-smi")
@click.option(
    "--batch-sizes",
    default="8,16,32,64,128,256",
    help="Batch sizes for benchmarking, separated by commas",
)
def main(
    url,
    load_onnx_directory,
    save_torch_directory,
    model_name,
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

    # # Enable GPU growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Converting Batch_Sizes to Int
    batch_sizes = [int(i) for i in batch_sizes.split(",")]

    # Save Dirc Creation
    if not os.path.exists(os.path.join(results_save_dir, model_name)):
        print("Results Save Dir Doesn't Exist Creating !!")
        os.makedirs(os.path.join(results_save_dir, model_name), exist_ok=True)

    load_path = os.path.join(load_onnx_directory, model_name)

    # Save Torch model if not exists
    save_path = os.path.join(save_torch_directory, f"{model_name}")
    if not os.path.exists(save_path):
        os.makedirs(save_torch_directory, exist_ok=True)
        print(
            f"\n=============== Saving Torch {model_name} Model ===============\n"
        )
        st_time = time.time()

        load_path = os.path.join(load_onnx_directory, model_name)
        torch_model = convert(f"{load_path}.onnx")
        torch.save(torch_model, f"{save_path}.pth")
        end_time = time.time()
        total_time = end_time - st_time
        print(
            f"\n=============== {model_name} converted to Torch and saved at {save_path} in time {total_time} secs ===============\n"
        )

    # Release GPU memory before benchmarking
    tf.keras.backend.clear_session()
    print(
        f"\n=============== Benchmarking Torch {model_name} Model ===============\n"
    )

    # Load the model
    st_load_time = time.time()
    torch_model = torch.load(f"{save_path}.pth").cuda()
    torch_model.eval()  # Set to eval mode
    end_load_time = time.time()
    total_load_time = end_load_time - st_load_time

    # Compiling the model
    preprocessor = tfk.builders_preprocs().get(model_name)[1]
    img = io.imread(url)
    pred_time = utils.make_preds(
        torch_model,
        model_name="Torch",
        preprocessor=preprocessor,
        torch=True,
        img=img,
    )

    total_model_compilation_time = total_load_time + pred_time
    print(
        f"\n=============== Torch Compile Time: {total_model_compilation_time} secs ===============\n"
    )

    # Benchmarking Torch Model
    img = io.imread(url)
    input_data, data_load_time = utils.batch_sigle_img(
        img,
        target_size=(224, 224),
        num_images=num_images,
        preprocessor=preprocessor,
    )
    input_data = input_data.astype(np.float32)

    # Results save path
    save_file_name = f"torch_{num_iterations}_it_{result_suffix}.csv"
    results_save_path = os.path.join(
        results_save_dir, model_name, save_file_name
    )

    utils.batch_model_performances(
        framework_name="torch",
        model=torch_model,
        input_data=input_data,
        batch_sizes=batch_sizes,
        csv_path=results_save_path,
        num_warmup_runs=num_warmup_runs,
        num_model_runs=num_iterations,
        trt=False,
        onnx=False,
        torch=True,
        gpu_id=gpu_id,
    )

    print(
        f"\n =============== Torch {model_name} Model Benchmarked =============== \n\n"
    )
    EMA_finalize()


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
