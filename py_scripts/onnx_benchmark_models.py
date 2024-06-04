# Base Libraries
from utility_scripts import utils
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

# Model Converting & Runtime libraries
import tf2onnx
import onnxruntime as rt

from EMA import (
    EMA_finalize,
    EMA_init,
)


# 3m 33 sec
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
@click.option(
    "--save-onnx-directory",
    default="models_lib/onnx_models",
    help="Directory to save onnx models",
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
    save_onnx_directory,
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

    # Save Dirs Creation
    if not os.path.exists(os.path.join(results_save_dir, model_name)):
        print("Results Save Dir Doesn't Exist Creating !!")
        os.makedirs(os.path.join(results_save_dir, model_name), exist_ok=True)

    # Loading and compiling tf Model
    tf_model, _ = btfk.load_tf_and_keras_models(
        load_directory=load_directory, model_name=model_name, img_url=url
    )

    # Save ONNX model if not exists
    save_path = os.path.join(save_onnx_directory, f"{model_name}")
    if not os.path.exists(save_onnx_directory):
        os.makedirs(save_onnx_directory, exist_ok=True)
        print(
            f"\n\n=============== Saving ONNX {model_name} Model "
            f"==============="
        )
        input_shape = (None, 224, 224, 3)
        st_time = time.time()
        # Convert the TensorFlow model to ONNX format
        onnx_model, _ = tf2onnx.convert.from_keras(
            tf_model,
            input_signature=[
                tf.TensorSpec(
                    shape=input_shape, dtype=tf.float32, name="input"
                )
            ],
        )

        # Save the ONNX model to a file
        with open(f"{save_path}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        end_time = time.time()
        total_time = end_time - st_time
        print(
            f"\n=============== {model_name} converted to onnx and saved at "
            f"{save_path} in time {total_time} secs ===============\n"
        )

    # Release GPU memory before benchmarking
    tf.keras.backend.clear_session()
    print(
        f"\n=============== Benchmarking ONNX {model_name} Model "
        f"===============\n"
    )
    st_time = time.time()
    preprocessor = tfk.builders_preprocs().get(model_name)[1]
    onnx_model_path = f"{save_path}.onnx"
    providers = ["CUDAExecutionProvider"]
    session = rt.InferenceSession(onnx_model_path, providers=providers)
    end_time = time.time()
    load_time = end_time - st_time
    img = io.imread(url)
    onnx = True
    pred_time = utils.make_preds(
        session,
        model_name="onnxrt",
        preprocessor=preprocessor,
        onnx=onnx,
        img=img,
    )

    total_model_compilation_time = pred_time + load_time
    print(
        f"\n=============== OnnxRT Compile Time: "
        f"{total_model_compilation_time} secs ===============\n"
    )

    # Benchmark the onnx model
    img = io.imread(url)
    input_data, _ = utils.batch_sigle_img(
        img, num_images=num_images, preprocessor=preprocessor
    )
    input_data = input_data.astype(np.float32)

    csv_path = os.path.join(
        results_save_dir,
        model_name,
        f"onnxrt_{num_iterations}_it_{result_suffix}.csv",
    )
    utils.batch_model_performances(
        framework_name="onnxrt",
        model=session,
        batch_sizes=batch_sizes,
        num_warmup_runs=num_warmup_runs,
        num_model_runs=num_iterations,
        input_data=input_data,
        csv_path=csv_path,
        onnx=True,
        trt=False,
        torch=False,
        gpu_id=gpu_id,
    )

    print(
        f"\n=============== ONNX {model_name} Model Benchmarked && results "
        f"saved to {results_save_dir}===============\n\n"
    )
    EMA_finalize()


class GlobalVars:
    result_suffix = ""


if __name__ == "__main__":
    nb = int(input("\n\nIf using a jnb enter 1 else 0\n"))
    uni = int(input("\n\nIf using uni gpu enter 1 else 0\n"))
    # If everything is run from jupyter nb then results file generated
    # will have a suffix of uni
    GlobalVars.result_suffix = "uni" if uni else "work"
    GlobalVars.result_suffix += "_nb" if nb else "_py"

    print(
        f"==================== Suffix used with result files will be "
        f"{GlobalVars.result_suffix}!! ============================="
    )

    main()
