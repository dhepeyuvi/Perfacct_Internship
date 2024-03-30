# Base Libraries
import os
import click
import time
from skimage import io

# Utility py files
from utility_scripts import tft_optimizer as tft
from utility_scripts import utils
from py_scripts import create_tf_and_keras_modes as tfk
import benchmark_tf_keras_normal_xla_models as btfk
import tensorflow as tf

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
    "--load-directory",
    default="models_lib/tf_models",
    help="Directory to load models",
)
@click.option(
    "--save-trt-directory",
    default="models_lib/trt_models",
    help="Directory to save trt models",
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
    "--gpu-id", required=True, help="GPU to use, get ID from nvidia-smi"
)
@click.option(
    "--batch-sizes",
    default="8,16,32,64,128,256,512",
    help="Batch sizes for benchmarking, separated by commas",
)
def main(
    url,
    load_directory,
    save_trt_directory,
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

    # Loading and Compiling FP32 Model
    PRECISION = "FP32"
    save_dir = os.path.join(
        save_trt_directory, f"{model_name}_TFTRT_{PRECISION}"
    )
    if not os.path.exists(save_dir):
        print(
            f"\n=============== Saving FP32 TF-TRT {model_name} Model ===============\n"
        )
        st_time = time.time()
        tf_path = os.path.join(load_directory, model_name)
        opt_model = tft.ModelOptimizer(tf_path)
        trt_fp32 = opt_model.convert(save_dir, precision=PRECISION)
        end_time = time.time()
        total_time = end_time - st_time
        print(
            f"\n=============== {model_name} converted to TF-TRT {PRECISION} precision and saved at {save_dir} in time {total_time} secs ===============\n"
        )

    # Benchmarking the FP32 model
    print(
        f"\n=============== Benchmarking FP32 TF-TRT {model_name} Model ===============\n"
    )

    # Release GPU memory before benchmarking
    tf.keras.backend.clear_session()
    preprocessor = tfk.builders_preprocs().get(model_name)[1]
    st_load_time = time.time()
    trt_fp32 = tft.OptimizedModel(save_dir)
    end_load_time = time.time()
    total_time_to_load = end_load_time - st_load_time

    img = io.imread(url)
    preds_time = utils.make_preds(
        trt_fp32, "trtfp32_" + model_name, img, preprocessor, trt=True
    )

    total_model_compilation_time = total_time_to_load + preds_time
    print(
        f"\n=============== TRTFP32 Compile Time: {total_model_compilation_time} secs ===============\n"
    )

    btfk.benchmark_model(
        framework_name=f"TFTRT{PRECISION}",
        url=url,
        model_name=model_name,
        model=trt_fp32,
        xla_enabled=False,
        results_save_dir=results_save_dir,
        save_file_name=f"TFTRT{PRECISION}_{num_iterations}_it_{result_suffix}.csv",
        num_images=num_images,
        batch_sizes=batch_sizes,
        num_warmup_runs=num_warmup_runs,
        num_model_runs=num_iterations,
        trt=True,
        torch=False,
        gpu_id=gpu_id,
    )

    # Loading and Compiling FP16 Model
    PRECISION = "FP16"
    save_dir = os.path.join(
        save_trt_directory, f"{model_name}_TFTRT_{PRECISION}"
    )
    if not os.path.exists(save_dir):
        print(
            f"\n=============== Saving FP16 TF-TRT {model_name} Model ===============\n"
        )
        st_time = time.time()
        tf_path = os.path.join(load_directory, model_name)
        opt_model = tft.ModelOptimizer(tf_path)
        trt_fp16 = opt_model.convert(save_dir, precision=PRECISION)
        end_time = time.time()
        total_time = end_time - st_time
        print(
            f"\n=============== {model_name} converted to TF-TRT {PRECISION} precision and saved at {save_dir} in time {total_time} secs ===============\n"
        )

    # Benchmarking the FP16 model
    print(
        f"\n=============== Benchmarking FP16 TF-TRT {model_name} Model ===============\n"
    )

    # Release GPU memory before benchmarking
    tf.keras.backend.clear_session()
    st_load_time = time.time()
    trt_fp16 = tft.OptimizedModel(save_dir)
    end_load_time = time.time()
    total_time_to_load = end_load_time - st_load_time

    img = io.imread(url)
    preds_time = utils.make_preds(
        trt_fp16, "trtfp16_" + model_name, img, preprocessor, trt=True
    )

    total_model_compilation_time = total_time_to_load + preds_time
    print(
        f"\n=============== TRTFP16 Compile Time: {total_model_compilation_time} secs ===============\n"
    )

    btfk.benchmark_model(
        framework_name=f"TFTRT{PRECISION}",
        url=url,
        model_name=model_name,
        model=trt_fp16,
        xla_enabled=False,
        results_save_dir=results_save_dir,
        save_file_name=f"TFTRT{PRECISION}_{num_iterations}_it_{result_suffix}.csv",
        num_images=num_images,
        batch_sizes=batch_sizes,
        num_warmup_runs=num_warmup_runs,
        num_model_runs=num_iterations,
        trt=True,
        torch=False,
        gpu_id=gpu_id,
    )

    print(
        f"\n=============== TF-TRT {model_name} Model Benchmarked && results saved to {results_save_dir} ===============\n\n"
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
