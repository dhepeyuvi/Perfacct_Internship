import click
import tensorflow as tf
import tensorflow.keras.applications as tf_app
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
from skimage import io
from tensorflow.keras.applications.mobilenet_v3 import (
    preprocess_input,
    decode_predictions,
)
from utility_scripts import utils
import os


@click.command()
@click.option(
    "--save-dir",
    default="models_lib/tf_models",
    help="Directory to save models",
    required=True,
)
@click.option(
    "--optimize",
    is_flag=True,
    help="Optimize Model by removing non-essentials",
    required=True,
    default=False,
)
@click.option(
    "--img-url",
    default="https://i.pinimg.com/originals/56/ea/2b/56ea2bb991a7446776ac2f2f27fdc397.jpg",
    help="Image URL for predictions",
)
def download_and_save(save_dir, img_url, optimize):
    print("=== Downloading Models ===")

    models = builders_preprocs()

    for model_name, (builder, preprocessor) in models.items():
        model = builder(weights="imagenet", include_top=True)

        if optimize:
            # NOTE:Optimization is not working for the time being
            model = optimize_model_for_inference(model)

        tf_keras_model_save(model, save_dir, model_name, img_url, preprocessor)
        print(
            f"\n\n===============Saved {model_name} to {save_dir}==============="
        )


def builders_preprocs():

    models = {"MobileNetV3L": [tf_app.MobileNetV3Large, preprocess_input]}

    return models


def tf_keras_model_save(model, save_dir, model_name, img_url, preprocessor):
    if not os.path.exists(save_dir):
        print(f"Save Dir Doesn't exist. Creating !!")
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, model_name)
    img = io.imread(img_url)
    utils.make_preds(model, model_name, img, preprocessor)
    model.save(save_path)
    model.save(f"{save_path}.keras")


# def optimize_model_for_inference(model):
#     # Convert the Keras model to a TensorFlow GraphDef
#     concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
#         flatten_input=tf.TensorSpec(
#             shape=[None, 224, 224, 3], dtype=tf.float32
#         )
#     )

#     frozen_graph = convert_variables_to_constants_v2(
#         concrete_func.graph.as_graph_def(),
#         [node.op.name for node in concrete_func.graph.get_operations()],
#     )


#     # Optimize the frozen graph for inference
#     optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
#         frozen_graph,
#         input_names=[
#             input_tensor.name for input_tensor in concrete_func.inputs
#         ],
#         output_names=[
#             output_tensor.name for output_tensor in concrete_func.outputs
#         ],
#         placeholder_type_enum=tf.float32.as_datatype_enum,
#     )

#     # Load the optimized model back into Keras
#     optimized_model = tf.keras.models.model_from_config(
#         tf.compat.v1.graph_util.import_graph_def(
#             optimized_graph_def
#         ).as_graph_def()
#     )

#     return optimized_model


if __name__ == "__main__":
    download_and_save()
