import tensorflow as tf
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine(onnx_path, max_batch_size, max_workspace_size):
    # Load ONNX model
    model = onnx.load(onnx_path)

    # Create a TensorRT builder and network
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(1)

    # Set the maximum batch size
    builder.max_batch_size = max_batch_size

    # Set the maximum workspace size
    builder.max_workspace_size = max_workspace_size

    # Create an ONNX parser
    parser = trt.OnnxParser(network, builder)

    # Parse the ONNX model
    with open(onnx_path, 'rb') as model_file:
        success = parser.parse(model_file.read())

    if not success:
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    # Specify dynamic dimensions for input
    profile = builder.create_optimization_profile()
    profile.set_shape("input_1", (1, 3, 224, 224), (16, 3, 224, 224), (32, 3, 224, 224))
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)

    # Build the TensorRT engine
    engine = builder.build_engine(network, config=config)

    return engine

def save_engine(engine, output_path):
    with open(output_path, "wb") as f:
        f.write(engine.serialize())

# Example usage
onnx_model_path = "path/to/your/model.onnx"
max_batch_size = 32
max_workspace_size = 1 << 30  # 1 GB

trt_engine = build_engine(onnx_model_path, max_batch_size, max_workspace_size)

if trt_engine:
    trt_engine_path = "path/to/save/trt_engine.plan"
    save_engine(trt_engine, trt_engine_path)
