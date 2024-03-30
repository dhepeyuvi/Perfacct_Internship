## TF Keras
=============== Using GPU: 1 ===============

2024-02-21 04:23:48.764562: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 11549 MB memory:  -> device: 0, name: NVIDIA TITAN X (Pascal), pci bus id: 0000:81:00.0, compute capability: 6.1
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.

================ Time to load the data onto CPU 0.04385495185852051 secs ================


2024-02-21 04:23:58.725506: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8700
2024-02-21 04:23:58.818466: I tensorflow/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2024-02-21 04:23:58.959732: I tensorflow/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory

=============== Top 3 predictions by TF_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402355), ('n04254680', 'soccer_ball', 0.21120338), ('n01871265', 'tusker', 0.10708933)]



================ Time to load the data onto CPU 0.02092576026916504 secs ================



=============== Top 3 predictions by Keras_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402355), ('n04254680', 'soccer_ball', 0.21120338), ('n01871265', 'tusker', 0.10708933)]



=============== TF Model Compile Time 7.604979038238525 secs && Keras Model Compile Time 3.2954609394073486 secs


=============== TF & Keras formats of MobileNetV3L Model Loading Completed ===============



=============== Benchmarking TensorFlow MobileNetV3L Model ===============


================ Time to load the data onto CPU 0.38904809951782227 secs ================



================ Batch Size 8 WarmUp Time 5.862186431884766 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 16 WarmUp Time 5.435409307479858 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 32 WarmUp Time 5.660342216491699 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 64 WarmUp Time 7.567151308059692 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 128 WarmUp Time 9.75763726234436 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 256 WarmUp Time 14.474512577056885 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 512 WarmUp Time 30.642561435699463 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Total time to run batch_performance_func 242.66100001335144 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/tf_10_it_work_py.csv ================



=============== Benchmarking Keras MobileNetV3L Model ===============

Benchmarking Keras MobileNetV3L Model

================ Time to load the data onto CPU 0.4543793201446533 secs ================



================ Batch Size 8 WarmUp Time 5.368754863739014 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 16 WarmUp Time 5.373613357543945 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 32 WarmUp Time 6.331328868865967 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 64 WarmUp Time 6.978407859802246 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 128 WarmUp Time 9.634269714355469 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 256 WarmUp Time 15.732005834579468 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Batch Size 512 WarmUp Time 30.673927307128906 secs ================


region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)
region ('/home/dhepe/work/internship', 653206)

================ Total time to run batch_performance_func 240.18738079071045 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/keras_10_it_work_py.csv ================



=============== TF & Keras MobileNetV3L Benchmarked  && results saved to benchmark_results ===============
## TF Keras XLA
=============== Using GPU: 1 ===============

2024-02-21 09:15:23.702781: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 11549 MB memory:  -> device: 0, name: NVIDIA TITAN X (Pascal), pci bus id: 0000:81:00.0, compute capability: 6.1
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.

================ Time to load the data onto CPU 0.047231435775756836 secs ================


2024-02-21 09:15:33.274605: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8700
2024-02-21 09:15:33.399818: I tensorflow/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2024-02-21 09:15:33.564362: I tensorflow/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory

=============== Top 3 predictions by TF_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402355), ('n04254680', 'soccer_ball', 0.21120338), ('n01871265', 'tusker', 0.10708933)]



================ Time to load the data onto CPU 0.029880285263061523 secs ================



=============== Top 3 predictions by Keras_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402355), ('n04254680', 'soccer_ball', 0.21120338), ('n01871265', 'tusker', 0.10708933)]



=============== TF Model Compile Time 7.25564980506897 secs && Keras Model Compile Time 3.1648857593536377 secs


=============== TF & Keras formats of MobileNetV3L Model Loading Completed ===============



=============== Benchmarking TensorFlow MobileNetV3L Model ===============


================ Time to load the data onto CPU 0.3996124267578125 secs ================


2024-02-21 09:15:35.697745: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560028170dc0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-02-21 09:15:35.697808: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA TITAN X (Pascal), Compute Capability 6.1
2024-02-21 09:15:35.723190: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2024-02-21 09:15:39.130030: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

================ Batch Size 8 WarmUp Time 4.9439287185668945 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 16 WarmUp Time 5.371102571487427 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 32 WarmUp Time 5.88367772102356 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 64 WarmUp Time 7.929189443588257 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 128 WarmUp Time 11.406505107879639 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 256 WarmUp Time 21.698920011520386 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
2024-02-21 09:17:13.956318: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 4.08GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.

================ Batch Size 512 WarmUp Time 40.258705615997314 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Total time to run batch_performance_func 137.7811415195465 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/tfxla_10_it_work_py.csv ================



=============== Benchmarking Keras MobileNetV3L Model ===============

Benchmarking Keras MobileNetV3L Model

================ Time to load the data onto CPU 0.4009075164794922 secs ================



================ Batch Size 8 WarmUp Time 3.3233253955841064 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 16 WarmUp Time 3.4923360347747803 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 32 WarmUp Time 4.367738485336304 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 64 WarmUp Time 5.327124834060669 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 128 WarmUp Time 7.638680458068848 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 256 WarmUp Time 11.903432130813599 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Batch Size 512 WarmUp Time 20.694905996322632 secs ================


region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)
region ('/home/dhepe/work/internship', 715906)

================ Total time to run batch_performance_func 97.64325022697449 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/kerasxla_10_it_work_py.csv ================



=============== TF & Keras MobileNetV3L Benchmarked  && results saved to benchmark_results ===============

## onnxrt
=============== Using GPU: 1 ===============

2024-02-21 07:07:42.329149: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 11549 MB memory:  -> device: 0, name: NVIDIA TITAN X (Pascal), pci bus id: 0000:81:00.0, compute capability: 6.1
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.

================ Time to load the data onto CPU 0.044257164001464844 secs ================


2024-02-21 07:07:52.344365: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8700
2024-02-21 07:07:52.440842: I tensorflow/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2024-02-21 07:07:52.581276: I tensorflow/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory

=============== Top 3 predictions by TF_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402355), ('n04254680', 'soccer_ball', 0.21120338), ('n01871265', 'tusker', 0.10708933)]



================ Time to load the data onto CPU 0.020885467529296875 secs ================



=============== Top 3 predictions by Keras_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402355), ('n04254680', 'soccer_ball', 0.21120338), ('n01871265', 'tusker', 0.10708933)]



=============== TF Model Compile Time 7.493459939956665 secs && Keras Model Compile Time 3.3267009258270264 secs


=============== TF & Keras formats of MobileNetV3L Model Loading Completed ===============




=============== Saving ONNX MobileNetV3L Model ===============
Could not search for non-variable resources. Concrete function internal representation may have changed.
2024-02-21 07:07:53.621843: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 1
2024-02-21 07:07:53.622026: I tensorflow/core/grappler/clusters/single_machine.cc:361] Starting new session
2024-02-21 07:07:53.623327: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 11549 MB memory:  -> device: 0, name: NVIDIA TITAN X (Pascal), pci bus id: 0000:81:00.0, compute capability: 6.1
2024-02-21 07:07:54.774531: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 11549 MB memory:  -> device: 0, name: NVIDIA TITAN X (Pascal), pci bus id: 0000:81:00.0, compute capability: 6.1
2024-02-21 07:07:54.934516: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 1
2024-02-21 07:07:54.934720: I tensorflow/core/grappler/clusters/single_machine.cc:361] Starting new session
2024-02-21 07:07:54.935938: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 11549 MB memory:  -> device: 0, name: NVIDIA TITAN X (Pascal), pci bus id: 0000:81:00.0, compute capability: 6.1

=============== MobileNetV3L converted to onnx and saved at models_lib/onnx_models/MobileNetV3L in time 4.9318060874938965 secs ===============


=============== Benchmarking ONNX MobileNetV3L Model ===============


================ Time to load the data onto CPU 0.046860694885253906 secs ================



=============== Top 3 predictions by onnxrt ===============
[('n02504013', 'Indian_elephant', 0.3840248), ('n04254680', 'soccer_ball', 0.21120246), ('n01871265', 'tusker', 0.107089214)]



=============== OnnxRT Compile Time: 1.4018518924713135 secs ===============


================ Time to load the data onto CPU 0.398104190826416 secs ================



================ Batch Size 8 WarmUp Time 1.6570992469787598 secs ================


region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)

================ Batch Size 16 WarmUp Time 2.016859531402588 secs ================


region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)

================ Batch Size 32 WarmUp Time 7.644717693328857 secs ================


region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)

================ Batch Size 64 WarmUp Time 9.383137226104736 secs ================


region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)

================ Batch Size 128 WarmUp Time 13.129862546920776 secs ================


region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)

================ Batch Size 256 WarmUp Time 15.988042116165161 secs ================


region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)

================ Batch Size 512 WarmUp Time 31.215376615524292 secs ================


region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)
region ('/home/dhepe/work/internship', 687672)

================ Total time to run batch_performance_func 107.81231141090393 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/onnxrt_10_it_uni_py.csv ================



=============== ONNX MobileNetV3L Model Benchmarked && results saved to benchmark_results===============

## TRT
=============== Using GPU: 1 ===============


=============== Benchmarking FP32 TF-TRT MobileNetV3L Model ===============

2024-02-21 09:26:50.382483: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 11549 MB memory:  -> device: 0, name: NVIDIA TITAN X (Pascal), pci bus id: 0000:81:00.0, compute capability: 6.1

================ Time to load the data onto CPU 0.04562973976135254 secs ================


2024-02-21 09:26:51.306629: I tensorflow/compiler/tf2tensorrt/common/utils.cc:106] Linked TensorRT version: 8.4.3
2024-02-21 09:26:51.307942: I tensorflow/compiler/tf2tensorrt/common/utils.cc:108] Loaded TensorRT version: 8.6.1
2024-02-21 09:26:57.637717: I tensorflow/compiler/tf2tensorrt/convert/convert_nodes.cc:1329] [TF-TRT] Sparse compute capability: enabled.

=============== Top 3 predictions by trtfp32_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.384024), ('n04254680', 'soccer_ball', 0.21120393), ('n01871265', 'tusker', 0.10708859)]



=============== TRTFP32 Compile Time: 90.55075573921204 secs ===============


================ Time to load the data onto CPU 0.40692996978759766 secs ================



================ Batch Size 8 WarmUp Time 77.25600457191467 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 16 WarmUp Time 72.76937055587769 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 32 WarmUp Time 98.52594828605652 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 64 WarmUp Time 141.67463183403015 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 128 WarmUp Time 235.8474202156067 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 256 WarmUp Time 432.30977058410645 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 512 WarmUp Time 814.4404311180115 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Total time to run batch_performance_func 1918.2969465255737 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/TFTRTFP32_10_it_work_py.csv ================



=============== Benchmarking FP16 TF-TRT MobileNetV3L Model ===============


================ Time to load the data onto CPU 0.04318976402282715 secs ================



=============== Top 3 predictions by trtfp16_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402408), ('n04254680', 'soccer_ball', 0.21120387), ('n01871265', 'tusker', 0.10708867)]



=============== TRTFP16 Compile Time: 0.5045609474182129 secs ===============


================ Time to load the data onto CPU 0.38135552406311035 secs ================



================ Batch Size 8 WarmUp Time 0.3886682987213135 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 16 WarmUp Time 0.5338482856750488 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 32 WarmUp Time 1.096088171005249 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 64 WarmUp Time 3.7444560527801514 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 128 WarmUp Time 8.47697401046753 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 256 WarmUp Time 16.656837701797485 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Batch Size 512 WarmUp Time 33.254194259643555 secs ================


region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)
region ('/home/dhepe/work/internship', 719083)

================ Total time to run batch_performance_func 109.53168773651123 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/TFTRTFP16_10_it_work_py.csv ================



=============== TF-TRT MobileNetV3L Model Benchmarked && results saved to benchmark_results ===============


## Torch
=============== Using GPU: 1 ===============


=============== Saving Torch MobileNetV3L Model ===============


=============== MobileNetV3L converted to Torch and saved at models_lib/torch_models/MobileNetV3L in time 0.400662899017334 secs ===============


=============== Benchmarking Torch MobileNetV3L Model ===============


================ Time to load the data onto CPU 0.045998334884643555 secs ================



=============== Top 3 predictions by Torch ===============
[('n02412080', 'ram', 0.15587954), ('n02504013', 'Indian_elephant', 0.09531607), ('n02415577', 'bighorn', 0.08223383)]



=============== Torch Compile Time: 2.0260751247406006 secs ===============


================ Time to load the data onto CPU 0.1648387908935547 secs ================



================ Batch Size 8 WarmUp Time 0.7230291366577148 secs ================


region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)

================ Batch Size 16 WarmUp Time 0.7975242137908936 secs ================


region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)

================ Batch Size 32 WarmUp Time 1.3528447151184082 secs ================


region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)

================ Batch Size 64 WarmUp Time 2.893559694290161 secs ================


region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)

================ Batch Size 128 WarmUp Time 5.600443124771118 secs ================


region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)

================ Batch Size 256 WarmUp Time 5.619304418563843 secs ================


region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)
region ('/home/dhepe/work/internship', 715321)

================ Total time to run batch_performance_func 25.38673162460327 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/torch_10_it_work_py.csv ================



 =============== Torch MobileNetV3L Model Benchmarked ===============
