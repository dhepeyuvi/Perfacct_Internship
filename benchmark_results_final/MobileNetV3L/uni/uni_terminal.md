# TF & Keras
=============== Using GPU: 0 ===============

2024-02-21 05:24:49.472490: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 12971 MB memory:  -> device: 0, name: NVIDIA A2, pci bus id: 0000:82:00.0, compute capability: 8.6
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.

================ Time to load the data onto CPU 0.04291200637817383 secs ================


2024-02-21 05:25:05.024770: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8700

=============== Top 3 predictions by TF_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402322), ('n04254680', 'soccer_ball', 0.21120352), ('n01871265', 'tusker', 0.107089184)]



================ Time to load the data onto CPU 0.03075861930847168 secs ================



=============== Top 3 predictions by Keras_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402322), ('n04254680', 'soccer_ball', 0.21120352), ('n01871265', 'tusker', 0.107089184)]



=============== TF Model Compile Time 26.08983087539673 secs && Keras Model Compile Time 4.740495920181274 secs


=============== TF & Keras formats of MobileNetV3L Model Loading Completed ===============



=============== Benchmarking TensorFlow MobileNetV3L Model ===============


================ Time to load the data onto CPU 0.48580455780029297 secs ================



================ Batch Size 8 WarmUp Time 6.384582281112671 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 16 WarmUp Time 6.5648674964904785 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 32 WarmUp Time 7.059735536575317 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 64 WarmUp Time 8.632713317871094 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 128 WarmUp Time 11.116826295852661 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 256 WarmUp Time 20.321717977523804 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 512 WarmUp Time 40.66022300720215 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Total time to run batch_performance_func 302.36481261253357 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/tf_10_it_uni_py.csv ================



=============== Benchmarking Keras MobileNetV3L Model ===============

Benchmarking Keras MobileNetV3L Model

================ Time to load the data onto CPU 0.45188117027282715 secs ================



================ Batch Size 8 WarmUp Time 6.150623559951782 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 16 WarmUp Time 6.2877278327941895 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 32 WarmUp Time 6.750884771347046 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 64 WarmUp Time 7.73676609992981 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 128 WarmUp Time 10.534793376922607 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 256 WarmUp Time 19.2990984916687 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Batch Size 512 WarmUp Time 38.62923789024353 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)
region ('/mnt/beegfs/home/dhepe/work/internship', 16910)

================ Total time to run batch_performance_func 295.782395362854 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/keras_10_it_uni_py.csv ================



=============== TF & Keras MobileNetV3L Benchmarked  && results saved to benchmark_results ===============

## TF & Keras XLA
=============== Using GPU: 0 ===============

2024-02-21 09:19:08.001074: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 12971 MB memory:  -> device: 0, name: NVIDIA A2, pci bus id: 0000:82:00.0, compute capability: 8.6
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.

================ Time to load the data onto CPU 0.02817702293395996 secs ================


2024-02-21 09:19:23.576102: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8700

=============== Top 3 predictions by TF_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402322), ('n04254680', 'soccer_ball', 0.21120352), ('n01871265', 'tusker', 0.107089184)]



================ Time to load the data onto CPU 0.03202033042907715 secs ================



=============== Top 3 predictions by Keras_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402322), ('n04254680', 'soccer_ball', 0.21120352), ('n01871265', 'tusker', 0.107089184)]



=============== TF Model Compile Time 19.677757501602173 secs && Keras Model Compile Time 4.6948769092559814 secs


=============== TF & Keras formats of MobileNetV3L Model Loading Completed ===============



=============== Benchmarking TensorFlow MobileNetV3L Model ===============


================ Time to load the data onto CPU 0.4960029125213623 secs ================


2024-02-21 09:19:31.182043: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5569fedcd670 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-02-21 09:19:31.182140: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA A2, Compute Capability 8.6
2024-02-21 09:19:31.327968: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2024-02-21 09:21:09.943416: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

================ Batch Size 8 WarmUp Time 123.26613926887512 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 16 WarmUp Time 122.90442633628845 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 32 WarmUp Time 123.446120262146 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 64 WarmUp Time 127.96874761581421 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 128 WarmUp Time 138.98862075805664 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 256 WarmUp Time 157.6819610595703 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
2024-02-21 09:34:04.110923: W tensorflow/tsl/framework/bfc_allocator.cc:366] Garbage collection: deallocate free memory regions (i.e., allocations) so that we can re-allocate a larger region to avoid OOM due to memory fragmentation. If you see this message frequently, you are running near the threshold of the available device memory and re-allocation may incur great performance overhead. You may try smaller batch sizes to observe the performance impact. Set TF_ENABLE_GPU_GARBAGE_COLLECTION=false if you'd like to disable this feature.

================ Batch Size 512 WarmUp Time 201.74535131454468 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Total time to run batch_performance_func 1049.2341842651367 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/tfxla_10_it_uni_py.csv ================



=============== Benchmarking Keras MobileNetV3L Model ===============

Benchmarking Keras MobileNetV3L Model

================ Time to load the data onto CPU 0.4974677562713623 secs ================



================ Batch Size 8 WarmUp Time 25.926514387130737 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 16 WarmUp Time 26.102861166000366 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 32 WarmUp Time 28.047643184661865 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 64 WarmUp Time 29.49739146232605 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 128 WarmUp Time 31.749541997909546 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 256 WarmUp Time 36.35616493225098 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Batch Size 512 WarmUp Time 48.68704628944397 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)
region ('/mnt/beegfs/home/dhepe/work/internship', 21499)

================ Total time to run batch_performance_func 279.54700088500977 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/kerasxla_10_it_uni_py.csv ================



=============== TF & Keras MobileNetV3L Benchmarked  && results saved to benchmark_results ===============
## Onnxrt
=============== Using GPU: 0 ===============

2024-02-21 09:13:10.180504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 12971 MB memory:  -> device: 0, name: NVIDIA A2, pci bus id: 0000:82:00.0, compute capability: 8.6
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.

================ Time to load the data onto CPU 0.023236989974975586 secs ================


2024-02-21 09:13:26.385237: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8700

=============== Top 3 predictions by TF_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402322), ('n04254680', 'soccer_ball', 0.21120352), ('n01871265', 'tusker', 0.107089184)]



================ Time to load the data onto CPU 0.03161764144897461 secs ================



=============== Top 3 predictions by Keras_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402322), ('n04254680', 'soccer_ball', 0.21120352), ('n01871265', 'tusker', 0.107089184)]



=============== TF Model Compile Time 22.77018904685974 secs && Keras Model Compile Time 4.873161792755127 secs


=============== TF & Keras formats of MobileNetV3L Model Loading Completed ===============



=============== Benchmarking ONNX MobileNetV3L Model ===============


================ Time to load the data onto CPU 0.02801990509033203 secs ================



=============== Top 3 predictions by onnxrt ===============
[('n02504013', 'Indian_elephant', 0.3840248), ('n04254680', 'soccer_ball', 0.21120267), ('n01871265', 'tusker', 0.107089214)]



=============== OnnxRT Compile Time: 4.197840929031372 secs ===============


================ Time to load the data onto CPU 0.4388468265533447 secs ================



================ Batch Size 8 WarmUp Time 1.605288028717041 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)

================ Batch Size 16 WarmUp Time 2.433199405670166 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)

================ Batch Size 32 WarmUp Time 3.8461709022521973 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)

================ Batch Size 64 WarmUp Time 6.77294659614563 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)

================ Batch Size 128 WarmUp Time 12.69863748550415 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)

================ Batch Size 256 WarmUp Time 24.63998246192932 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)

================ Batch Size 512 WarmUp Time 48.87656617164612 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)
region ('/mnt/beegfs/home/dhepe/work/internship', 20671)

================ Total time to run batch_performance_func 152.1394579410553 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/onnxrt_10_it_uni_py.csv ================



=============== ONNX MobileNetV3L Model Benchmarked && results saved to benchmark_results===============
## TRT
=============== Using GPU: 0 ===============


=============== Benchmarking FP32 TF-TRT MobileNetV3L Model ===============

2024-02-21 08:14:58.480707: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13177 MB memory:  -> device: 0, name: NVIDIA A2, pci bus id: 0000:82:00.0, compute capability: 8.6

================ Time to load the data onto CPU 0.027800798416137695 secs ================


2024-02-21 08:14:59.658611: I tensorflow/compiler/tf2tensorrt/common/utils.cc:106] Linked TensorRT version: 8.4.3
2024-02-21 08:14:59.660492: I tensorflow/compiler/tf2tensorrt/common/utils.cc:108] Loaded TensorRT version: 8.5.3
2024-02-21 08:15:06.221039: I tensorflow/compiler/tf2tensorrt/convert/convert_nodes.cc:1329] [TF-TRT] Sparse compute capability: enabled.

=============== Top 3 predictions by trtfp32_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.3832771), ('n04254680', 'soccer_ball', 0.20888723), ('n01871265', 'tusker', 0.10830025)]



=============== TRTFP32 Compile Time: 96.42725133895874 secs ===============


================ Time to load the data onto CPU 0.4591343402862549 secs ================



================ Batch Size 8 WarmUp Time 63.697370290756226 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)

================ Batch Size 16 WarmUp Time 53.96844291687012 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)

================ Batch Size 32 WarmUp Time 67.31198906898499 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)

================ Batch Size 64 WarmUp Time 96.97826790809631 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)

================ Batch Size 128 WarmUp Time 155.62933945655823 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)

================ Batch Size 256 WarmUp Time 274.7408492565155 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)

================ Batch Size 512 WarmUp Time 518.8352072238922 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)

================ Total time to run batch_performance_func 1285.548363685608 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/TFTRTFP32_10_it_uni_py.csv ================



=============== Benchmarking FP16 TF-TRT MobileNetV3L Model ===============


================ Time to load the data onto CPU 0.04842972755432129 secs ================


2024-02-21 08:41:15.713781: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger TensorRT encountered issues when converting weights between types and that could affect accuracy.
2024-02-21 08:41:15.713835: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger Check verbose logs for the list of affected weights.
2024-02-21 08:41:15.713848: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 55 weights are affected by this issue: Detected subnormal FP16 values.
2024-02-21 08:41:15.714030: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 24 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.

=============== Top 3 predictions by trtfp16_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38521132), ('n04254680', 'soccer_ball', 0.21524152), ('n01871265', 'tusker', 0.104900345)]



=============== TRTFP16 Compile Time: 196.52820253372192 secs ===============


================ Time to load the data onto CPU 0.4956698417663574 secs ================


2024-02-21 08:43:14.124489: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger TensorRT encountered issues when converting weights between types and that could affect accuracy.
2024-02-21 08:43:14.124566: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger Check verbose logs for the list of affected weights.
2024-02-21 08:43:14.124598: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 55 weights are affected by this issue: Detected subnormal FP16 values.
2024-02-21 08:43:14.124784: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 24 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.

================ Batch Size 8 WarmUp Time 118.20445990562439 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
2024-02-21 08:45:06.104270: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger TensorRT encountered issues when converting weights between types and that could affect accuracy.
2024-02-21 08:45:06.104329: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger Check verbose logs for the list of affected weights.
2024-02-21 08:45:06.104355: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 55 weights are affected by this issue: Detected subnormal FP16 values.
2024-02-21 08:45:06.104563: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 24 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.

================ Batch Size 16 WarmUp Time 103.92092275619507 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
2024-02-21 08:47:06.757325: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger TensorRT encountered issues when converting weights between types and that could affect accuracy.
2024-02-21 08:47:06.757387: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger Check verbose logs for the list of affected weights.
2024-02-21 08:47:06.757399: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 55 weights are affected by this issue: Detected subnormal FP16 values.
2024-02-21 08:47:06.757537: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 24 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.

================ Batch Size 32 WarmUp Time 114.75322151184082 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
2024-02-21 08:49:40.673223: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger TensorRT encountered issues when converting weights between types and that could affect accuracy.
2024-02-21 08:49:40.673297: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger Check verbose logs for the list of affected weights.
2024-02-21 08:49:40.673318: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 55 weights are affected by this issue: Detected subnormal FP16 values.
2024-02-21 08:49:40.673494: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 24 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.

================ Batch Size 64 WarmUp Time 150.74453449249268 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
2024-02-21 08:53:21.726031: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger TensorRT encountered issues when converting weights between types and that could affect accuracy.
2024-02-21 08:53:21.726091: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger Check verbose logs for the list of affected weights.
2024-02-21 08:53:21.726104: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 55 weights are affected by this issue: Detected subnormal FP16 values.
2024-02-21 08:53:21.726281: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 24 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.

================ Batch Size 128 WarmUp Time 216.9086742401123 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
2024-02-21 08:59:11.444105: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger TensorRT encountered issues when converting weights between types and that could affect accuracy.
2024-02-21 08:59:11.444167: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger Check verbose logs for the list of affected weights.
2024-02-21 08:59:11.444183: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 55 weights are affected by this issue: Detected subnormal FP16 values.
2024-02-21 08:59:11.444364: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 24 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.

================ Batch Size 256 WarmUp Time 349.43058013916016 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
2024-02-21 09:09:22.341020: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger TensorRT encountered issues when converting weights between types and that could affect accuracy.
2024-02-21 09:09:22.341087: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger Check verbose logs for the list of affected weights.
2024-02-21 09:09:22.341101: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 55 weights are affected by this issue: Detected subnormal FP16 values.
2024-02-21 09:09:22.341277: W tensorflow/compiler/tf2tensorrt/utils/trt_logger.cc:83] TF-TRT Warning: DefaultLogger - 24 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.

================ Batch Size 512 WarmUp Time 617.6952607631683 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)
region ('/mnt/beegfs/home/dhepe/work/internship', 11219)

================ Total time to run batch_performance_func 1724.234623670578 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/TFTRTFP16_10_it_uni_py.csv ================



=============== TF-TRT MobileNetV3L Model Benchmarked && results saved to benchmark_results ===============
## Torch
==================== Suffix used with result files will be uni_py!! =============================
No read permissions for /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj.
No read permissions for /sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj.
No access to RAPL devices.

=============== Using GPU: 0 ===============


=============== Saving Torch MobileNetV3L Model ===============


=============== MobileNetV3L converted to Torch and saved at models_lib/torch_models/MobileNetV3L in time 0.882225513458252 secs ===============


=============== Benchmarking Torch MobileNetV3L Model ===============


================ Time to load the data onto CPU 0.07602834701538086 secs ================



=============== Top 3 predictions by Torch ===============
[('n02412080', 'ram', 0.15328614), ('n02504013', 'Indian_elephant', 0.09575488), ('n02415577', 'bighorn', 0.08093311)]



=============== Torch Compile Time: 8.162209510803223 secs ===============


================ Time to load the data onto CPU 0.15341639518737793 secs ================



================ Batch Size 8 WarmUp Time 1.0107030868530273 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)

================ Batch Size 16 WarmUp Time 1.410574197769165 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)

================ Batch Size 32 WarmUp Time 2.5779001712799072 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)

================ Batch Size 64 WarmUp Time 5.139158487319946 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)

================ Batch Size 128 WarmUp Time 9.931269884109497 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)

================ Batch Size 256 WarmUp Time 9.89268183708191 secs ================


region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)
region ('/mnt/beegfs/home/dhepe/work/internship', 23663)

================ Total time to run batch_performance_func 43.38271713256836 secs ================


================ Performance results saved to benchmark_results/MobileNetV3L/torch_10_it_uni_py.csv ================



 =============== Torch MobileNetV3L Model Benchmarked ===============