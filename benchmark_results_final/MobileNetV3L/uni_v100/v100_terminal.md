# Terminal
## TF & Keras V100
=============== Using GPU: 0 ===============
================ Time to load the data onto CPU 0.01851677894592285 secs ================
=============== Top 3 predictions by TF_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402352), ('n04254680', 'soccer_ball', 0.21120337), ('n01871265', 'tusker', 0.107089326)]
================ Time to load the data onto CPU 0.024542570114135742 secs ================
=============== Top 3 predictions by Keras_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402352), ('n04254680', 'soccer_ball', 0.21120337), ('n01871265', 'tusker', 0.107089326)]
=============== TF Model Compile Time 11.192299365997314 secs && Keras Model Compile Time 13.316596269607544 secs
=============== TF & Keras formats of MobileNetV3L Model Loading Completed ===============
=============== Benchmarking TensorFlow MobileNetV3L Model ===============
================ Time to load the data onto CPU 0.3491792678833008 secs ================
================ Batch Size 8 WarmUp Time 4.903225421905518 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 16 WarmUp Time 4.943378210067749 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 32 WarmUp Time 5.066892623901367 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 64 WarmUp Time 6.413240909576416 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 128 WarmUp Time 7.965727806091309 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 256 WarmUp Time 11.052647590637207 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 512 WarmUp Time 17.13770318031311 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Total time to run batch_performance_func 210.18677592277527 secs ================
================ Performance results saved to benchmark_results/MobileNetV3L/tf_10_it_uni_py.csv ================



=============== Benchmarking Keras MobileNetV3L Model ===============
Benchmarking Keras MobileNetV3L Model
================ Time to load the data onto CPU 0.3499736785888672 secs ================
================ Batch Size 8 WarmUp Time 4.755535840988159 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 16 WarmUp Time 4.8616743087768555 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 32 WarmUp Time 5.144608497619629 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 64 WarmUp Time 6.175721168518066 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 128 WarmUp Time 7.711963891983032 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 256 WarmUp Time 10.80111289024353 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Batch Size 512 WarmUp Time 16.823407888412476 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 581363)
================ Total time to run batch_performance_func 206.02983117103577 secs ================


## TF & Keras XLA V100
=============== Using GPU: 0 ===============
================ Time to load the data onto CPU 0.018967151641845703 secs ================
=============== Top 3 predictions by TF_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402352), ('n04254680', 'soccer_ball', 0.21120337), ('n01871265', 'tusker', 0.107089326)]
================ Time to load the data onto CPU 0.0241701602935791 secs ================
=============== Top 3 predictions by Keras_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402352), ('n04254680', 'soccer_ball', 0.21120337), ('n01871265', 'tusker', 0.107089326)]
=============== TF Model Compile Time 9.15047550201416 secs && Keras Model Compile Time 13.303657293319702 secs
=============== TF & Keras formats of MobileNetV3L Model Loading Completed ===============
=============== Benchmarking TensorFlow MobileNetV3L Model ===============
================ Time to load the data onto CPU 0.32565927505493164 secs ================
================ Batch Size 8 WarmUp Time 6.426370620727539 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 16 WarmUp Time 4.441643953323364 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 32 WarmUp Time 4.910478115081787 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 64 WarmUp Time 6.607644081115723 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 128 WarmUp Time 10.2920663356781 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 256 WarmUp Time 17.070328950881958 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 512 WarmUp Time 30.622048377990723 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Total time to run batch_performance_func 120.00288248062134 secs ================
================ Performance results saved to benchmark_results/MobileNetV3L/tfxla_10_it_uni_py.csv ================

=============== Benchmarking Keras MobileNetV3L Model ===============
Benchmarking Keras MobileNetV3L Model
================ Time to load the data onto CPU 0.33551645278930664 secs ================
================ Batch Size 8 WarmUp Time 2.4900550842285156 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 16 WarmUp Time 2.717503547668457 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 32 WarmUp Time 3.032484292984009 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 64 WarmUp Time 4.022627353668213 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 128 WarmUp Time 5.546788215637207 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 256 WarmUp Time 8.727567672729492 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Batch Size 512 WarmUp Time 14.58624005317688 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 590069)
================ Total time to run batch_performance_func 81.26945209503174 secs ================
================ Performance results saved to benchmark_results/MobileNetV3L/kerasxla_10_it_uni_py.csv ================
=============== TF & Keras MobileNetV3L Benchmarked  && results saved to benchmark_results ===============

## Onnxrt V100
=============== Using GPU: 0 ===============
================ Time to load the data onto CPU 0.019052505493164062 secs ================
=============== Top 3 predictions by TF_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402352), ('n04254680', 'soccer_ball', 0.21120337), ('n01871265', 'tusker', 0.107089326)]
================ Time to load the data onto CPU 0.021513938903808594 secs ================
=============== Top 3 predictions by Keras_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402352), ('n04254680', 'soccer_ball', 0.21120337), ('n01871265', 'tusker', 0.107089326)]
=============== TF Model Compile Time 8.729480981826782 secs && Keras Model Compile Time 13.254267930984497 secs


=============== TF & Keras formats of MobileNetV3L Model Loading Completed ===============
=============== Saving ONNX MobileNetV3L Model ===============
=============== MobileNetV3L converted to onnx and saved at models_lib/onnx_models/MobileNetV3L in time 4.155555486679077 secs ===============
=============== Benchmarking ONNX MobileNetV3L Model ===============
================ Time to load the data onto CPU 0.020626068115234375 secs ================
=============== Top 3 predictions by onnxrt ===============
[('n02504013', 'Indian_elephant', 0.3840251), ('n04254680', 'soccer_ball', 0.21120231), ('n01871265', 'tusker', 0.107089356)]

=============== OnnxRT Compile Time: 4.179460048675537 secs ===============
================ Time to load the data onto CPU 0.3350636959075928 secs ================
================ Batch Size 8 WarmUp Time 1.947298288345337 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 596022)
================ Batch Size 16 WarmUp Time 2.5033648014068604 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 596022)
================ Batch Size 32 WarmUp Time 3.0496304035186768 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 596022)
================ Batch Size 64 WarmUp Time 3.888066530227661 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 596022)
================ Batch Size 128 WarmUp Time 5.6300530433654785 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 596022)
================ Batch Size 256 WarmUp Time 9.651694297790527 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 596022)
================ Batch Size 512 WarmUp Time 16.18315100669861 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 596022)
================ Total time to run batch_performance_func 68.45357632637024 secs ================
================ Performance results saved to benchmark_results/MobileNetV3L/onnxrt_10_it_uni_py.csv ================
=============== ONNX MobileNetV3L Model Benchmarked && results saved to benchmark_results===============

## TRT FP 32 & 16 v100
=============== Using GPU: 0 ===============
=============== Saving FP32 TF-TRT MobileNetV3L Model ===============
=============== MobileNetV3L converted to TF-TRT FP32 precision and saved at models_lib/trt_models/MobileNetV3L_TFTRT_FP32 in time 7.68590784072876 secs ===============
=============== Benchmarking FP32 TF-TRT MobileNetV3L Model ===============
================ Time to load the data onto CPU 0.02854323387145996 secs ================
=============== Top 3 predictions by trtfp32_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.38402387), ('n04254680', 'soccer_ball', 0.21120405), ('n01871265', 'tusker', 0.10708865)]

=============== TRTFP32 Compile Time: 75.40795254707336 secs ===============
================ Time to load the data onto CPU 0.34550929069519043 secs ================
================ Batch Size 8 WarmUp Time 49.664515018463135 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 16 WarmUp Time 36.53905940055847 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 32 WarmUp Time 39.410847663879395 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 64 WarmUp Time 45.033101320266724 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 128 WarmUp Time 57.020777463912964 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 256 WarmUp Time 79.29660606384277 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 512 WarmUp Time 125.25580549240112 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Total time to run batch_performance_func 479.9645345211029 secs ================
================ Performance results saved to benchmark_results/MobileNetV3L/TFTRTFP32_10_it_uni_py.csv ================
=============== Saving FP16 TF-TRT MobileNetV3L Model ===============


=============== MobileNetV3L converted to TF-TRT FP16 precision and saved at models_lib/trt_models/MobileNetV3L_TFTRT_FP16 in time 6.983251333236694 secs ===============
=============== Benchmarking FP16 TF-TRT MobileNetV3L Model ===============
================ Time to load the data onto CPU 0.018540143966674805 secs ================
=============== Top 3 predictions by trtfp16_MobileNetV3L ===============
[('n02504013', 'Indian_elephant', 0.3823841), ('n04254680', 'soccer_ball', 0.21449803), ('n01871265', 'tusker', 0.10618422)]
=============== TRTFP16 Compile Time: 88.64108991622925 secs ===============
================ Time to load the data onto CPU 0.32892823219299316 secs ================
================ Batch Size 8 WarmUp Time 65.42250037193298 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 16 WarmUp Time 51.609381675720215 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 32 WarmUp Time 51.72873592376709 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 64 WarmUp Time 58.035547971725464 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 128 WarmUp Time 66.41375994682312 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 256 WarmUp Time 86.06984448432922 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Batch Size 512 WarmUp Time 128.765319108963 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 603219)
================ Total time to run batch_performance_func 554.1874232292175 secs ================
================ Performance results saved to benchmark_results/MobileNetV3L/TFTRTFP16_10_it_uni_py.csv ================
=============== TF-TRT MobileNetV3L Model Benchmarked && results saved to benchmark_results ===============

## Torch v100
=============== Using GPU: 0 ===============
=============== Saving Torch MobileNetV3L Model ===============
=============== MobileNetV3L converted to Torch and saved at models_lib/torch_models/MobileNetV3L in time 1.1041498184204102 secs ===============
=============== Benchmarking Torch MobileNetV3L Model ===============
================ Time to load the data onto CPU 0.022124290466308594 secs ================
=============== Top 3 predictions by Torch ===============
[('n02412080', 'ram', 0.15587981), ('n02504013', 'Indian_elephant', 0.09531617), ('n02415577', 'bighorn', 0.08223365)]
=============== Torch Compile Time: 3.9291763305664062 secs ===============
================ Time to load the data onto CPU 0.11188459396362305 secs ================
================ Batch Size 8 WarmUp Time 0.7276136875152588 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 620977)
================ Batch Size 16 WarmUp Time 0.49582505226135254 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 620977)
================ Batch Size 32 WarmUp Time 0.7260885238647461 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 620977)
================ Batch Size 64 WarmUp Time 1.3987085819244385 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 620977)
================ Batch Size 128 WarmUp Time 2.6273887157440186 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 620977)
================ Batch Size 256 WarmUp Time 2.533320665359497 secs ================
region ('/mnt/beegfs/home/dhepe/work/internship', 620977)
================ Total time to run batch_performance_func 17.719592332839966 secs ================
================ Performance results saved to benchmark_results/MobileNetV3L/torch_10_it_uni_py.csv ================
 =============== Torch MobileNetV3L Model Benchmarked ===============

 # JNB