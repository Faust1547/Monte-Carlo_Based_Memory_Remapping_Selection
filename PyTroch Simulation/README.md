# 軟體端 (Python) - 蒙地卡羅模擬實驗 #

## 執行流程： ##
1. Train Models
2. Run Monte Carlo Simulation
3. Generate Constraint Data

## Train Models ##
用於訓練基於 CIFAR-10 資料集的 AlexNet 與 VGG16 模型，將會產出 .pth 檔案。
* AlexNet_CIFAR-10.py 
* VGG16_CIFAR-10.py

## Monte Carlo Simulation ##
.py 檔案用於執行蒙地卡羅模擬實驗，.json 檔案用於設置實驗參數，例如：Fault model、Fault Distribution、Bit Error Score等
* AlexNet
  * alexnet_global_mode_accuracy_experiment.py 
  * alexNet_global_mode_accuracy_config.json   
* VGG16
  * vgg16_global_mode_accuracy_experiment.py   
  * vgg16_global_mode_accuracy_config.json

## Monte Carlo Simulation ##
.py 檔案用於執行蒙地卡羅模擬實驗，.json 檔案用於設置實驗參數，例如：Fault model、Fault Distribution、Bit Error Score等
* AlexNet
  * alexnet_global_mode_accuracy_experiment.py 
  * alexNet_global_mode_accuracy_config.json   
* VGG16
  * vgg16_global_mode_accuracy_experiment.py   
  * vgg16_global_mode_accuracy_config.json    
