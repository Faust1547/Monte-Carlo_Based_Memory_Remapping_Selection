# Post-sim Result Screenshot
共 46 筆錯誤資訊，依據區間選擇 Inter-bank 模式進行重新映射。

<img width="539" height="212" alt="Post-sim_Result" src="https://github.com/user-attachments/assets/204a52ea-c54c-4337-b248-f38a59003687" />

# Mode Selection BER-based Constraint Plot (Accuracy Target 0.98)
根據蒙地卡羅實驗最後歸納出的 BER-based 重新映射模式選擇的區間，在此以 VGG16 + CIFAR-10 為例。

<img width="1620" height="864" alt="selected_mode_target_0 980" src="https://github.com/user-attachments/assets/8b8b50df-d5b5-4c60-92e6-ab1fe6c186a8" />

# Mode Selection Fault-count Threshold (Accuracy Target 0.98)
BER-based 重新映射模式選擇區間根據 DNN 模型有效位元數換算的 Fault-count Threshold 區間。
``` text
000000 ~ 000001  ->  BASE 
000001 ~ 00053F  ->  INTER
00053F ~ 001A3B  ->  INTRA
```
