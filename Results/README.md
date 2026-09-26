## Hierarchy
``` text
Results
├── README.md
├── Post-sim Result
│   ├── README.md
│   └── vgg16_cifar10_r980_constraint.dat
└── VLSI Implement
    ├── Area
    ├── Chip
    ├── LVS
    ├── Power
    └── Timing
```
## Post-sim Result
透過 Testbench 驗證 Fault-Count Threshold 的重新映射模式選擇功能，並以偵測到 46 筆故障資訊的案例，比對預期與實際選擇結果。

## Physical Implementation
| Specification | TSMC 90 nm 1P9M |
|---|---|
| Frequency | 200 MHz |
|Page Buffer Size| 16 KB |
| Timing Closure | Setup / Hold Met |
| Dynamic Power | 37.2271 mW |
| Cell Leakage Power | 2.1886 mW |
| Core Area | 835,875.489 μm² |
| Chip Area | 1,450,397.276 μm² | 
| LVS | Correct |
