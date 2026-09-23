## Hierarchy
``` text
Results
├── README.md
├── RTL Simulation Result
│   └── Post-sim_RTL_Result.png
└── VLSI Implement
    ├── Area
    ├── Chip
    ├── LVS
    ├── Power
    └── Timing
```
## RTL Simulation
透過 Testbench 驗證 Fault-count Throshold 選擇重新映射模式正確性與運算符合對應模式進行處理，展示以檢測出 46 筆故障資訊輸入為測試案例之 Post-simulation 正確選擇預期之重新映射模式結果。

## VLSI Implement
| Specification | TSMC 90 nm 1P9M | TSMC N16 ADFP |
|---|---|---|
| Frequency | 200 MHz | 1.25 GHz |
| Timing Closure | Setup / Hold Met | Setup / Hold Met|
| Dynamic Power | 17.7559 mW | 10.6 mW |
| Core Area | 557,343.647 μm² | 10,777.54 μm² |
| Chip Area | 1,044,749.730 μm² | 21,025.00 μm² |
| LVS | Correct | Correct |
