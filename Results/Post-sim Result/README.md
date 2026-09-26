# Post-sim Result Screenshot
### 1. Memory Initialization & Instruction Execution

<img width="414" height="235" alt="image" src="https://github.com/user-attachments/assets/ee60c1be-8c2b-4774-9395-59e8cb1fd479" />


### 2. Final Data Memory
<img width="321" height="697" alt="image" src="https://github.com/user-attachments/assets/24638c93-8d06-4c84-bae2-590c8d27e3d4" />


### 3. Verification Summary
<img width="545" height="111" alt="image" src="https://github.com/user-attachments/assets/9de9ee7e-3dd1-4645-88a0-60ed06edaf4e" />

# Load-Use Hazard Test Case
``` text
lw $1, 0($0)
add  $2, $1, $1
sw $2, 4($0)
```
## Load-Use Hazard Verification
本測試使用連續的 Load、Add 與 Store 指令，驗證 Hazard Detection 與 Forwarding 機制能否正確處理資料相依，避免後續指令使用尚未更新的暫存器資料。
|Dependency|	Expected Behavior|
|---|---|
|lw → add	|偵測 Load-use Hazard，插入必要的 Stall，待資料可用後透過 Forwarding 提供給 ALU|
|add → sw	|將更新後的 $2 傳遞至 Store 指令，確保寫入資料正確|


## Input Data
| Data Type | Data Name |
|---|---|
| Instruction input file | `IM_32bit.txt ` |
| Data Memory input file | `DM_32bit.txt ` |

## Verification Data
| Data Type | Data Name |
|---|---|
| Data Memory  | `expected_DM.dat ` |
| Register File | `expected_RF.dat` |
| Write Back Data  | `expected_WB.dat` |
| Store Word Data | `expected_SW.dat` |

## Output Data
| Data Type | Data Name |
|---|---|
| Data Memory Output file | `DM_out.dat` |
| Register File Output file | `RF_out.dat` |
| Write Back Data Output file  | `WB_out.dat` |
| Store Word Data Output file| `SW_out.dat` |
