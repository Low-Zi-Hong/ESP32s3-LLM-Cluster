# Flashing and Preprocessing

- pls navigate to the python_tools directory to proceed
- flash the master program to master node and node program to rest of the esp.
  - This will create partition which is important for the next step.
  - Important notes: pls remove the rst pin and leave it disconnected/floating out before flashing. 

[[crop_token.py]]
- crop the tokenizer to 32K tokens
  - Note that 32K token is the maximum size of ESP32s3 can handle, larger than this will need more than 16MB of flash. If possible you can split the embeded matric to 2 equivalent matric and use 2 ESP32S3 to run it.

[[crop_model_weight.py]]
- crop the embed to fit 32K tokens
  
[[update_config.py]]
- update the config.json of the new quantized model

[[qat_158.py]]
- train the model to quantize to bit1.58
- Disclaimer: This script just partially train, the loss will hit somekind of 8.0. It will be just spitting out random tokens.. This can be verify through the [[run_model158.py]] script. If u have a stronger computer, just edit the script and train the model again. thanks :D

[[bit4_embedding.py]]
- pack the embedding layer to bit4 quantize

[[pack_tokenizer_bin.py]]
- pack the tokenizer to .bin file to flash to esp32

[[pack_model_bin.py]]
- pack the model to .bin file to flash to different esp32

- use [[flash_embed.bat]] and [[flash_tokenizer]] to flash the tokenizer and embed to master node
- use [[flash_layers.bat]] to flash the layers to respective esp32s3. Do manually change the COMX to ur COM and the layers to each and every esp node.
  - Posible to use the Device Manager on windows to check and keep tracking of the COM port.
  - The flashing process will take some time... pls be patient lol depends on your model size.


* I have other .py file which work as tools to spy the model and check the model file. Their usage can refer the py file itself and ask GPT if possible lol.

# Model Specifications

| Parameter | Value | Description / Note |
| :--- | :--- | :--- |
| **Base Architecture** | Qwen2-0.5B (Transformer Decoder) | Pruned & Quantized for MCU cluster |
| **Total Layers** | 24 Layers | Distributed as 4 layers per Compute Node (6 nodes) |
| **Hidden Size ($d_{model}$)** | 896 | Native Qwen2-0.5B hidden dimension |
| **Intermediate Size** | 4,864 | SwiGLU MLP intermediate dimension |
| **Attention Heads** | 14 Q-Heads / 2 KV-Heads (GQA) | Grouped-Query Attention with $d_{head}=64$ |
| **KV Dimension** | 128 | $2 \times 64$ per token step |
| **Vocabulary Size** | 32,000 (32K) | Cropped from original 151K BPE vocab |
| **Quantization Precision** | **1.58-bit (Ternary)** + **INT4** | BitNet $\{-1, 0, 1\}$ linear layers + INT4 Embeddings |
| **Weight Packing** | 4 ternary weights / byte (2-bit packed) | CPU/MCU 4-byte aligned SIMD layout |
| **Layer Memory Footprint** | ~3.82 MB / Layer | Packed ternary weights + FP16 scales & Norm |
| **Node Flash Allocation** | ~15.3 MB / Node (4 Layers) | Fits within 16MB SPI Flash partition |
| **Master Flash Allocation** | ~14.0 MB (INT4 Embed) + 64 KB (Final Norm) | Flash memory map on Master Node |
| **Context Window** | 512 Tokens | Dynamic KV Cache in Node PSRAM |

# Wiring Things Up

```text

=============================================================================
                          ⚡ COMMON GROUND (GND) ⚡
             (CRITICAL: All boards MUST share the same GND pin!)
=============================================================================
       |                         |                         |
+--------------+          +--------------+          +--------------+
|              |          |              |          |              |
| MASTER NODE  |          | COMPUTE NODE |          | LAST NODE    |
|              |          | (Node 1...N) |          | (Node N)     |
+--------------+          +--------------+          +--------------+

[ 1. SPI DATA PIPELINE (Daisy-Chain) ]
  (CH A - TX)               (CH B - RX)               (CH B - RX)
  GPIO 4 (CS)   ----------> GPIO 15 (CS)
  GPIO 5 (MOSI) ----------> GPIO 16 (MOSI)
  GPIO 7 (CLK)  ----------> GPIO 18 (CLK)
                            (CH A - TX)
                            GPIO 4 (CS)   ----------> GPIO 15 (CS)
                            GPIO 5 (MOSI) ----------> GPIO 16 (MOSI)
                            GPIO 7 (CLK)  ----------> GPIO 18 (CLK)

[ 2. CONTROL & SYNC LOOP ]
  GPIO 1 (RST)  ----------> EN / RST Pin 
                            GPIO 1 (RST)  ----------> EN / RST Pin
  GPIO 3 (Ready)<------------------------------------ GPIO 3 (Ready)
  
[ 3. STATUS INDICATOR ]
                            GPIO 8 -----------------> Status LED
                                                      GPIO 8 ---> LED

```

- This project use Serial Peripheral Interface (SPI) communicate between ESP and wire things up.
  - Each node consist 2 channel of SPI one master and one slave. Below are the default value:
  - for Channel A (acting as TX sending node):
    - Chip select   - GPIO 4
    - MOSI          - GPIO 5
    - CLK           - GPIO 7
  - for Channel B (acting as RX receive node):
    - Chip select   - GPIO 15
    - MOSI          - GPIO 16
    - CLK           - GPIO 18
  - To connect nodes, wire Node 1's Channel A directly to Node 2's Channel B (e.g., GPIO 5 to GPIO 16, GPIO 7 to GPIO 18).
  - fell free to check it out at [[spi_bus.h]] and alter the pin for ur convinient 
- The master node also will reset other ESP32 on initiate
  - on master node:
    - master's GPIO 1 sends reset signal to the Node's RST pin.
    - it will receive signal from node from GPIO 3
  - on other node board
    - node's GPIO 1 is also fire reset signal to rst pin of other node, the last node GPIO 1 is kept floating
    - GPIO 3 of the last node is connect back to the master node GPIO 3 to tell the master node all the node are ready.
- A LED pin on node board is prepared, GPIO 8 is connected to an LED and it will light up when the node is running.

# Powering

- In my case I am using 6 node and 1 master in total of 7 ESP32S3. The master board is connected to my computer; power through my computer. for the rest of the nodes I power using a DC power supply. On idle it runs on 5V 0.23A around 1.17W. While inferencing, the cluster consume 0.5V and 0.3A around 1.53W of power.
- It is possible to power the cluster through type-C cable. (if u have enough of cable lol)

# Running

- possible to use /bench to view the benchmark of the model
  - type /bench
  - type your prompt
  - benchmark will generate after your answer.
- You can change the max token length at [[main.cpp]] of master_board
  - edit the ```MAX_GEN_LEN``` and rebuild the project

# Scaling Up

- This project makes it possible to scale up. It can possibly run any model size just that u need more ESP32. Each ESP32s3 able to run 4 layers, if u have 6 then it is 24 layers. if u have 100 of them then it will be 400 layers of model. Just that the time of inference will linearly growth while u scale up. Currently each node spent 1.3s to inference.