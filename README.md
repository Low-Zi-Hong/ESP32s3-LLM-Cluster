# ESP32s3-LLM-Cluster

A distributed pipeline inference engine on multiple ESP32S3 running 1.58-bit (BitNet) Language model. 

![ESP32S3 boards](docs/images/Picture.JPG)

## Architecture

This project runs a sliced 0.5B LLM across a cluster of 7 ESP32s3. One act as master and others are node. The master node runs the tokenizer and embeding and the other attention layer and MLP ran on the nodes. The master and node communicate through high speed SPI Daisy-Chain.

```text
┌─────────────────────────────────────────────────────────┐
│                     MASTER NODE                         │
│                                                         │
│  [ Prompt ] ---> BPE Tokenizer                          │
│                       │                                 │
│                 Token Embedding                         │
│             (INT4, ~14MB in Flash)                      │
│                       │                                 │
│             (SPI CH A - TX to Node 1)                   │
└───────────────────────┬─────────────────────────────────┘
                        │ Hidden State Vector (FP32)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    COMPUTE NODE 1                       │
│             (SPI CH B - RX from Master)                 │
│                                                         │
│  ► Layer 0 to 3 (4x Transformer Blocks)                 │
│    • RMSNorm (FP16 scaled to FP32)                      │
│    • 1.58-bit Attention (Q, K, V, O proj) + RoPE        │
│    • KV Cache (PSRAM)                                   │
│    • 1.58-bit MLP (Gate, Up, Down proj)                 │
│                                                         │
│             (SPI CH A - TX to Node 2)                   │
└───────────────────────┬─────────────────────────────────┘
                        │
                       ... (Nodes 2 to 5)
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    COMPUTE NODE 6                       │
│             (SPI CH B - RX from Node 5)                 │
│                                                         │
│  ► Layer 20 to 23 (4x Transformer Blocks)               │
│    • Same 1.58-bit Architecture                         │
│                                                         │
│             (SPI CH A - TX back to Master)              │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                     MASTER NODE                         │
│             (SPI CH B - RX from Node 6)                 │
│                                                         │
│                 Final RMS Norm                          │
│             (FP16, 64KB in 'fnorm' partition)           │
│                       │                                 │
│         LM Head (Tied to INT4 Embeddings)               │
│                       │                                 │
│               Greedy Sampling                           │
│                       │                                 │
│  [ Output ] <--- Next Token ID                          │
└─────────────────────────────────────────────────────────┘
```

## Getting Started

pls refer [workflow guide](workflow.md) to start with the project.

## Project Structure

```text
.
├── README.md                   # Project documentation
├── workflow.md                 # Step-by-step flashing, model prep & wiring guide
├── .gitignore                  # Git ignore rules for build files & binaries
│
├── docs/                       
│   └── images/                 # Architecture diagrams and hardware photos
│
├── master_board/               # Firmware for the Master Node (ESP-IDF)
│   ├── main/
│   │   ├── main.cpp            # Master orchestrator, user I/O & BPE tokenizer
│   │   ├── embedding.cpp       # INT4 embedding lookup logic
│   │   ├── lm_head.cpp         # LM Head mapping and greedy sampling
│   │   └── spi_bus.cpp         # Master dual-channel SPI driver
│   ├── partitions.csv          # Custom partition table (token, model, fnorm)
│   └── CMakeLists.txt
│
├── node_firmware/              # Firmware for the Compute Nodes (ESP-IDF)
│   ├── main/
│   │   ├── main.cpp            # Node worker entry point & inference loop
│   │   ├── bitlinear.cpp       # 1.58-bit ternary linear layer implementation
│   │   ├── bitlinear_forward.S # Assembly optimized MAC ops for 1.58-bit
│   │   ├── qwen_attention.cpp  # Qwen Attention, RoPE & KV-Cache runtime
│   │   ├── lut_table.cpp       # Look-up tables for extreme optimization
│   │   └── spi_bus.cpp         # Daisy-chain SPI DMA receiver/transmitter
│   ├── partitions.csv          # Layer partition layout for Node
│   └── CMakeLists.txt
│
├── python_tools/               # PC-side quantization & preprocessing suite
    ├── crop_token.py           # Vocabulary pruning (scales down to 32K tokens)
    ├── crop_model_weight.py    # Embedding matrix slicing
    ├── qat_158.py              # BitNet QAT (Quantization-Aware Training) fine-tuning
    ├── bit4_embedding.py       # INT4 weight packer for embeddings
    ├── pack_tokenizer_bin.py   # Serializes tokenizer rules into ESP32 .bin
    ├── pack_model_bin.py       # Packs 1.58-bit layer chunks for physical alignment
    ├── look_model_structure.py # Debug tool for inspecting .safetensors
    └── flash_*.bat             # Multi-threaded fast flashing scripts
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspiration, related works, and references:
* [ESP-32-s3-Story-maker-LLM](https://github.com/harmansingh4163-ai/ESP-32-s3-Story-maker-LLM) - Inspiration for single-node quantized LLM deployment on ESP32-S3.
* [esp32s3-distributed-ai](https://github.com/wladimiravila/esp32s3-distributed-ai) - Inspiration for multi-node distributed AI architecture on microcontrollers.
* [BitNet](https://github.com/microsoft/BitNet) - 1.58-bit ternary quantization concept and architecture.