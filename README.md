# OmniSVG: A Unified Scalable Vector Graphics Generation Model

<div align="center">
<a href='https://arxiv.org/abs/2504.06263'><img src='https://img.shields.io/badge/arXiv-2504.06263-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://omnisvg.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/OmniSVG/OmniSVG1.1_8B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/OmniSVG"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset%20-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/OmniSVG/MMSVGBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Bench-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/spaces/OmniSVG/OmniSVG-3B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo%20-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://github.com/OpenVGLab/OmniSVG-train'><img src='https://img.shields.io/badge/Training-Code-blue?logo=github'></a>
</div>

## 🔥🔥🔥 News !!
- [2025/12/31] 👋 We have released the training code of OmniSVG. 
- [2025/12/22] We have updated **MMSVG-Icon** (264K→904K) and **MMSVG-Illustration** (66K→255K) datasets with enhanced captions and PNG previews! Check out [MMSVG-Icon](https://huggingface.co/datasets/OmniSVG/MMSVG-Icon) and [MMSVG-Illustration](https://huggingface.co/datasets/OmniSVG/MMSVG-Illustration).
- [2025/12/02] We have released the **OmniSVG1.1_8B** weights and updated **OmniSVG1.1_4B** model weights! Check out [OmniSVG1.1_8B](https://huggingface.co/OmniSVG/OmniSVG1.1_8B) and [OmniSVG1.1_4B](https://huggingface.co/OmniSVG/OmniSVG1.1_4B).
- [2025/12/02] We have released **MMSVGBench** benchmark dataset and evaluation code! Check out [MMSVGBench](https://huggingface.co/datasets/OmniSVG/MMSVGBench) and [Evaluation](https://github.com/OmniSVG/OmniSVG?tab=readme-ov-file#6-evaluation).
- [2025/09/18] OmniSVG is accepted to **NeurIPS 2025**🔥! See you in San Diego!
- [2025/07/22] 👋 We have released the Huggingface Demo. 🤗[Demo](https://huggingface.co/spaces/OmniSVG/OmniSVG-3B).
- [2025/07/22] 👋 We have released the inference code and model weight of MMSVG-Icon and MMSVG-Illustration dataset. 🤗[Weight](https://huggingface.co/OmniSVG/OmniSVG).
- [2025/04/09] 👋 Release MMSVG-Icon and MMSVG-Illustration 🤗[Dataset](https://huggingface.co/OmniSVG).
- [2025/04/09] 👋 Upload paper and init project. [Read](https://arxiv.org/abs/2504.06263)



## 🧩 Community Contributions
If you are developing / using OmniSVG in your projects, or you want to contribute to OmniSVG, please let us know 🎉.

- If you find data issues when using MMSVG dataset, please drop an issue in this [form](https://npqawhh9ht.feishu.cn/wiki/KHv2wDqAxiSV8skpkANcbmlwnqc?from=from_copylink).
- 👋 OmniSVG ComfyUI Plugin by [@smthemex](https://github.com/smthemex) [ComfyUI_OmniSVG](https://github.com/smthemex/ComfyUI_OmniSVG).

## 📑 Open-source Plan
- [x] Project Page & Technical Report
- [x] MMSVG-Icon and MMSVG-Illustration Dataset Release
- [x] Inference Code & Model Weight of MMSVG-Icon and MMSVG-Illustration Dataset
- [x] Online Demo (Gradio deployed on Huggingface)
- [x] Model Weight of OmniSVG1.1_8B Release
- [x] Model Weight of OmniSVG1.1_4B Release
- [x] MMSVGBench Benchmark & Evaluation Code Release
- [x] Training Code Release


## 1. Introduction

**OmniSVG** is the first family of end-to-end multimodal SVG generators that leverage pre-trained Vision-Language Models (VLMs), capable of generating complex and detailed SVGs, from simple icons to intricate anime characters. We also introduce MMSVG-2M, a multimodal dataset with two million richly annotated SVG assets, along with a standardized evaluation protocol for conditional SVG generation tasks. 


## 2. Models

### Model Variants

OmniSVG supports two model sizes with different base models:

| Model | Base Model | Base Vocab Size | Extended Vocab Size | Download | Size | Update |
|-------|------------|-----------------|---------------------|----------|------|--------|
| **OmniSVG1.1_8B** | Qwen2.5-VL-7B-Instruct | 152064 | 197000 | [HuggingFace](https://huggingface.co/OmniSVG/OmniSVG1.1_8B) | 17.2 GB | 2025-12-02 |
| **OmniSVG1.1_4B** | Qwen2.5-VL-3B-Instruct | 151936 | 197000 | [HuggingFace](https://huggingface.co/OmniSVG/OmniSVG1.1_4B) | 7.69 GB | 2025-12-02 |


## 3. Dependencies and Installation

### 3.1 Clone the Repository
```bash
git clone https://github.com/OpenVGLab/OmniSVG-train.git
cd OmniSVG-train
```

### 3.2 Create Conda Environment
```bash
conda create -n omnisvg python=3.10
conda activate omnisvg
```

### 3.3 Install Dependencies

#### System Dependencies

**macOS:**
```bash
brew install cairo
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install libcairo2 libcairo2-dev
```

#### Python Dependencies

Install PyTorch with CUDA 12.1 support:
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Install remaining dependencies:
```bash
pip install -r requirements.txt
```

Install picosvg for SVG preprocessing:
```bash
pip install picosvg
```

#### (Optional) Flash Attention 2

For faster training and inference, install Flash Attention 2:
```bash
pip install flash-attn --no-build-isolation
```


## 4. Inference

### Performance

|                   | GPU Memory | Time per 256/512/1024/2048/4096 tokens |
|-------------------|------------|----------------------------------------|
| OmniSVG1.1_8B     | 26G        | 5.38/9.02/20.11/40.34/98.11 seconds    |
| OmniSVG1.1_4B     | 17G        | 4.08/8.68/18.07/37.51/82.70 seconds    |

> **Note:** The inference time shown here is measured per OmniSVG SVG tokens, while the inference time reported in our paper is measured per XML code tokens for fair comparison with baseline methods.

### Download Model Weights

```bash
pip install huggingface-hub

# Download OmniSVG1.1-8B
huggingface-cli download OmniSVG/OmniSVG1.1_8B --local-dir /PATH/TO/OmniSVG1.1_8B

# Download OmniSVG1.1-4B
huggingface-cli download OmniSVG/OmniSVG1.1_4B --local-dir /PATH/TO/OmniSVG1.1_4B
```

### Text-to-SVG Generation

```bash
# Using 8B model (default)
python inference.py --task text-to-svg --input prompts.txt --output ./output_text --save-all-candidates

# Using 4B model
python inference.py --task text-to-svg --input prompts.txt --output ./output_text --model-size 4B --save-all-candidates

# Custom generation parameters
python inference.py --task text-to-svg --input prompts.txt --output ./output_text \
    --temperature 0.5 --top-p 0.9 --top-k 50 --repetition-penalty 1.05
```

### Image-to-SVG Generation

```bash
python inference.py --task image-to-svg --input ./examples --output ./output_image --save-all-candidates
```

### Interactive Demo

```bash
# Local deployment
python app.py
```

Or try our [Online Demo on Hugging Face Spaces](https://huggingface.co/spaces/OmniSVG/OmniSVG-3B).



## 5. Training

### 5.1 Data Preparation

#### Data Directory Structure

```
data/
├── train_meta.csv      # Training metadata
├── val_meta.csv        # Validation metadata
├── svg/                # SVG files
│   ├── 000001.svg
│   ├── 000002.svg
│   └── ...
└── png/                # Rendered PNG images
    ├── 000001.png
    ├── 000002.png
    └── ...
```

#### Metadata CSV Format

```csv
id,desc_en,detail,keywords,len_pix
000001,"A red apple","A detailed description of a red apple with stem and leaf","apple,fruit,red",256
000002,"Blue star","A simple five-pointed blue star","star,blue,shape",128
```

#### Download MMSVG Dataset

```bash
# Download illustration dataset
huggingface-cli download OmniSVG/MMSVG-Illustration --repo-type dataset --local-dir ./data/illustration

# Download icon dataset
huggingface-cli download OmniSVG/MMSVG-Icon --repo-type dataset --local-dir ./data/icon
```

Or use the built-in data downloader:
```bash
python -m utils.data_downloader --output_dir ./data --datasets illustration icon
```

### 5.2 SVG Data Preprocessing

Before training, SVG files need to be preprocessed to ensure compatibility with the model. We provide a preprocessing script that:

- **Simplifies SVG syntax** using `picosvg` (removes unnecessary groups, transforms, rect elements, etc.)
- **Normalizes dimensions** to 200×200 pixels
- **Optionally simplifies paths** for more efficient tokenization

#### Single File Processing

```bash
# Basic preprocessing
python preprocess_svg.py --input file.svg --output processed.svg

# With custom dimensions
python preprocess_svg.py --input file.svg --output processed.svg --width 200 --height 200

```

#### Batch Directory Processing

```bash
# Process all SVGs in a directory
python preprocess_svg.py --input_dir ./raw_svgs --output_dir ./processed_svgs

```

#### Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` / `-i` | - | Single SVG file to process |
| `--output` / `-o` | - | Output path for single file (auto-generated if not specified) |
| `--input_dir` | - | Directory containing SVG files for batch processing |
| `--output_dir` | - | Output directory for batch processing |
| `--scale` | 1.0 | SVG zoom scale factor |
| `--width` | 200 | Output SVG width in pixels |
| `--height` | 200 | Output SVG height in pixels |
| `--simplify` | False | Enable path simplification (arcs, heuristics, splitting) |
| `--max_dist` | 5 | Maximum path length before splitting (used with `--simplify`) |

#### What the Preprocessing Does

1. **picosvg preprocessing**: Converts complex SVG features to simple paths
   - Removes `<g>` groups and flattens structure
   - Converts `<rect>`, `<circle>`, `<ellipse>` to `<path>` elements
   - Removes transforms by baking them into coordinates
   - Strips unsupported attributes and elements

2. **Normalization**: Scales and centers the SVG to fit within the target dimensions (200×200 by default)

3. **Path simplification** (optional):
   - Simplifies arc commands
   - Applies heuristic simplification
   - Splits long paths for better tokenization

### 5.3 Configuration

The training system uses YAML configuration files located in the `configs/` directory:

- `configs/tokenization.yaml` - Model-specific tokenization settings
- `configs/train_config.yaml` - Training hyperparameters

#### Key Configuration Options

```yaml
# configs/train_config.yaml
model:
  size: "4B"                    # Model size: "4B" or "8B"
  use_flash_attn: true          # Enable Flash Attention 2

data:
  data_dir: "./data"            # Data directory path
  max_seq_length: 2048          # Maximum SVG sequence length, decrease if cuda out pf memory

training:
  learning_rate: 1.0e-5
  epochs: 100
  gradient_accumulation_steps: 4
```

### 5.4 Training Commands

#### Using run.sh (Recommended)

Edit `run.sh` to configure your settings:

```bash
# Configuration in run.sh
MODEL_SIZE="4B"          # "4B" or "8B"
USE_FLASH_ATTN="true"    # Enable Flash Attention
NUM_GPUS=8               # Number of GPUs
BATCH_SIZE=4             # Batch size per GPU
DATA_DIR="./data"        # Data directory

# Run training
bash run.sh
```

#### Using Command Line

```bash
# Train 4B model
accelerate launch --num_processes 8 --mixed_precision bf16 \
    train.py \
    --model_size 4B \
    --use_flash_attn \
    --data_dir ./data \
    --output_dir ./output \
    --batch_size 4

# Train 8B model
accelerate launch --num_processes 8 --mixed_precision bf16 \
    train.py \
    --model_size 8B \
    --use_flash_attn \
    --data_dir ./data \
    --output_dir ./output \
    --batch_size 2
```

#### Download and Use HuggingFace Data

```bash
accelerate launch train.py \
    --model_size 4B \
    --use_flash_attn \
    --use_hf_data \
    --datasets illustration icon \
    --data_dir ./data
```

#### Resume from Checkpoint

```bash
# Resume from official OmniSVG checkpoint (auto-download)
accelerate launch train.py \
    --model_size 4B \
    --resume_from_checkpoint auto \
    --data_dir ./data

# Resume from local checkpoint
accelerate launch train.py \
    --model_size 4B \
    --resume_from_checkpoint /path/to/checkpoint \
    --data_dir ./data
```

### 5.5 Training Examples


```bash
# Single GPU training (for debugging)
python train.py --model_size 4B --use_flash_attn --data_dir ./data --batch_size 1

# Multi-GPU training with DeepSpeed
accelerate launch --config_file ./configs/zero_stage2.yaml \
    train.py --model_size 8B --use_flash_attn --data_dir ./data

# List available models and datasets
python train.py --list_models
python train.py --list_datasets
```

### 5.6 Training Output

Checkpoints and logs are saved to the output directory:

```
output/omnisvg_4b_YYYYMMDD_HHMMSS/
├── config.yaml           # Saved configuration
├── args.json             # Command line arguments
├── logs/                 # TensorBoard logs
├── step_3000/            # Checkpoint at step 3000
├── step_6000/            # Checkpoint at step 6000
└── best_model/           # Best validation checkpoint
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir ./output/omnisvg_4b/logs
```


## 6. Evaluation

We provide **MMSVGBench** for standardized evaluation of SVG generation models.

**Download MMSVGBench:**
```bash
huggingface-cli download OmniSVG/MMSVGBench --repo-type dataset --local-dir /PATH/TO/MMSVGBench
```

### Benchmark Overview

MMSVGBench is a **purely synthetic benchmark** where all prompts and images are generated using GPT models, ensuring the data is **unseen** during model training for fair generalization evaluation.

| Task | Complexity Level | Samples | Description |
|------|------------------|---------|-------------|
| Text-to-SVG | Icon | 150 | Simple icons (1-2 elements) |
| Text-to-SVG | Illustration | 150 | Complex illustrations (1-3 interacting elements) |
| Image-to-SVG | Icon | 150 | GPT-4o generated icon images |
| Image-to-SVG | Illustration | 150 | GPT-4o generated illustration images |

The evaluation code is available in the `metrics` directory. For more details, see [MMSVGBench](https://huggingface.co/datasets/OmniSVG/MMSVGBench/blob/main/README.md).


## 7. Project Structure

```
OmniSVG/
├── configs/
│   ├── tokenization.yaml      # Tokenization config for 4B/8B models
│   └── train_config.yaml      # Training hyperparameters
├── utils/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── dataset.py             # Dataset and data loading
│   └── data_downloader.py     # HuggingFace data downloading
├── model/
│   └── decoder.py             # Model architecture
├── metrics/                   # Evaluation metrics
├── train.py                   # Training script
├── inference.py               # Inference script
├── preprocess_svg.py             # SVG data preprocessing script
├── app.py                     # Gradio demo
├── run.sh                     # Training launch script
└── requirements.txt
```


## 8. License

OmniSVG is licensed under the [**Apache License 2.0**](https://www.apache.org/licenses/LICENSE-2.0), while MMSVG dataset is under [**Creative Commons Attribution Non Commercial Share Alike 4.0 License**](https://spdx.org/licenses/CC-BY-NC-SA-4.0).


## Citation

```bibtex
@article{yang2025omnisvg,
  title={OmniSVG: A Unified Scalable Vector Graphics Generation Model}, 
  author={Yiying Yang and Wei Cheng and Sijin Chen and Xianfang Zeng and Jiaxu Zhang and Liao Wang and Gang Yu and Xinjun Ma and Yu-Gang Jiang},
  journal={arXiv preprint arxiv:2504.06263},
  year={2025}
}
```


## Acknowledgments

We thank the following excellent open-source works:

- [IconShop](https://icon-shop.github.io/): The first advanced work that leverages LLMs to generate monochrome, icon-level SVGs.
- [LLM4SVG](https://arxiv.org/abs/2412.11102): Treats SVG coordinates as number strings for higher spatial accuracy.
- [StarVector](https://starvector.github.io/): Equips LLM with an image encoder for Image-to-SVG generation.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=OpenVGLab/OmniSVG-train&type=Date)](https://www.star-history.com/#OpenVGLab/OmniSVG-train&Date)