# AI-Challenge-Ovis2_5

**아주소중한딥러닝챌린지: 2025 SW중심대학사업 참여대학 공동 AI 경진대회 - Private LB 1st Place**

This repository contains the code and resources for the 1st place solution in the 2025 AI Challenge. The project focuses on fine-tuning the `AIDC-AI/Ovis2.5-9B` multimodal model to perform a variety of tasks including visual question answering, image captioning, text summarization, and mathematical reasoning.

## Handling Large Files

Due to GitHub's file size limitations, the model checkpoints, processed datasets, and downloaded images are not included in this repository. Please download them from the following links and place them in the appropriate directories.

*   **[Download Checkpoints (TBD)]()** -> Unzip and place the contents into `code/output_dft_loss/v0-20250828-230749/`
*   **[Download Converted Data](https://drive.google.com/file/d/1HMUs2MQiu8okNZOwsCdFG1c9_PBXifdL/view?usp=sharing)** -> Unzip and place the `.parquet` files into `data/converted/` (You don't need this directory for just an inference.)
*   **[Download Train/Test Ready Dataset](https://drive.google.com/file/d/1kMtcuviMQbMLHtbYl7gIs04fGIJ-tEDL/view?usp=sharing)** -> Unzip and place the `.jsonl` files into `data/dataset/`
*   **[Download Image Data](https://drive.google.com/file/d/1EjHIfgIpWW5jZF2FXDMe6mfWXWxysGlr/view?usp=sharing)** -> Unzip and place the image folders into `data/image/sample_images`, `data/image/train_images`, and `data/image/test_images`.

---

## Environment Setup

### System Specifications
The model was trained on a machine with the following specifications:
*   **OS:** Ubuntu 22.04.4 LTS (Jammy Jellyfish)
*   **Processor:** 16 vCore
*   **Memory:** 128 GiB
*   **Accelerator:** 1x NVIDIA A100 80GB PCIe
*   **Python:** 3.10.14

### Dependencies
All required Python packages are listed in `requirements.txt`. You can install the necessary packages using pip:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── code
│   ├── logs
│   ├── output_dft_loss
│   │   └── v0-20250828-230749
│   │       ├── checkpoint-2000 (download via Google Drive)
│   │       ├── ...
│   │       └── checkpoint-2733 (download via Google Drive)
│   └── Ovis2.5-9B.ipynb
├── data
│   ├── converted (download via Google Drive)
│   ├── dataset (download via Google Drive)
│   ├── image (download via Google Drive)
│   └── raw (download via Kaggle competition page)
├── environment.txt
├── prediction
│   └── submission_BEST.csv
├── README.md
└── requirements.txt
```

---

## How to Use

Follow these steps to replicate the training and inference pipeline.

### Step 1: Data Preparation
Place the raw competition data files into the `data/raw/` directory.

### Step 2: Run the Notebook
Open and execute the cells in the `code/Ovis2.5-9B.ipynb` Jupyter Notebook. The notebook is structured to handle the entire workflow from data preprocessing to inference.

#### a. Preprocessing
The initial cells in the notebook download all images from URLs provided in the raw dataset and save them locally to the `data/image/` directory. This step is crucial as it converts the dataset into a format where images are referenced by local file paths, which is necessary for training.

***Note:*** *This preprocessing step can be prone to failures due to network issues or broken URLs. If you encounter problems, please use the pre-processed and converted data files provided in the download links above.*

#### b. Custom Dataset Creation
The script then transforms the preprocessed data into a `.jsonl` format tailored for the MS-Swift training framework. Task-specific prompts are engineered to guide the model's responses for each of the five tasks (captioning, vqa, summarization, text_qa, math_reasoning). During the process, easy (short) text_qa questions and 30% of the text_qa questions will be dropped for stratification purpose. 10% of the whole dataset will be reserved for the validation set additionally.

#### c. Model Training
The training process is divided into two phases using the MS-Swift framework.

1.  **Supervised Fine-Tuning (SFT) with Cross-Entropy Loss:** The model is first fine-tuned on the custom dataset for one epoch. This phase adapts the model to the specific tasks and prompt formats.
2.  **Dynamic Fine-Tuning (DFT) with DFT Loss:** Training is then resumed from the best SFT checkpoint (in this case, `checkpoint-2600`) for a short duration. This phase uses DFT to improve model's CoT chain and penalize the model for selecting low probability token (more like temperature management during training).

The base model used is **AIDC-AI/Ovis2.5-9B**, a powerful multimodal model known for its native-resolution visual perception and strong reasoning capabilities.

#### d. Inference
After training, the final checkpoint (`checkpoint-2733`) is used to generate predictions on the test set. The notebook loads the fine-tuned LoRA adapters and runs inference, saving the results to a `.csv` file in the `prediction/` directory.

---

## Training Results

The model was trained for a total of 2733 steps.

### Final Training Metrics
| Metric | Value |
| :--- | :--- |
| **Total Train Runtime** | 8h 12m |
| **Min Train Loss** | 0.179783 at 2104/2733 steps |
Minimum train loss is not an evident indication of saturated model as we use randomized batch selection with potentially different number of examples per a task. In fact, the model was actually able to gradually reduce the train loss and improve train accuracy as seen on the smoothened train loss plot (w/ smoothing weight 0.9, Exponential Moving Average (EMA) was used.

### Performance Graphs

The following graphs from the training log illustrate the model's performance during the fine-tuning process.

| Training Loss | Token Acc. | Grad_Norm | LR |
| :---: | :---: | :---: | :---: |
| ![](code/output_dft_loss/v0-20250828-230749/images/train_loss.png) | ![](code/output_dft_loss/v0-20250828-230749/images/train_token_acc.png) | ![](code/output_dft_loss/v0-20250828-230749/images/train_grad_norm.png) | ![](code/output_dft_loss/v0-20250828-230749/images/train_learning_rate.png) |

### Model Details
The fine-tuning was performed using Rank-Stabilized Low-Rank Adaptation (rsLoRA) with PISSA initialization to efficiently adapt the large model.

*   **Total Parameters:** 9,350M ≈ 9.4B
*   **Trainable Parameters:** 174.59M (1.87%)
