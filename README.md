# Tamil-ASR
LOW-RANK ADAPTATION ON WHISPER MODEL FOR TAMIL ASR


This repository contains the code and related materials for the project **LOW-RANK ADAPTATION ON WHISPER MODEL FOR TAMIL ASR**. 


## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Example of Data](#example-of-data)
5. [Experiment](#experiment)
6. [Installation and Usage](#installation-and-usage)

## Introduction

Advances in speech processing have led to significant improvements in technologies like virtual assistants, automated transcription services, and real-time translation tools. This work mainly focuses on extracting learnable features, enhancing accuracy, and reducing the computational complexity of Tamil ASR.

## Problem Statement

With limited work and dataset on dravidian languages, my project aims to finetune the publicly available Whisper Tamil Model using Low Rank Adaptation

## Dataset

The low-rank adaptation technique discussed in the previous chapter was implemented by finetuning the Whisper Small Tamil model using publicly available Common Voice Tamil dataset 13.

### Example of data:
{'audio': 
{'path':'/root/.cache/huggingface/datasets/downloads/extracted/26d01089476eefe8f1950b403fe01fb35c17249845931bb35227afa2fe442bdd/ta_train_0/common_voice_ta_26650298.mp3', 
'array': array([0., 0., 0., ..., 0., 0., 0.]), 
'sampling_rate': 16000},
 'sentence': 'அவரைப் பொதுமக்கள் விடாமல் பின்னாலேயே துரத்திக் கொண்டே ஓடினார்கள்.'}

## Experiment

LoRA is implemented on key and output projections, for fine-tuning Whisper Small Tamil model with 3M parameters.
 
![image](https://github.com/user-attachments/assets/8ecb4f7f-e37d-4a08-9956-3472effe0d58)

The publicly available Common Voice 13 Tamil speech datasets were used for finetuning the whisper model for Tamil transcription task. Implementation is based on the publicly available Huggingface Transformers3 code base. All the experiments are conducted on free available NVIDIA T4 GPUs in the Collab platform.

![image](https://github.com/user-attachments/assets/eb4278ff-44ee-42e9-ac20-5134f1d34223)
 
Typically, α is set as 64 and rank as 32. Besides, in Algorithm 1, we prune singular values every ∆T step (e.g., ∆T = 100) such that the pruned triplets can still get updated within these intervals and possibly reactivated in future iterations. The number of trainable parameters is controlled by the rank r and the number of adapted weight matrices n. The dropout rate is fixed as 0.05 for all experiments

 ![image](https://github.com/user-attachments/assets/4bfda4a7-f960-4a70-ab8c-fe12eceaa925)

### Training Parameters:
TrainOutput(
global_step=100, 
training_loss=0.1413337230682373, 
metrics={'train_runtime': 3350.6723, 
'train_samples_per_second': 0.239, 
'train_steps_per_second': 0.03, 
'total_flos': 2.34945183744e+17, 
'train_loss': 0.1413337230682373, 
'epoch': 0.018453589223103892})

### Evaluation Results:
The Word Error Rate by testing the finetuned model with the Common Voice 13 Tamil Test Data is obtained as 39.44160
 
![image](https://github.com/user-attachments/assets/7c1bde06-2d3f-45b0-8e3f-b5eb3340c6f9)


## Installation and Usage

### Prerequisites:
- Python 3.x
- Numpy
  ```bash
  !pip install -q transformers datasets librosa evaluate jiwer gradio bitsandbytes==0.37 accelerate
  !pip install -q git+https://github.com/huggingface/peft.git@main
  !apt-get install -y nvidia-cuda-toolkit
  !pip install bitsandbytes --upgrade   

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/syed-azim-git/Tamil-ASR/edit/main.git
