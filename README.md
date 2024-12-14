## Overview

This directory is designed for training a SD classifier. Below is the directories structure and how to run

---

## Directory Structure

```plaintext
code/
├── AutoEncoder/       # Files for our autoencoder architecture and training 
├── Data/              # Where to place data
├── Helper/            # Where helper functions and data prossessing code lives
├── report_assets/     # Figures and data from training
├── SignDistanceModel/ # Files for our sign distance model architecture and training 
└── README.md          # Repository overview (this file)
```

## Running the Project

1. Create a Data folder in the root of the code directory

2. Download the [ModelNet40](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset) and place in Data folder  

3. Run the repair script on the data
   ```bash
   python3 Helpers/repair_dataset.py
   ```

4. Train the autoencoder model
   ```bash
   python3 AutoEncoder/train.py
   ```

5. Download the [SDF dataset](https://drive.google.com/file/d/1L-h7KjIdfOEd_aui1a9p4v5c856TkNsJ/view?usp=sharing) from our google drive and place it in the Data folder

6. Train the SDF classification model
   ```bash
   python3 SignDistanceModel/train.py
   ```