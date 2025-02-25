# ProteinRPN

## Overview

ProteinRPN is a deep learning-based framework designed for predicting Gene Ontology (GO) terms of proteins. It achieves this by first extracting and localizing functional residues in proteins, leveraging a graph-based region proposal network. The model effectively refines functional regions within protein residue graphs to improve prediction accuracy across three GO subontologies:

- **Biological Process (BP)**
- **Molecular Function (MF)**
- **Cellular Component (CC)**

## Dataset

The dataset required for training can be found [here](https://drive.google.com/file/d/1jyDd4yTVOJBL19vXZeLutV7PBpcCo4Cd/view?usp=sharing). Make sure to download and place it in an appropriate directory.

## Installation & Dependencies

Ensure you have the following dependencies installed before running the model:

```bash
pip install -r requirements.txt
```

## Usage

### Setting Up

Modify `GO_combined_train.py` according to your needs:

- **Task Selection**: Set the `task` variable to `'bp'`, `'mf'`, or `'cc'` depending on the GO term to predict.
- **Model Naming**: Change the `suffix` variable to specify the model name during saving.
- **Device Configuration**: Set `device` to `'cuda'` for GPU or `'cpu'` for CPU.
- **Dataset Path**: Update the dataset path inside the script to match the local directory where datasets are stored. Alternatively, modify the `PATH` variable to point to the dataset directory.

### Training the Model

Run the following command to train the model:

```bash
python GO_combined_train.py --task bp --suffix bp_test --device cuda --contrast True --batch_size 48 --model_save_path '.' --polynormer False
```

#### Arguments:

- `--task`: Specifies the GO term subontology (`bp`, `mf`, or `cc`).
- `--suffix`: Defines the model save name.
- `--device`: Chooses between `cuda` or `cpu`.
- `--contrast`: Enables contrastive learning if set to `True`.
- `--batch_size`: Sets the batch size for training.
- `--model_save_path`: Directory where the trained model will be stored.
- `--polynormer`: Determines whether to use Polynormer-based modeling.

### Outputs

- The trained model is saved as a `.pt` file in the specified `model_save_path`.
- Outputs are stored in the `output_dicts` directory as `.pkl` files, named based on `--suffix`.

### Evaluation

Use `evaluation_script.ipynb` to evaluate the trained model. Load the `.pkl` output files for analysis.
