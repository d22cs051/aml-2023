# AML PA 3

## Introduction

Code of PA-3 AML Couse

## Requirements

- Python 3.x
- PyTorch
- PyTorch Lightning
- torchmetrics
- torchvision


## Dataset

### Flower-102

The Flower-103 dataset is used for training, validation, and testing. The dataset is split as follows:

- **Training Set:** 80%
- **Validation Set:** 10%
- **Test Set:** 10%

### Food-101

The Food-101 dataset is used for training, validation, and testing. The dataset is split as follows:

- **Training Set:** 75%
- **Validation Set:** 15%
- **Test Set:** 10%


### Dataset Configuration (dataset.py)

Adjust the dataset-related configurations in the `dataset.py` file, including the data directory, batch size, and number of workers.

## How to Use

1. Clone the repository:

```bash
git clone https://github.com/d22cs051/aml-2023.git
cd D22CS051_AML_PA_3
```

2. Configure the dataset and training hyperparameters in `config.py`.

3. Run the main script:

```bash
python train.py
```

## Model Architecture

The neural network architecture consists of a DINO backbone combined with a linear head. You can choose from different DINO backbones and linear heads in the `dino_backbones` and `NN` classes.

## Training and Evaluation

The training script (`model.py`) utilizes PyTorch Lightning to manage the modeling process. The script includes training, validation, and testing steps.

```bash
python model.py
```

## TensorBoard Logging

Training progress can be visualized using TensorBoard. Run the following command to launch TensorBoard:

```bash
tensorboard --logdir=tb_logs
```

Access TensorBoard in your web browser at [http://localhost:6006/](http://localhost:6006/).

## Author

- Bikash Dutta
- Contact: d22cs051@iitj.ac.in

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.