# MRD
# Project Usage Guide

This repository provides the training and testing code for anomaly detection experiments. The workflow mainly consists of three steps: selecting the best sensitivity factor `alpha`, training the model with the selected `alpha`, and testing the trained model.

## Files

The main files are listed below:

```text
README.md
de_resnet.py
fun.py
model.py
mvtec.py
resnet.py
test.py
train.py
train_best_alpha.py
utils.py
```

## 1. Select the Best Alpha

Before formal training, run `train_best_alpha.py` to search for the best `alpha` value for each category.

```bash
python train_best_alpha.py
```

After running this script, the best `alpha` value for each class will be obtained. Please record these values, as they will be used in the main training stage.

## 2. Train the Model

After obtaining the best `alpha` values, manually set the corresponding `alpha` for each category in `train.py`.

Then run:

```bash
python train.py
```

The model will be trained using the selected `alpha` values. After training, the model parameters will be saved automatically.

## 3. Test the Model

After training is completed, run `test.py` with the corresponding saved model parameters to evaluate the performance.

```bash
python test.py
```

Make sure that the model path in `test.py` is correctly set to the saved checkpoint from the training stage.

## Overall Workflow

```text
Step 1: Run train_best_alpha.py
        ↓
Step 2: Obtain the best alpha value for each category
        ↓
Step 3: Set the recorded alpha values in train.py
        ↓
Step 4: Run train.py for model training
        ↓
Step 5: Use the saved model parameters for testing with test.py
```

## Notes

* `train_best_alpha.py` is used to determine the optimal `alpha` value for each category.
* The selected `alpha` values should be manually recorded and placed into `train.py`.
* The trained model parameters will be saved after training.
* During testing, please ensure that `test.py` loads the correct saved model checkpoint.

