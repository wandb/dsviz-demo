# dsviz-demo

W&B dataset and prediction viz demo

This software is a work in progress.

We will disable UI pages that visualize data logged using these features as we change storage formats during development. (we will not preserve backward compatibility of the visualization UI during the development phase)

## Setup

Install the client-ng branch to get a compatible wandb client.

```
pip install git+https://github.com/wandb/client.git@feature/artifact-media
```

install requirements

```
pip install -r requirements.txt
```

Get bdd100k.tgz from Shawn, and then

```
tar xzvf bdd100k.tgz
```

## UI

Navigate to the Files tab in the Artifacts UI to visualize these examples.

## Running examples

Create v0 and v1 of a dataset

```
python create_dataset.py bdd100k-1 --image_count 1000
python create_dataset.py bdd100k-1 --image_count 1182
```

Log two different predictions on a growing dataset

```
python log_predictions.py preds-1 --image_count 500
python log_predictions.py preds-1 --image_count 550
```

Create other comparison example artifacts

```
wandb artifact put -n dsviz-demo/various-media filetypes/v0
wandb artifact put -n dsviz-demo/various-media filetypes/v1
```
