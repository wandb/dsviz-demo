# dsviz-demo

dataset and prediction viz demo

This software is work in progress.

We will disable UI pages that visualize data logged using these features as we change storage formats during development. (we will not preserve backward compatibility of the visualization UI during the development phase)

## instructions

Install the client-ng branch to get a compatible wandb client.

```
pip install git+https://github.com/wandb/client-ng.git@feature/artifact-tables
```

install requirements

```
pip install -r requirements.txt
```

Get bdd100k.tgz from Shawn, and then

```
tar xzvf bdd100k.tgz
```

create v0 and v1 of a dataset

```
python create_dataset.py bdd100k-1 --image_count 1000
python create_dataset.py bdd100k-1 --image_count 1182
```
