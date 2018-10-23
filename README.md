# Monty

[![CircleCI](https://circleci.com/gh/DiscoverAI/mouse-autoencoder.svg?style=svg)](https://circleci.com/gh/DiscoverAI/mouse-autoencoder)

> If it's trouble you want, pally, you've come to the right mouse!
>
> ~ Monterey Jack (Chip 'n Dale Rescue Rangers)

A python tensorflow autoencoder.

![Monty](https://vignette.wikia.nocookie.net/disney/images/8/82/Chip-n-dale-rescue-rangers-volume-2-20061114045022993-000.jpg/revision/latest?cb=20111206182410&format=original)

## Install
Make sure you have pipenv and python 3.6.0 installed.

Then just run:
```bash
pipenv install --dev
```

## Test
Test it with:
```bash
pipenv run pytest
```

## Run
Run it with:
```bash
pipenv run monty/main.py
```
Hyperparameters and paths are configurable via a flagfile or via command
line params. For example:
```bash
pipenv run monty/main.py --flagfile=config/default_flags.txt
```
```bash
pipenv run monty/main.py --batch_size=64
```
All flags have default values set in `monty/main.py` 

## Run notebooks
### Install notebooks requirements
```bash
pipenv run pip install jupiter ipython
```

### Run the actual notebooks
Install the kernel:
```bash
ipython kernel install --user --name=monty
```

Run the notebooks:
```bash
jupyter notebook
```
And change the kernel to monty when you have a notebook open.
