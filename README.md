# resource-management-env
OpenAI Gym environment for resource management

## Updates

* Need to penalise invalid selection, otherwise agent won't learn.
* Following [this guide](https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952).
* Almost finished but refactoring and adding unit tests.

## How to run (TODO:)

```shell
conda env create -f environment.yml
```

Register the Environment
```shell
cd .. # or cd to where contains resource-management-env
pip install -e resource-management-env
```

Create conda environment with Jupyter Lab
```shell
conda env create -f environment-dev.yml
```
