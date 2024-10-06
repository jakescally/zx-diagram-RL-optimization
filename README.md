# ZX Calclulus RL Circuit Simplification
 Repository for my reinforcement learning algorithm to simplify ZX diagrams. Part of undergraduate research at Florida State University.

 ## Setup/Installation

Ensure that you have conda installed. In particular, this was verified on conda 22.9.0. This specific version can be installed with the following two commands:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh```
```
Then create a new conda environment for the project using
```
conda create --name env_name python=3.10
```
It must be on python version 3.10. Activate your env using ```conda activate env_name```. From within the conda environment and verify that you have pip version 24.2 installed using ```pip --version```. Also, you may want to install jupyter for this new conda environment. Do this by running ```conda install jupyter```.

Clone or download the github repository into whatever folder you want. Navigate to that folder and run
```
pip install -r requirements.txt
```
This should install all of the necessary prerequisites.

## Provided Files
jake_gnn_v1 was the first "successful" model I created, but it only had access to some of the ZX actions. ALl this model really shows is that sometimes it is better to change the connections of the graph for further simplification in the long term instead of greedily fusing everything through. The v2 model is much more robust, allowing the agent to perform more ZX actions, as well as giving it the option of stopping.
