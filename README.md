# LWP-WL-Pytorch


This is the repository for the paper "LWP-WL: Link weight prediction based on
CNNs and Weisfeiler-Lehman algorithm for node ordering"

## How to Docker

**Please check *requirements.txt* to check the requisites!**

**Please after installation run ´python3.8 main.py --help´ to check every hyperparameter and argument values**. 

Ideally we want to run the Docker installation, so first build the image,

``docker image build --tag lwp_wl_pytorch -f LWP-WL_Dockerfile ./``

then run the container,

``docker run --ipc=host --gpus "device=3" -it --name LWP-WL -v ~/:/code lwp_wl_pytorch bin/bash``

The *exps.sh* file will run standard experiments for LWP-WL, GCNs and Node2Vec (please take into account that optimal parameters for LWP-WL might not be the default ones on each of the dataset).

If you are looking to edit the code he highly suggest to use a binding folder between the machine holding the code and the container.

## How to

Otherwise, you can run the code without using Docker whenever you fill the *requirements.txt* file. The code was tested and published under *python3.8*.

The *exps.sh* file will run standard experiments for LWP-WL, GCNs and Node2Vec (please take into account that optimal parameters for LWP-WL might not be the default ones on each of the dataset).