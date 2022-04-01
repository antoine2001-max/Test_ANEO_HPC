# !/bin/sh

sudo apt update -y
sudo apt install python3-pip -y
sudo apt install python3-pandas
pip install ipyparallel
sudo apt-get install -y python3-mpi4py
pip3 install numpy
sudo apt-get install python3-matplotlib
sudo apt-get install -y --no-install-recommends openmpi-bin
sudo apt-get install -y libopenmpi-dev
sudo pip3 install -r requirements_node.txt
