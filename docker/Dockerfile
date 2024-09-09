FROM continuumio/miniconda3:24.7.1-0

WORKDIR /biharmonic

RUN apt update -y && apt upgrade -y
RUN apt install -y wget git build-essential

RUN conda create -y -n fenics
RUN conda install -n fenics -q -y -c conda-forge scipy fenics tabulate
RUN conda run -n fenics pip install "block @ git+https://github.com/blocknics/cbc.block"

RUN git clone https://github.com/wagnandr/pyuzawamg.git
WORKDIR pyuzawamg
RUN git checkout mg-for-cahn-hilliard
RUN conda run -n fenics python3 -m pip install -e .

RUN echo "conda activate fenics" >> ~/.bashrc
