FROM nvidia/cuda:11.3.0-base-ubuntu20.04

RUN apt-get update &&  \
    apt install --no-install-recommends --no-install-suggests -y curl &&  \
    apt install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx libx11-6

RUN apt install -y pip

# Setup - Install Pip and Venv for packaging
RUN apt install -y python3-pip && apt install -y python3-venv

# Create a Venv and activate it
RUN python3 -m venv /opt/app-env
ENV PATH /opt/app-env/bin:$PATH
ENV VIRTUAL_ENV /opt/app-env

# Set working directory
WORKDIR /app

# Cache the install of packages
COPY requirements.txt requirements.txt

RUN pip install torch torchvision torchaudio
RUN pip install -r requirements.txt

COPY src src

CMD ["/bin/bash"]