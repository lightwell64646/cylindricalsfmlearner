FROM leesharma/cylindricalsfm:gpu
RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN pip install scikit-image tqdm keras
RUN pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
RUN apt-get install unzip -y
RUN curl https://rclone.org/install.sh | bash
RUN apt-get install unrar -y
RUN pip install --upgrade scipy
RUN apt-get install ffmpeg -y
RUN pip install requests
RUN pip install plyfile
RUN pip install tensorflow

WORKDIR /

RUN apt-get install llvm-6.0 freeglut3 freeglut3-dev -y ;
RUN apt-get install wget -y ;
RUN wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb ;
RUN apt update ; \
    dpkg -i ./mesa_18.3.3-0.deb || true ; \
    apt install -f -y ;

RUN git clone https://github.com/mmatl/pyopengl ;\
    pip install ./pyopengl

RUN git clone https://github.com/mikedh/trimesh.git ;\
    pip install ./trimesh

RUN pip install --upgrade pyrender

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
RUN apt-get update
RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
RUN apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
RUN apt-get update

# Install NVIDIA driver
RUN apt-get install --no-install-recommends nvidia-driver-430
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
RUN apt-get install --no-install-recommends \
    cuda-10-1 \
    libcudnn7=7.6.4.38-1+cuda10.1  \
    libcudnn7-dev=7.6.4.38-1+cuda10.1


# Install TensorRT. Requires that libcudnn7 is installed above.
RUN apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1