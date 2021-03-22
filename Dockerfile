FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt update && apt install -y vim curl git build-essential

# For opencv, else throws error 'libgl1.so caanot open ...'. Soln from: https://stackoverflow.com/a/63377623/1874627
RUN apt update && apt-get install libtiff5 ffmpeg libsm6 libxext6  -y

# By default installs 0.7.0 which does not have transforms implemented for both PIL/Tensor inputs
RUN pip install \
    texttable \
    tqdm \
    matplotlib \
    tensorboard \
    scikit-image \
    torchray

# Install taskfile to run go task
USER root
RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /bin

# Setup matplotlibrc file for saving plot in non-gui mode
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

WORKDIR /workspace
