FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

# install torchelastic
WORKDIR /opt/torchelastic
COPY . .
RUN pip uninstall -y -qqq torch
RUN pip install --progress-bar off torch==1.5.0
RUN pip install -v .

WORKDIR /workspace
RUN chmod -R a+w .
