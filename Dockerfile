FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

# install torchelastic
WORKDIR /opt/torchelastic
COPY . .
RUN pip install -v .

WORKDIR /workspace
RUN chmod -R a+w .
