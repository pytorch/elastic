FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

# install torchelastic
WORKDIR /opt/torchelastic
COPY . .
RUN pip install -v .

WORKDIR /workspace
RUN chmod -R a+w .
