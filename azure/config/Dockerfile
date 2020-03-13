FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04
 
RUN apt-get update -y && \
    apt-get install -y curl
 
RUN ldconfig /usr/local/cuda/lib64/stubs && \
    conda install -y conda=4.6.14 python=3.6.2 && conda clean -ay && \
    # Install AzureML SDK
    pip install --no-cache-dir azureml-defaults && \
    pip install pyyaml && \
    # Install PyTorch
    pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html && \
    pip install --pre torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html && \
    pip install --no-cache-dir mkl==2018.0.3 && \
    ldconfig && \
    pip install tensorboard==1.14.0 && \
    pip install future==0.17.1 && \
    pip install python-etcd==0.4.5 && \
    pip install torchelastic
    
