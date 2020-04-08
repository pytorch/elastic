FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

# install torchelastic
WORKDIR /opt/torchelastic
COPY . .
# TODO remove torch nightly install when 1.5.0 releases (also update requirements.txt)
RUN pip uninstall -y -qqq torch
RUN pip install --progress-bar off --pre torch -f https://download.pytorch.org/whl/test/cu101/torch_test.html
RUN pip install -v .

WORKDIR /workspace
RUN chmod -R a+w .
