FROM python:3.8-buster
WORKDIR /workspace
RUN pip install numpy python-etcd
RUN pip install torch==1.5.0
# TODO Replace this with the PIP version when available
ADD torchelastic torchelastic
ADD echo.py echo.py
ENV PYTHONPATH /workspace
ENV ALLOW_NONE_AUTHENTICATION yes
ENTRYPOINT ["python", "/workspace/torchelastic/distributed/launch.py"]
