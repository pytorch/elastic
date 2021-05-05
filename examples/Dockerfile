ARG BASE_IMAGE=pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
FROM $BASE_IMAGE

# install utilities and dependencies
RUN pip install awscli --upgrade
RUN pip install classy-vision

RUN pip uninstall -y torch
# TODO remove and make the BASE_IMAGE pytorch:1.9.0-cuda11.1-cudnn8-runtime when torch-1.9 releases
RUN pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html

WORKDIR /workspace

# download imagenet tiny for data
RUN apt-get -q update && apt-get -q install -y wget unzip
RUN wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip -q tiny-imagenet-200.zip -d data && rm tiny-imagenet-200.zip

COPY . ./examples
RUN chmod -R u+x ./examples/bin
RUN examples/bin/install_etcd -d examples/bin
ENV PATH=/workspace/examples/bin:${PATH}

# create a template classy project in /workspace/classy_vision
# (see https://classyvision.ai/#quickstart)
RUN classy-project classy_vision

USER root
ENTRYPOINT ["python", "-m", "torch.distributed.run"]
CMD ["--help"]
