ARG BASE_IMAGE=pytorch/elastic:0.2.0rc0
FROM $BASE_IMAGE

# install utilities and dependencies
RUN pip install awscli --upgrade
RUN pip install classy-vision

WORKDIR /workspace
COPY . ./examples
RUN chmod -R u+x ./examples/bin
RUN examples/bin/install_etcd -d examples/bin
ENV PATH=/workspace/examples/bin:${PATH}

# create a template classy project in /workspace/classy_vision
# (see https://classyvision.ai/#quickstart)
RUN classy-project classy_vision

# download imagenet tiny for data
RUN apt-get -q update && apt-get -q install -y wget unzip
RUN wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip -q tiny-imagenet-200.zip -d data && rm tiny-imagenet-200.zip

USER nobody
ENTRYPOINT ["python", "-m", "torchelastic.distributed.launch"]
CMD ["--help"]
