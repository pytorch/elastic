ARG BASE_IMAGE=pytorch/elastic:0.1.0-rc
FROM $BASE_IMAGE

ARG WITH_CLASSY_VISION=1

# install utilities
RUN pip install awscli --upgrade

# TODO (after classy vision open sources)
# 1. remove ssh_private_key hack when classy is open sourced
ARG GITHUB_SSH_KEY

RUN if [ "$WITH_CLASSY_VISION" = "1" ]; then \
      apt-get -yq update && apt-get -yqq install ssh && \
      mkdir /root/.ssh/ && \
      echo "${GITHUB_SSH_KEY}" > /root/.ssh/id_rsa && chmod 600 /root/.ssh/id_rsa && \
      ssh-keyscan github.com > /root/.ssh/known_hosts && \
      pip install git+ssh://git@github.com/facebookresearch/ClassyVision.git && \
      rm -rf /root/.ssh ; \
    else \
      echo "building without ClassyVision" ; \
    fi

WORKDIR /workspace
COPY . ./examples
RUN chmod -R u+x ./examples/bin
ENV PATH=/workspace/examples/bin:${PATH}

USER nobody
ENTRYPOINT ["/workspace/examples/bin/fetch_and_run"]
