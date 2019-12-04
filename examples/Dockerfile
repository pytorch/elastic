ARG BASE_IMAGE=pytorch/elastic:0.1.0rc1
FROM $BASE_IMAGE

ARG WITH_CLASSY_VISION=1

# install utilities and dependencies
RUN pip install awscli --upgrade

RUN if [ "$WITH_CLASSY_VISION" = "1" ]; then \
      pip install classy-vision
    else \
      echo "building without ClassyVision" ; \
    fi

WORKDIR /workspace
COPY . ./examples
RUN chmod -R u+x ./examples/bin
ENV PATH=/workspace/examples/bin:${PATH}

USER nobody
ENTRYPOINT ["/workspace/examples/bin/fetch_and_run"]
