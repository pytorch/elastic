FROM torchelastic/examples:0.1.0rc1

USER root
RUN apt update && apt install -y wget unzip
RUN wget http://cs231n.stanford.edu/tiny-imagenet-200.zip &&  unzip tiny-imagenet-200.zip -d /data