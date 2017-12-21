FROM ubuntu:16.04

RUN apt-get update && \
    apt-get upgrade -qy && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl && \
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      tensorflow-model-server && \
    apt-get clean && \
    rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

EXPOSE 8500
