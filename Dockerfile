FROM radioastro/cuda:devel
MAINTAINER gijsmolenaar@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
    python-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# implicit numpy install fails
run pip install --upgrade numpy

# this step is not cached so we do it as late as possible
ADD https://codeload.github.com/NVlabs/cub/zip/1.5.2 /

ADD . ./src

RUN pip install /src
