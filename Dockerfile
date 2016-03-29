FROM radioastro/cuda:devel
MAINTAINER gijsmolenaar@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
    python-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# implicit numpy install fails
run pip install --upgrade numpy

ADD . ./src

RUN pip install /src

CMD /usr/local/bin/montblanc
