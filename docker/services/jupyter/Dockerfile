# Set the base image using miniconda
FROM jupyter/datascience-notebook

# Set environmental variable(s)
ENV ACCEPT_INTEL_PYTHON_EULA=yes

# Set working directory
WORKDIR /home/notebooks


## Add requirements file
#ADD requirements.txt  ../../../source/
#
## Installs, clean, and update
#RUN apt-get update \
#    && apt-get clean \
#    && pip install --upgrade pip \
#    && pip install -r ../../../source/requirements.txt

# Run shell command for notebook on start
CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root

