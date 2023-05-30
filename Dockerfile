FROM tensorflow/tensorflow:latest-gpu

# install git
RUN sudo apt-get update && sudo apt-get install -y git
# create ssh public key
RUN ssh-keygen -t rsa

# create working dir
RUN mkdir /home/tf

WORKDIR /home/tf
