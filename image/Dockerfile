FROM tiangolo/python-machine-learning:latest

VOLUME "/input"
VOLUME "/output/datasets"
VOLUME "/output/tables_and_plots"

ADD run.sh /run.sh
RUN chmod u+x /run.sh
ENTRYPOINT /run.sh

#######################
# Customization start #
#######################

RUN git clone https://github.com/ngarneau/vecmap.git /vecmap
WORKDIR /vecmap

#######################
# Customization end   #
#######################
