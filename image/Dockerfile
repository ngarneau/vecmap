FROM tiangolo/python-machine-learning:latest

VOLUME "/input"
VOLUME "/output/datasets"
VOLUME "/output/tables_and_plots"

ADD run.sh /run.sh
RUN chmod u+x /run.sh

#######################
# Customization start #
#######################

RUN git clone https://github.com/ngarneau/vecmap.git /vecmap
WORKDIR /vecmap
RUN pip install -r requirements.txt

CMD /run.sh

#######################
# Customization end   #
#######################
