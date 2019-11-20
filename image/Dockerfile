FROM ngarneau/cupy-python3

RUN apt install -y git

RUN git clone https://github.com/ngarneau/vecmap.git /vecmap
WORKDIR /vecmap
RUN pip3 install -r requirements.txt

VOLUME "/input"
VOLUME "/output/datasets"
VOLUME "/output/tables_and_plots"

ADD run.sh /run.sh
RUN chmod u+x /run.sh

CMD /run.sh
