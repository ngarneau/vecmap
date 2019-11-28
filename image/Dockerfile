FROM ngarneau/cupy-python3

RUN apt install -y git
RUN apt install -y texlive-latex-base

RUN git clone https://github.com/ngarneau/vecmap.git /vecmap
WORKDIR /vecmap
RUN pip3 install -r requirements.txt

VOLUME "/input"
VOLUME "/output/datasets"
VOLUME "/output/tables_and_plots"

ADD run.sh /run.sh
RUN chmod u+x /run.sh

ADD run_all.sh /run_all.sh
RUN chmod u+x /run_all.sh

CMD /run.sh
