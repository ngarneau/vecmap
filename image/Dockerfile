FROM nvidia/cuda:10.1-cudnn7-devel

RUN apt update -y
RUN apt install software-properties-common -y
RUN apt install python3.7 -y
RUN apt install python3-pip -y
RUN apt install -y git

RUN git clone https://github.com/ngarneau/vecmap.git /vecmap
WORKDIR /vecmap
RUN pip3 install -r requirements.txt
RUN pip3 install -r nvidia_requirements.txt

VOLUME "/input"
VOLUME "/output/datasets"
VOLUME "/output/tables_and_plots"

ADD run.sh /run.sh
RUN chmod u+x /run.sh

CMD /run.sh
