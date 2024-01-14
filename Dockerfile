FROM ml


RUN mkdir -pv /spectrogram_network/src
RUN mkdir -pv /spectrogram_network/data
RUN mkdir -pv /spectrogram_network/notebooks
RUN mkdir -pv /spectrogram_network/artifacts
RUN mkdir -pv /spectrogram_network/models

# Where computed results and plots can go
# When should we use VOLUME?
#VOLUME /local/artifacts


WORKDIR /spectrogram_network/
ADD requirements.txt /spectrogram_network/
RUN pip3 install -r requirements.txt

# good for the final paper container
#ADD src /local/src
#ADD notebooks /local/notebooks
# for developping it's better to just mount the volumes anonymously. See docker ressources for a better exp.

ENV PYTHONPATH=/spectrogram_network/src

ENTRYPOINT [ "/bin/zsh" ]