FROM ml


RUN mkdir -pv /nids/src
RUN mkdir -pv /nids/data
RUN mkdir -pv /nids/notebooks
RUN mkdir -pv /nids/artifacts
RUN mkdir -pv /nids/models

# Where computed results and plots can go
# When should we use VOLUME?
#VOLUME /local/artifacts


WORKDIR /nids/
ADD requirements.txt /nids/
RUN pip3 install -r requirements.txt

# good for the final paper container
#ADD src /local/src
#ADD notebooks /local/notebooks
# for developping it's better to just mount the volumes anonymously. See docker ressources for a better exp.

ENV PYTHONPATH=/nids/src

ENTRYPOINT [ "/bin/zsh" ]