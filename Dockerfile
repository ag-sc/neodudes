FROM python:3.12

ENV PYTHONUNBUFFERED=1 \
    SRC_DIR=/neodudes

ENV TZ=Europe/Berlin

COPY . $SRC_DIR/
WORKDIR $SRC_DIR

RUN apt-get update && apt-get install -y python3-dev default-mysql-client default-libmysqlclient-dev zstd build-essential vim python-is-python3 graphviz python3-graphviz pkg-config libgirepository1.0-dev gcc libcairo2-dev pkg-config libgraphviz-dev cmake sqlite3 tmux z3 curl python3-gi gobject-introspection gir1.2-gtk-3.0 ripgrep
RUN pip install --no-cache-dir -r requirements.txt 
RUN pip install --no-cache-dir compress_pickle[lz4] 

ENV PYTHONPATH=/neodudes/src

RUN python -m spacy download en_core_web_trf
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download en_core_web_sm
RUN python -c "import stanza; stanza.download('en')"

RUN wget -O - -o /dev/null https://ag-sc.techfak.uni-bielefeld.de/download/dudes/entity_predicates.sqlite.zst | zstd -d - -o /neodudes/src/lemon/resources/entity_predicates.sqlite
RUN curl -s https://ag-sc.techfak.uni-bielefeld.de/download/dudes/labels_trie_tagger_fr.cpl -o /neodudes/src/lemon/resources/labels_trie_tagger_fr.cpl
RUN mkdir -p /neodudes/src/lemon/resources/qald/query_score_models/
RUN wget -O - -o /dev/null https://ag-sc.techfak.uni-bielefeld.de/download/dudes/query_score_models.tar.zst | tar --zstd -xvf - -C /neodudes/src/lemon/resources/qald/query_score_models/

RUN printenv > /etc/environment

# Run the init script
CMD [ "bash" ]
