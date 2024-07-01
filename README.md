# NeoDUDES - A Compositional Question Answering System Using DUDES

## Setup

The easiest way to run the NeoDUDES question answering system is using Docker. Either run 

```bash
docker pull neodudes:latest
``` 

on your machine or build the image yourself:

```bash
ocker build . -t neodudes
``` 

After that, you can run the container as follows:

```bash 
docker run -e DBPEDIA_SPOTLIGHT_ENDPOINT='http://172.17.0.1:2222/rest' -e DBPEDIA_ENDPOINT='http://172.17.0.1:8890/sparql' -it neodudes
```

The container expects two environment variables to be set: 

- DBPEDIA_SPOTLIGHT_ENDPOINT: URL of a running DBpedia Spotlight [https://www.dbpedia-spotlight.org/](https://www.dbpedia-spotlight.org/) instance accessible for the Docker container
- DBPEDIA_ENDPOINT: URL of a SPARQL endpoint serving the necessary triples for the benchmark. 

You can start a DBpedia Spotlight instance for English locally in the background using the following command:

```bash
docker run -tid --restart unless-stopped --name dbpedia-spotlight.en --mount source=spotlight-model,target=/opt/spotlight -p 2222:80 dbpedia/dbpedia-spotlight spotlight.sh en
```

For the SPARQL endpoint, in case of QALD-9, the triples used by our approach can be found here: [https://ag-sc.techfak.uni-bielefeld.de/download/dudes/2016.ttl.zst](https://ag-sc.techfak.uni-bielefeld.de/download/dudes/2016.ttl.zst). To serve them, you can use, e.g., [Virtuoso](https://hub.docker.com/r/openlink/virtuoso-opensource-7) for which you can find a prepared data directory here: [https://ag-sc.techfak.uni-bielefeld.de/download/dudes/virtuoso2016.tar.zst](https://ag-sc.techfak.uni-bielefeld.de/download/dudes/virtuoso2016.tar.zst)

Please note that in case of local instances you need to take care that they are accessible from the container, e.g. by finding out the corresponding IP addresses or using some kind of docker-compose setup. URLs like 'http://localhost:8890/sparql' will likely not work (see [https://stackoverflow.com/questions/24319662/from-inside-of-a-docker-container-how-do-i-connect-to-the-localhost-of-the-mach](https://stackoverflow.com/questions/24319662/from-inside-of-a-docker-container-how-do-i-connect-to-the-localhost-of-the-mach)).

In case you want to setup the project locally, the `Dockerfile` and `requirements.txt` might give you a few hints which packages you need to install.

## Benchmark

Inside the container, there are a few prepared scripts which you can run:

- `make_docs.sh` to generate the Sphinx documentation of the project
- `qald-eval-test.sh` and 'qald-eval-train.sh' for running the QALD-9 train or test benchmark, respectively
- `qald-rpc.sh` for starting the DUDES RPC server dealing with tagging input questions and scoring SPARQL queries with the provided models
- `src/llm/prompting.py` for experiments with GPT - for this make sure to provide valid OpenAI organization and API key in the `.env` file. An example evnironment file can be found in `sample.env`
- `src/llm/query_score_training.py` for training query scoring models - a Slurm job array is given with `query_score_train.sarray` and `query_score_train_best.sarray`, also illustrating relevant command line options
- `qald-eval-newpipeline.py` for running the QALD-9 benchmark, using the files `src/lemon/resources/qald/QALD9_train-dataset-raw.csv` and `src/lemon/resources/qald/QALD9_test-dataset-raw.csv`

Running the benchmark can be done using qald-eval-newpipeline.py, which by default launches four processes running in parralel. To change the number of spawned processes, you need to change the `cpu_count` variable.




