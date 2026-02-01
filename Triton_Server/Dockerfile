FROM nvcr.io/nvidia/tritonserver:22.12-py3

ENV MODEL_REPOSITORY=/model_repository
RUN mkdir -p ${MODEL_REPOSITORY}
COPY model_repository ${MODEL_REPOSITORY}

COPY scripts/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh
EXPOSE 8000 8001 8002

ENTRYPOINT [ "/entrypoint.sh" ]

CMD [ "tritonserver","--model-repository=/model_repository","--model-control-mode=poll","--repository-poll-secs=5", "--strict-model-config=true", \
    "--log-verbose=1" ,"--allow-metrics=true","--allow-gpu-metrics=true"]