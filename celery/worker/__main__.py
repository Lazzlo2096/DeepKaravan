import logging

from worker.inference import get_engine
from worker.tasks import app


def main():
    logging.basicConfig(level=logging.INFO)
    # Load exactly one model, before consuming messages. The solo pool avoids
    # forking a GPU context and never starts additional inference processes.
    get_engine()
    app.worker_main(
        [
            "worker",
            "--pool=solo",
            "--concurrency=1",
            "--prefetch-multiplier=1",
            "--without-gossip",
            "--without-mingle",
            "--loglevel=INFO",
        ]
    )


if __name__ == "__main__":
    main()
