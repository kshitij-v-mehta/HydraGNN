import pickle
import sys
import traceback

from logger import logger

import mpi_utils
from threading import Thread
from queue import Queue

from hydragnn.utils.datasets import AdiosDataset
from hydragnn.utils.distributed import nsplit

from dbm import write_to_db, close_db


max_q_size = int(1e6)
db_cache_size = int(1e3)
db_path = f"./hydragnn_pyg_{mpi_utils.global_rank}.db"


def adios_reader(adios_in, q: Queue):
    try:
        comm = mpi_utils.global_comm
        nproc = mpi_utils.global_size
        rank = mpi_utils.global_rank

        # Open ADIOS file and create datasets
        trainset = AdiosDataset(adios_in, "trainset", comm)
        valset = AdiosDataset(adios_in, "valset", comm)
        testset = AdiosDataset(adios_in, "testset", comm)

        # Loop through datasets, populate PyG Data objects, and add them to queue
        for i, dataset in enumerate([trainset, valset, testset]):
            rx = list(nsplit(range(len(dataset)), nproc))[rank]
            logger.debug(f"Rank {rank} on {mpi_utils.hostname} reading indices {rx[0]} to {rx[-1]} of {len(dataset)}")
            dataset.setsubset(rx[0], rx[-1] + 1, preload=True)

            # Iterate to populate the PyG Data object
            for pyg in dataset:
                serialized_pyg = pickle.dumps(pyg)
                q.put( (i, serialized_pyg) )

            logger.info(f"Reader thread has put all objects for {dataset.label} to queue")

        # All done. Indicate completion.
        q.put( (None, None) )
        logger.info(f"Reader thread done adding all objects for {adios_in} to queue")

    except Exception as e:
        logger.error(f"adios_reader thread encountered {e} at {traceback.format_exc()}. Terminating.")
        mpi_utils.global_comm.Abort(1)


def main():
    try:
        assert len(sys.argv) == 2, f"Run as {sys.argv[0]} adios-input"
        adios_in = sys.argv[1]

        # launch adios reader thread
        q = Queue(maxsize=max_q_size)
        adios_reader_t = Thread(target=adios_reader, args=(adios_in, q))
        adios_reader_t.start()

        # Start getting ojects from queue and write to db
        while True:
            set_type, pyg = q.get(timeout=None)
            q.task_done()

            if pyg is None:
                break

            write_to_db(db_path, set_type, pyg, db_cache_size)

        adios_reader_t.join()
        close_db()

        logger.info(f"All objects for {adios_in} successfully added to database. Goodbye.")

    except Exception as e:
        logger.error(f"Encountered {e} at {traceback.format_exc()}. Terminating.")
        mpi_utils.global_comm.Abort(1)


if __name__ == '__main__':
    main()
