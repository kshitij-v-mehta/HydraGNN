import glob, time
import os.path

from mpi4py import MPI
from sqlite_helper import DB
import torch


def main():
    rank = MPI.COMM_WORLD.Get_rank()
    db = DB(f"odac23_{rank}.db")

    while True:
        mdfiles = glob.glob("./tmp/*DONE")
        for fname in mdfiles:
            if 'ALL' not in fname:
                data = torch.load(fname)
                path_id = os.path.basename(fname).split('.pt-DONE')[0].replace('-','/')
                db.write(path_id, data)
                os.remove(fname)

            if fname == 'ALL-DONE' and len(mdfiles) == 1:
                break

        time.sleep(1)


if __name__ == '__main__':
    main()
