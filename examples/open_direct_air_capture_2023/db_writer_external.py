import glob, time, pickle
import os.path

from mpi4py import MPI
from sqlite_helper import DB
import torch


def main():
    rank = MPI.COMM_WORLD.Get_rank()
    print(f"Rank {rank} of the external db writer here")
    db = DB(f"odac23_{rank}.db")

    while True:
        mdfiles = glob.glob("/tmp/*DONE")
        for fname in mdfiles:
            if 'ALL' not in fname:
                print(f"external db writer found new file {fname}")
                data = torch.load(fname)
                blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                path_id = os.path.basename(fname).split('.pt-DONE')[0].replace('-','/')
                db.write(path_id, blob)
                os.remove(fname)
                print(f"external db writer successfully wrote {fname} info to db")

            if fname == 'ALL-DONE' and len(mdfiles) == 1:
                print("external db writer sees all files are done. quitting.", flush=True)
                break

        time.sleep(1)


if __name__ == '__main__':
    main()
