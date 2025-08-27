import os, glob, time, pickle
import os.path

from mpi4py import MPI
from sqlite_helper import DB
import torch


def main():
    rank = MPI.COMM_WORLD.Get_rank()
    print(f"Rank {rank} of the external db writer here")
    db = DB(f"./db/odac23_{rank}.db")

    wait_count = 0
    while True:
        mdfiles = glob.glob("/tmp/*DONE")
        for fname in mdfiles:
            print(f"external db writer {rank} found new file {fname}")
            newfname = fname.replace('-DONE','')
            os.rename(fname, newfname)
            data = torch.load(newfname, weights_only=False)
            blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            path_id = os.path.basename(newfname).split('.pt')[0].split('.pt')[0].replace('-','/')
            db.write(path_id, blob)
            os.remove(newfname)
            print(f"external db writer {rank} successfully wrote {newfname} info to db")

        if os.path.exists('ALL-DONE'):
            wait_count += 1
            if wait_count == 3:
                print(f"external db writer {rank} sees all files are done. quitting.", flush=True)
                break

        time.sleep(1)

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        os.remove('ALL-DONE')


if __name__ == '__main__':
    main()
