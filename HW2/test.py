import mpi4py as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)
