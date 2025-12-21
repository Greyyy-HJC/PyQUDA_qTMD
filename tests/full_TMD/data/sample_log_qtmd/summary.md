Based on the tests of dirac.invert(src_point)

1. local is exactly same as local numpy
2. aurora is different from local
3. aurora mpi is different from aurora
4. aurora numpy is exactly same as aurora
5. aurora mpi numpy is exactly same as aurora mpi

So, it is not about the backend, but it is related to the mpi and different machine (hardware and QUDA version).