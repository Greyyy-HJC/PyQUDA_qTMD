Based on the tests of dirac.invert(src_point)

1. local numpy is exactly same as local
2. aurora is different from local
3. aurora mpi is different from aurora
4. aurora numpy is exactly same as aurora
5. aurora mpi numpy is exactly same as aurora mpi

So, it is not about the backend, but it is related to the mpi and different machine (hardware and QUDA version).


Based on the tests of dirac.mat(src_point)

1. local numpy is exactly same as local
2. aurora is slightly different from local ~ 1e-11
3. aurora mpi is different from aurora ~ 1e+00
4. aurora numpy is exactly same as aurora 
5. aurora mpi numpy is exactly same as aurora mpi 

So, it is not about the backend, slightly related to the different machine (hardware and QUDA version), but it is strongly related to the mpi.