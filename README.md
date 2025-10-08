# NextSilicon 3d Stenctil Benchmarks

This is a collection of 3d-Stencils. The goal is to find best optimizations
for higher order 3d stencil codes.

## 7-point isotropic cross stencil

This is the simplest, lowest order 3d-stencil using a cross-shape suitable for solving the accoustic iso-tropic wave equation.

```
const float c1 = (c*dt/dx)*(c*dt/dx);  // c: speed of sound, dt: time step, dx: grid spacing
const float c2 = dt*dt;
const float c0 = (2.0f - 6.0f*c1);
for (unsigned int i = 0; i < N;i++) {
  for (unsigned int i = 1;i < N;j++) {
    for (unsigned int k = 0;k < N;k++) {
      const float stencil = (u[i+1][j][k] + u[i-1][j][k] +  // 5 ADDS
            	  	           u[i][j+1][k] + u[i][j-1][k] +
            		             u[i][j][k+1] + u[i][j][k-1]);
      v[i][j][k] = c0 * u[i][j][k] - v[i][j][k] + c1 * stencil + c2 * s[i][j][k]; // 3 FMA
    }
  }
}	  
```

The stencil has three streams: u[],v[] and s[]).
- u[] is the current pressure field at time t,
- s[] is a source term,
- v[] contains the pressure field at time t-1 as input and will be updated to hold the pressure field at time t+1.

The time integration scheme is of 2nd order.

The stencil has the shape of a 3d-cross around the central point u[i][j][k] and has a total of 7 points (including the central one).
It requires a total of 3 multiplications and 8 additions or a total of 5 ADDS + 3 FMA operations or a total of 8 FMA operations.
If only FMA operations are used the depth of the dependency graph of the computations is 8 to compute the result for v[i][j][k].
which is preferable since the number of FCB slots is also 8.

The arithmetic intensity (AI) for floats is 12 Flops / (3 x 4 Bytes + 1 x 4 Bytes) = 3/4 Flops/Byte (or 3/8 for doubles), when considering only loads from memory.
Arithmetic instensity considering all loads & stores is  12 Flops / (9 x 4B + 1 x 4B) = 3/10 Flops/Byte (3/20 for doubles).

```
make clean
DEFINES="-DVLEN=16 -DSTENCIL=7 -DHWITER=1 -DPAR2D=1" make
# train
nextloader --cfg-file nsd/new_runtime.cfg -- ./stencil3d --xs 528 --ys 528 --zs 528 --zb 48 --yb 33 --xh 7 --yh 7 --zh 16 --iter 20 --train --dups 4
# run
nextloader --cfg-file nsd/new_runtime.cfg -- ./stencil3d --xs 528 --ys 528 --zs 528 --zb 48 --yb 33 --xh 7 --yh 7 --zh 16 --iter 2000 --train --dups 4
```

Note, the --dups parameter makes sure that the 4 duplications are mapped to alternating x-planes in the grid, so the threads of duplications actualy
work closely together in the cache. It is one of the only cases where a quasi-2D HW iteration counter with part of the inner k-loop left in place, is
actually better than a 3D HW iteration counter.


## Building the benchmark

The benchmark uses GNU Make to build the utility. It has been successfully tested with a recent master release from 2025 Sep 28.
It assumes that nextutils is properly installed already.

There are two Makefiles:
1) ``Makefile`` for building the NextSilicon version. Just run ``make`` to build ``stencil3d``
2) ``Makefile.host`` for building the host CPU reference version. Just run ``make -f Makefile.host`` to build ``hstencil3d``
You will need to run ``make clean`` with either Makefile to get rid of .o files to switch between the two.

Note, by default the VLEN=1 STENCIL_SIZE=7 SUBTYPE=ref with a 3d iterator is build using the command:
```
DEFINES="-DSCHED_STATIC=1 -DCOLL=2" make
```
This has been giving the best performance with 4 CBUs per mill and 12 dups using a window of "2x2".

For building an optimized vector version compile with
```
DEFINES="-DVLEN=16 -DSUBTYPE=vec -DHWITER -DPAR3D" make
```
This will use a linearized 3d iteration index over vectorized blocks using HW iteration counter
with dups cooperativity and jblocking.

For building an optimized binary on the host use the following build command:
```
DEFINES="-DSCHED_STATIC=1" make -f Makefile.host
```
This will use the 3d-Iterator using default OpenMP static scheduling without chunk size, but
only uses the 

## Runnning on the host

The following is an example on a Spapphire Rapids machine with 32 cores per socket using 24 or 32 cores to run the benchmark.

```
export OMP_NUM_TTHREADS=24  # I usually stay away from using all cores because OS jitter is high on internal VMs
export OMP_PROC_BIND
./hstencil3d --xs 1040 --ys 1040 --zs 1040 --zb 1040 --yb 26 --xh 7 --yh 7 --zh 16 --iter 20 --train
```

This should deliver 300-350 GFlop/s on a single Intel Xeon &448Y (Sapphire Rapids) CPU.

### Running on Maverick

The new_systemd.cfg file contains a some tweaks that limit execution to a single tile (numa domain).
The projection window may need to be tweaked from 4,7 to get best possible results.

Launch ``nextsystemd --cfg-file new_systemd.cfg``

Assuming you have nextloader configured in your Linux kernel
```
export OMP_NUM_TTHREADS=24  # I usually stay away from using all cores because OS jitter is high on internal VMs
export OMP_PROC_BIND=1
nextloader --cfg-file new_runtime.conf -- ./stencil3d --xs 1040 --ys 1040 --zs 1040 --zb 16 --yb 26 --xh 3 --yh 3 --zh 16 --iter 20 --train
```
Note, with the 3d-itertor --zb VLEN is enforced, --yb must divide --ys without remainder.
