#ifndef _CONFIG_H_
#define _CONFIG_H_

#define DEFAULT_HALO_XY 4        // to fit 25-point stencil
#define DEFAULT_HALO_Z 16        // to support VLEN=16
#define DEFAULT_GRIDSIZE_XY 945  // 51 cachelines incl halo
#define DEFAULT_GRIDSIZE_Z 880   // 55+2 cachelines incl halo
#define DEFAULT_OFS 16           // base multiplier for shuft between u,s,v arrays
#define DEFAULT_ITER 100         // default number of outer iterations
#define DEFAULT_CACHE_ITER 1     // default number of inner cache spinloops
#define PREFETCH_DIST_CLIST 64   // prefetching distance for hilbert and checkerboard indices

#define X_TILE_SZ 1              // default tile size x-direction is 1
#define Y_TILE_SZ 1              // default tile size y-direction is 1
#define Z_TILE_SZ 16             // default tile size z-direction is the entire grid

#define SUPERBLOCK_MIB_TARGET 12 // try to fit superblocks into this MiB window

#ifndef COLL
#define COLL 1
#endif

#ifndef SCHED
#define SCHED 1                  // default STATIC scheduling chunk size for reference iterator
#endif

// the default iterator is 2d jk-blocked with static OMP scheduling with an allocation size of SCHED

//#define PAR2D
//#define HWITER 1               // use HW iteration counter (3d)
//#define CHECKER 1              // use 3d colored checkerboard iterator (3d)
//#define HILBERT 1              // use 2d hilbert curve iterator
//#define SHUFFLE 1              // shuffle hilbert index
//#define SCHED_STATIC 1         // static OMP scheduling without a chunk size
//#define NODUPS 1               // no re-shuffling of dups iterations

#ifndef STENCIL_SIZE
#define STENCIL_SIZE 7           // select the stencil size: 2,7,13 or 25
#endif
#ifndef SUBTYPE
#define SUBTYPE ref          // subtype of the kernel: ref | simd | vec | vec_pipe
#endif
#ifndef VLEN
#define VLEN 1                  // vector length of kernel: 1 | 8 | 16
#endif
#ifndef JUNROLL
#define JUNROLL 1                // unrolling level in j-direction
#endif
// auto generate the stencil kernel name

#define _SFCT(SZ,VL,ST) stencil_##SZ##_##ST##_##VL
#define SFCT(...) _SFCT(__VA_ARGS__)
#define STENCIL_FCT SFCT(STENCIL_SIZE,VLEN,SUBTYPE)

#if defined(PAR2D)
#define ITERATE iterate_2d_jk_blocked
#elif defined(PAR3D)
#define ITERATE iterate_3d_jk_blocked
#elif defined(HILBERT)
#define ITERATE iterate_2d_hilbert
#elif defined(CHECKER)
#define ITERATE iterate_3d_checkerboard
#else
#define ITERATE iterate_3d
#endif

#if STENCIL_SIZE == 2

#define FLOPS_PER_STENCIL 2
#define MEM_BYTES_PER_STENCIL (2*sizeof(real))
#define CACHE_BYTES_PER_STENCIL (2*sizeof(real))

#elif STENCIL_SIZE == 7

#define FLOPS_PER_STENCIL 12
#define MEM_BYTES_PER_STENCIL (4*sizeof(real))
#define CACHE_BYTES_PER_STENCIL (10*sizeof(real))

#elif STENCIL_SIZE == 13

#define FLOPS_PER_STENCIL 29
#define MEM_BYTES_PER_STENCIL (4*sizeof(real))
#define CACHE_BYTES_PER_STENCIL (16*sizeof(real))

#else // STENCIL_SIZE == 25

#define FLOPS_PER_STENCIL 41
#define MEM_BYTES_PER_STENCIL (4*sizeof(real))
#define CACHE_BYTES_PER_STENCIL (28*sizeof(real))

#endif

// C++ doesn't know restrict keyword, but knows __restrict__
#define restrict __restrict__

#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

// define vector types supported by GCC and Clang compilers
#ifdef __GNUC__
typedef real vec2 __attribute__ ((vector_size (2 * sizeof(real))));
typedef real vec4 __attribute__ ((vector_size (4 * sizeof(real))));
typedef real vec8 __attribute__ ((vector_size (8 * sizeof(real))));
typedef real vec16 __attribute__ ((vector_size (16 * sizeof(real))));
#else
typedef real vec2 __attribute__((ext_vector_type(2)));
typedef real vec4 __attribute__((ext_vector_type(4)));
typedef real vec8 __attribute__((ext_vector_type(8)));
typedef real vec16 __attribute__((ext_vector_type(16)));
#endif

// min/max macros
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) < (b)) ? (b) : (a))

#endif
