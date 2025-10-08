#ifndef _STENCIL_2_H_
#define _STENCIL_2_H_

#define CONSTANTS_2 \
  const real c0 = 0.5f
  
#define STENCIL_2(i,j,k)	     \
  const real u0   = u[IDX(i,j,k)];   \
  const real up1k = u[IDX(i,j,k+1)]; \
  const real stencil = c0*(u0 + up1k)

// stencil implementatons
// _ref: default reference variant without any manual tuning
// _simd: forced OMP SIMD vectorization
// _vec: manually optimized vectorization avoiding alignment faults
// _vec_nopipe: like vec, but without loop pipelining

// reference kernel with default vectorization - relies completely on compiler optimizations

static inline __attribute__((always_inline))
void stencil_2_ref_1(real * restrict v, const real * restrict s, const real * restrict u,
		     const unsigned int i, const unsigned int j, const unsigned int k,
		     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_2;

  STENCIL_2(i,j,k);
  v[IDX(i,j,k)] = stencil;
}

// forced vectorization via OMP SIMD
 
static inline __attribute__((always_inline))
void stencil_2_simd_8(real * restrict v, const real * restrict s, const real * restrict u,
		      const unsigned int i, const int j, const unsigned int kmin,
		      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_2;
#pragma omp simd simdlen(8) 
  for(unsigned int k = kmin;k < kmin+8;k++) {
    STENCIL_2(i,j,k);
    v[IDX(i,j,k)] = stencil;
  }
}

static inline __attribute__((always_inline))
void stencil_2_simd_16(real * restrict v, const real * restrict s, const real * restrict u,
		       const unsigned int i, const int j, const unsigned int kmin,
		       const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_2;
#pragma omp simd simdlen(16) 
  for(unsigned int k = kmin;k < kmin+16;k++) {
    STENCIL_2(i,j,k);
    v[IDX(i,j,k)] = stencil;
  }
}

// manually vectorized kernels taking care of vector alignment in memory
// pipelined and non-pipelined variants

static inline __attribute__((always_inline))
void stencil_2_vec_8(real * restrict v, const real * restrict s, const real * restrict u,
		     const unsigned int i, const int j, const unsigned int k,
		     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_2;

  vec8 u0 = *((vec8 *)(&u[IDX(i,j,k)]));
  const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u[IDX(i,j,k+8)]};
  const vec8 stencil = c0*(u0 + up1k);
  *((vec8 *)(&v[IDX(i,j,k)])) = stencil;
}

static inline __attribute__((always_inline))
void stencil_2_vec_16(real * restrict v, const real * restrict s, const real * restrict u,
		      const unsigned int i, const int j, const unsigned int k,
		      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_2;

  vec16 u0 = *((vec16 *)(&u[IDX(i,j,k)]));
  const vec16 up1k = (vec16){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],
    u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u[IDX(i,j,k+16)]};
  const vec16 stencil = c0*(u0 + up1k);
  *((vec16 *)(&v[IDX(i,j,k)])) = stencil;
}



static inline __attribute__((always_inline))
void stencil_2_pipe_vec_pipe_8(real * restrict v, const real * restrict s, const real * restrict u,
			       const unsigned int i, const int j, const unsigned int kmin, const unsigned int kmax,
			       const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_2;

  // peel the unaligned access out of the first loop iteration
  // preload um1k and u0 to feed 1st iteration

  vec8 u0 = *((vec8 *)(&u[IDX(i,j,kmin)]));

  for (unsigned int k = kmin; k < kmax; k+=8) {
    const vec8 un = *((vec8 *)(&u[IDX(i,j,kmin+8)]));
    const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],un[0]};
    const vec8 stencil = c0*(u0 + up1k);
    *((vec8 *)(&v[IDX(i,j,k)])) = stencil;
    // create pipeline to next iteration
    u0 = un;
  }
}

static inline __attribute__((always_inline))
void stencil_2_vec_pipe_16(real * restrict v, const real * restrict s, const real * restrict u,
			   const unsigned int i, const int j, const unsigned int kmin,const unsigned int kmax,
			   const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_2;
  
  // peel the unaligned access out of the first loop iteration
  // preload um1k and u0 to feed 1st iteration

  vec16 u0 = *((vec16 *)(&u[IDX(i,j,kmin)]));

  for (unsigned int k = kmin; k < kmax; k+=16) {
    const vec16 un = *((vec16 *)(&u[IDX(i,j,k+16)]));
    const vec16 up1k = (vec16){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],
      u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],un[0]};
    const vec16 stencil = c0*(u0+up1k);
    *((vec16 *)(&v[IDX(i,j,k)])) = stencil;
    // create pipeline to next iteration
    u0 = un;
  }
}

#endif
