#ifndef _STENCIL_7_H_
#define _STENCIL_7_H_

#define CONSTANTS_7 \
  const real coef0 = 1.5f;	     \
  const real cx1 = 0.5f;	     \
  const real cy1 = 0.5f;             \
  const real cz1 = 0.5f

#define STENCIL_7(i,j,k)	     \
  const real um1k = u[IDX(i,j,k-1)]; \
  const real up1k = u[IDX(i,j,k+1)]; \
  const real um1i = u[IDX(i-1,j,k)]; \
  const real up1i = u[IDX(i+1,j,k)]; \
  const real um1j = u[IDX(i,j-1,k)]; \
  const real up1j = u[IDX(i,j+1,k)]; \
  const real u0   = u[IDX(i,j,k)];   \
  const real s0   = s[IDX(i,j,k)];   \
  const real v0   = v[IDX(i,j,k)];					\
  const real stencil = coef0*u0 + cx1*(um1i + up1i) + cy1*(um1j + up1j) + cz1*(um1k + up1k)

// stencil implementatons
// _ref: default reference variant without any manual tuning
// _simd: forced OMP SIMD vectorization
// _vec: manually alignment-aware vectorized variants
// _vec_pipe: manually alignment-aware and pipleined vectorized variants

// reference kernel - relies completely on compiler optimizations
// expectation is that compiler turns this into equivalent of _vec kernels

static inline __attribute__((always_inline))
void stencil_7_ref_1(real * restrict v, const real * restrict s, const real * restrict u,
		     const unsigned int i, const unsigned int j, const unsigned int k,
		     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_7;
  STENCIL_7(i,j,k);
  v[IDX(i,j,k)] = v0 - 3.5f*u0 + s0*stencil;
}

// forced vectorization via OMP SIMD
 
static inline __attribute__((always_inline))
void stencil_7_simd_4(real * restrict v, const real * restrict s, const real * restrict u,
		      const unsigned int i, const unsigned int j, const unsigned int kmin,
		      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_7;
#pragma omp simd simdlen(4) 
  for(unsigned int k = kmin;k < kmin+4;k++) {
    STENCIL_7(i,j,k);
    v[IDX(i,j,k)] = v0 - 3.5f*u0 + s0*stencil;
  }
}

static inline __attribute__((always_inline))
void stencil_7_simd_8(real * restrict v, const real * restrict s, const real * restrict u,
		      const unsigned int i, const unsigned int j, const unsigned int kmin,
		      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_7;
#pragma omp simd simdlen(8) 
  for(unsigned int k = kmin;k < kmin+8;k++) {
    STENCIL_7(i,j,k);
    v[IDX(i,j,k)] = v0 - 3.5f*u0 + s0*stencil;
  }
}

static inline __attribute__((always_inline))
void stencil_7_simd_16(real * restrict v, const real * restrict s, const real * restrict u,
		       const unsigned int i, const unsigned int j, const unsigned int kmin,
		       const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_7;
#pragma omp simd simdlen(16) 
  for(unsigned int k = kmin;k < kmin+16;k++) {
    STENCIL_7(i,j,k);
    v[IDX(i,j,k)] = v0 - 3.5f*u0 + s0*stencil;
  }
}

// manually vectorized kernels taking care of vector alignment in memory

static inline __attribute__((always_inline))
void stencil_7_vec_8(real * restrict v, const real * restrict s, const real * restrict u,
		     const unsigned int i, const unsigned int j, const unsigned int k,
		     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_7;

  // issue cache line prefetches in i-direction
  //__builtin_prefetch(&u[IDX(i+2,j,k)]);
  //__builtin_prefetch(&s[IDX(i+2,j,k)]);
  //__builtin_prefetch(&v[IDX(i+2,j,k)]);

  const vec8 u0 = *((vec8 *)(&u[IDX(i,j,k)]));
  // avoid unaligned access by using a scalar 1-element load + shuffle for k-1 and k+1 elements
  const vec8 um1k = (vec8){u[IDX(i,j,k-1)],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6]};
  const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u[IDX(i,j,k+8)]};
  // aligned vector loads
  const vec8 um1i = *((vec8 *)(&u[IDX(i-1,j,k)]));
  const vec8 up1i = *((vec8 *)(&u[IDX(i+1,j,k)]));
  const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
  const vec8 up1j = *((vec8 *)(&u[IDX(i,j+1,k)]));
  const vec8 s0   = *((vec8 *)(&s[IDX(i,j,k)]));
  const vec8 v0   = *((vec8 *)(&v[IDX(i,j,k)]));
  // compute stencil update
  const vec8 stencil = coef0*u0 + cx1*(um1i + up1i) + cy1*(um1j + up1j) + cz1*(um1k + up1k);
  *(vec8 *)(&v[IDX(i,j,k)]) = v0 - 3.5f*u0 + s0*stencil;
}

static inline __attribute__((always_inline))
void stencil_7_vec_8(real * restrict v, const real * restrict s, const real * restrict u,
		     const unsigned int i, const unsigned int j, const unsigned int kmin, const unsigned int kmax,
		     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_7;
  
  for(unsigned int k = kmin;k < kmax;k += 8) {
    // issue cache line prefetches in i-direction
    // __builtin_prefetch(&u[IDX(i+2,j,k)]);
    // __builtin_prefetch(&s[IDX(i+4,j,k)]);
    // __builtin_prefetch(&v[IDX(i+4,j,k)]);
    
    const vec8 u0 = *((vec8 *)(&u[IDX(i,j,k)]));
    // avoid unaligned access by using a scalar 1-element load + shuffle for k-1 and k+1 elements
    const vec8 um1k = (vec8){u[IDX(i,j,k-1)],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6]};
    const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u[IDX(i,j,k+8)]};
    // aligned vector loads
    const vec8 um1i = *((vec8 *)(&u[IDX(i-1,j,k)]));
    const vec8 up1i = *((vec8 *)(&u[IDX(i+1,j,k)]));
    const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
    const vec8 up1j = *((vec8 *)(&u[IDX(i,j+1,k)]));
    const vec8 s0   = *((vec8 *)(&s[IDX(i,j,k)]));
    const vec8 v0   = *((vec8 *)(&v[IDX(i,j,k)]));
    // compute stencil update
    const vec8 stencil = coef0*u0 + cx1*(um1i + up1i) + cy1*(um1j + up1j) + cz1*(um1k + up1k);
    *(vec8 *)(&v[IDX(i,j,k)]) = v0 - 3.5f*u0 + s0*stencil;
  }
}

static inline __attribute__((always_inline))
void stencil_7_vec_j2_8(real * restrict v, const real * restrict s, const real * restrict u,
			const unsigned int i, const unsigned int j, const unsigned int k,
			const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_7;

  // issue cache line prefetches in i-direction
  // __builtin_prefetch(&u[IDX(i+5,j,k)]);
  // __builtin_prefetch(&s[IDX(i+4,j,k)]);
  // __builtin_prefetch(&v[IDX(i+4,j,k)]);

  const vec8 u00 = *((vec8 *)(&u[IDX(i,j  ,k)]));
  const vec8 u01 = *((vec8 *)(&u[IDX(i,j+1,k)]));
  // avoid unaligned access by using a scalar 1-element load + shuffle for k-1 and k+1 elements
  const vec8 um1k0 = (vec8){u[IDX(i,j,k-1)],u00[0],u00[1],u00[2],u00[3],u00[4],u00[5],u00[6]};
  const vec8 up1k0 = (vec8){u00[1],u00[2],u00[3],u00[4],u00[5],u00[6],u00[7],u[IDX(i,j,k+8)]};
  const vec8 um1k1 = (vec8){u[IDX(i,j+1,k-1)],u00[0],u00[1],u00[2],u00[3],u00[4],u00[5],u00[6]};
  const vec8 up1k1 = (vec8){u01[1],u01[2],u01[3],u01[4],u01[5],u01[6],u01[7],u[IDX(i,j+1,k+8)]};
  // aligned vector loads
  const vec8 um1i0 = *((vec8 *)(&u[IDX(i-1,j,k)]));
  const vec8 um1i1 = *((vec8 *)(&u[IDX(i-1,j+1,k)]));
  const vec8 up1i0 = *((vec8 *)(&u[IDX(i+1,j,k)]));
  const vec8 up1i1 = *((vec8 *)(&u[IDX(i+1,j+1,k)]));
  const vec8 um1j  = *((vec8 *)(&u[IDX(i,j-1,k)]));
  const vec8 up2j  = *((vec8 *)(&u[IDX(i,j+2,k)]));
  const vec8 s00   = *((vec8 *)(&s[IDX(i,j,k)]));
  const vec8 s01   = *((vec8 *)(&s[IDX(i,j,k)]));
  const vec8 v00   = *((vec8 *)(&v[IDX(i,j,k)]));
  const vec8 v01   = *((vec8 *)(&v[IDX(i,j,k)]));
  // compute stencil update
  const vec8 stencil0 = coef0*u00 + cx1*(um1i0 + up1i0) + cy1*(um1j + u01) + cz1*(um1k0 + up1k0);
  const vec8 stencil1 = coef0*u01 + cx1*(um1i1 + up1i1) + cy1*(u00 + up2j) + cz1*(um1k1 + up1k1);
  *(vec8 *)(&v[IDX(i,j  ,k)]) = v00 - 3.5f*u00 + s00*stencil0;
  *(vec8 *)(&v[IDX(i,j+1,k)]) = v01 - 3.5f*u01 + s01*stencil1;
}

static inline __attribute__((always_inline))
void stencil_7_vec_j2_8(real * restrict v, const real * restrict s, const real * restrict u,
			const unsigned int i, const unsigned int j, const unsigned int kmin, const unsigned int kmax,
			const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_7;

  for(unsigned int k = kmin;k < kmax;k += 8) {
    // issue cache line prefetches in i-direction
    // __builtin_prefetch(&u[IDX(i+5,j,k)]);
    // __builtin_prefetch(&s[IDX(i+4,j,k)]);
    // __builtin_prefetch(&v[IDX(i+4,j,k)]);
    
    const vec8 u00 = *((vec8 *)(&u[IDX(i,j  ,k)]));
    const vec8 u01 = *((vec8 *)(&u[IDX(i,j+1,k)]));
    // avoid unaligned access by using a scalar 1-element load + shuffle for k-1 and k+1 elements
    const vec8 um1k0 = (vec8){u[IDX(i,j,k-1)],u00[0],u00[1],u00[2],u00[3],u00[4],u00[5],u00[6]};
    const vec8 up1k0 = (vec8){u00[1],u00[2],u00[3],u00[4],u00[5],u00[6],u00[7],u[IDX(i,j,k+8)]};
    const vec8 um1k1 = (vec8){u[IDX(i,j+1,k-1)],u00[0],u00[1],u00[2],u00[3],u00[4],u00[5],u00[6]};
    const vec8 up1k1 = (vec8){u01[1],u01[2],u01[3],u01[4],u01[5],u01[6],u01[7],u[IDX(i,j+1,k+8)]};
    // aligned vector loads
    const vec8 um1i0 = *((vec8 *)(&u[IDX(i-1,j,k)]));
    const vec8 um1i1 = *((vec8 *)(&u[IDX(i-1,j+1,k)]));
    const vec8 up1i0 = *((vec8 *)(&u[IDX(i+1,j,k)]));
    const vec8 up1i1 = *((vec8 *)(&u[IDX(i+1,j+1,k)]));
    const vec8 um1j  = *((vec8 *)(&u[IDX(i,j-1,k)]));
    const vec8 up2j  = *((vec8 *)(&u[IDX(i,j+2,k)]));
    const vec8 s00   = *((vec8 *)(&s[IDX(i,j,k)]));
    const vec8 s01   = *((vec8 *)(&s[IDX(i,j,k)]));
    const vec8 v00   = *((vec8 *)(&v[IDX(i,j,k)]));
    const vec8 v01   = *((vec8 *)(&v[IDX(i,j,k)]));
    // compute stencil update
    const vec8 stencil0 = coef0*u00 + cx1*(um1i0 + up1i0) + cy1*(um1j + u01) + cz1*(um1k0 + up1k0);
    const vec8 stencil1 = coef0*u01 + cx1*(um1i1 + up1i1) + cy1*(u00 + up2j) + cz1*(um1k1 + up1k1);
    *(vec8 *)(&v[IDX(i,j  ,k)]) = v00 - 3.5f*u00 + s00*stencil0;
    *(vec8 *)(&v[IDX(i,j+1,k)]) = v01 - 3.5f*u01 + s01*stencil1;
  }
}

static inline __attribute__((always_inline))
void stencil_7_vec_16(real * restrict v, const real * restrict s, const real * restrict u,
		      const unsigned int i, const unsigned int j, const unsigned int k,
		      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_7;

  // issue cache line prefetches in i-direction
  __builtin_prefetch(&u[IDX(i+4,j,k)]);
  __builtin_prefetch(&s[IDX(i+4,j,k)]);
  __builtin_prefetch(&v[IDX(i+4,j,k)]);

  const vec16 u0 = *((vec16 *)(&u[IDX(i,j,k)]));
  // avoid unaligned access by using a scalar 1-element load + shuffle for k-1 and k+1 elements
  const vec16 um1k = (vec16){u[IDX(i,j,k-1)],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14]};
  const vec16 up1k = (vec16){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u[IDX(i,j,k+16)]};
  // aligned vector loads
  const vec16 um1i = *((vec16 *)(&u[IDX(i-1,j,k)]));
  const vec16 up1i = *((vec16 *)(&u[IDX(i+1,j,k)]));
  const vec16 um1j = *((vec16 *)(&u[IDX(i  ,j-1,k)]));
  const vec16 up1j = *((vec16 *)(&u[IDX(i  ,j+1,k)]));
  const vec16 s0   = *((vec16 *)(&s[IDX(i  ,j,k)]));
  const vec16 v0   = *((vec16 *)(&v[IDX(i  ,j,k)]));
  // compute stencil update
  const vec16 stencil = coef0*u0 + cx1*(um1i + up1i) + cy1*(um1j + up1j) + cz1*(um1k + up1k);
  *(vec16 *)(&v[IDX(i,j,k)]) = v0 - 3.5f*u0 + s0*stencil;
}

static inline __attribute__((always_inline))
void stencil_7_vec_16(real * restrict v, const real * restrict s, const real * restrict u,
		      const unsigned int i, const unsigned int j, const unsigned int kmin, const unsigned int kmax,
		      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_7;

  for(unsigned int k = kmin;k < kmax;k += 16) {
    // issue cache line prefetches in i-direction
    // __builtin_prefetch(&u[IDX(i+5,j,k)]);
    // __builtin_prefetch(&s[IDX(i+4,j,k)]);
    // __builtin_prefetch(&v[IDX(i+4,j,k)]);
    
    const vec16 u0 = *((vec16 *)(&u[IDX(i,j,k)]));
    // avoid unaligned access by using a scalar 1-element load + shuffle for k-1 and k+1 elements
    const vec16 um1k = (vec16){u[IDX(i,j,k-1)],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14]};
    const vec16 up1k = (vec16){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u[IDX(i,j,k+16)]};
    // aligned vector loads
    const vec16 um1i = *((vec16 *)(&u[IDX(i-1,j,k)]));
    const vec16 up1i = *((vec16 *)(&u[IDX(i+1,j,k)]));
    const vec16 um1j = *((vec16 *)(&u[IDX(i  ,j-1,k)]));
    const vec16 up1j = *((vec16 *)(&u[IDX(i  ,j+1,k)]));
    const vec16 s0   = *((vec16 *)(&s[IDX(i  ,j,k)]));
    const vec16 v0   = *((vec16 *)(&v[IDX(i  ,j,k)]));
    // compute stencil update
    const vec16 stencil = coef0*u0 + cx1*(um1i + up1i) + cy1*(um1j + up1j) + cz1*(um1k + up1k);
    *(vec16 *)(&v[IDX(i,j,k)]) = v0 - 3.5f*u0 + s0*stencil;
  }
}
// pipelined vector variants

static inline __attribute__((always_inline))
void stencil_7_vec_pipe_8(real * restrict v, const real * restrict s, const real * restrict u,
			  const unsigned int i, const unsigned int j, const unsigned int kmin, const unsigned int kmax,
			  const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_7;
  // peel the unaligned access out of the first loop iteration
  // preload um1k and u0 to feed 1st iteration
  vec8 u0 = *((vec8 *)(&u[IDX(i,j,0)]));
  // avoid unaligned access by using a smaller 1-element load + shuffle
  vec8 um1k = (vec8){u[IDX(i,j,-1)],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6]};
  
  for(unsigned int k = kmin;k < kmax;k+=8) {
    // aligned vector loads
    const vec8 u0n = *((vec8 *)(&u[IDX(i,j,k+8)]));
    const vec8 um1i = *((vec8 *)(&u[IDX(i-1,j,k)]));
    const vec8 up1i = *((vec8 *)(&u[IDX(i+1,j,k)]));
    const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
    const vec8 up1j = *((vec8 *)(&u[IDX(i,j+1,k)]));
    const vec8 s0   = *((vec8 *)(&s[IDX(i,j,k)]));
    const vec8 v0   = *((vec8 *)(&v[IDX(i,j,k)]));
    // avoid unaligned vector load by shuffle of u0 and u0n
    const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0n[0]};
    const vec8 stencil = coef0*u0 + cx1*(um1i + up1i) + cy1*(um1j + up1j) + cz1*(um1k + up1k);
    *(vec8 *)(&v[IDX(i,j,k)]) = v0 - 3.5f*u0 + s0*stencil;
    // initialize um1k and u0 to feed the next iteration (NO LOADS!!)
    // shuffle from u0 and u0n to get um1k
    um1k = (vec8){u0[7],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5],u0n[6]};
    u0 = u0n;
  }
}

static inline __attribute__((always_inline))
void stencil_7_vec_pipe_16(real * restrict v, const real * restrict s, const real * restrict u,
			   const unsigned int i, const unsigned int j, const unsigned int kmin, unsigned int kmax,
			   const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_7;
  // peel the unaligned access out of the first loop iteration
  // preload um1k and u0 to feed 1st iteration
  vec16 u0 = *((vec16 *)(&u[IDX(i,j,kmin)]));
  // avoid unaligned access by using a smaller 1-element load + shuffle
  vec16 um1k = (vec16){u[IDX(i,j,kmin-1)],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14]};
	
  for(unsigned int k = kmin;k < kmax;k+=16) {
    // aligned vector loads
    const vec16 u0n  = *((vec16 *)(&u[IDX(i,j,k+16)]));
    const vec16 um1i = *((vec16 *)(&u[IDX(i-1,j,k)]));
    const vec16 up1i = *((vec16 *)(&u[IDX(i+1,j,k)]));
    const vec16 um1j = *((vec16 *)(&u[IDX(i  ,j-1,k)]));
    const vec16 up1j = *((vec16 *)(&u[IDX(i  ,j+1,k)]));
    const vec16 s0   = *((vec16 *)(&s[IDX(i  ,j,k)]));
    const vec16 v0   = *((vec16 *)(&v[IDX(i  ,j,k)]));
    // avoid unaligned vector load by shuffling of u0 and u0n
    const vec16 up1k = (vec16){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0]};
    const vec16 stencil = coef0*u0 + cx1*(um1i + up1i) + cy1*(um1j + up1j) + cz1*(um1k + up1k);
    *(vec16 *)(&v[IDX(i,j,k)]) = v0 - 3.5f*u0 + s0*stencil;
    // initialize um1k and u0 to feed the next iteration (NO LOADS!!)
    // shuffle from u0 and u0n to get um1k
    um1k = (vec16){u0[15],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5],u0n[6],
      u0n[7],u0n[8],u0n[9],u0n[10],u0n[11],u0n[12],u0n[13],u0n[14]};
    u0 = u0n;
  }
}


#endif
