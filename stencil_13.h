#ifndef _STENCIL_13_H_
#define _STENCIL_13_H_

#ifdef ISOTROPIC
#define CONSTANTS_13			\
  const real coef0 = 1.5f;		\
  const real cx1 = 0.3f;		\
  const real cx2 = 0.2f;		\
  const real cy1 = cx1;			\
  const real cy2 = cx2;			\
  const real cz1 = cx1;			\
  const real cz2 = cx2
#else
#define CONSTANTS_13			\
  const real coef0 = 1.5f;		\
  const real cx1 = 0.3f;		\
  const real cx2 = 0.2f;		\
  const real cy1 = cx1*1.1f;		\
  const real cy2 = cx2*1.1f;		\
  const real cz1 = cx1*0.9f;		\
  const real cz2 = cx2*0.9f
#endif

#define STENCIL_13(i,j,k)		     \
  const real um2k = u[IDX(i,j,k-2)];			\
  const real um1k = u[IDX(i,j,k-1)];			\
  const real up1k = u[IDX(i,j,k+1)];			\
  const real up2k = u[IDX(i,j,k+2)];			\
  const real um2i = u[IDX(i-2,j,k)];			\
  const real um1i = u[IDX(i-1,j,k)];			\
  const real up1i = u[IDX(i+1,j,k)];			\
  const real up2i = u[IDX(i+2,j,k)];			\
  const real um2j = u[IDX(i,j-2,k)];			\
  const real um1j = u[IDX(i,j-1,k)];			\
  const real up1j = u[IDX(i,j+1,k)];			\
  const real up2j = u[IDX(i,j+2,k)];			\
  const real u0   = u[IDX(i,j,k)];			\
  const real s0   = s[IDX(i,j,k)];			\
  const real v0   = v[IDX(i,j,k)];			\
  const real stencil  = coef0*u0 +				\
    cz2*(um2k + up2k) + cz1*(um1k + up1k) +		\
    cy2*(um2j + up2j) + cy1*(um1j + up1j) +		\
    cx2*(um2i + up2i) + cx1*(um1i + up1i)
  
// stencil implementatons
// _ref: VANILLA reference variant without any manual tuning
// _simd: forced OMP SIMD vectorization
// _vlen: manually optimized variants

inline void stencil_13_ref_1(real * restrict v, const real * restrict s, const real * restrict u,
			     const unsigned int i, const unsigned int j, const unsigned int k,
			     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_13;

  // compiler may want to vectorize however it likes
  // ideally it should auto-generate a vector version that is close in performance to the manual vec kernels
  
  STENCIL_13(i,j,k);
  v[IDX(i,j,k)] = v0 - 3.5f*u0 + s0*stencil;
}


inline void stencil_13_simd_8(real * restrict v, const real * restrict s, const real * restrict u,
			      const unsigned int i, const unsigned int j, const unsigned int kmin,
			      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_13;

#pragma omp simd simdlen(8) 
  for(unsigned int k = kmin;k < kmin+8;k++) {
    STENCIL_13(i,j,k);
    v[IDX(i,j,k)] = v0 - 3.5f*u0 + s0*stencil;
  }
}

inline void stencil_13_simd_16(real * restrict v, const real * restrict s, const real * restrict u,
			       const unsigned int i, const unsigned int j, const unsigned int kmin,
			       const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_13;

#pragma omp simd simdlen(16) 
  for(unsigned int k = kmin;k < kmin+16;k++) {
    STENCIL_13(i,j,k);
    v[IDX(i,j,k)] = v0 - 3.5f*u0 + s0*stencil;
  }
}

inline void stencil_13_vec_4(real * restrict v, const real * restrict s, const real * restrict u,
			     const unsigned int i, const unsigned int j, const unsigned int k,
			     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_13;

  // __builtin_prefetch(&u[IDX(i+4,j,k)]);

  // combination of aligned vlen=2 and vlen=8 loads for k-direction
  const vec2 u0m2 = *((vec2 *)(&u[IDX(i,j,k-2)]));
  const vec4 u0   = *((vec4 *)(&u[IDX(i,j,k)]));
  const vec2 u0p4 = *((vec2 *)(&u[IDX(i,j,k+4)]));
  const vec4 s0   = *((vec4 *)(&s[IDX(i,j,k)]));
  const vec4 v0   = *((vec4 *)(&v[IDX(i,j,k)]));
  // getting the shifted vectors in k-direction
  const vec4 um2k = (vec4){u0m2[0],u0m2[1],u0[0],u0[1]};
  const vec4 um1k = (vec4){u0m2[1],u0[0],u0[1],u0[2]};
  const vec4 up1k = (vec4){u0[1],u0[2],u0[3],u0p4[0]};
  const vec4 up2k = (vec4){u0[2],u0[3],u0p4[0],u0p4[1]};
  // loads in the i-j-directions are all aligned
  const vec4 um2i = *((vec4 *)(&u[IDX(i-2,j,k)]));
  const vec4 um1i = *((vec4 *)(&u[IDX(i-1,j,k)]));
  const vec4 up1i = *((vec4 *)(&u[IDX(i+1,j,k)]));
  const vec4 up2i = *((vec4 *)(&u[IDX(i+2,j,k)]));
  const vec4 um2j = *((vec4 *)(&u[IDX(i,j-2,k)]));
  const vec4 um1j = *((vec4 *)(&u[IDX(i,j-1,k)]));
  const vec4 up1j = *((vec4 *)(&u[IDX(i,j+1,k)]));
  const vec4 up2j = *((vec4 *)(&u[IDX(i,j+2,k)]));
  const vec4 stencil  = coef0*u0 +
    cz2*(um2k + up2k) + cz1*(um1k + up1k) +
    cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx2*(um2i + up2i) + cx1*(um1i + up1i);
  
  *(vec4 *)(&v[IDX(i,j,k)]) = v0 - 3.5f*u0 + s0*stencil;
}


inline void stencil_13_vec_4(real * restrict v, const real * restrict s, const real * restrict u,
			     const unsigned int i, const unsigned int j, const unsigned int kmin, const unsigned int kmax,
			     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_13;

  for(unsigned int k = kmin;k < kmax;k += 4) {
    // __builtin_prefetch(&u[IDX(i+4,j,k)]);
    
    // combination of aligned vlen=2 and vlen=8 loads for k-direction
    const vec2 u0m2 = *((vec2 *)(&u[IDX(i,j,k-2)]));
    const vec4 u0   = *((vec4 *)(&u[IDX(i,j,k)]));
    const vec2 u0p4 = *((vec2 *)(&u[IDX(i,j,k+4)]));
    const vec4 s0   = *((vec4 *)(&s[IDX(i,j,k)]));
    const vec4 v0   = *((vec4 *)(&v[IDX(i,j,k)]));
    // getting the shifted vectors in k-direction
    const vec4 um2k = (vec4){u0m2[0],u0m2[1],u0[0],u0[1]};
    const vec4 um1k = (vec4){u0m2[1],u0[0],u0[1],u0[2]};
    const vec4 up1k = (vec4){u0[1],u0[2],u0[3],u0p4[0]};
    const vec4 up2k = (vec4){u0[2],u0[3],u0p4[0],u0p4[1]};
    // loads in the i-j-directions are all aligned
    const vec4 um2i = *((vec4 *)(&u[IDX(i-2,j,k)]));
    const vec4 um1i = *((vec4 *)(&u[IDX(i-1,j,k)]));
    const vec4 up1i = *((vec4 *)(&u[IDX(i+1,j,k)]));
    const vec4 up2i = *((vec4 *)(&u[IDX(i+2,j,k)]));
    const vec4 um2j = *((vec4 *)(&u[IDX(i,j-2,k)]));
    const vec4 um1j = *((vec4 *)(&u[IDX(i,j-1,k)]));
    const vec4 up1j = *((vec4 *)(&u[IDX(i,j+1,k)]));
    const vec4 up2j = *((vec4 *)(&u[IDX(i,j+2,k)]));
    const vec4 stencil  = coef0*u0 +
      cz2*(um2k + up2k) + cz1*(um1k + up1k) +
      cy2*(um2j + up2j) + cy1*(um1j + up1j) +
      cx2*(um2i + up2i) + cx1*(um1i + up1i);
    
    *(vec4 *)(&v[IDX(i,j,k)]) = v0 - 3.5f*u0 + s0*stencil;
  }
}


inline void stencil_13_vec_8(real * restrict v, const real * restrict s, const real * restrict u,
			     const unsigned int i, const unsigned int j, const unsigned int k,
			     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_13;

  __builtin_prefetch(&u[IDX(i+4,j,k)]);
  __builtin_prefetch(&u[IDX(i+4,j+1,k)]);

  // combination of aligned vlen=2 and vlen=8 loads for k-direction
  const vec2 u0m2 = *((vec2 *)(&u[IDX(i,j,k-2)]));
  const vec8 u0 = *((vec8 *)(&u[IDX(i,j,k)]));
  const vec2 u0p8 = *((vec2 *)(&u[IDX(i,j,k+8)]));
  const vec8 s0   = *((vec8 *)(&s[IDX(i,j,k)]));
  const vec8 v0   = *((vec8 *)(&v[IDX(i,j,k)]));
  // getting the shifted vectors in k-direction
  const vec8 um2k = (vec8){u0m2[0],u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5]};
  const vec8 um1k = (vec8){u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6]};
  const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0p8[0]};
  const vec8 up2k = (vec8){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0p8[0],u0p8[1]};
  // loads in the i-j-directions are all aligned
  const vec8 um2i = *((vec8 *)(&u[IDX(i-2,j,k)]));
  const vec8 um1i = *((vec8 *)(&u[IDX(i-1,j,k)]));
  const vec8 up1i = *((vec8 *)(&u[IDX(i+1,j,k)]));
  const vec8 up2i = *((vec8 *)(&u[IDX(i+2,j,k)]));
  const vec8 um2j = *((vec8 *)(&u[IDX(i,j-2,k)]));
  const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
  const vec8 up1j = *((vec8 *)(&u[IDX(i,j+1,k)]));
  const vec8 up2j = *((vec8 *)(&u[IDX(i,j+2,k)]));
  const vec8 stencil  = coef0*u0 +
    cz2*(um2k + up2k) + cz1*(um1k + up1k) +
    cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx2*(um2i + up2i) + cx1*(um1i + up1i);
  
  *(vec8 *)(&v[IDX(i,j,k)]) = v0 - 3.5f*u0 + s0*stencil;
}


inline void stencil_13_vec_j2_8(real * restrict v, const real * restrict s, const real * restrict u,
				const unsigned int i, const unsigned int j, const unsigned int k,
				const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_13;

  __builtin_prefetch(&u[IDX(i+4,j,k)]);
  __builtin_prefetch(&u[IDX(i+4,j,k)]);
  
  // combination of aligned vlen=2 and vlen=8 loads for k-direction
  const vec2 u0m20 = *((vec2 *)(&u[IDX(i,j,k-2)]));
  const vec8 u00   = *((vec8 *)(&u[IDX(i,j,k)]));
  const vec2 u0p80 = *((vec2 *)(&u[IDX(i,j,k+8)]));
  const vec8 s00   = *((vec8 *)(&s[IDX(i,j,k)]));
  const vec8 v00   = *((vec8 *)(&v[IDX(i,j,k)]));

  const vec2 u0m21 = *((vec2 *)(&u[IDX(i,j+1,k-2)]));
  const vec8 u01   = *((vec8 *)(&u[IDX(i,j+1,k)]));
  const vec2 u0p81 = *((vec2 *)(&u[IDX(i,j+1,k+8)]));
  const vec8 s01   = *((vec8 *)(&s[IDX(i,j+1,k)]));
  const vec8 v01   = *((vec8 *)(&v[IDX(i,j+1,k)]));
  // getting the shifted vectors in k-direction
  const vec8 um2k0 = (vec8){u0m20[0],u0m20[1],u00[0],u00[1],u00[2],u00[3],u00[4],u00[5]};
  const vec8 um1k0 = (vec8){u0m20[1],u00[0],u00[1],u00[2],u00[3],u00[4],u00[5],u00[6]};
  const vec8 up1k0 = (vec8){u00[1],u00[2],u00[3],u00[4],u00[5],u00[6],u00[7],u0p80[0]};
  const vec8 up2k0 = (vec8){u00[2],u00[3],u00[4],u00[5],u00[6],u00[7],u0p80[0],u0p80[1]};

  const vec8 um2k1 = (vec8){u0m21[0],u0m21[1],u01[0],u01[1],u01[2],u01[3],u01[4],u01[5]};
  const vec8 um1k1 = (vec8){u0m21[1],u01[0],u01[1],u01[2],u01[3],u01[4],u01[5],u01[6]};
  const vec8 up1k1 = (vec8){u01[1],u01[2],u01[3],u01[4],u01[5],u01[6],u01[7],u0p81[0]};
  const vec8 up2k1 = (vec8){u01[2],u01[3],u01[4],u01[5],u01[6],u01[7],u0p81[0],u0p81[1]};
  // loads in the i-j-directions are all aligned
  const vec8 um2i0 = *((vec8 *)(&u[IDX(i-2,j,k)]));
  const vec8 um1i0 = *((vec8 *)(&u[IDX(i-1,j,k)]));
  const vec8 up1i0 = *((vec8 *)(&u[IDX(i+1,j,k)]));
  const vec8 up2i0 = *((vec8 *)(&u[IDX(i+2,j,k)]));

  const vec8 um2i1 = *((vec8 *)(&u[IDX(i-2,j+1,k)]));
  const vec8 um1i1 = *((vec8 *)(&u[IDX(i-1,j+1,k)]));
  const vec8 up1i1 = *((vec8 *)(&u[IDX(i+1,j+1,k)]));
  const vec8 up2i1 = *((vec8 *)(&u[IDX(i+2,j+1,k)]));

  const vec8 um2j = *((vec8 *)(&u[IDX(i,j-2,k)]));
  const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
  const vec8 up1j = u01;
  const vec8 up2j = *((vec8 *)(&u[IDX(i,j+2,k)]));
  const vec8 up3j = *((vec8 *)(&u[IDX(i,j+3,k)]));
  const vec8 stencil0 = coef0*u00 +
    cz2*(um2k0 + up2k0) + cz1*(um1k0 + up1k0) +
    cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx2*(um2i0 + up2i0) + cx1*(um1i0 + up1i0);
  const vec8 stencil1 = coef0*u01 +
    cz2*(um2k1 + up2k1) + cz1*(um1k1 + up1k1) +
    cy2*(um1j + up3j) + cy1*(u00 + up2j) +
    cx2*(um2i1 + up2i1) + cx1*(um1i1 + up1i1);
  
  *(vec8 *)(&v[IDX(i,j,k)]) = v00 - 3.5f*u00 + s00*stencil0;
  *(vec8 *)(&v[IDX(i,j+1,k)]) = v01 - 3.5f*u01 + s01*stencil1;
}


inline void stencil_13_vec_j2_8(real * restrict v, const real * restrict s, const real * restrict u,
				const unsigned int i, const unsigned int j, const unsigned int kmin, const unsigned int kmax,
				const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_13;

  for(unsigned int k = kmin;k < kmax;k += 16) {
    // combination of aligned vlen=2 and vlen=8 loads for k-direction
    const vec2 u0m20 = *((vec2 *)(&u[IDX(i,j,k-2)]));
    const vec8 u00   = *((vec8 *)(&u[IDX(i,j,k)]));
    const vec2 u0p80 = *((vec2 *)(&u[IDX(i,j,k+8)]));
    const vec8 s00   = *((vec8 *)(&s[IDX(i,j,k)]));
    const vec8 v00   = *((vec8 *)(&v[IDX(i,j,k)]));
    const vec2 u0m21 = *((vec2 *)(&u[IDX(i,j+1,k-2)]));
    const vec8 u01   = *((vec8 *)(&u[IDX(i,j+1,k)]));
    const vec2 u0p81 = *((vec2 *)(&u[IDX(i,j+1,k+8)]));
    const vec8 s01   = *((vec8 *)(&s[IDX(i,j+1,k)]));
    const vec8 v01   = *((vec8 *)(&v[IDX(i,j+1,k)]));
    // getting the shifted vectors in k-direction
    const vec8 um2k0 = (vec8){u0m20[0],u0m20[1],u00[0],u00[1],u00[2],u00[3],u00[4],u00[5]};
    const vec8 um1k0 = (vec8){u0m20[1],u00[0],u00[1],u00[2],u00[3],u00[4],u00[5],u00[6]};
    const vec8 up1k0 = (vec8){u00[1],u00[2],u00[3],u00[4],u00[5],u00[6],u00[7],u0p80[0]};
    const vec8 up2k0 = (vec8){u00[2],u00[3],u00[4],u00[5],u00[6],u00[7],u0p80[0],u0p80[1]};
    const vec8 um2k1 = (vec8){u0m21[0],u0m21[1],u01[0],u01[1],u01[2],u01[3],u01[4],u01[5]};
    const vec8 um1k1 = (vec8){u0m21[1],u01[0],u01[1],u01[2],u01[3],u01[4],u01[5],u01[6]};
    const vec8 up1k1 = (vec8){u01[1],u01[2],u01[3],u01[4],u01[5],u01[6],u01[7],u0p81[0]};
    const vec8 up2k1 = (vec8){u01[2],u01[3],u01[4],u01[5],u01[6],u01[7],u0p81[0],u0p81[1]};
    // loads in the i-j-directions are all aligned
    const vec8 um2i0 = *((vec8 *)(&u[IDX(i-2,j,k)]));
    const vec8 um1i0 = *((vec8 *)(&u[IDX(i-1,j,k)]));
    const vec8 up1i0 = *((vec8 *)(&u[IDX(i+1,j,k)]));
    const vec8 up2i0 = *((vec8 *)(&u[IDX(i+2,j,k)]));
    const vec8 um2i1 = *((vec8 *)(&u[IDX(i-2,j+1,k)]));
    const vec8 um1i1 = *((vec8 *)(&u[IDX(i-1,j+1,k)]));
    const vec8 up1i1 = *((vec8 *)(&u[IDX(i+1,j+1,k)]));
    const vec8 up2i1 = *((vec8 *)(&u[IDX(i+2,j+1,k)]));
    
    const vec8 um2j = *((vec8 *)(&u[IDX(i,j-2,k)]));
    const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
    const vec8 up1j = u01;
    const vec8 up2j = *((vec8 *)(&u[IDX(i,j+2,k)]));
    const vec8 up3j = *((vec8 *)(&u[IDX(i,j+3,k)]));
    const vec8 stencil0 = coef0*u00 +
      cz2*(um2k0 + up2k0) + cz1*(um1k0 + up1k0) +
      cy2*(um2j + up2j) + cy1*(um1j + up1j) +
      cx2*(um2i0 + up2i0) + cx1*(um1i0 + up1i0);
    const vec8 stencil1 = coef0*u01 +
      cz2*(um2k1 + up2k1) + cz1*(um1k1 + up1k1) +
      cy2*(um1j + up3j) + cy1*(u00 + up2j) +
      cx2*(um2i1 + up2i1) + cx1*(um1i1 + up1i1);
    
    *(vec8 *)(&v[IDX(i,j,k)]) = v00 - 3.5f*u00 + s00*stencil0;
    *(vec8 *)(&v[IDX(i,j+1,k)]) = v01 - 3.5f*u01 + s01*stencil1;
  }
}


inline void stencil_13_vec_8(real * restrict v, const real * restrict s, const real * restrict u,
			     const unsigned int i, const unsigned int j, const unsigned int kmin, const unsigned int kmax,
			     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_13;

  for(unsigned int k = kmin;k < kmax;k += 8) {
    // combination of aligned vlen=2 and vlen=8 loads for k-direction
    const vec2 u0m2 = *((vec2 *)(&u[IDX(i,j,k-2)]));
    const vec8 u0 = *((vec8 *)(&u[IDX(i,j,k)]));
    const vec2 u0p8 = *((vec2 *)(&u[IDX(i,j,k+8)]));
    // getting the shifted vectors in k-direction
    const vec8 um2k = (vec8){u0m2[0],u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5]};
    const vec8 um1k = (vec8){u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6]};
    const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0p8[0]};
    const vec8 up2k = (vec8){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0p8[0],u0p8[1]};
    // loads in the i-j-directions are all aligned
    const vec8 um2i = *((vec8 *)(&u[IDX(i-2,j,k)]));
    const vec8 um1i = *((vec8 *)(&u[IDX(i-1,j,k)]));
    const vec8 up1i = *((vec8 *)(&u[IDX(i+1,j,k)]));
    const vec8 up2i = *((vec8 *)(&u[IDX(i+2,j,k)]));
    const vec8 um2j = *((vec8 *)(&u[IDX(i,j-2,k)]));
    const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
    const vec8 up1j = *((vec8 *)(&u[IDX(i,j+1,k)]));
    const vec8 up2j = *((vec8 *)(&u[IDX(i,j+2,k)]));
    const vec8 s0   = *((vec8 *)(&s[IDX(i,j,k)]));
    const vec8 v0   = *((vec8 *)(&v[IDX(i,j,k)]));
    const vec8 stencil  = coef0*u0 +
      cz2*(um2k + up2k) + cz1*(um1k + up1k) +
      cy2*(um2j + up2j) + cy1*(um1j + up1j) +
      cx2*(um2i + up2i) + cx1*(um1i + up1i);
    
    *(vec8 *)(&v[IDX(i,j,k)]) = v0 - 3.5f*u0 + s0*stencil;
  }
}


inline void stencil_13_vec_16(real * restrict v, const real * restrict s, const real * restrict u,
			      const unsigned int i, const unsigned int j, const unsigned int k,
			      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_13;

  //__builtin_prefetch(&u[IDX(i+6,j,k)]);

  // aligned vector loads
  const vec2  u0m2 = *((vec2 *)(&u[IDX(i,j,k-2)]));
  const vec16 u0   = *((vec16 *)(&u[IDX(i,j,k)]));
  const vec2  u0p16 = *((vec2 *)(&u[IDX(i,j,k+16)]));
  const vec16 um2i = *((vec16 *)(&u[IDX(i-2,j,k)]));
  const vec16 um1i = *((vec16 *)(&u[IDX(i-1,j,k)]));
  const vec16 up1i = *((vec16 *)(&u[IDX(i+1,j,k)]));
  const vec16 up2i = *((vec16 *)(&u[IDX(i+2,j,k)]));
  const vec16 um2j = *((vec16 *)(&u[IDX(i,j-2,k)]));
  const vec16 um1j = *((vec16 *)(&u[IDX(i,j-1,k)]));
  const vec16 up1j = *((vec16 *)(&u[IDX(i,j+1,k)]));
  const vec16 up2j = *((vec16 *)(&u[IDX(i,j+2,k)]));
  const vec16 s0   = *((vec16 *)(&s[IDX(i,j,k)]));
  const vec16 v0   = *((vec16 *)(&v[IDX(i,j,k)]));
  // avoid unaligned vector load by shuffle of u0m2, u0 and u0p16
  const vec16 um2k = (vec16){u0m2[0],u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13]};
  const vec16 um1k = (vec16){u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14]};
  const vec16 up1k = (vec16){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0p16[0]};
  const vec16 up2k = (vec16){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0p16[0],u0p16[1]};
  const vec16 stencil  = coef0*u0 + // 1 MUL + 12 ADDS + 12 FMA (37 FLOPS) 
    cz2*(um2k + up2k) + cz1*(um1k + up1k) +
    cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx2*(um2i + up2i) + cx1*(um1i + up1i);
    
  *(vec16 *)(&v[IDX(i,j,k)]) = v0 - 6.25f*u0 + s0*stencil; // 2 FMA (4 FLOPS)
}


inline void stencil_13_vec_16(real * restrict v, const real * restrict s, const real * restrict u,
			      const unsigned int i, const unsigned int j, const unsigned int kmin, const unsigned int kmax,
			      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_13;

  for(unsigned int k = kmin;k < kmax;k += 16) {
    //__builtin_prefetch(&u[IDX(i+6,j,k)]);
    
    // aligned vector loads
    const vec2  u0m2 = *((vec2 *)(&u[IDX(i,j,k-2)]));
    const vec16 u0   = *((vec16 *)(&u[IDX(i,j,k)]));
    const vec2  u0p16 = *((vec2 *)(&u[IDX(i,j,k+16)]));
    const vec16 um2i = *((vec16 *)(&u[IDX(i-2,j,k)]));
    const vec16 um1i = *((vec16 *)(&u[IDX(i-1,j,k)]));
    const vec16 up1i = *((vec16 *)(&u[IDX(i+1,j,k)]));
    const vec16 up2i = *((vec16 *)(&u[IDX(i+2,j,k)]));
    const vec16 um2j = *((vec16 *)(&u[IDX(i,j-2,k)]));
    const vec16 um1j = *((vec16 *)(&u[IDX(i,j-1,k)]));
    const vec16 up1j = *((vec16 *)(&u[IDX(i,j+1,k)]));
    const vec16 up2j = *((vec16 *)(&u[IDX(i,j+2,k)]));
    const vec16 s0   = *((vec16 *)(&s[IDX(i,j,k)]));
    const vec16 v0   = *((vec16 *)(&v[IDX(i,j,k)]));
    // avoid unaligned vector load by shuffle of u0m2, u0 and u0p16
    const vec16 um2k = (vec16){u0m2[0],u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13]};
    const vec16 um1k = (vec16){u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14]};
    const vec16 up1k = (vec16){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0p16[0]};
    const vec16 up2k = (vec16){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0p16[0],u0p16[1]};
    const vec16 stencil  = coef0*u0 + // 1 MUL + 12 ADDS + 12 FMA (37 FLOPS) 
      cz2*(um2k + up2k) + cz1*(um1k + up1k) +
      cy2*(um2j + up2j) + cy1*(um1j + up1j) +
      cx2*(um2i + up2i) + cx1*(um1i + up1i);
    
    *(vec16 *)(&v[IDX(i,j,k)]) = v0 - 6.25f*u0 + s0*stencil; // 2 FMA (4 FLOPS)
  }
}


inline void stencil_13_vec_pipe_8(real * restrict v, const real * restrict s, const real * restrict u,
				  const unsigned int i, const unsigned int j, const unsigned int kmin, const unsigned int kmax,
				  const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_13;

  // preload um4k, um3k, um2k and um1k and u0 to feed 1st iteration
  vec2 u0m2 = *((vec2 *)(&u[IDX(i,j,kmin-2)]));
  vec8 u0 = *((vec8 *)(&u[IDX(i,j,kmin)]));
  // avoid unaligned access by using a smaller 1-element load + shuffle
  vec8 um2k = (vec8){u0m2[0],u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5]};
  vec8 um1k = (vec8){u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6]};
	
  for(unsigned int k = kmin;k < kmax;k+=8) {
    // aligned vector loads
    const vec8 u0n = *((vec8 *)(&u[IDX(i,j,k+8)]));
    const vec8 um2i = *((vec8 *)(&u[IDX(i-2,j,k)]));
    const vec8 um1i = *((vec8 *)(&u[IDX(i-1,j,k)]));
    const vec8 up1i = *((vec8 *)(&u[IDX(i+1,j,k)]));
    const vec8 up2i = *((vec8 *)(&u[IDX(i+2,j,k)]));
    const vec8 um2j = *((vec8 *)(&u[IDX(i,j-2,k)]));
    const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
    const vec8 up1j = *((vec8 *)(&u[IDX(i,j+1,k)]));
    const vec8 up2j = *((vec8 *)(&u[IDX(i,j+2,k)]));
    const vec8 s0   = *((vec8 *)(&s[IDX(i,j,k)]));
    const vec8 v0   = *((vec8 *)(&v[IDX(i,j,k)]));
    // avoid unaligned vector load by shuffle of u0 and u0n
    const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0n[0]};
    const vec8 up2k = (vec8){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0n[0],u0n[1]};
    const vec8 stencil  = coef0*u0 +
      cz2*(um2k + up2k) + cz1*(um1k + up1k) +
      cy2*(um2j + up2j) + cy1*(um1j + up1j) +
      cx2*(um2i + up2i) + cx1*(um1i + up1i);
    
    *(vec8 *)(&v[IDX(i,j,k)]) = v0 - 6.25f*u0 + s0*stencil;
    // initialize um1k,um2k and u0 to feed the next iteration (NO LOADS!!)
    // shuffle from u0 and u0n to get um1k,um2k
    um1k = (vec8){u0[7],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5],u0n[6]};
    um2k = (vec8){u0[6],u0[6],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5]};
    u0 = u0n;
  }
}


inline void stencil_13_vec_nopipe_16(real * restrict v, const real * restrict s, const real * restrict u,
				     const unsigned int i, const unsigned int j, const unsigned int kmin, unsigned int kmax,
				     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_13;

  // preload um4k, um3k, um2k and um1k and u0 to feed 1st iteration
  vec2 u0m2 = *((vec2 *)(&u[IDX(i,j,kmin-2)]));
  vec16 u0 = *((vec16 *)(&u[IDX(i,j,kmin)]));
  // avoid unaligned access by using a smaller 1-element load + shuffle
  vec16 um2k = (vec16){u0m2[0],u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13]};
  vec16 um1k = (vec16){u0m2[1],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14]};
	
  for(unsigned int k = kmin;k < kmax;k+=16) {
    // aligned vector loads
    const vec16 u0n = *((vec16 *)(&u[IDX(i,j,k+16)]));
    const vec16 um2i = *((vec16 *)(&u[IDX(i-2,j,k)]));
    const vec16 um1i = *((vec16 *)(&u[IDX(i-1,j,k)]));
    const vec16 up1i = *((vec16 *)(&u[IDX(i+1,j,k)]));
    const vec16 up2i = *((vec16 *)(&u[IDX(i+2,j,k)]));
    const vec16 um2j = *((vec16 *)(&u[IDX(i,j-2,k)]));
    const vec16 um1j = *((vec16 *)(&u[IDX(i,j-1,k)]));
    const vec16 up1j = *((vec16 *)(&u[IDX(i,j+1,k)]));
    const vec16 up2j = *((vec16 *)(&u[IDX(i,j+2,k)]));
    const vec16 s0   = *((vec16 *)(&s[IDX(i,j,k)]));
    const vec16 v0   = *((vec16 *)(&v[IDX(i,j,k)]));
    // avoid unaligned vector load by shuffle of u0 and u0n
    const vec16 up1k = (vec16){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0]};
    const vec16 up2k = (vec16){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0],u0n[1]};
    const vec16 stencil  = coef0*u0 + // 1 MUL + 12 ADDS + 12 FMA (37 FLOPS) 
      cz2*(um2k + up2k) + cz1*(um1k + up1k) +
      cy2*(um2j + up2j) + cy1*(um1j + up1j) +
      cx2*(um2i + up2i) + cx1*(um1i + up1i);
    
    *(vec16 *)(&v[IDX(i,j,k)]) = v0 - 6.25f*u0 + s0*stencil; // 2 FMA (4 FLOPS)
    // initialize um1k and u0 to feed the next iteration (NO LOADS!!)
    // shuffle from u0 and u0n to get um1k,um2k,um3k and um4k
    um1k = (vec16){u0[15],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5],u0n[6],
      u0n[7],u0n[8],u0n[9],u0n[10],u0n[11],u0n[12],u0n[13],u0n[14]};
    um2k = (vec16){u0[14],u0[15],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5],u0n[6],
      u0n[7],u0n[8],u0n[9],u0n[10],u0n[11],u0n[12],u0n[13]};
    u0 = u0n;
  }
}
#endif
