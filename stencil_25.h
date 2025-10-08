#ifndef _STENCIL_25_H_
#define _STENCIL_25_H_

#ifdef ISOTROPIC
#define CONSTANTS_25		       \
          const real coef0 = 1.25f;    \
          const real cx1 = 0.4f;       \
	  const real cx2 = 0.3f;       \
	  const real cx3 = 0.2f;       \
	  const real cx4 = 0.1f;       \
	  const real cy1 = cx1;	       \
	  const real cy2 = cx2;	       \
	  const real cy3 = cx3;	       \
	  const real cy4 = cx4;	       \
	  const real cz1 = cz1;	       \
	  const real cz2 = cz2;	       \
	  const real cz3 = cz3;	       \
	  const real cz4 = cz4
#else
#define CONSTANTS_25		       \
          const real coef0 = 1.25f;    \
          const real cx1 = 0.4f;       \
	  const real cx2 = 0.3f;       \
	  const real cx3 = 0.2f;       \
	  const real cx4 = 0.1f;       \
	  const real cy1 = cx1*1.1f;   \
	  const real cy2 = cx2*1.1f;   \
	  const real cy3 = cx3*1.1f;   \
	  const real cy4 = cx4*1.1f;   \
	  const real cz1 = cx1*0.9f;   \
	  const real cz2 = cx2*0.9f;   \
	  const real cz3 = cx3*0.9f;   \
	  const real cz4 = cx4*0.9f
#endif

#define STENCIL_25(i,j,k)				\
  const real um4k = u[IDX(i,j,k-4)];			\
  const real um3k = u[IDX(i,j,k-3)];			\
  const real um2k = u[IDX(i,j,k-2)];			\
  const real um1k = u[IDX(i,j,k-1)];			\
  const real up1k = u[IDX(i,j,k+1)];			\
  const real up2k = u[IDX(i,j,k+2)];			\
  const real up3k = u[IDX(i,j,k+3)];			\
  const real up4k = u[IDX(i,j,k+4)];			\
  const real um4i = u[IDX(i-4,j,k)];			\
  const real um3i = u[IDX(i-3,j,k)];			\
  const real um2i = u[IDX(i-2,j,k)];			\
  const real um1i = u[IDX(i-1,j,k)];			\
  const real up1i = u[IDX(i+1,j,k)];			\
  const real up2i = u[IDX(i+2,j,k)];			\
  const real up3i = u[IDX(i+3,j,k)];			\
  const real up4i = u[IDX(i+4,j,k)];			\
  const real um4j = u[IDX(i,j-4,k)];			\
  const real um3j = u[IDX(i,j-3,k)];			\
  const real um2j = u[IDX(i,j-2,k)];			\
  const real um1j = u[IDX(i,j-1,k)];			\
  const real up1j = u[IDX(i,j+1,k)];			\
  const real up2j = u[IDX(i,j+2,k)];			\
  const real up3j = u[IDX(i,j+3,k)];			\
  const real up4j = u[IDX(i,j+4,k)];			\
  const real u0   = u[IDX(i,j,k)];			\
  const real s0   = s[IDX(i,j,k)];			\
  const real v0   = v[IDX(i,j,k)];			\
  const real stencil = coef0*u0 +			\
    cz4*(um4k + up4k) + cz3*(um3k + up3k) + cz2*(um2k + up2k) + cz1*(um1k + up1k) + \
    cy4*(um4j + up4j) + cy3*(um3j + up3j) + cy2*(um2j + up2j) + cy1*(um1j + up1j) + \
    cx4*(um4i + up4i) + cx3*(um3i + up3i) + cx2*(um2i + up2i) + cx1*(um1i + up1i)
  
// stencil implementatons
// _ref: VANILLA reference variant without any manual tuning
// _simd: forced OMP SIMD vectorization
// _vec: manually optimized variants
// _vec_nopipe: manually optimized variants without pipelining

inline void stencil_25_ref_1(real * restrict v, const real * restrict s, const real * restrict u,
			     const unsigned int i, const int j, const unsigned int k,
			     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;
  
  STENCIL_25(i,j,k);
  v[IDX(i,j,k)] = v0 - 6.25f*u0 + s0*stencil;
}

inline void stencil_25_ref_1(real * restrict v, const real * restrict s, const real * restrict u,
			     const unsigned int i, const int j, const unsigned int kmin, const unsigned int kmax,
			     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;
  
  for (unsigned int k = kmin;k < kmax;k++) {
    STENCIL_25(i,j,k);
    v[IDX(i,j,k)] = v0 - 6.25f*u0 + s0*stencil;
  }
}


inline void stencil_25_simd_8(real * restrict v, const real * restrict s, const real * restrict u,
			      const unsigned int i, const int j, const unsigned int kmin,
			      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_25;

#pragma omp simd simdlen(8) 
  for(unsigned int k = kmin;k < kmin+8;k++) {
    STENCIL_25(i,j,k);
    v[IDX(i,j,k)] = v0 - 6.25f*u0 + s0*stencil;
  }
}

inline void stencil_25_simd_8(real * restrict v, const real * restrict s, const real * restrict u,
			      const unsigned int i, const int j, const unsigned int kmin, const unsigned int kmax,
			      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_25;

#pragma omp simd simdlen(8) 
  for(unsigned int k = kmin;k < kmax;k+=8) {
    STENCIL_25(i,j,k);
    v[IDX(i,j,k)] = v0 - 6.25f*u0 + s0*stencil;
  }
}

inline void stencil_25_simd_16(real * restrict v, const real * restrict s, const real * restrict u,
			       const unsigned int i, const int j, const unsigned int kmin,
			      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_25;

#pragma omp simd simdlen(16) 
  for(unsigned int k = kmin;k < kmin+16;k++) {
    STENCIL_25(i,j,k);
    v[IDX(i,j,k)] = v0 - 6.25f*u0 + s0*stencil;
  }
}


inline void stencil_25_simd_16(real * restrict v, const real * restrict s, const real * restrict u,
			       const unsigned int i, const int j, const unsigned int kmin,const unsigned int kmax,
			      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {

  CONSTANTS_25;

#pragma omp simd simdlen(16) 
  for(unsigned int k = kmin;k < kmax;k++) {
    STENCIL_25(i,j,k);
    v[IDX(i,j,k)] = v0 - 6.25f*u0 + s0*stencil;
  }
}


inline void stencil_25_vec_j2_4(real * restrict v, const real * restrict s, const real * restrict u,
				const unsigned int i, const int j, const unsigned int k,
				const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;

  // combination of aligned vlen=4 and vlen=8 loads for k-direction
  const vec4 u0m40 = *((vec4 *)(&u[IDX(i,j,k-4)]));
  const vec4 u00   = *((vec4 *)(&u[IDX(i,j,k)]));
  const vec4 u0p40 = *((vec4 *)(&u[IDX(i,j,k+4)]));
  const vec4 u0m41 = *((vec4 *)(&u[IDX(i,j+1,k-4)]));
  const vec4 u01   = *((vec4 *)(&u[IDX(i,j+1,k)]));
  const vec4 u0p41 = *((vec4 *)(&u[IDX(i,j+1,k+4)]));
  // getting the shifted vectors in k-direction
  const vec4 um4k0 = u0m40;
  const vec4 um3k0 = (vec4){u0m40[1],u0m40[2],u0m40[3],u00[0]};
  const vec4 um2k0 = (vec4){u0m40[2],u0m40[3],u00[0],u00[1]};
  const vec4 um1k0 = (vec4){u0m40[3],u00[0],u00[1],u00[2]};
  const vec4 up1k0 = (vec4){u00[1],u00[2],u00[3],u0p40[0]};
  const vec4 up2k0 = (vec4){u00[2],u00[3],u0p40[0],u0p40[1]};
  const vec4 up3k0 = (vec4){u00[3],u0p40[0],u0p40[1],u0p40[2]};
  const vec4 up4k0 = u0p40;
  const vec4 um4k1 = u0m41;
  const vec4 um3k1 = (vec4){u0m41[1],u0m41[2],u0m41[3],u01[0]};
  const vec4 um2k1 = (vec4){u0m41[2],u0m41[3],u01[0],u01[1]};
  const vec4 um1k1 = (vec4){u0m41[3],u01[0],u01[1],u01[2]};
  const vec4 up1k1 = (vec4){u01[1],u01[2],u01[3],u0p41[0]};
  const vec4 up2k1 = (vec4){u01[2],u01[3],u0p41[0],u0p41[1]};
  const vec4 up3k1 = (vec4){u01[3],u0p41[0],u0p41[1],u0p41[2]};
  const vec4 up4k1 = u0p41;
   // loads in the i-j-directions are all aligned
  const vec4 um4i0 = *((vec4 *)(&u[IDX(i-4,j,k)]));
  const vec4 um3i0 = *((vec4 *)(&u[IDX(i-3,j,k)]));
  const vec4 um2i0 = *((vec4 *)(&u[IDX(i-2,j,k)]));
  const vec4 um1i0 = *((vec4 *)(&u[IDX(i-1,j,k)]));
  const vec4 up1i0 = *((vec4 *)(&u[IDX(i+1,j,k)]));
  const vec4 up2i0 = *((vec4 *)(&u[IDX(i+2,j,k)]));
  const vec4 up3i0 = *((vec4 *)(&u[IDX(i+3,j,k)]));
  const vec4 up4i0 = *((vec4 *)(&u[IDX(i+4,j,k)]));
  const vec4 um4i1 = *((vec4 *)(&u[IDX(i-4,j+1,k)]));
  const vec4 um3i1 = *((vec4 *)(&u[IDX(i-3,j+1,k)]));
  const vec4 um2i1 = *((vec4 *)(&u[IDX(i-2,j+1,k)]));
  const vec4 um1i1 = *((vec4 *)(&u[IDX(i-1,j+1,k)]));
  const vec4 up1i1 = *((vec4 *)(&u[IDX(i+1,j+1,k)]));
  const vec4 up2i1 = *((vec4 *)(&u[IDX(i+2,j+1,k)]));
  const vec4 up3i1 = *((vec4 *)(&u[IDX(i+3,j+1,k)]));
  const vec4 up4i1 = *((vec4 *)(&u[IDX(i+4,j+1,k)]));

  const vec4 um4j = *((vec4 *)(&u[IDX(i,j-4,k)]));
  const vec4 um3j = *((vec4 *)(&u[IDX(i,j-3,k)]));
  const vec4 um2j = *((vec4 *)(&u[IDX(i,j-2,k)]));
  const vec4 um1j = *((vec4 *)(&u[IDX(i,j-1,k)]));
  const vec4 up1j = u01;
  const vec4 up2j = *((vec4 *)(&u[IDX(i,j+2,k)]));
  const vec4 up3j = *((vec4 *)(&u[IDX(i,j+3,k)]));
  const vec4 up4j = *((vec4 *)(&u[IDX(i,j+4,k)]));
  const vec4 up5j = *((vec4 *)(&u[IDX(i,j+5,k)]));
  const vec4 s00   = *((vec4 *)(&s[IDX(i,j,k)]));
  const vec4 v00   = *((vec4 *)(&v[IDX(i,j,k)]));
  const vec4 s01   = *((vec4 *)(&s[IDX(i,j+1,k)]));
  const vec4 v01   = *((vec4 *)(&v[IDX(i,j+1,k)]));
  
  const vec4 stencil0 = coef0*u00 +
    cz4*(um4k0 + up4k0) + cz3*(um3k0 + up3k0) + cz2*(um2k0 + up2k0) + cz1*(um1k0 + up1k0) +
    cy4*(um4j + up4j) + cy3*(um3j + up3j) + cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx4*(um4i0 + up4i0) + cx3*(um3i0 + up3i0) + cx2*(um2i0 + up2i0) + cx1*(um1i0 + up1i0);
  const vec4 stencil1 = coef0*u01 +
    cz4*(um4k1 + up4k1) + cz3*(um3k1 + up3k1) + cz2*(um2k1 + up2k1) + cz1*(um1k1 + up1k1) +
    cy4*(um3j + up5j) + cy3*(um2j + up3j) + cy2*(um1j + up2j) + cy1*(u00 + up1j) +
    cx4*(um4i1 + up4i1) + cx3*(um3i1 + up3i1) + cx2*(um2i1 + up2i1) + cx1*(um1i1 + up1i1);
    
  *(vec4 *)(&v[IDX(i,j,k)]) = v00 - 6.25f*u00 + s00*stencil0;
  *(vec4 *)(&v[IDX(i,j+1,k)]) = v01 - 6.25f*u01 + s01*stencil1;
}


inline void stencil_25_vec_j2_4(real * restrict v, const real * restrict s, const real * restrict u,
				const unsigned int i, const int j, const unsigned int kmin, const unsigned int kmax,
				const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;

  for(unsigned int k = kmin;k < kmax;k += 4) {
  // combination of aligned vlen=4 and vlen=8 loads for k-direction
  const vec4 u0m40 = *((vec4 *)(&u[IDX(i,j,k-4)]));
  const vec4 u00   = *((vec4 *)(&u[IDX(i,j,k)]));
  const vec4 u0p40 = *((vec4 *)(&u[IDX(i,j,k+4)]));
  const vec4 u0m41 = *((vec4 *)(&u[IDX(i,j+1,k-4)]));
  const vec4 u01   = *((vec4 *)(&u[IDX(i,j+1,k)]));
  const vec4 u0p41 = *((vec4 *)(&u[IDX(i,j+1,k+4)]));
  // getting the shifted vectors in k-direction
  const vec4 um4k0 = u0m40;
  const vec4 um3k0 = (vec4){u0m40[1],u0m40[2],u0m40[3],u00[0]};
  const vec4 um2k0 = (vec4){u0m40[2],u0m40[3],u00[0],u00[1]};
  const vec4 um1k0 = (vec4){u0m40[3],u00[0],u00[1],u00[2]};
  const vec4 up1k0 = (vec4){u00[1],u00[2],u00[3],u0p40[0]};
  const vec4 up2k0 = (vec4){u00[2],u00[3],u0p40[0],u0p40[1]};
  const vec4 up3k0 = (vec4){u00[3],u0p40[0],u0p40[1],u0p40[2]};
  const vec4 up4k0 = u0p40;
  const vec4 um4k1 = u0m41;
  const vec4 um3k1 = (vec4){u0m41[1],u0m41[2],u0m41[3],u01[0]};
  const vec4 um2k1 = (vec4){u0m41[2],u0m41[3],u01[0],u01[1]};
  const vec4 um1k1 = (vec4){u0m41[3],u01[0],u01[1],u01[2]};
  const vec4 up1k1 = (vec4){u01[1],u01[2],u01[3],u0p41[0]};
  const vec4 up2k1 = (vec4){u01[2],u01[3],u0p41[0],u0p41[1]};
  const vec4 up3k1 = (vec4){u01[3],u0p41[0],u0p41[1],u0p41[2]};
  const vec4 up4k1 = u0p41;
   // loads in the i-j-directions are all aligned
  const vec4 um4i0 = *((vec4 *)(&u[IDX(i-4,j,k)]));
  const vec4 um3i0 = *((vec4 *)(&u[IDX(i-3,j,k)]));
  const vec4 um2i0 = *((vec4 *)(&u[IDX(i-2,j,k)]));
  const vec4 um1i0 = *((vec4 *)(&u[IDX(i-1,j,k)]));
  const vec4 up1i0 = *((vec4 *)(&u[IDX(i+1,j,k)]));
  const vec4 up2i0 = *((vec4 *)(&u[IDX(i+2,j,k)]));
  const vec4 up3i0 = *((vec4 *)(&u[IDX(i+3,j,k)]));
  const vec4 up4i0 = *((vec4 *)(&u[IDX(i+4,j,k)]));
  const vec4 um4i1 = *((vec4 *)(&u[IDX(i-4,j+1,k)]));
  const vec4 um3i1 = *((vec4 *)(&u[IDX(i-3,j+1,k)]));
  const vec4 um2i1 = *((vec4 *)(&u[IDX(i-2,j+1,k)]));
  const vec4 um1i1 = *((vec4 *)(&u[IDX(i-1,j+1,k)]));
  const vec4 up1i1 = *((vec4 *)(&u[IDX(i+1,j+1,k)]));
  const vec4 up2i1 = *((vec4 *)(&u[IDX(i+2,j+1,k)]));
  const vec4 up3i1 = *((vec4 *)(&u[IDX(i+3,j+1,k)]));
  const vec4 up4i1 = *((vec4 *)(&u[IDX(i+4,j+1,k)]));

  const vec4 um4j = *((vec4 *)(&u[IDX(i,j-4,k)]));
  const vec4 um3j = *((vec4 *)(&u[IDX(i,j-3,k)]));
  const vec4 um2j = *((vec4 *)(&u[IDX(i,j-2,k)]));
  const vec4 um1j = *((vec4 *)(&u[IDX(i,j-1,k)]));
  const vec4 up1j = u01;
  const vec4 up2j = *((vec4 *)(&u[IDX(i,j+2,k)]));
  const vec4 up3j = *((vec4 *)(&u[IDX(i,j+3,k)]));
  const vec4 up4j = *((vec4 *)(&u[IDX(i,j+4,k)]));
  const vec4 up5j = *((vec4 *)(&u[IDX(i,j+5,k)]));
  const vec4 s00   = *((vec4 *)(&s[IDX(i,j,k)]));
  const vec4 v00   = *((vec4 *)(&v[IDX(i,j,k)]));
  const vec4 s01   = *((vec4 *)(&s[IDX(i,j+1,k)]));
  const vec4 v01   = *((vec4 *)(&v[IDX(i,j+1,k)]));
  
  const vec4 stencil0 = coef0*u00 +
    cz4*(um4k0 + up4k0) + cz3*(um3k0 + up3k0) + cz2*(um2k0 + up2k0) + cz1*(um1k0 + up1k0) +
    cy4*(um4j + up4j) + cy3*(um3j + up3j) + cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx4*(um4i0 + up4i0) + cx3*(um3i0 + up3i0) + cx2*(um2i0 + up2i0) + cx1*(um1i0 + up1i0);
  const vec4 stencil1 = coef0*u01 +
    cz4*(um4k1 + up4k1) + cz3*(um3k1 + up3k1) + cz2*(um2k1 + up2k1) + cz1*(um1k1 + up1k1) +
    cy4*(um3j + up5j) + cy3*(um2j + up3j) + cy2*(um1j + up2j) + cy1*(u00 + up1j) +
    cx4*(um4i1 + up4i1) + cx3*(um3i1 + up3i1) + cx2*(um2i1 + up2i1) + cx1*(um1i1 + up1i1);
    
  *(vec4 *)(&v[IDX(i,j,k)]) = v00 - 6.25f*u00 + s00*stencil0;
  *(vec4 *)(&v[IDX(i,j+1,k)]) = v01 - 6.25f*u01 + s01*stencil1;
  }
}


inline void stencil_25_vec_4(real * restrict v, const real * restrict s, const real * restrict u,
			     const unsigned int i, const int j, const unsigned int k,
			     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;

  __builtin_prefetch(&u[IDX(i+6,j,k)]);

  const vec4 u0m4 = *((vec4 *)(&u[IDX(i,j,k-4)]));
  const vec4 u0   = *((vec4 *)(&u[IDX(i,j,k)]));
  const vec4 u0p4 = *((vec4 *)(&u[IDX(i,j,k+4)]));
  // getting the shifted vectors in k-direction
  const vec4 um4k = u0m4;
  const vec4 um3k = (vec4){u0m4[1],u0m4[2],u0m4[3],u0[0]};
  const vec4 um2k = (vec4){u0m4[2],u0m4[3],u0[0],u0[1]};
  const vec4 um1k = (vec4){u0m4[3],u0[0],u0[1],u0[2]};
  const vec4 up1k = (vec4){u0[1],u0[2],u0[3],u0p4[0]};
  const vec4 up2k = (vec4){u0[2],u0[3],u0p4[0],u0p4[1]};
  const vec4 up3k = (vec4){u0[3],u0p4[0],u0p4[1],u0p4[2]};
  const vec4 up4k = u0p4;
  // loads in the i-j-directions are all aligned
  const vec4 um4i = *((vec4 *)(&u[IDX(i-4,j,k)]));
  const vec4 um3i = *((vec4 *)(&u[IDX(i-3,j,k)]));
  const vec4 um2i = *((vec4 *)(&u[IDX(i-2,j,k)]));
  const vec4 um1i = *((vec4 *)(&u[IDX(i-1,j,k)]));
  const vec4 up1i = *((vec4 *)(&u[IDX(i+1,j,k)]));
  const vec4 up2i = *((vec4 *)(&u[IDX(i+2,j,k)]));
  const vec4 up3i = *((vec4 *)(&u[IDX(i+3,j,k)]));
  const vec4 up4i = *((vec4 *)(&u[IDX(i+4,j,k)]));
  const vec4 um4j = *((vec4 *)(&u[IDX(i,j-4,k)]));
  const vec4 um3j = *((vec4 *)(&u[IDX(i,j-3,k)]));
  const vec4 um2j = *((vec4 *)(&u[IDX(i,j-2,k)]));
  const vec4 um1j = *((vec4 *)(&u[IDX(i,j-1,k)]));
  const vec4 up1j = *((vec4 *)(&u[IDX(i,j+1,k)]));
  const vec4 up2j = *((vec4 *)(&u[IDX(i,j+2,k)]));
  const vec4 up3j = *((vec4 *)(&u[IDX(i,j+3,k)]));
  const vec4 up4j = *((vec4 *)(&u[IDX(i,j+4,k)]));
  const vec4 s0   = *((vec4 *)(&s[IDX(i,j,k)]));
  const vec4 v0   = *((vec4 *)(&v[IDX(i,j,k)]));
  const vec4 stencil = coef0*u0 +
    cz4*(um4k + up4k) + cz3*(um3k + up3k) + cz2*(um2k + up2k) + cz1*(um1k + up1k) +
    cy4*(um4j + up4j) + cy3*(um3j + up3j) + cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx4*(um4i + up4i) + cx3*(um3i + up3i) + cx2*(um2i + up2i) + cx1*(um1i + up1i);
  *(vec4 *)(&v[IDX(i,j,k)]) = v0 - 6.25f*u0 + s0*stencil;
}


inline void stencil_25_vec_j2_8(real * restrict v, const real * restrict s, const real * restrict u,
				const unsigned int i, const int j, const unsigned int k,
				const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;

  __builtin_prefetch(&u[IDX(i+6,j  ,k)]);
  __builtin_prefetch(&u[IDX(i+6,j+1,k)]);

  const vec4 u0m40 = *((vec4 *)(&u[IDX(i,j,k-4)]));
  const vec8 u00   = *((vec8 *)(&u[IDX(i,j,k)]));
  const vec4 u0p40 = *((vec4 *)(&u[IDX(i,j,k+8)]));
  const vec4 u0m41 = *((vec4 *)(&u[IDX(i,j+1,k-4)]));
  const vec8 u01   = *((vec8 *)(&u[IDX(i,j+1,k)]));
  const vec4 u0p41 = *((vec4 *)(&u[IDX(i,j+1,k+8)]));
  // getting the shifted vectors in k-direction
  const vec8 um4k0 = (vec8){u0m40[0],u0m40[1],u0m40[2],u0m40[3],u00[0],u00[1],u00[2],u00[3]};
  const vec8 um3k0 = (vec8){u0m40[1],u0m40[2],u0m40[3],u00[0],u00[1],u00[2],u00[3],u00[4]};
  const vec8 um2k0 = (vec8){u0m40[2],u0m40[3],u00[0],u00[1],u00[2],u00[3],u00[4],u00[5]};
  const vec8 um1k0 = (vec8){u0m40[3],u00[0],u00[1],u00[2],u00[3],u00[4],u00[5],u00[6]};
  const vec8 up1k0 = (vec8){u00[1],u00[2],u00[3],u00[4],u00[5],u00[6],u00[7],u0p40[0]};
  const vec8 up2k0 = (vec8){u00[2],u00[3],u00[4],u00[5],u00[6],u00[7],u0p40[0],u0p40[1]};
  const vec8 up3k0 = (vec8){u00[3],u00[4],u00[5],u00[6],u00[7],u0p40[0],u0p40[1],u0p40[2]};
  const vec8 up4k0 = (vec8){u00[4],u00[5],u00[6],u00[7],u0p40[0],u0p40[1],u0p40[2],u0p40[3]};

  const vec8 um4k1 = (vec8){u0m41[0],u0m41[1],u0m41[2],u0m41[3],u01[0],u01[1],u01[2],u01[3]};
  const vec8 um3k1 = (vec8){u0m41[1],u0m41[2],u0m41[3],u01[0],u01[1],u01[2],u01[3],u01[4]};
  const vec8 um2k1 = (vec8){u0m41[2],u0m41[3],u01[0],u01[1],u01[2],u01[3],u01[4],u01[5]};
  const vec8 um1k1 = (vec8){u0m41[3],u01[0],u01[1],u01[2],u01[3],u01[4],u01[5],u01[6]};
  const vec8 up1k1 = (vec8){u01[1],u01[2],u01[3],u01[4],u01[5],u01[6],u01[7],u0p41[0]};
  const vec8 up2k1 = (vec8){u01[2],u01[3],u01[4],u01[5],u01[6],u01[7],u0p41[0],u0p41[1]};
  const vec8 up3k1 = (vec8){u01[3],u01[4],u01[5],u01[6],u01[7],u0p41[0],u0p41[1],u0p41[2]};
  const vec8 up4k1 = (vec8){u01[4],u01[5],u01[6],u01[7],u0p41[0],u0p41[1],u0p41[2],u0p41[3]};

  // loads in the i-j-directions are all aligned
  const vec8 um4i0 = *((vec8 *)(&u[IDX(i-4,j,k)]));
  const vec8 um3i0 = *((vec8 *)(&u[IDX(i-3,j,k)]));
  const vec8 um2i0 = *((vec8 *)(&u[IDX(i-2,j,k)]));
  const vec8 um1i0 = *((vec8 *)(&u[IDX(i-1,j,k)]));
  const vec8 up1i0 = *((vec8 *)(&u[IDX(i+1,j,k)]));
  const vec8 up2i0 = *((vec8 *)(&u[IDX(i+2,j,k)]));
  const vec8 up3i0 = *((vec8 *)(&u[IDX(i+3,j,k)]));
  const vec8 up4i0 = *((vec8 *)(&u[IDX(i+4,j,k)]));

  const vec8 um4i1 = *((vec8 *)(&u[IDX(i-4,j+1,k)]));
  const vec8 um3i1 = *((vec8 *)(&u[IDX(i-3,j+1,k)]));
  const vec8 um2i1 = *((vec8 *)(&u[IDX(i-2,j+1,k)]));
  const vec8 um1i1 = *((vec8 *)(&u[IDX(i-1,j+1,k)]));
  const vec8 up1i1 = *((vec8 *)(&u[IDX(i+1,j+1,k)]));
  const vec8 up2i1 = *((vec8 *)(&u[IDX(i+2,j+1,k)]));
  const vec8 up3i1 = *((vec8 *)(&u[IDX(i+3,j+1,k)]));
  const vec8 up4i1 = *((vec8 *)(&u[IDX(i+4,j+1,k)]));

  // loads in j-direction we are saving 7 loads due to unrolling by 2
  const vec8 um4j  = *((vec8 *)(&u[IDX(i,j-4,k)]));
  const vec8 um3j  = *((vec8 *)(&u[IDX(i,j-3,k)]));
  const vec8 um2j  = *((vec8 *)(&u[IDX(i,j-2,k)]));
  const vec8 um1j  = *((vec8 *)(&u[IDX(i,j-1,k)]));
  const vec8 up1j  = u01;
  const vec8 up2j  = *((vec8 *)(&u[IDX(i,j+2,k)]));
  const vec8 up3j  = *((vec8 *)(&u[IDX(i,j+3,k)]));
  const vec8 up4j  = *((vec8 *)(&u[IDX(i,j+4,k)]));
  const vec8 up5j  = *((vec8 *)(&u[IDX(i,j+5,k)]));

  const vec8 s00   = *((vec8 *)(&s[IDX(i,j,k)]));
  const vec8 v00   = *((vec8 *)(&v[IDX(i,j,k)]));
  const vec8 s01   = *((vec8 *)(&s[IDX(i,j+1,k)]));
  const vec8 v01   = *((vec8 *)(&v[IDX(i,j+1,k)]));
  
  const vec8 stencil0 = coef0*u00 +
    cz4*(um4k0 + up4k0) + cz3*(um3k0 + up3k0) + cz2*(um2k0 + up2k0) + cz1*(um1k0 + up1k0) +
    cy4*(um4j + up4j) + cy3*(um3j + up3j) + cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx4*(um4i0 + up4i0) + cx3*(um3i0 + up3i0) + cx2*(um2i0 + up2i0) + cx1*(um1i0 + up1i0);
  const vec8 stencil1 = coef0*u01 +
    cz4*(um4k1 + up4k1) + cz3*(um3k1 + up3k1) + cz2*(um2k1 + up2k1) + cz1*(um1k1 + up1k1) +
    cy4*(um3j + up5j) + cy3*(um2j + up3j) + cy2*(um1j + up2j) + cy1*(u00 + up1j) +
    cx4*(um4i1 + up4i1) + cx3*(um3i1 + up3i1) + cx2*(um2i1 + up2i1) + cx1*(um1i1 + up1i1);
  
  *(vec8 *)(&v[IDX(i,j,k)]) = v00 - 6.25f*u00 + s00*stencil0;
  *(vec8 *)(&v[IDX(i,j+1,k)]) = v01 - 6.25f*u01 + s01*stencil1;
}


inline void stencil_25_vec_8(real * restrict v, const real * restrict s, const real * restrict u,
			     const unsigned int i, const int j, const unsigned int kmin, const unsigned int kmax,
			     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;

  for(unsigned int k = kmin;k < kmax;k += 8) {
    //  __builtin_prefetch(&u[IDX(i+5,j,k)]);
  //__builtin_prefetch(&v[IDX(i+2,j,k)]);
  //__builtin_prefetch(&s[IDX(i+2,j,k)]);

  // combination of aligned vlen=4 and vlen=8 loads for k-direction
  const vec4 u0m4 = *((vec4 *)(&u[IDX(i,j,k-4)]));
  const vec8 u0   = *((vec8 *)(&u[IDX(i,j,k)]));
  const vec4 u0p8 = *((vec4 *)(&u[IDX(i,j,k+8)]));
  // getting the shifted vectors in k-direction
  const vec8 um4k = (vec8){u0m4[0],u0m4[1],u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3]};
  const vec8 um3k = (vec8){u0m4[1],u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4]};
  const vec8 um2k = (vec8){u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5]};
  const vec8 um1k = (vec8){u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6]};
  const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0p8[0]};
  const vec8 up2k = (vec8){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0p8[0],u0p8[1]};
  const vec8 up3k = (vec8){u0[3],u0[4],u0[5],u0[6],u0[7],u0p8[0],u0p8[1],u0p8[2]};
  const vec8 up4k = (vec8){u0[4],u0[5],u0[6],u0[7],u0p8[0],u0p8[1],u0p8[2],u0p8[3]};
  // loads in the i-j-directions are all aligned
  const vec8 um4i = *((vec8 *)(&u[IDX(i-4,j,k)]));
  const vec8 um3i = *((vec8 *)(&u[IDX(i-3,j,k)]));
  const vec8 um2i = *((vec8 *)(&u[IDX(i-2,j,k)]));
  const vec8 um1i = *((vec8 *)(&u[IDX(i-1,j,k)]));
  const vec8 up1i = *((vec8 *)(&u[IDX(i+1,j,k)]));
  const vec8 up2i = *((vec8 *)(&u[IDX(i+2,j,k)]));
  const vec8 up3i = *((vec8 *)(&u[IDX(i+3,j,k)]));
  const vec8 up4i = *((vec8 *)(&u[IDX(i+4,j,k)]));
  const vec8 um4j = *((vec8 *)(&u[IDX(i,j-4,k)]));
  const vec8 um3j = *((vec8 *)(&u[IDX(i,j-3,k)]));
  const vec8 um2j = *((vec8 *)(&u[IDX(i,j-2,k)]));
  const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
  const vec8 up1j = *((vec8 *)(&u[IDX(i,j+1,k)]));
  const vec8 up2j = *((vec8 *)(&u[IDX(i,j+2,k)]));
  const vec8 up3j = *((vec8 *)(&u[IDX(i,j+3,k)]));
  const vec8 up4j = *((vec8 *)(&u[IDX(i,j+4,k)]));
  const vec8 s0   = *((vec8 *)(&s[IDX(i,j,k)]));
  const vec8 v0   = *((vec8 *)(&v[IDX(i,j,k)]));
  const vec8 stencil = coef0*u0 +
    cz4*(um4k + up4k) + cz3*(um3k + up3k) + cz2*(um2k + up2k) + cz1*(um1k + up1k) +
    cy4*(um4j + up4j) + cy3*(um3j + up3j) + cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx4*(um4i + up4i) + cx3*(um3i + up3i) + cx2*(um2i + up2i) + cx1*(um1i + up1i);
  
  *(vec8 *)(&v[IDX(i,j,k)]) = v0 - 6.25f*u0 + s0*stencil;
  }
}


inline void stencil_25_vec_8(real * restrict v, const real * restrict s, const real * restrict u,
			     const unsigned int i, const int j, const unsigned int k,
			     const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;

  // adding one prefetch improved projection significantly (from ~16 CBUs to ~13) with NODUPS=1
  __builtin_prefetch(&u[IDX(i+5,j,k)]); 
  //adding more prefetches destroys this property again
  // __builtin_prefetch(&v[IDX(i+1,j,k)]);
  //__builtin_prefetch(&s[IDX(i+2,j,k)]);

  // load central elements via combination of aligned vlen=4 and vlen=8 loads for k-direction
  const vec4 u0m4 = *((vec4 *)(&u[IDX(i,j,k-4)]));
  const vec8 u0   = *((vec8 *)(&u[IDX(i,j,k)]));
  const vec4 u0p8 = *((vec4 *)(&u[IDX(i,j,k+8)]));
  const vec8 s0   = *((vec8 *)(&s[IDX(i,j,k)]));
  const vec8 v0   = *((vec8 *)(&v[IDX(i,j,k)]));
  // getting the shifted vectors in k-direction
  const vec8 um4k = (vec8){u0m4[0],u0m4[1],u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3]};
  const vec8 um3k = (vec8){u0m4[1],u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4]};
  const vec8 um2k = (vec8){u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5]};
  const vec8 um1k = (vec8){u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6]};
  const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0p8[0]};
  const vec8 up2k = (vec8){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0p8[0],u0p8[1]};
  const vec8 up3k = (vec8){u0[3],u0[4],u0[5],u0[6],u0[7],u0p8[0],u0p8[1],u0p8[2]};
  const vec8 up4k = (vec8){u0[4],u0[5],u0[6],u0[7],u0p8[0],u0p8[1],u0p8[2],u0p8[3]};
  // loads in the i-j-directions are all aligned
  const vec8 um4i = *((vec8 *)(&u[IDX(i-4,j,k)]));
  const vec8 um3i = *((vec8 *)(&u[IDX(i-3,j,k)]));
  const vec8 um2i = *((vec8 *)(&u[IDX(i-2,j,k)]));
  const vec8 um1i = *((vec8 *)(&u[IDX(i-1,j,k)]));
  const vec8 up1i = *((vec8 *)(&u[IDX(i+1,j,k)]));
  const vec8 up2i = *((vec8 *)(&u[IDX(i+2,j,k)]));
  const vec8 up3i = *((vec8 *)(&u[IDX(i+3,j,k)]));
  const vec8 up4i = *((vec8 *)(&u[IDX(i+4,j,k)]));
  const vec8 um4j = *((vec8 *)(&u[IDX(i,j-4,k)]));
  const vec8 um3j = *((vec8 *)(&u[IDX(i,j-3,k)]));
  const vec8 um2j = *((vec8 *)(&u[IDX(i,j-2,k)]));
  const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
  const vec8 up1j = *((vec8 *)(&u[IDX(i,j+1,k)]));
  const vec8 up2j = *((vec8 *)(&u[IDX(i,j+2,k)]));
  const vec8 up3j = *((vec8 *)(&u[IDX(i,j+3,k)]));
  const vec8 up4j = *((vec8 *)(&u[IDX(i,j+4,k)]));
  const vec8 stencil = coef0*u0 +
    cz4*(um4k + up4k) + cz3*(um3k + up3k) + cz2*(um2k + up2k) + cz1*(um1k + up1k) +
    cy4*(um4j + up4j) + cy3*(um3j + up3j) + cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx4*(um4i + up4i) + cx3*(um3i + up3i) + cx2*(um2i + up2i) + cx1*(um1i + up1i);
  
  *(vec8 *)(&v[IDX(i,j,k)]) = v0 - 6.25f*u0 + s0*stencil;
}


inline void stencil_25_vec_16(real * restrict v, const real * restrict s, const real * restrict u,
			      const unsigned int i, const int j, const unsigned int k,
			      const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;

  __builtin_prefetch(&u[IDX(i+6,j,k)]);

  const vec4  u0m4 = *((vec4 *)(&u[IDX(i,j,k-4)]));
  const vec16 u0   = *((vec16 *)(&u[IDX(i,j,k)]));
  const vec4  u0n  = *((vec4 *)(&u[IDX(i,j,k+16)]));
  const vec16 um4i = *((vec16 *)(&u[IDX(i-4,j,k)]));
  const vec16 um3i = *((vec16 *)(&u[IDX(i-3,j,k)]));
  const vec16 um2i = *((vec16 *)(&u[IDX(i-2,j,k)]));
  const vec16 um1i = *((vec16 *)(&u[IDX(i-1,j,k)]));
  const vec16 up1i = *((vec16 *)(&u[IDX(i+1,j,k)]));
  const vec16 up2i = *((vec16 *)(&u[IDX(i+2,j,k)]));
  const vec16 up3i = *((vec16 *)(&u[IDX(i+3,j,k)]));
  const vec16 up4i = *((vec16 *)(&u[IDX(i+4,j,k)]));
  const vec16 um4j = *((vec16 *)(&u[IDX(i,j-4,k)]));
  const vec16 um3j = *((vec16 *)(&u[IDX(i,j-3,k)]));
  const vec16 um2j = *((vec16 *)(&u[IDX(i,j-2,k)]));
  const vec16 um1j = *((vec16 *)(&u[IDX(i,j-1,k)]));
  const vec16 up1j = *((vec16 *)(&u[IDX(i,j+1,k)]));
  const vec16 up2j = *((vec16 *)(&u[IDX(i,j+2,k)]));
  const vec16 up3j = *((vec16 *)(&u[IDX(i,j+3,k)]));
  const vec16 up4j = *((vec16 *)(&u[IDX(i,j+4,k)]));
  const vec16 s0   = *((vec16 *)(&s[IDX(i,j,k)]));
  const vec16 v0   = *((vec16 *)(&v[IDX(i,j,k)]));
  // avoid unaligned vector load by shuffle of u0 and u0n
  const vec16 um4k = (vec16){u0m4[0],u0m4[1],u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11]};
  const vec16 um3k = (vec16){u0m4[1],u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12]};
  const vec16 um2k = (vec16){u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13]};
  const vec16 um1k = (vec16){u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14]};	
  const vec16 up1k = (vec16){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0]};
  const vec16 up2k = (vec16){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0],u0n[1]};
  const vec16 up3k = (vec16){u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0],u0n[1],u0n[2]};
  const vec16 up4k = (vec16){u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0],u0n[1],u0n[2],u0n[3]};
  const vec16 stencil = coef0*u0 + // 1 MUL + 12 ADDS + 12 FMA (37 FLOPS) 
    cz4*(um4k + up4k) + cz3*(um3k + up3k) + cz2*(um2k + up2k) + cz1*(um1k + up1k) +
    cy4*(um4j + up4j) + cy3*(um3j + up3j) + cy2*(um2j + up2j) + cy1*(um1j + up1j) +
    cx4*(um4i + up4i) + cx3*(um3i + up3i) + cx2*(um2i + up2i) + cx1*(um1i + up1i);
  *(vec16 *)(&v[IDX(i,j,k)]) = v0 - 6.25f*u0 + s0*stencil; // 2 FMA (4 FLOPS)
}


inline void stencil_25_vec_pipe_8(real * restrict v, const real * restrict s, const real * restrict u,
				  const unsigned int i, const int j, const unsigned int kmin,const unsigned int kmax,
				  const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;
  
  // preload um4k, um3k, um2k and um1k and u0 to feed 1st iteration
  vec4 u0m4 = *((vec4 *)(&u[IDX(i,j,kmin-4)]));
  vec8 u0 = *((vec8 *)(&u[IDX(i,j,kmin)]));
  // avoid unaligned access by using a smaller 1-element load + shuffle
  vec8 um4k = (vec8){u0m4[0],u0m4[1],u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3]};
  vec8 um3k = (vec8){u0m4[1],u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4]};
  vec8 um2k = (vec8){u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5]};
  vec8 um1k = (vec8){u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6]};
	
  for(unsigned int k = kmin;k < kmax;k+=8) {
    // aligned vector loads
    const vec8 u0n = *((vec8 *)(&u[IDX(i,j,k+8)]));
    const vec8 um4i = *((vec8 *)(&u[IDX(i-4,j,k)]));
    const vec8 um3i = *((vec8 *)(&u[IDX(i-3,j,k)]));
    const vec8 um2i = *((vec8 *)(&u[IDX(i-2,j,k)]));
    const vec8 um1i = *((vec8 *)(&u[IDX(i-1,j,k)]));
    const vec8 up1i = *((vec8 *)(&u[IDX(i+1,j,k)]));
    const vec8 up2i = *((vec8 *)(&u[IDX(i+2,j,k)]));
    const vec8 up3i = *((vec8 *)(&u[IDX(i+3,j,k)]));
    const vec8 up4i = *((vec8 *)(&u[IDX(i+4,j,k)]));
    const vec8 um4j = *((vec8 *)(&u[IDX(i,j-4,k)]));
    const vec8 um3j = *((vec8 *)(&u[IDX(i,j-3,k)]));
    const vec8 um2j = *((vec8 *)(&u[IDX(i,j-2,k)]));
    const vec8 um1j = *((vec8 *)(&u[IDX(i,j-1,k)]));
    const vec8 up1j = *((vec8 *)(&u[IDX(i,j+1,k)]));
    const vec8 up2j = *((vec8 *)(&u[IDX(i,j+2,k)]));
    const vec8 up3j = *((vec8 *)(&u[IDX(i,j+3,k)]));
    const vec8 up4j = *((vec8 *)(&u[IDX(i,j+4,k)]));
    const vec8 s0   = *((vec8 *)(&s[IDX(i,j,k)]));
    const vec8 v0   = *((vec8 *)(&v[IDX(i,j,k)]));
    // avoid unaligned vector load by shuffle of u0 and u0n
    const vec8 up1k = (vec8){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0n[0]};
    const vec8 up2k = (vec8){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0n[0],u0n[1]};
    const vec8 up3k = (vec8){u0[3],u0[4],u0[5],u0[6],u0[7],u0n[0],u0n[1],u0n[2]};
    const vec8 up4k = (vec8){u0[4],u0[5],u0[6],u0[7],u0n[0],u0n[1],u0n[2],u0n[3]};
    const vec8 stencil = coef0*u0 +
      cz4*(um4k + up4k) + cz3*(um3k + up3k) + cz2*(um2k + up2k) + cz1*(um1k + up1k) +
      cy4*(um4j + up4j) + cy3*(um3j + up3j) + cy2*(um2j + up2j) + cy1*(um1j + up1j) +
      cx4*(um4i + up4i) + cx3*(um3i + up3i) + cx2*(um2i + up2i) + cx1*(um1i + up1i);
    
    *(vec8 *)(&v[IDX(i,j,k)]) = v0 - 6.25f*u0 + s0*stencil;
    // initialize um1k and u0 to feed the next iteration (NO LOADS!!)
    // shuffle from u0 and u0n to get um1k,um2k,um3k and um4k
    um1k = (vec8){u0[7],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5],u0n[6]};
    um2k = (vec8){u0[6],u0[6],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5]};
    um3k = (vec8){u0[5],u0[6],u0[7],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4]};
    um4k = up4k;
    u0 = u0n;
  }
}


inline void stencil_25_vec_pipe_16(real * restrict v, const real * restrict s, const real * restrict u,
				   const unsigned int i, const int j, const unsigned int kmin, unsigned int kmax,
				   const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  CONSTANTS_25;
  
  // preload um4k, um3k, um2k and um1k and u0 to feed 1st iteration
  vec4 u0m4 = *((vec4 *)(&u[IDX(i,j,kmin-4)]));
  vec16 u0 = *((vec16 *)(&u[IDX(i,j,kmin)]));
  // avoid unaligned access by using a smaller 1-element load + shuffle
  vec16 um4k = (vec16){u0m4[0],u0m4[1],u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11]};
  vec16 um3k = (vec16){u0m4[1],u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12]};
  vec16 um2k = (vec16){u0m4[2],u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13]};
  vec16 um1k = (vec16){u0m4[3],u0[0],u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14]};
  
  for(unsigned int k = kmin;k < kmax;k+=16) {
    // aligned vector loads
    const vec16 u0n = *((vec16 *)(&u[IDX(i,j,k+16)]));
    const vec16 um4i = *((vec16 *)(&u[IDX(i-4,j,k)]));
    const vec16 um3i = *((vec16 *)(&u[IDX(i-3,j,k)]));
    const vec16 um2i = *((vec16 *)(&u[IDX(i-2,j,k)]));
    const vec16 um1i = *((vec16 *)(&u[IDX(i-1,j,k)]));
    const vec16 up1i = *((vec16 *)(&u[IDX(i+1,j,k)]));
    const vec16 up2i = *((vec16 *)(&u[IDX(i+2,j,k)]));
    const vec16 up3i = *((vec16 *)(&u[IDX(i+3,j,k)]));
    const vec16 up4i = *((vec16 *)(&u[IDX(i+4,j,k)]));
    const vec16 um4j = *((vec16 *)(&u[IDX(i,j-4,k)]));
    const vec16 um3j = *((vec16 *)(&u[IDX(i,j-3,k)]));
    const vec16 um2j = *((vec16 *)(&u[IDX(i,j-2,k)]));
    const vec16 um1j = *((vec16 *)(&u[IDX(i,j-1,k)]));
    const vec16 up1j = *((vec16 *)(&u[IDX(i,j+1,k)]));
    const vec16 up2j = *((vec16 *)(&u[IDX(i,j+2,k)]));
    const vec16 up3j = *((vec16 *)(&u[IDX(i,j+3,k)]));
    const vec16 up4j = *((vec16 *)(&u[IDX(i,j+4,k)]));
    const vec16 s0   = *((vec16 *)(&s[IDX(i,j,k)]));
    const vec16 v0   = *((vec16 *)(&v[IDX(i,j,k)]));
    // avoid unaligned vector load by shuffle of u0 and u0n
    const vec16 up1k = (vec16){u0[1],u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0]};
    const vec16 up2k = (vec16){u0[2],u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0],u0n[1]};
    const vec16 up3k = (vec16){u0[3],u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0],u0n[1],u0n[2]};
    const vec16 up4k = (vec16){u0[4],u0[5],u0[6],u0[7],u0[8],u0[9],u0[10],u0[11],u0[12],u0[13],u0[14],u0[15],u0n[0],u0n[1],u0n[2],u0n[3]};
    const vec16 stencil = coef0*u0 + // 1 MUL + 12 ADDS + 12 FMA (37 FLOPS) 
      cz4*(um4k + up4k) + cz3*(um3k + up3k) + cz2*(um2k + up2k) + cz1*(um1k + up1k) +
      cy4*(um4j + up4j) + cy3*(um3j + up3j) + cy2*(um2j + up2j) + cy1*(um1j + up1j) +
      cx4*(um4i + up4i) + cx3*(um3i + up3i) + cx2*(um2i + up2i) + cx1*(um1i + up1i);
    
    *(vec16 *)(&v[IDX(i,j,k)]) = v0 - 6.25f*u0 + s0*stencil; // 2 FMA (4 FLOPS)
    // initialize um1k and u0 to feed the next iteration (NO LOADS!!)
    // shuffle from u0 and u0n to get um1k,um2k,um3k and um4k
    um1k = (vec16){u0[15],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5],u0n[6],
      u0n[7],u0n[8],u0n[9],u0n[10],u0n[11],u0n[12],u0n[13],u0n[14]};
    um2k = (vec16){u0[14],u0[15],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5],u0n[6],
      u0n[7],u0n[8],u0n[9],u0n[10],u0n[11],u0n[12],u0n[13]};
    um3k = (vec16){u0[13],u0[14],u0[15],u0n[0],u0n[1],u0n[2],u0n[3],u0n[4],u0n[5],u0n[6],
      u0n[7],u0n[8],u0n[9],u0n[10],u0n[11],u0n[12]};
    um4k = up4k;
    u0 = u0n;
  }
}
#endif
