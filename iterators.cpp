#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#ifdef USE_NS_API
#include <nsapi/parallelism.hpp>
#endif
#include "iterators.h"
//#define DEBUG

#define ITERATION_HEADER(grid)						\
  const unsigned int imult = grid.imult;				\
  const unsigned int jmult = grid.jmult;				\
  const unsigned int ijk_ofs = grid.ijk_ofs;				\
  const unsigned int ksize = grid.zsize;				\
  const unsigned int jsize = grid.ysize/JUNROLL;			\
  const unsigned int isize = grid.xsize;				\
  const unsigned int ijsize = isize*jsize;				\
  const unsigned int iblocksize = grid.xb_size;				\
  const unsigned int jblocksize = grid.yb_size/JUNROLL;			\
  const unsigned int kblocksize = grid.zb_size;				\
  const unsigned int iblocks = grid.xblocks;				\
  const unsigned int jblocks = grid.yblocks;				\
  const unsigned int kblocks = grid.zblocks;				\
  const unsigned int jshuffle = grid.jshuf;				\
  const unsigned int kshuffle = grid.kshuf;				\
  const unsigned int nblocks = grid.nblocks;				\
  const unsigned int ijksize = ijsize*kblocks;				\
  const unsigned int isize_dups = isize / dups;				\
  unsigned int iter = 0;						\
  __builtin_assume(jmult > 0);						\
  __builtin_assume(imult > 0);						\
  __builtin_assume(ijk_ofs > 0);					\
  __builtin_assume(nblocks > 0);					\
  __builtin_assume(ijksize > 0);					\
  __builtin_assume(ijsize > 0);						\
  __builtin_assume(isize > 0);						\
  __builtin_assume(jsize > 0);                                          \
  __builtin_assume(ksize > 0)

#ifdef NODUPS
#ifdef SHUFFLE
// dijk iteration space:  jblocks * isize/dups * jblocksize * kblocks = isize * jsize * kblocks;
#define DECODE_DIJK							\
  const unsigned int dup    = 1;					\
  const unsigned int jbi    = dijk / (jblocksize * kblocks);		\
  const unsigned int jkb    = dijk - jbi * jblocksize * kblocks;	\
  const unsigned int jb     = jbi / isize;				\
  const unsigned int i      = jbi - jb * isize;				\
  const unsigned int j      = jkb / kblocks;				\
  const unsigned int ks     = jkb - j * kblocks;			\
  const unsigned int kb     = (ks * kshuffle) % kblocks;		\
  const unsigned int kmin   = kb * kblocksize;				\
  const unsigned int kmax   = MIN(kmin + kblocksize,ksize);		\
  const unsigned int jofs   = (j * jshuffle) % jblocksize;			\
  const unsigned int j0     = (jofs + jb*jblocksize)*JUNROLL
#else
// dijk iteration space:  jblocks * isize/dups * jblocksize * kblocks = isize * jsize * kblocks;
#define DECODE_DIJK							\
  const unsigned int dup    = 1;					\
  const unsigned int jbi    = dijk / (jblocksize * kblocks);		\
  const unsigned int jkb    = dijk - jbi * jblocksize * kblocks;	\
  const unsigned int jb     = jbi / isize;				\
  const unsigned int i      = jbi - jb * isize;				\
  const unsigned int j      = jkb / kblocks;				\
  const unsigned int ks     = jkb - j * kblocks;			\
  const unsigned int kb     = ks;					\
  const unsigned int kmin   = kb * kblocksize;				\
  const unsigned int kmax   = MIN(kmin + kblocksize,ksize);		\
  const unsigned int jofs   = j;					\
  const unsigned int j0     = (jofs + jb*jblocksize)*JUNROLL
#endif

#else
// dijk iteration space:  dups * jblocks * isize/dups * jblocksize * kblocks = isize * jsize * kblocks;
#define DECODE_DIJK							\
  const unsigned int djbi   = dijk / (jblocksize * kblocks);		\
  const unsigned int jkb    = dijk - djbi * jblocksize * kblocks;	\
  const unsigned int dup    = djbi / (isize_dups * jblocks);		\
  const unsigned int jbi    = djbi - dup * isize_dups * jblocks;	\
  const unsigned int jb     = jbi / isize_dups;				\
  const unsigned int i      = (jbi - jb * isize_dups)*dups + dup;	\
  const unsigned int jofs   = jkb / kblocks;				\
  const unsigned int kb     = jkb - jofs * kblocks;			\
  const unsigned int kmin   = kb * kblocksize;				\
  const unsigned int kmax   = MIN(kmin + kblocksize,ksize);		\
  const unsigned int j0     = (jofs + jb*jblocksize)*JUNROLL
#endif


void inner_kernel_kmax(const unsigned int dijk,
		       real * restrict v, const real * restrict s, const real * restrict u,
		       const unsigned int isize, const unsigned int jsize, const unsigned int ksize,
		       const unsigned int dups, const unsigned int isize_dups,
		       const unsigned int jblocks, const unsigned int kblocks,
		       const unsigned int jblocksize, const unsigned int kblocksize,
		       const unsigned int jshuffle, const unsigned int kshuffle,
		       const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  DECODE_DIJK;

#ifdef DEBUG
  printf("%s: dijk = %d, dup = %d, i = %d , j = %d, kmin/kmax = %d/%d\n",__func__,dijk,dup,i,j0,kmin,kmax); 
#endif
      
  for (unsigned int k = kmin;k < kmax;k += VLEN) {
    STENCIL_FCT(v,s,u,i,j0,k,imult,jmult,ijk_ofs);
  }
}

void inner_kernel_vlen(const unsigned int dijk,
		       real * restrict v, const real * restrict s, const real * restrict u,
		       const unsigned int isize, const unsigned int jsize, const unsigned int ksize,
		       const unsigned int dups, const unsigned int isize_dups,
		       const unsigned int jblocks, const unsigned int kblocks,
		       const unsigned int jblocksize, const unsigned int kblocksize,
		       const unsigned int jshuffle, const unsigned int kshuffle,
		       const unsigned int imult, const unsigned int jmult, const unsigned int ijk_ofs) {
  
  DECODE_DIJK;

#ifdef DEBUG
      printf("%s: dijk = %d, dup = %d, i = %d , j = %d, kmin/kmax = %d/%d\n",__func__,dijk,dup,i,j0,kmin,kmax); 
#endif
      
      STENCIL_FCT(v,s,u,i,j0,kmin,imult,jmult,ijk_ofs);
}

__attribute__((noinline))
void iterate_3d(real * restrict v, const real * restrict s, const real * restrict u,
		const s_grid & grid,const unsigned int prefetch_dist, const unsigned int dups,
		const unsigned int iterations, const unsigned int cacheiter)
{
  ITERATION_HEADER(grid);

  do {
#if defined(SCHED_STATIC)
#pragma omp parallel for firstprivate(v,s,u,ijk_ofs,imult,jmult,dups,isize_dups,isize,jsize,ksize,ijksize,kblocks,jblocks,kblocksize,jblocksize,jshuffle,kshuffle) default(none) schedule(static) collapse(COLL)
#else
#pragma omp parallel for firstprivate(v,s,u,ijk_ofs,imult,jmult,dups,isize_dups,isize,jsize,ksize,ijksize,kblocks,jblocks,kblocksize,jblocksize,jshuffle,kshuffle) default(none) schedule(static,SCHED) collapse(COLL)
#endif
    //    for (unsigned int jb = 0;jb < jsize;jb+=jblocksize) {
      for (unsigned int i = 0;i < isize;i++) {
	//	for (unsigned int j = jb;j < jb+jblocksize;j++) {
	  for (unsigned int j = 0;j < isize;j++) {
	    //#pragma ns vectorize predicate
#pragma unroll(2)
	    for (unsigned int k = 0;k < ksize;k++) {
	      STENCIL_FCT(v,s,u,i,j,k,imult,jmult,ijk_ofs);
	    }
	  }
	}
      //   }
  //    }
  } while (++iter < iterations);
}

__attribute__((noinline))
void iterate_3d_jk_blocked(real * restrict v, const real * restrict s, const real * restrict u,
			   const s_grid & grid,const unsigned int prefetch_dist, const unsigned int dups,
			   const unsigned int iterations, const unsigned int cacheiter)
{
  ITERATION_HEADER(grid);

  do {
    // using HW iteration counter
#ifdef HWITER
    nsapi::parallel_for<nsapi::work_distribution::STATIC_DYNAMIC>(ijksize,
								  inner_kernel_vlen,    
								  v,s,u,
								  isize,jsize,ksize,
								  dups,isize_dups,
								  jblocks,kblocks,
								  jblocksize,kblocksize,
								  jshuffle,kshuffle,
								  imult,jmult,ijk_ofs);
#else
#if defined(SCHED_STATIC)
#pragma omp parallel for firstprivate(v,s,u,ijk_ofs,imult,jmult,dups,isize_dups,isize,jsize,ksize,ijksize,kblocks,jblocks,kblocksize,jblocksize,jshuffle,kshuffle) default(none) schedule(static)
#else
#pragma omp parallel for firstprivate(v,s,u,ijk_ofs,imult,jmult,dups,isize_dups,isize,jsize,ksize,ijksize,kblocks,jblocks,kblocksize,jblocksize,jshuffle,kshuffle) default(none) schedule(static,SCHED)
#endif
    for (unsigned int dijk = 0; dijk < ijksize; dijk++) {
      DECODE_DIJK;

#ifdef DEBUG
      printf("%s: dijk = %d, dup = %d, i = %d , j = %d, kmin/kmax = %d/%d\n",__func__,dijk,dup,i,j0,kmin,kmax); 
#endif
    
      STENCIL_FCT(v,s,u,i,j0,kmin,imult,jmult,ijk_ofs);
    }
#endif
  } while (++iter < iterations);
}

__attribute__((noinline))
void iterate_2d_jk_blocked(real * restrict v, const real * restrict s, const real * restrict u,
			   const s_grid & grid, const unsigned int prefetch_dist, const unsigned int dups,
			   const unsigned int iterations, const unsigned int cacheiter)
{
  ITERATION_HEADER(grid);

  do {
    // using HW iteration counter
#ifdef HWITER
    nsapi::parallel_for<nsapi::work_distribution::STATIC_DYNAMIC>(ijksize,
								  inner_kernel_kmax,    
								  v,s,u,
								  isize,jsize,ksize,
								  dups,isize_dups,
								  jblocks,kblocks,
								  jblocksize,kblocksize,
								  jshuffle,kshuffle,
								  imult,jmult,ijk_ofs);
#else
#if defined(SCHED_STATIC)
#pragma omp parallel for firstprivate(v,s,u,ijk_ofs,imult,jmult,dups,isize_dups,isize,jsize,ksize,ijksize,kblocks,jblocks,kblocksize,jblocksize,jshuffle,kshuffle) default(none) schedule(static)
#else
#pragma omp parallel for firstprivate(v,s,u,ijk_ofs,imult,jmult,dups,isize_dups,isize,jsize,ksize,ijksize,kblocks,jblocks,kblocksize,jblocksize,jshuffle,kshuffle) default(none) schedule(static,SCHED)
#endif
    for (unsigned int dijk = 0; dijk < ijksize; dijk++) {
      DECODE_DIJK;

#ifdef DEBUG
      printf("%s: dijk = %d, dup = %d, i = %d , j = %d, kmin/kmax = %d/%d\n",__func__,dijk,dup,i,j0,kmin,kmax); 
#endif
      for (unsigned int k = kmin;k < kmax;k += VLEN) {
	STENCIL_FCT(v,s,u,i,j0,k,imult,jmult,ijk_ofs);
      }
    }
#endif
  } while (++iter < iterations);
}


__attribute__((noinline))
void iterate_2d_hilbert(real * restrict v, const real * restrict s, const real * restrict u,
			const s_grid & grid, const unsigned int prefetch_dist, const unsigned int dups,
			const unsigned int iterations, const unsigned int cacheiter)
{
  ITERATION_HEADER(grid);
  const gilbert2d::s_index2d * pair = grid.get_gilbert_index();

  do {
#ifdef SCHED_STATIC
#pragma omp parallel for firstprivate(v,s,u,pair,ijk_ofs,imult,jmult,ijsize,ksize) default(none) schedule(static)
#else
#pragma omp parallel for firstprivate(v,s,u,pair,ijk_ofs,imult,jmult,ijsize,ksize) default(none) schedule(static,1)
#endif
    for (unsigned int ij = 0; ij < ijsize; ij++) {
      const unsigned int i = pair[ij].x;
      const unsigned int j = pair[ij].y;
      
      for (unsigned int k = 0;k < ksize;k+=VLEN) {
	STENCIL_FCT(v,s,u,i,j,k,imult,jmult,ijk_ofs);
      }
    }   
  } while (++iter < iterations);
}

__attribute__((noinline))
void iterate_2d_checkerboard(real * restrict v, const real * restrict s, const real * restrict u,
			     const s_grid & grid,const unsigned int prefetch_dist, const unsigned int dups,
			     const unsigned int iterations, const unsigned int cacheiter)
{
  ITERATION_HEADER(grid);
  const s_grid::s_block * restrict clist = grid.get_checkerboard_index();

  do {
#pragma omp parallel for firstprivate(v,s,u,clist,ijk_ofs,imult,jmult,nblocks,cacheiter,prefetch_dist) default(none) schedule(static,1)
    for (unsigned int ijk = 0; ijk < nblocks; ijk++) {
      const unsigned int imin = clist[ijk].imin;
      const unsigned int imax = clist[ijk].imax;
      const unsigned int jmin = clist[ijk].jmin;
      const unsigned int jmax = clist[ijk].jmax;
      const unsigned int kmin = clist[ijk].kmin;
      const unsigned int kmax = clist[ijk].kmax;

      __builtin_prefetch(clist + ijk + prefetch_dist);

      for (unsigned int i = imin;i < imax;i++) {
	for (unsigned int j = jmin;j < jmax;j++) {
	  STENCIL_FCT(v,s,u,i,j,kmin,imult,jmult,ijk_ofs);
	}
      }
    }
  } while (++iter < iterations);
}


__attribute__((noinline))
void init_arrays(real * restrict v, real * restrict s, real * restrict u,unsigned int gsize) {

#pragma omp parallel for firstprivate(u,s,v,gsize) schedule(static)
  for(unsigned int i = 0;i < gsize;i++) {
    v[i] = 0.0f;
    u[i] = 1.0f;
    s[i] = 1.0f;
  }
}


__attribute__((noinline))
void check(const real * restrict v, const s_grid & grid, const unsigned int maxiter) {

  const unsigned int isize = grid.xsize;
  const unsigned int jsize = grid.ysize;
  const unsigned int ksize = grid.zsize;
  const unsigned int imult = grid.imult;
  const unsigned int jmult = grid.jmult;
  const unsigned int ijk_ofs = grid.ijk_ofs;
  unsigned int error = 0;
  
#pragma omp parallel for firstprivate(v,isize,jsize,ksize,imult,jmult,ijk_ofs,maxiter) collapse(2) reduction(+:error) schedule(static,1)
  for(unsigned int i = 0;i < isize;i++) {
    for(unsigned int j = 0;j < jsize;j++) {
      for(unsigned int k = 0;k < ksize;k++) {
	if (fabs(v[IDX(i,j,k)] - (real)maxiter) > 0.1f) error++;
      }
    }
  }

  printf("check: %d errors found\n",error);
}
