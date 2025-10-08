#ifndef _ITERATORS_H_
#define _ITERATORS_H_

#include "stencils.h"

__attribute__((noinline))
void iterate_3d(real * restrict v, const real * restrict s, const real * restrict u,
		const s_grid & grid, const unsigned int prefetch_dist, const unsigned int dups,
		const unsigned int iterations,const unsigned int cacheiter);

__attribute__((noinline))
void iterate_3d_jk_blocked(real * restrict v, const real * restrict s, const real * restrict u,
			   const s_grid & grid, const unsigned int prefetch_dist, const unsigned int dups,
			   const unsigned int iterations,const unsigned int cacheiter);

__attribute__((noinline))
void iterate_2d_jk_blocked(real * restrict v, const real * restrict s, const real * restrict u,
			   const s_grid & grid, const unsigned int prefetch_dist, const unsigned int dups,
			   const unsigned int iterations,const unsigned int cacheiter);

__attribute__((noinline))
void iterate_2d_hilbert(real * restrict v, const real * restrict s, const real * restrict u,
			const s_grid & grid,  const unsigned int prefetch_dist, const unsigned int dups,
			const unsigned int iterations,const unsigned int cacheiter);

__attribute__((noinline))
void iterate_2d_checkerboard(real * restrict v, const real * restrict s, const real * restrict u,
			     const s_grid & grid,  const unsigned int prefetch_dist, const unsigned int dups,
			     const unsigned int iterations,const unsigned int cacheiter);

__attribute__((noinline))
void init_arrays(real * restrict v,real * restrict s,real * restrict u,unsigned int gsize);

__attribute__((noinline))
void check(const real * restrict v, const s_grid & grid, const unsigned int maxiter);

#endif
