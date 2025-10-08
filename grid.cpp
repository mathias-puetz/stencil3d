#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "grid.h"

s_grid::s_grid(const unsigned int xs,const unsigned int ys, const unsigned int zs,
	       const unsigned int xh,const unsigned int yh, const unsigned int zh,
	       const unsigned int xb,const unsigned int yb, const unsigned int zb,
	       const real sm, const unsigned int kshuffle,const unsigned int jshuffle)
  : xsize(xs),ysize(ys),zsize(zs),xhalo(xh),yhalo(yh),zhalo(zh),
    xb_size(xb),yb_size(yb),zb_size(zb),sb_mib(sm),jshuf(jshuffle),kshuf(kshuffle),
    checkerboard(NULL),glist(NULL)
{
  
  printf("Initializing grid structure\n");

  jmult = zsize + 2*zhalo;
  imult = jmult*(ysize + 2*yhalo);
  ijk_ofs = zhalo + yhalo*jmult + xhalo*imult;

  gsize = (zsize+2*zhalo)*(ysize+2*yhalo)*(xsize+2*xhalo);
  gvol = zsize*ysize*xsize;

  printf("... grid dimensions %d x %d x %d with halo %d x %d x %d\n",xsize,ysize,zsize,xhalo,yhalo,zhalo);
  printf("... with %d inner grid points and %d total gridpoints\n",gvol,gsize);
  
  init_blocks();

  glist = new gilbert2d(xsize,ysize);
  
#ifdef CSHUFFLE
  // check compatibility of shuffle parameter with yblocksize and isize
  const unsigned int grem = (yb_size*xsize) & 511;
  if ((yb_size % shuffle) == 0) {
    fprintf(stderr,"shuffle parameter %d must not be a divisor of y-blocksize %d\n",shuffle,yb_size);
    exit(2);
  }
  if ((grem % shuffle) == 0) {
    fprintf(stderr,"shuffle parameter %d must not be a divisor of xy-block 512-group remainder %d x %d mod 512 = %d\n",shuffle,yb_size,xsize,grem);
    exit(2);
  }
#endif

  init_superblocks();
}

s_grid::~s_grid(void) {
  
  if (checkerboard) free(checkerboard);
  if (glist) delete glist;
}

void s_grid::init_blocks() {

  xb_size = MIN(xb_size,xsize);
  yb_size = MIN(yb_size,ysize);
  zb_size = MIN(zb_size,zsize);
  xblocks = xsize / xb_size + MIN(xsize % xb_size,1);
  yblocks = ysize / yb_size + MIN(ysize % yb_size,1);
  zblocks = zsize / zb_size + MIN(zsize % zb_size,1);
  nblocks = xblocks * yblocks * zblocks;  
  
  bvol = zb_size*yb_size*xb_size;
  bsize = bvol + zb_size*yb_size*8 + zb_size*xb_size*8 + yb_size*xb_size*8;
}

void s_grid::init_superblocks() {
  
  // total grid size including halos, blocksize and super-blocks
  printf("Optimizing superblocks\n");

  // determine the super-block dimensions, while trying to fit it into sb_mib MiB

  unsigned int svol = (sb_mib*1024*1024)/(3*sizeof(real));
  unsigned int asize = cbrt((double)svol); // initially prefer cubic shaped super-blocks

  printf("...  target superblock volume = %d\n",svol);
  printf("...  target superblock cubic box size = %d\n",asize);

  // determine super-block size in z-direction first
  if (asize > 0.5*zsize) {  // prefer full z-columns, if super-block > zsize/2
    printf("...  z: expanding superblock to entire grid size %d\n",zsize);
    zsb_size = zsize;
    zsblocks = 1;
    zblocks_per_sb = zblocks;
  }
  else {
    if (asize < zb_size) asize = zb_size;
    zsblocks = zsize / asize + MIN(zsize % asize,1);                           // first estimate
    zsb_size = zsize / zsblocks + MIN(zsize % zsblocks,1);           // first estimate
    zblocks_per_sb = zsb_size / zb_size + MIN(zsb_size % zb_size,1); // round up to nearest block size multiple
    zsb_size = zb_size * zblocks_per_sb;                                       // correction
    zsblocks = zsize / zsb_size + MIN(zsize % zsb_size,1);           // correction
  }
  printf("...  z: %d superblocks of size %d with %d blocks\n",zsblocks,zsb_size,zblocks_per_sb);
  
  // then in y-direction
  asize = MIN((unsigned int)sqrt(svol/zsb_size),ysize);
  if (asize > 0.5*ysize) { // prefer full yz-planes, if super-block > ysize/2
    printf("...  y: expanding superblock to entire grid size %d\n",ysize);
    ysb_size = ysize;
    ysblocks = 1;
    yblocks_per_sb = yblocks;
  }
  else {
    if (asize < yb_size) asize = yb_size;
    ysblocks = ysize / asize + MIN(ysize % asize,1);
    ysb_size = ysize / ysblocks + MIN(ysize % ysblocks,1);           // first estimate
    yblocks_per_sb = ysb_size / yb_size + MIN(ysb_size % yb_size,1); // round up to nearest block size multiple
    ysb_size = yb_size * yblocks_per_sb;                             // correction
    ysblocks = ysize / ysb_size + MIN(ysize % ysb_size,1);           // correction
  }
  printf("...  y: %d superblocks of size %d with %d blocks\n",ysblocks,ysb_size,yblocks_per_sb);

  // finally in x-direction
  asize = MIN(svol/(ysb_size*zsb_size),xsize);
  if (asize < yb_size) asize = xb_size;
  xsblocks = xsize / asize + MIN(xsize % asize,1);
  xsb_size = xsize / xsblocks + MIN(xsize % xsblocks,1);           // first estimate
  xblocks_per_sb = xsb_size / xb_size + MIN(xsb_size % xb_size,1); // round up to nearest block size multiple
  xsb_size = xb_size * xblocks_per_sb;                             // correction
  xsblocks = xsize / xsb_size + MIN(xsize % xsb_size,1);           // correction
  printf("...  x: %d superblocks of size %d with %d blocks\n",xsblocks,xsb_size,xblocks_per_sb);

  sbvol = xsb_size*ysb_size*zsb_size;
  sbsize = sbvol + 2*4*(xsb_size*ysb_size + xsb_size*zsb_size + ysb_size*zsb_size);

  printf("...  superblock cache foot print = %ld MiB\n",(sbsize*3*sizeof(real))/(1024*1024));
}

void s_grid::init_checkerboard(void)
{
  const unsigned int nodd = nblocks / 2;
  const unsigned int neven = nblocks - nodd;
  const unsigned int nalloc = ((sizeof(s_block)*nblocks)/64+1)*64; // allocate cache line aligned
  
  unsigned int * list_even = (unsigned int *)malloc(sizeof(unsigned int)*neven);
  unsigned int * list_odd = (unsigned int *)malloc(sizeof(unsigned int)*nodd);
  checkerboard = (s_block *)aligned_alloc(64,nalloc);

  assert(list_even);
  assert(list_odd);
  assert(checkerboard);

  printf("Initializing checkerboard index\n");
  printf("...      blocks %d = %d x %d x %d of grid size %d x %d x %d\n",
	 nblocks,xblocks,yblocks,zblocks,xb_size,yb_size,zb_size);
  printf("... superblocks %d = %d x %d x %d of grid size %d x %d x %d organized in %d x %d x %d blocks\n",
	 xsblocks*ysblocks*zsblocks,xsblocks,ysblocks,zsblocks,xsb_size,ysb_size,zsb_size,
	 xblocks_per_sb,yblocks_per_sb,zblocks_per_sb);
  
  unsigned int odd = 0;
  unsigned int even = 0;

  // outer loop over starting point (is,js,ks) of superblocks
  for(unsigned int is = 0;is < xsize;is += xsb_size)
    for(unsigned int js = 0;js < ysize;js += ysb_size)
      for(unsigned int ks = 0;ks < zsize;ks += zsb_size) {
	// inner loop over blocks within superblock
	for(unsigned int i = is;i < MIN(is + xsb_size,xsize);i += xb_size)
	  for(unsigned int j = js;j < MIN(js + ysb_size,ysize);j += yb_size)
	    for(unsigned int k = ks;k < MIN(ks + zsb_size,zsize);k += zb_size) {
	      unsigned int ofs = k + zsize*(j + ysize*i);
#ifdef DEBUG
	      printf("(%d,%d,%d) (%d,%d,%d) ofs = %d odd = %d\n",
		     is,js,ks,i,j,k,ofs,((i/xb_size + j/yb_size + k/zb_size) & 1));
#endif
	      if (((i/xb_size + j/yb_size + k/zb_size) & 1) == 0) {
		if (even < neven) {
		  list_even[even++] = ofs;
		}
		else {
		  fprintf(stderr,"(%d,%d,%d) (%d,%d,%d) ofs = %d odd = %d\n",
			  is,js,ks,i,j,k,ofs,((i/xb_size + j/yb_size + k/zb_size) & 1));
		  fprintf(stderr,"even(%d) >= neven(%d)\n",even,neven);
		  fflush(stdout);
		  exit(3);
		}
	      }
	      else {
		if (odd < nodd) {
		  list_odd[odd++] = ofs;
		}
		else {
		  fprintf(stderr,"(%d,%d,%d) (%d,%d,%d) ofs = %d odd = %d\n",
			  is,js,ks,i,j,k,ofs,((i/xb_size + j/yb_size + k/zb_size) & 1));
		  fprintf(stderr,"odd(%d) >= nodd(%d)\n",odd,nodd);
		  fflush(stdout);
		  exit(3);
		}
	      }
	    }
      }

  unsigned int sbblocks = (xblocks_per_sb * yblocks_per_sb* zblocks_per_sb)/2;
  printf("... shuffling checkerboard in chunks of size %d blocks\n",sbblocks);

  // shuffling even and odd blocks in chunks of superblocks
  // This ensures cache re-use, while the odd-even coloring minimizes cacheline conflicts in shared cache

  odd = 0;
  even = 0;
  for(unsigned int ijk = 0;ijk < nblocks;ijk++) {
    unsigned int ofs = (((ijk/sbblocks) & 1) == 0 && even < neven) ? list_even[even++] : list_odd[odd++];
    unsigned int ij = ofs / zsize;
    checkerboard[ijk].imin = ij / ysize;
    checkerboard[ijk].jmin = ij % ysize;
    checkerboard[ijk].kmin = ofs % zsize;
    // safeguard the max boundaries
    checkerboard[ijk].imax = MIN(checkerboard[ijk].imin + xb_size,xsize);
    checkerboard[ijk].jmax = MIN(checkerboard[ijk].jmin + yb_size,ysize);
    checkerboard[ijk].kmax = MIN(checkerboard[ijk].kmin + zb_size,zsize);
#ifdef DEBUG
    printf("%d (%d,%d) ofs = %d i = (%d,%d) j = (%d,%d) k = (%d,%d)\n",
	   ijk,even-1,odd-1,ofs,
	   checkerboard[ijk].imin,checkerboard[ijk].imax,
	   checkerboard[ijk].jmin,checkerboard[ijk].jmax,
	   checkerboard[ijk].kmin,checkerboard[ijk].kmax);
#endif
  }
#ifdef DEBUG
  fflush(stdout);
#endif
  
  free(list_even);
  free(list_odd);
}
