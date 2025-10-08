#ifndef _GRID_H_
#define _GRID_H_

#include "config.h"
#include "gilbert2d.h"

class s_grid {

 public:

  typedef struct {
    
    unsigned int imin;
    unsigned int imax;
    unsigned int jmin;
    unsigned int jmax;
    unsigned int kmin;
    unsigned int kmax;
    unsigned int reserved[2];
    
  } s_block;

  s_grid(const unsigned int xs,const unsigned int ys, const unsigned int zs,
	 const unsigned int xh,const unsigned int yh, const unsigned int zh,
	 const unsigned int xb,const unsigned int yb, const unsigned int zb,
	 const real sm, const unsigned int jshuffle, const unsigned int kshuffle);
  
  ~s_grid(void);

  void init_checkerboard(void);
  
  inline const unsigned int idx(unsigned int i,unsigned int j,unsigned int k) {
    return ijk_ofs + k + jmult*j + imult*i;
  }

  inline const gilbert2d::s_index2d * get_gilbert_index(void) const { return glist->get_index(); }
  inline const s_block * get_checkerboard_index(void) const { return checkerboard; }
  
  unsigned int xsize,ysize,zsize;              // grid dimensions, zsize must be a multiple of 16
  unsigned int xhalo,yhalo,zhalo;              // halo cells (additional to grid dimensions, x/yhalo >= 4, zhalo must be a 8 (vlen=8) or 16 (vlen=16)
  unsigned int gsize;                          // total number of gridpoints including halo cells
  unsigned int gvol;                           // total number of inner gridpoints excluding halo cells
  unsigned int xb_size,yb_size,zb_size;        // the size of tiles
  unsigned int xblocks,yblocks,zblocks;        // the number of tiles in each direction
  unsigned int nblocks;                        // the total number of tiles
  unsigned int xsb_size,ysb_size,zsb_size;     // the number of tiles per super-block in each direction
  unsigned int bsize;                          // number of grid points in a block tile including its stencil halo
  unsigned int bvol;                           // number of inner grid points in a block tile excluding its stencil halo
  unsigned int xsblocks,ysblocks,zsblocks;     // the number of super-blocks in each direction
  unsigned int xblocks_per_sb,yblocks_per_sb,zblocks_per_sb; // the number blocks per super-block in each direction
  unsigned int nsblocks;                       // the total number of super-blocks
  unsigned int sbvol;                          // number of inner grid points in a superblock excluding its stencil halo
  unsigned int sbsize;                         // number of grid points in a superblock including its stencil halo
  unsigned int z_last;                         // flag: if 1, then zblocks are iterated last
  unsigned int jshuf;                          // prime number for shuffling j-indexes within a jblock
  unsigned int kshuf;                          // prime number for shuffling k-blocks
  real sb_mib;                                 // target size in MiB for a superblock

  unsigned int ijk_ofs;
  unsigned int jmult;
  unsigned int imult;

 private:

  s_block * checkerboard;
  gilbert2d * glist;
  
  void init_blocks(void);
  void init_superblocks(void);

};

#endif
