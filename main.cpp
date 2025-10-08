#include <assert.h>
#include <math.h>
#include <sys/mman.h>
#ifdef MPI
#include <mpi.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#ifdef USE_NS_API
#include <nsapi/memory.h>
#endif

#ifdef TELEM_REGION
#include "nsapi/telem.h"
#endif

#include "aux.h"
#include "iterators.h"

int is_prime(unsigned int number)
{
  unsigned int maxp = sqrt((double)number);
  unsigned int p = 2;
  while (p <= maxp)
    if ((number % p++) == 0) return 0;
  return 1;
}

void migrate_arrays(real * u, real * s, real * v,unsigned int asize,unsigned int direction) {
#ifdef USE_NS_API
  int err;
  double t0,t1;

  if (direction == 1) {
    device_location_request loc = {};
    
    // migrate to quad 0 only balancing evenly across 4 double rows
    t0 = seconds();
    for (unsigned int dup;dup < 4;dup++) {
      loc.location.chip = 0;
      loc.location.quad = 0;
      loc.location.row = dup*4;
      loc.location.col = 0;
      loc.location.quad = 0;
      loc.location.dcode = 1;
      loc.rpt_location = RPT_DIRECTIVE;
      loc.rpt_dcode = RPT_DIRECTIVE;
      loc.lp = RLP_CLUSTER;
      
      err = nsapi_mem_migrate_ex(v+dup*asize/4,asize/4,NSAPI_PAGE_LOC_DEVICE,false,loc);
      if (err) {
	fprintf(stderr,"error %d migrating v[]\n",err);
	exit(3);
      }
      err = nsapi_mem_migrate_ex(s+dup*asize/4,asize/4,NSAPI_PAGE_LOC_DEVICE,false,loc);
      if (err) {
	fprintf(stderr,"error %d migrating s[]\n",err);
	exit(3);
      }
      err = nsapi_mem_migrate_ex(u+dup*asize/4,asize/4,NSAPI_PAGE_LOC_DEVICE,false,loc);
      if (err) {
	fprintf(stderr,"error %d migrating u[]\n",err);
	exit(3);
      }
    }
    t1 = seconds();
    printf("migration to device took %f secs: %f GB/s\n", t1-t0,(double)asize*1e-9/(t1-t0));
  }
  else {
    t0 = seconds();
    err = nsapi_mem_migrate(v,asize,NSAPI_PAGE_LOC_HOST,false);
    if (err) {
      fprintf(stderr,"error %d migrating v[]\n",err);
      exit(3);
    }
    err = nsapi_mem_migrate(s,asize,NSAPI_PAGE_LOC_HOST,false);
    if (err) {
      fprintf(stderr,"error %d migrating s[]\n",err);
      exit(3);
    }
    err = nsapi_mem_migrate(u,asize,NSAPI_PAGE_LOC_HOST,false);
    if (err) {
      fprintf(stderr,"error %d migrating u[]\n",err);
      exit(3);
    }
    t1 = seconds();
      printf("migration from device took %f secs: %f (GB/s)\n", t1-t0,(double)asize*1e-9/(t1-t0));
  }
#endif
}

int main(int argc, char **argv)
{
  unsigned int train = 0;
  unsigned int offset = DEFAULT_OFS;        // offset is measured in number of reals
  unsigned int maxiter = DEFAULT_ITER;      // total number of iteration per measurement (refined below)
  unsigned int cacheiter = 1;               // total number of iteration inside the mill for emulating cache re-use
  unsigned int dups = 1;
  unsigned int prefetch = PREFETCH_DIST_CLIST;
  unsigned int xs = DEFAULT_GRIDSIZE_XY;
  unsigned int ys = DEFAULT_GRIDSIZE_XY;
  unsigned int zs = DEFAULT_GRIDSIZE_Z;
  unsigned int xh = DEFAULT_HALO_XY;
  unsigned int yh = DEFAULT_HALO_XY;
  unsigned int zh = DEFAULT_HALO_Z;
  unsigned int xb = X_TILE_SZ;
  unsigned int yb = Y_TILE_SZ;
  unsigned int zb  = (Z_TILE_SZ) ? Z_TILE_SZ : zs;
  real sbm = SUPERBLOCK_MIB_TARGET;
  unsigned int jshuffle = 13;
  unsigned int kshuffle = 3;
  int my_rank = 0;
  int size = 1;
  real *ov,*os,*ou;
  real *v,*s,*u;
  double t0,t1;
  double flops,mem_bytes,cache_bytes;
  char unit[] = "GFlop/s";
  
  int iarg = 1;  
  while (iarg < argc) {
    unsigned int args = 0;
    args += get_int_arg(argv,iarg,argc,"--iter",maxiter,1,100000,1);
    args += get_int_arg(argv,iarg,argc,"--citer",cacheiter,1,100000,1);
    args += get_int_arg(argv,iarg,argc,"--dups",dups,1,64,1);
    args += get_int_arg(argv,iarg,argc,"--ofs",offset,0,16384,1);
    args += get_int_arg(argv,iarg,argc,"--pref",prefetch,1,512,1);
#ifdef SHUFFLE
    args += get_int_arg(argv,iarg,argc,"--jshuffle",jshuffle,1,31,1);
    args += get_int_arg(argv,iarg,argc,"--kshuffle",kshuffle,1,31,1);
#endif
    args += get_int_arg(argv,iarg,argc,"--xh",xh,4,16,1);
    args += get_int_arg(argv,iarg,argc,"--yh",yh,4,16,1);
    args += get_int_arg(argv,iarg,argc,"--zh",zh,VLEN,16,VLEN);
    args += get_int_arg(argv,iarg,argc,"--xs",xs,1,1264,1);
    args += get_int_arg(argv,iarg,argc,"--ys",ys,1,1264,1);
    args += get_int_arg(argv,iarg,argc,"--zs",zs,VLEN,1264,VLEN);
    args += get_int_arg(argv,iarg,argc,"--xb",xb,1,1264,1);
    args += get_int_arg(argv,iarg,argc,"--yb",yb,1,1264,1);
    args += get_int_arg(argv,iarg,argc,"--zb",zb,VLEN,1264,VLEN);
    args += get_real_arg(argv,iarg,argc,"--sbm",sbm,1.0,10.0);
    args += get_flag(argv,iarg,argc,"--train",train);
    if (args > 0) {
      iarg += args;
    }
    else {
      fprintf(stderr,"unsupported argument %s\n",argv[iarg]);
      return 2;
    }
  }

#if defined(PAR2D) || defined(HILBERT) || defined(CHECKERBOARD)
#else
  // HW iterator requires that zb = VLEN
  if (VLEN != zb) {
    fprintf(stderr,"WARNING: When compiled with HWITER=1, then zb must be forced to match VLEN. Force zb = %d\n",VLEN);
    zb = VLEN;
  }
#endif

#ifdef NODUPS
  // HW iterator requires that zb = VLEN
  if (dups != 1) {
    fprintf(stderr,"WARNING: When compiled with NODUPS=1, the only dups = %d is supported\n",dups);
    dups = 1;
  }
#endif

  if ((ys % yb) != 0) {
    fprintf(stderr,"ERROR: --ys must by an exact multiple of --yb\n");
    exit(2);
  }

  if ((JUNROLL > 1) && ((zb % JUNROLL) != 0)) {
    fprintf(stderr,"ERROR: --zb must by an exact multiple of j-unroll factor %d\n",JUNROLL);
    exit(2);
  }
  
#ifdef SHUFFLE
  if (!is_prime(jshuffle)) {
    fprintf(stderr,"--jshuffle parameter %d is not a prime number\n",jshuffle);
    exit(2);
  }
  if ((jshuffle > 1) && ((yb % jshuffle) == 0)) {
    fprintf(stderr,"--jshuffle parameter %d must not devide --jb %d\n",jshuffle,yb);
    exit(2);
  }
    
  if (!is_prime(kshuffle)) {
    fprintf(stderr,"--kshuffle parameter %d is not a prime number\n",kshuffle);
    exit(2);
  }
  if ((kshuffle > 1) && (((zs/zb) % kshuffle) == 0)) {
    fprintf(stderr,"--kshuffle parameter %d must not devide the number of zblocks = zs/zb = %d\n",kshuffle,zs/zb);
    exit(2);
  }
#endif
  
  s_grid grid = s_grid(xs,ys,zs,xh,yh,zh,xb,yb,zb,sbm,jshuffle,kshuffle);
  
  // the number of GFlop/s computed in one measurement
  flops = FLOPS_PER_STENCIL*1.0e-9*(double)grid.gvol*(double)maxiter*(double)cacheiter;
  mem_bytes = MEM_BYTES_PER_STENCIL*1.0e-9*(double)grid.gvol*(double)maxiter*(double)cacheiter;
  cache_bytes = CACHE_BYTES_PER_STENCIL*1.0e-9*(double)grid.gvol*(double)maxiter*(double)cacheiter;
  
  // grid.checkerboard_init();
  
#ifdef MPI  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (my_rank == 0) {
    printf("[rank%i] Phase #%i: MPI initialization complete\n", my_rank, ++phase);
  }
#else
  my_rank = 0;
  size = 1;
#endif
  
  if (my_rank == 0) {
    printf("running STENCIL(%d) benchmark: \n",STENCIL_SIZE);
    printf(" -- using a grid size of %dx%dx%d with a halo of %dx%dx%d elements totaling %f GB\n",
	   grid.xsize,grid.ysize,grid.zsize,grid.xhalo,grid.yhalo,grid.zhalo,1e-9*(double)grid.gsize*sizeof(real));
    printf(" -- using block tile size of %d x %d x %d\n",grid.xb_size,grid.yb_size,grid.zb_size);
    printf(" -- each block tile has a cache footprint of %ld (%ld) kiB\n",(grid.bvol*sizeof(real))/1024,(grid.bsize*sizeof(real))/1024);
#if defined(HWITER)
    printf(" -- scheduling using HW iteration counter.\n");
#elif defined(HILBERT)
    printf(" -- using a Hilbert curve in X-Y plane\n");
#elif defined(CHECKER)
    printf(" -- using a 3d checkerboard block tiling with red-black update in super-blocks\n");
#else
#if defined(SCHED_STATIC)
    printf(" -- scheduling policy is OpenMP STATIC \n");
#else
    printf(" -- scheduling policy is OpenMP STATIC,%d \n",SCHED);
#endif
#endif
  }
  
  int phase = 0;
  // we allocate arrays on page boundaries and then shift by offsets
  // allocation size must be a multiple of alignment size
  const size_t page_size = 1UL << 16; // page size of 64kB
  size_t asize = (((grid.gsize+64*offset)*sizeof(real)/page_size)+1)*page_size;

  ov = (real *)aligned_alloc(page_size,asize);
  os = (real *)aligned_alloc(page_size,asize);
  ou = (real *)aligned_alloc(page_size,asize);
  
  printf("madvise ov: %d\n",madvise(ov,asize,MADV_WILLNEED | MADV_HUGEPAGE));
  printf("madvise ov: %d\n",madvise(os,asize,MADV_WILLNEED | MADV_HUGEPAGE));
  printf("madvise ov: %d\n",madvise(ou,asize,MADV_WILLNEED | MADV_HUGEPAGE));

#ifdef INIT_ON_HOST
#pragma omp parallel for schedule(static)
#pragma ns location host
  for(size_t ofs = 0;ofs < asize;ofs += page_size) {
    unsigned int rofs = ofs / sizeof(real);
    memset(ov + rofs,0,page_size);
    memset(os + rofs,0,page_size);
    memset(ou + rofs,0,page_size);
  }
#endif
  
  assert(ov);
  assert(os);
  assert(ou);

  // shift the array starts by some CLs to minimize CL bank conflicts
  v = ov;
  s = os + offset*13;
  u = ou + offset*42;

  // migrate allocated memory to the device, if not in training mode

#ifdef USE_NS_API
  if (!train) migrate_arrays(ou,os,ou,asize,1);
#endif
  
#ifdef TELEM_REGION
  nsapi_telem_region_enter();
#endif

  printf("[rank%i] Phase #%i: array initialization\n", my_rank, ++phase);
  t0 = seconds();
  init_arrays(ov,os,ou,asize/sizeof(real));
  t1 = seconds();
  if (my_rank == 0) {
    printf("[rank%i] Phase #%i: initialization took %f secs\n", my_rank, ++phase,t1-t0);
  }
  
  printf("[rank%i] Phase #%i: initial warmup\n", my_rank, ++phase);
  t0 = seconds();
  ITERATE(v,s,u,grid,prefetch,dups,1U,cacheiter);
  t1 = seconds();
  if (my_rank == 0) {
    printf("[rank%i] Phase #%i: finish warmup in %f secs\n", my_rank, ++phase,t1-t0);
  }
  check(v,grid,1U);
  
  printf("[rank%i] Phase #%i: start compute\n", my_rank, ++phase);

  t0 = seconds();
  ITERATE(v,s,u,grid,prefetch,dups,maxiter,cacheiter);
  t1 = seconds();
  if (my_rank == 0) printf("[rank%i] Phase #%i: finish compute in %f secs (%f %s, mem: %f GB/s, cache: %f GB/s)\n",
			   my_rank, ++phase,t1-t0,flops/(t1-t0),unit,mem_bytes/(t1-t0)/cacheiter,cache_bytes/(t1-t0));

  check(v,grid,maxiter+1);
  printf("freeing data arrays\n");

#ifdef USE_NS_API
  if (!train) migrate_arrays(ou,os,ou,asize,0);
#endif
  
  free(ov);
  free(os);
  free(ou);

  printf("freeing indices\n");

#ifdef TELEM_REGION
  nsapi_telem_region_exit();
#endif
#ifdef MPI
  MPI_Finalize();
#endif
  printf("[rank%i] Phase #%i: done!\n", my_rank, ++phase);
}
