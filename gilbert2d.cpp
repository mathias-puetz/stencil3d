#include <stdio.h>
#include <stdlib.h>
#include "gilbert2d.h"

gilbert2d::gilbert2d(const int width,const int height)
  : xsize(width),ysize(width),index(NULL),alloc_size((xsize*ysize*sizeof(s_index2d)/64 + 256)*64)
{
  s_index2d * list = (s_index2d *)aligned_alloc(64,alloc_size); // allocate on cacheline boundary
  s_index2d * top = list;

  printf("Initializing 2d hilbert curve\n");
  
  if (list) init(&top, 0, 0, 0, xsize, ysize, 0);

  index = list;
}  

void gilbert2d::shuffle(const int bin_size)
{
  s_index2d * shuffle = (s_index2d *)aligned_alloc(64,alloc_size);
  int nbins = (xsize*ysize) / bin_size;

  if (nbins * bin_size < xsize*ysize) nbins++;

  int j = 0;
  for(int ib = 0;ib < nbins;ib++) {
    int i = ib;
    while (i < xsize*ysize) {
      //      printf("ib = %d, i = %d, j = %d\n",ib,i,j);
      shuffle[j] = index[i];
      j++;
      i+=nbins;
    }
  }
  // copy shuffled pair list back
  for(int i=0;i < xsize*ysize;i++) index[i] = shuffle[i];

  free(shuffle);
}

gilbert2d::~gilbert2d() {

  if (index) free(index);
}

// private recursive method that populates the gilbert2d pair array
void gilbert2d::init(s_index2d ** list, int x, int y, int ax, int ay, int bx, int by) {

  int width = abs(ax + ay); // why abs()
  int height = abs(bx + by);
  int width2;
  int height2;
  int dax,day,dbx,dby;
  int ax2,ay2,bx2,by2;

  // unit major direction
  dax = sgn(ax); 
  day = sgn(ay);
  // unit ortho direction
  dbx = sgn(bx);
  dby = sgn(by);

  if (height == 1) {
    // trivial row fill
    for(int i = 0; i < width;i++) {
      (*list)->x = x;
      (*list)->y = y;
      (*list)++;

      x += dax;
      y += day;
    }
    return;
  }
  else if (width == 1) {
    // trivial column fill
    for(int i = 0; i < width;i++) {
      (*list)->x = x;
      (*list)->y = y;
      (*list)++;

      x += dbx;
      y += dby;
    }
    return;
  }

  ax2 = ax/2;
  ay2 = ay/2;
  bx2 = bx/2;
  by2 = by/2;
  
  width2 = abs(ax2 + ay2);
  height2 = abs(bx2 + by2);

  if (2*width > 3*height) {
    
    if (width2 % 2 > 0 && width > 2) {
      // prefer even steps
      ax2 += dax;
      ay2 += day;
    }
      
    // long case: split in two parts only
    init(list, x, y, ax2, ay2, bx, by);
    init(list, x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by);
  }
  else {
    if (height2 % 2 > 0 && height > 2) {
      // prefer even steps
      bx2 += dbx;
      by2 += dby;
    }
    // standard case: one step up, one long horizontal, one step down
    init(list, x, y, bx2, by2, ax2, ay2);
    init(list, x+bx2, y+by2, ax, ay, bx-bx2, by-by2);
    init(list, x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),-bx2, -by2, -(ax-ax2), -(ay-ay2));
  }
}

#ifdef UNIT_TEST
int main(int argc, char **argv)
{
  const unsigned int xsize = 7;
  const unsigned int ysize = 7;

  gilbert2d glist(xsize,ysize);

  for (int i = 0;i < glist.getSize();i++) printf("gilbert: %d, %d\n",glist.getX(i),glist.getY(i));

  glist.shuffle(6);
  
  for (int i = 0;i < glist.getSize();i++) printf("gilbert: %d, %d\n",glist.getX(i),glist.getY(i));
}
#endif
