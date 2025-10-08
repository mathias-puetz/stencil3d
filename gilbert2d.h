#ifndef _GILBERT2D_H_
#define _GILBERT2D_H_

// index class allocates and initializes an array containing xsize*ysize pair elements
// that form a space filling curve of the (0..xsize-1,0..ysize-1) plane

class gilbert2d {

 public:
  
  typedef struct {
    unsigned int x;
    unsigned int y;
  } s_index2d;
  
  gilbert2d(const int width,const int height);

  ~gilbert2d(void);

  void shuffle(const int bin_size);

  inline const s_index2d * get_index(void) const { return index; }
  inline const int get_size(void) const { return xsize*ysize; }
  inline const int get_x(unsigned int i) const { return index[i].x; }
  inline const int get_y(unsigned int i) const { return index[i].y; }

 private:

  void init(s_index2d ** list, int x, int y, int ax, int ay, int bx, int by);

  inline int sgn(const int x) {

    return (((x) > 0) ? 1 : (((x) < 0) ? -1 : 0));
  }

  s_index2d * index;

  unsigned int xsize;
  unsigned int ysize;
  unsigned int alloc_size;
};

#endif
