#ifndef _STENCILS_H_
#define _STENCILS_H_

// 3d array index access macro
#define IDX(i,j,k) (ijk_ofs + (k) + (j)*jmult + (i)*imult)

#include "grid.h"

#include "stencil_2.h"
#include "stencil_7.h"
#include "stencil_13.h"
#include "stencil_25.h"

#endif
