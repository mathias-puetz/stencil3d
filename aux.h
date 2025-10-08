#ifndef _AUX_H_
#define _AUX_H_

#include "config.h"

__attribute__((noinline))
double seconds();

__attribute__((noinline))
int get_int_arg(char *argv[], int iarg, int argc,
		const char * opt,unsigned int & val,unsigned int minval,unsigned int maxval,unsigned int modval);

__attribute__((noinline))
int get_real_arg(char *argv[], int iarg, int argc,
		 const char * opt,real & val,real minval,real maxval);


__attribute__((noinline))
int get_flag(char *argv[], int iarg, int argc, const char * opt,unsigned int & flag);

#endif
