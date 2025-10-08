#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include "aux.h"

__attribute__((noinline))
double seconds() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)tv.tv_sec+1e-6*tv.tv_usec;
}

__attribute__((noinline))
int get_int_arg(char *argv[], int iarg, int argc,
					const char * opt,unsigned int & val,unsigned int minval,unsigned int maxval,unsigned int modval) {

  if (strcmp(argv[iarg],opt) == 0) {
    if (iarg+1 < argc) {
      val = atoi(argv[iarg+1]);
      if (val < minval) {
	fprintf(stderr,"%s must be larger or equal to %d\n",opt,minval);
	exit(2);
      }
      if (val > maxval) {
	fprintf(stderr,"%s must be smaller or equal to %d\n",opt,maxval);
	exit(2);
      }
      if ((val % modval) != 0) {
	fprintf(stderr,"%s must be a multiple of %d\n",opt,modval);
	exit(2);
      }
      printf("%s set to %d\n",opt,val);
      return 2;
    }
    else {
      fprintf(stderr,"%s requires a numeric argument [%d - %d]\n",opt,minval,maxval);
      exit(2);
    }
  }
  return 0;
}

__attribute__((noinline))
int get_real_arg(char *argv[], int iarg, int argc,
                 const char * opt,real & val,real minval,real maxval) {

  if (strcmp(argv[iarg],opt) == 0) {
    if (iarg+1 < argc) {
      val = atof(argv[iarg+1]);
      if (val < minval) {
	fprintf(stderr,"%s must be larger or equal to %f\n",opt,minval);
	exit(2);
      }
      if (val > maxval) {
	fprintf(stderr,"%s must be smaller or equal to %f\n",opt,maxval);
	exit(2);
      }
      printf("%s set to %f\n",opt,val);
      return 2;
    }
    else {
      fprintf(stderr,"%s requires a float argument [%f - %f]\n",opt,minval,maxval);
      exit(2);
    }
  }
  return 0;
}


__attribute__((noinline))
int get_flag(char *argv[], int iarg, int argc, const char * opt,unsigned int & flag) {

  if (strcmp(argv[iarg],opt) == 0) {
    flag = 1;
    printf("%s set to ENABLED\n",opt);
    return 1;
  }
  return 0;
}
