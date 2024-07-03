// vim: tabstop=3: ai
///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: helper_funcs.h
//
// Description: Header-only misc function definitions not specific to FDTD simulation
//
///////////////////////////////////////////////////////////////////////////////

#include "helper_funcs.h"

//self-explanatory
bool check_file_exists(char *filename) {
   struct stat st;
   bool file_exists = stat(filename, &st)==0;
   if (!file_exists) printf("%s doesn't exist!\n",filename);
   return file_exists;
}

//malloc check malloc, and initialise to zero
//hard stop program if failed
void mymalloc(void **arr, uint64_t Nbytes) {
   *arr = malloc(Nbytes);
   if (*arr == NULL) {
      printf("Memory allocation failed");
      assert(true==false); //to break
   }
   //initialise to zero
   memset(*arr,0,(size_t)Nbytes);
}

//for sorting int64 arrays and returning keys
struct sort_int64_struct {
   int64_t val;
   int64_t idx;
};
//comparator (for FCC ABC nodes)
int cmpfunc_int64 (const void * a, const void * b) {
   if ( *(const int64_t*)a < *(const int64_t*)b ) return -1;
   if ( *(const int64_t*)a > *(const int64_t*)b ) return  1;
   return 0;
}
//comparator with indice keys (for FCC ABC nodes)
int cmpfunc_int64_keys (const void * a, const void * b) {
   if ( (*(const struct sort_int64_struct*)a).val < (*(const struct sort_int64_struct*)b).val) return -1;
   if ( (*(const struct sort_int64_struct*)a).val > (*(const struct sort_int64_struct*)b).val) return  1;
   return 0;
}

//sort and return indices
void qsort_keys(int64_t *val_arr, int64_t *key_arr, int64_t N) {
   struct sort_int64_struct *struct_arr;
   struct_arr = (struct sort_int64_struct*) malloc(N*sizeof(struct sort_int64_struct));
   if (struct_arr == NULL) {
      printf("Memory allocation failed");
      assert(true==false); //to break
   }
   for (int64_t i=0; i<N; i++) {
      struct_arr[i].val = val_arr[i];
      struct_arr[i].idx = i;
   }
   qsort(struct_arr,N,sizeof(struct sort_int64_struct),cmpfunc_int64_keys);
   for (int64_t i=0; i<N; i++) {
      val_arr[i] = struct_arr[i].val;
      key_arr[i] = struct_arr[i].idx;
   }
   free(struct_arr);
}
