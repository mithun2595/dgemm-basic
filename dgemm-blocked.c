/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */
#include <emmintrin.h>
#include <stdlib.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_2_SIZE 200
#define BLOCK_1_SIZE 36
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static void do_transpose(double* B, int lda, int K, int N, double* T) {
	int K_ind = (K%2==1) ? K+1 : K;
	for (int i=0; i < K; i++) {
		for (int j= 0; j < N; j++) {
		/*	double t = B[i*lda+j];
			B[i*lda+j] = B[j*lda+i];
			B[j*lda+i] = t;
		*/
			T[j*K_ind+i] = B[i*lda+j];
		}	
	}
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C, double* T)
{
	//K_ind  is the real unpadded index to which the matrix T goes to
	//K      is the padded length of matrix (used to skip to column)
	//K_iter is the iteration we go to for the multiplication
	//
	//if K_ind = 5, then K = 6 and K_iter would be 4
	//for the special case of k=5, we handle it separately
	//in the if-else statement below. This is so we can go through
	//the matrix with a max k value of 4 (hence k_iter). 
	 
	int K_ind = K;
	if (K%2==1) {
		K = K+1;
	}
	
	do_transpose(B, lda, K_ind, N, T);
	double ci1j1 = 0; 
	double ci2j1 = 0;
	double ci1j2 = 0;
	double ci2j2 = 0;
  
	double* s_i2j1 = NULL;
	double* s_i1j2 = NULL;
	double* s_i1j1 = NULL;
	double* s_i2j2 = NULL;

	posix_memalign(&s_i1j1, 16, 2*sizeof(double));
	posix_memalign(&s_i2j1, 16, 2*sizeof(double));
	posix_memalign(&s_i1j2, 16, 2*sizeof(double));
	posix_memalign(&s_i2j2, 16, 2*sizeof(double));

	for(int i=0;i<M;i+=2) {

		for(int j=0;j<N;j+=2) {
			int K_iter = K;

			ci1j1 = C[i*lda+j];
			if (j+1 >= N && i+1 >= M) {
				ci2j2 = 0;
				ci1j2 = 0;
				ci2j1 = 0;
			} else if (j+1 >= N) {
				ci1j2 = 0;
				ci2j2 = 0;
				ci2j1 = C[(i+1)*lda+j];
			} else if (i+1 >= M) {
				ci2j1 = 0;
				ci2j2 = 0;
				ci1j2 = C[i*lda+j+1];
			} else {
				ci1j2 = C[i*lda+j+1];
				ci2j1 = C[(i+1)*lda+j];
				ci2j2 = C[(i+1)*lda+j+1];
			}	

			if (K_ind%2 == 1) {
				ci1j1 += A[i*lda+K_ind-1] * T[K*j + K_ind-1];
				
				if (j+1 >= N && i+1 >= M) {
					//do nothing	
				} else if (i+1 >= M) {
					ci1j2 += A[i*lda+K_ind-1] * T[K*(j+1) + K_ind-1];
				} else if (j+1 >= N) {
					ci2j1 += A[(i+1)*lda+K_ind-1] * T[K*j+K_ind-1];
				} else {
					ci1j2 += A[i*lda+K_ind-1] * T[K*(j+1) + K_ind-1];
					ci2j1 += A[(i+1)*lda+K_ind-1] * T[K*j+K_ind-1];
					ci2j2 += A[(i+1)*lda+K_ind-1] * T[K*(j+1)+K_ind-1];
				}			
				K_iter = K_ind-1;
			}

			register __m128d sum_1 = _mm_setzero_pd();
			register __m128d sum_2 = _mm_setzero_pd();
			register __m128d sum_3 = _mm_setzero_pd();
			register __m128d sum_4 = _mm_setzero_pd();
			register __m128d a_1, a_2;
			register __m128d b_1, b_2;
			for(int k=0;k<K_iter;k += 2) {
				a_1 = _mm_loadu_pd(&A[i*lda+k]);
				b_1 = _mm_load_pd(&T[K*j+k]);
				
				if (i+1 >= M && j+1 >= N) {
					a_2 = _mm_setzero_pd();
					b_2 = _mm_setzero_pd();
				} else if (i+1 >= M) {
					b_2 = _mm_load_pd(&T[K*(j+1)+k]);
					a_2 = _mm_setzero_pd();
				} else if (j+1 >= N) {
					a_2 = _mm_loadu_pd(&A[(i+1)*lda+k]);
					b_2 = _mm_setzero_pd();
				} else {
					a_2 = _mm_loadu_pd(&A[(i+1)*lda+k]);
					b_2 = _mm_load_pd(&T[K*(j+1)+k]);
				}
				sum_1 = _mm_add_pd(sum_1, _mm_mul_pd(a_1,b_1));
				sum_2 = _mm_add_pd(sum_2, _mm_mul_pd(a_1,b_2));
				sum_3 = _mm_add_pd(sum_3, _mm_mul_pd(a_2,b_1));
				sum_4 = _mm_add_pd(sum_4, _mm_mul_pd(a_2,b_2));
				//cij += A[i*lda + k] * T[K*j + k];
			}

			_mm_store_pd(s_i1j1, sum_1);
			_mm_store_pd(s_i1j2, sum_2);
			_mm_store_pd(s_i2j1, sum_3);
			_mm_store_pd(s_i2j2, sum_4);
			
			C[i*lda+j] = ci1j1 + s_i1j1[0] + s_i1j1[1];
			if (j+1 >= N && i+1 >= M) {
				//do nothing
			} else if (i+1 >= M) {
				C[i*lda+j+1] = ci1j2 + s_i1j2[0] + s_i1j2[1];
			} else if (j+1 >= N) {
				C[(i+1)*lda+j] = ci2j1 + s_i2j1[0] + s_i2j1[1];	
			} else {
				C[i*lda+j+1] = ci1j2 + s_i1j2[0] + s_i1j2[1];
				C[(i+1)*lda+j] = ci2j1 + s_i2j1[0] + s_i2j1[1];
				C[(i+1)*lda+j+1] = ci2j2 + s_i2j2[0] + s_i2j2[1];
			}	

		//	C[i*lda+j] = cij;
		}
	}
	free(s_i1j1);
	free(s_i1j2);
	free(s_i2j1);
	free(s_i2j2);
}

static void second_block(int lda, int M, int N, int K, double* A, double *B, double *C) {
	for (int i=0; i < M; i += BLOCK_1_SIZE) {
		int M_2 = min (BLOCK_1_SIZE, M-i);
		for (int j=0; j < N; j += BLOCK_1_SIZE) {
			int N_2 = min (BLOCK_1_SIZE, N-j);
			double *c = C + i*lda + j;

			for (int k=0; k < K; k += BLOCK_1_SIZE) {
				double *a = A + i*lda + k;
				double *b = B + k*lda + j;
				int K_2 = min (BLOCK_1_SIZE, K-k);
				
				int K_mem = K_2;
				if (K_2 % 2 == 1) {
					K_mem = K_2 + 1;
				}

				double* T = NULL;
			
				if (posix_memalign(&T, 16, K_mem*N*sizeof(double))) {
					printf("ERROR ON USING POSIX_MEMALIGN \n");
					return;	
				}			
				do_block(lda, M_2, N_2, K_2, a, b, c, T);
				free(T);
			}
		}
	}
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C) {
	

	for (int i = 0; i < lda; i += BLOCK_2_SIZE) {
		int M = min (BLOCK_2_SIZE, lda - i);
		for (int j = 0; j < lda; j += BLOCK_2_SIZE) {
			int N = min (BLOCK_2_SIZE, lda - j);
			double *c = C + i*lda + j;
			for (int k = 0; k < lda; k += BLOCK_2_SIZE) {
				double *a = A + i*lda + k;
				double *b = B + k*lda + j;
				int K = min (BLOCK_2_SIZE, lda - k);
				
				second_block(lda, M, N, K, a, b, c);
			}
		}
	}
}
