/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */
#include <emmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_2_SIZE 256
#define BLOCK_1_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
	for(int i=0;i<M;++i) {

		for(int j=0;j<N;++j) {

			cij = C[i*lda+j];

			for(int k=0;k<K;++k) {
				cij += A[i*lda + k] * B[k*lda + j];
			}
			
			C[i*lda+j] = cij;
		}
	}
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

				do_block(lda, M_2, N_2, K_2, a, b, c);

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
