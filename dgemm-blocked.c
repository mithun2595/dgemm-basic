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
#define BLOCK_2_SIZE 128
#define BLOCK_1_SIZE 16
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static void do_transpose(double* B, int lda, int K, int N, double* T) {
	for (int i=0; i < K; i++) {
		for (int j= 0; j < N; j++) {
		/*	double t = B[i*lda+j];
			B[i*lda+j] = B[j*lda+i];
			B[j*lda+i] = t;
		*/
			T[j*K+i] = B[i*lda+j];
		}	
	}
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
	double* T = malloc(K*N*sizeof(double));
	do_transpose(B, lda, K, N, T);
	double cij = 0;
	double* s = malloc(2*sizeof(double)); 
	
	for(int i=0;i<M;++i) {

		for(int j=0;j<N;++j) {
			int K_iter = K;
			cij = C[i*lda+j];
			if (K%2 == 1) {
				cij += A[i*lda+K-1] * T[K*j + K-1];			
				K_iter = K-1;
			}
			register __m128d c = _mm_setzero_pd();
			register __m128d sum = _mm_setzero_pd();
			register __m128d a;
			register __m128d b;
			for(int k=0;k<K_iter;k += 2) {
				a = _mm_loadu_pd(&A[i*lda+k]);
				b = _mm_loadu_pd(&T[K*j+k]);
				c = _mm_mul_pd(a,b);
				sum = _mm_add_pd(sum, c);
				//cij += A[i*lda + k] * T[K*j + k];
			}
			_mm_storeu_pd(s, sum);
			C[i*lda+j] = cij + s[0] + s[1];
			
		//	C[i*lda+j] = cij;
		}
	}
	free(s);
	free(T);
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
