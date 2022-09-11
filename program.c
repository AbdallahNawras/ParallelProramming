#include <xmmintrin.h>
#include <omp.h>
#include <stdio.h>

#define SSE_WIDTH		4

//MulSum using SIMD
float
SimdMulSum( float *a, float *b, int len )
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;

	__m128 ss = _mm_loadu_ps( &sum[0] );
	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps( &sum[0], ss );

	for( int i = limit; i < len; i++ )
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}

//MulSum using without using SIMD
float
NonSimdMulSum( float *a, float *b, int len )
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;

	for( int i = 0; i < limit; i++ )
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0];
}

//main to populate arrays and perform computations
int main(void)
{
	int length = 1000;
	float a[length];
	float b[length];
	

	//populate the array upto length
	for(int i = 0; i< length; i++)
	{
		a[i] = (float)length/(i+1);
		b[i] = (float)a[i]/(i+1);
	}

	//start timer for SIMD operations on arrays
	double start_time_Simd = omp_get_wtime();
	double single_time_Simd = 0.0;
	double  min_time_Simd = 10000.0;

	for(int i = 0; i< length; i++)
	{
		SimdMulSum(a, b, length);
		
		single_time_Simd = omp_get_wtime() - start_time_Simd;
		if(single_time_Simd < min_time_Simd)
		{
			min_time_Simd = single_time_Simd;
		}
	}

	// end timer, get difference
	double timeSimd = omp_get_wtime() - start_time_Simd;

	//start timer for nonSIMD operations on arrays
	double start_time_NonSimd = omp_get_wtime();
	double single_time_NonSimd = 0.0;
	double  min_time_NonSimd = 10000.0;
	for(int i = 0; i< length; i++)
	{
		NonSimdMulSum(a, b, length);

		single_time_NonSimd = omp_get_wtime() - start_time_NonSimd;
		if(single_time_NonSimd < min_time_NonSimd)
		{
			min_time_NonSimd = single_time_NonSimd;
		}
	}
	//end timer and get difference
	double timeNonSimd = omp_get_wtime() - start_time_NonSimd;


	//get speedUp for vertical(y) axis of chart.
//float speedUp = (timeNonSimd / timeSimd);
	float speedUp = (min_time_NonSimd / min_time_Simd);
	printf("Array Size, Time SIMD, Time non-SIMD, SpeedUP\n");
	printf("%d, %f, %f, %f\n", length, timeSimd, timeNonSimd, speedUp);
	

}
