#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <omp.h>
#include <inttypes.h>

#ifndef N
#define N 2000000
#endif
#ifndef NTIMES
#define NTIMES 100
#endif
#ifndef OFFSET
#define OFFSET 0
#endif

#define USEC 1000000

static double bwData[N+OFFSET];

unsigned int bwStreamSize = N; // 2*N

int main (int argc, char **argv) { 
	// Usage: ./memBw <duration in sec>
	if (argc < 2) { 
		printf("Usage: ./memBw <duration in sec>\n"); 
		exit(0);
	}
	double scalar = 3.0;
	
	uint32_t maxThreads = omp_get_num_procs();
        printf("Total number of processors (cores): %" PRIu32 "\n", maxThreads);
        printf("Running with %d threads for %d seconds\n", maxThreads, atoi(argv[1]));

	unsigned int usr_timer = atoi(argv[1]) * maxThreads;
	omp_set_num_threads(maxThreads);
	#pragma omp parallel
	{
		double time_spent = 0.0;
		while (time_spent < usr_timer) {
			double *mid = bwData + (bwStreamSize/2);
			clock_t begin = clock();

			for (int i = 0; i < bwStreamSize/2; i++) {
				bwData[i] = scalar*mid[i];
			}
		
			for (int i = 0; i < bwStreamSize/2; i++) {
				mid[i] = scalar*bwData[i];
			}

			clock_t end = clock(); 
  			time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
			// printf("Time spent: %f\n", time_spent / maxThreads);

			// intensity adjuster
			usleep(USEC);
		}
	}
	return 0;
}
