#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <dlfcn.h>
#include <sys/time.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <memory>
#include <map>
#include <tuple>
#include <cufft.h>
#include <fftw3.h>
#include <cassert>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

	
typedef std::tuple<cudaStream_t, cufftHandle, cufftDoubleComplex *, fftw_complex *> fftStreamHandle;
struct DevMemory
{
	int n;
	int rank; 
	int inembed;
	int istride;
	int idist;
	int onembed;
	int ostride;
	int odist;
	int mem_size;
	cufftDoubleComplex * dev_ptr;
	int plansize;
	cufftHandle plan;
	int curPlanSize;
	std::vector<fftStreamHandle> avail_streams;
	std::vector<fftStreamHandle> used_streams;
};

std::shared_ptr<DevMemory> CMPI_local_mem;
std::shared_ptr<DevMemory> CMPI_local_mem_forward;

extern "C" {

	void InitializePlanForward(int rank, const int *n, int howmany,
                             fftw_complex *in, const int *inembed,
                             int istride, int idist,
                             fftw_complex *out, const int *onembed,
                             int ostride, int odist,
                             int sign, unsigned flags) {

		fprintf(stderr, "%s\n", "Allocating global structure");
		// Plan Information: 126 1 126 126 1 126 126 1 1
		fprintf(stderr, "%s: %d %d %d %d %d %d %d %d %d\n", "Plan Information",
		*n, howmany, *inembed, istride, idist, *onembed, ostride, odist, sign );
		fprintf(stderr, "%s %d\n", "SIZE OF INIT COMPLEX: ", sizeof(fftw_complex));
		CMPI_local_mem_forward.reset(new DevMemory[1]);
		CMPI_local_mem_forward.get()->rank = rank;
		CMPI_local_mem_forward.get()->n = *n;
		CMPI_local_mem_forward.get()->inembed = *inembed;
		CMPI_local_mem_forward.get()->istride = istride;
		CMPI_local_mem_forward.get()->idist = idist;
		CMPI_local_mem_forward.get()->onembed = *onembed;
		CMPI_local_mem_forward.get()->ostride = ostride;
		CMPI_local_mem_forward.get()->odist = odist;
		CMPI_local_mem_forward.get()->mem_size = *n;
		CMPI_local_mem_forward.get()->plansize = *n;
		CMPI_local_mem_forward.get()->curPlanSize = 0;
		gpuErrchk(cudaMalloc((void**)&(CMPI_local_mem_forward.get()->dev_ptr),
			sizeof(cufftDoubleComplex)*CMPI_local_mem_forward.get()->mem_size*CMPI_local_mem_forward.get()->plansize));
	}


	void InitializePlan(int rank, const int *n, int howmany,
                             fftw_complex *in, const int *inembed,
                             int istride, int idist,
                             fftw_complex *out, const int *onembed,
                             int ostride, int odist,
                             int sign, unsigned flags) {

		fprintf(stderr, "%s\n", "Allocating global structure");
		// Plan Information: 126 1 126 126 1 126 126 1 1
		fprintf(stderr, "%s: %d %d %d %d %d %d %d %d %d\n", "Plan Information",
		*n, howmany, *inembed, istride, idist, *onembed, ostride, odist, sign );
		fprintf(stderr, "%s %d\n", "SIZE OF INIT COMPLEX: ", sizeof(fftw_complex));
		CMPI_local_mem.reset(new DevMemory[1]);
		CMPI_local_mem.get()->rank = rank;
		CMPI_local_mem.get()->n = *n;
		CMPI_local_mem.get()->inembed = *inembed;
		CMPI_local_mem.get()->istride = istride;
		CMPI_local_mem.get()->idist = idist;
		CMPI_local_mem.get()->onembed = *onembed;
		CMPI_local_mem.get()->ostride = ostride;
		CMPI_local_mem.get()->odist = odist;
		CMPI_local_mem.get()->mem_size = *n;
		CMPI_local_mem.get()->plansize = *n;
		CMPI_local_mem.get()->curPlanSize = 0;
		gpuErrchk(cudaMalloc((void**)&(CMPI_local_mem.get()->dev_ptr),
			sizeof(cufftDoubleComplex)*CMPI_local_mem.get()->mem_size*CMPI_local_mem.get()->plansize));
	}

	fftStreamHandle GetNewStream() {
		std::vector<fftStreamHandle> * avail_streams = &(CMPI_local_mem.get()->avail_streams);
		std::vector<fftStreamHandle> * used_streams = &(CMPI_local_mem.get()->used_streams);
		// Return a cached stream
		if (avail_streams->size() > 0) {
			fftStreamHandle	in_gpu = avail_streams->back();
			avail_streams->pop_back();
			return in_gpu;
		}
		// otherwise create one and allocate memory
		cufftHandle plan;
		cudaStream_t stream;
		cufftDoubleComplex * dComplex;		
		gpuErrchk((cudaError_t)cufftPlanMany(&plan, CMPI_local_mem.get()->rank, &CMPI_local_mem.get()->n, 
			&CMPI_local_mem.get()->inembed, CMPI_local_mem.get()->istride,
			CMPI_local_mem.get()->idist,&CMPI_local_mem.get()->onembed,
			CMPI_local_mem.get()->ostride,CMPI_local_mem.get()->odist,
			CUFFT_Z2Z,1));
		gpuErrchk((cudaError_t)cudaStreamCreate(&stream));
		gpuErrchk((cudaError_t)cufftSetStream(plan,stream));
		gpuErrchk(cudaMalloc((void**)&dComplex, sizeof(cufftDoubleComplex)*CMPI_local_mem.get()->mem_size));
		fftStreamHandle ret = std::make_tuple(stream, plan, dComplex, (fftw_complex*)NULL); 
		return ret; 
	}

	void AddUsedStream(fftStreamHandle in){
		CMPI_local_mem.get()->used_streams.push_back(in);
	}


	fftw_plan CMPI_BUILD_PLAN_BACKWARD(int rank, const int *n, int howmany,
                             fftw_complex *in, const int *inembed,
                             int istride, int idist,
                             fftw_complex *out, const int *onembed,
                             int ostride, int odist,
                             int sign, unsigned flags) {
		if (CMPI_local_mem_forward.get() == NULL) {
			fprintf(stderr, "%s\n", "Initializing Plan");
			InitializePlanForward(rank, n, howmany, in, inembed, istride, idist, out, 
					onembed, ostride, odist, sign, flags);
		}
		fftw_plan plan;
		return (fftw_plan)plan;
	}
	
	fftw_plan CMPI_BUILD_PLAN(int rank, const int *n, int howmany,
                             fftw_complex *in, const int *inembed,
                             int istride, int idist,
                             fftw_complex *out, const int *onembed,
                             int ostride, int odist,
                             int sign, unsigned flags) {
		if (CMPI_local_mem.get() == NULL) {
			fprintf(stderr, "%s\n", "Initializing Plan");
			InitializePlan(rank, n, howmany, in, inembed, istride, idist, out, 
					onembed, ostride, odist, sign, flags);
		}
		fftw_plan plan;
		return (fftw_plan)plan;
	}

	void CMPI_SETUP_TRANSFORM_BACKWARD(int plansize, fftw_complex * in) {
		if (CMPI_local_mem_forward.get() == NULL) {
				// Allocate Arrays:
				assert("WE SHOULD NEVER BE HERE" == 0);
				fprintf(stderr, "%s\n", "Allocating global structure");
				CMPI_local_mem_forward.reset(new DevMemory[1]);
		}
		if (CMPI_local_mem_forward.get()->curPlanSize != plansize) {
			fprintf(stderr, "%s\n","Creating a new plansize" );
			if (CMPI_local_mem_forward.get()->curPlanSize != 0)
				cufftDestroy(CMPI_local_mem_forward.get()->plan);
			gpuErrchk((cudaError_t)cufftPlanMany(&(CMPI_local_mem_forward.get()->plan), 
					CMPI_local_mem_forward.get()->rank, &CMPI_local_mem_forward.get()->n, 
					&CMPI_local_mem_forward.get()->inembed, CMPI_local_mem_forward.get()->istride,
					CMPI_local_mem_forward.get()->idist,&CMPI_local_mem_forward.get()->onembed,
					CMPI_local_mem_forward.get()->ostride,CMPI_local_mem_forward.get()->odist,
					CUFFT_Z2Z,plansize));
			CMPI_local_mem_forward.get()->curPlanSize = plansize;
		}
		if(CMPI_local_mem_forward.get()->plansize < plansize) {
			if(CMPI_local_mem_forward.get()->plansize > 0){
				cudaFree(CMPI_local_mem_forward.get()->dev_ptr);
			}
			gpuErrchk(cudaMalloc((void**)&(CMPI_local_mem_forward.get()->dev_ptr), 
				sizeof(cufftDoubleComplex)*CMPI_local_mem_forward.get()->mem_size*plansize));
			CMPI_local_mem_forward.get()->plansize = plansize;
		}
		gpuErrchk(cudaMemcpy(CMPI_local_mem_forward.get()->dev_ptr,in,sizeof(cufftDoubleComplex)*CMPI_local_mem_forward.get()->mem_size*plansize, cudaMemcpyHostToDevice));
		gpuErrchk((cudaError_t)cufftExecZ2Z(CMPI_local_mem_forward.get()->plan, CMPI_local_mem_forward.get()->dev_ptr,CMPI_local_mem_forward.get()->dev_ptr,CUFFT_INVERSE));
		gpuErrchk(cudaMemcpy(in,CMPI_local_mem_forward.get()->dev_ptr,sizeof(cufftDoubleComplex)*CMPI_local_mem_forward.get()->mem_size*plansize, cudaMemcpyDeviceToHost));

	}

	void CMPI_SETUP_TRANSFORM(int plansize, fftw_complex * in) {
		if (CMPI_local_mem.get() == NULL) {
				// Allocate Arrays:
				assert("WE SHOULD NEVER BE HERE" == 0);
				fprintf(stderr, "%s\n", "Allocating global structure");
				CMPI_local_mem.reset(new DevMemory[1]);
		}
		if (CMPI_local_mem.get()->curPlanSize != plansize) {
			fprintf(stderr, "%s\n","Creating a new plansize" );
			if (CMPI_local_mem.get()->curPlanSize != 0)
				cufftDestroy(CMPI_local_mem.get()->plan);
			gpuErrchk((cudaError_t)cufftPlanMany(&(CMPI_local_mem.get()->plan), 
					CMPI_local_mem.get()->rank, &CMPI_local_mem.get()->n, 
					&CMPI_local_mem.get()->inembed, CMPI_local_mem.get()->istride,
					CMPI_local_mem.get()->idist,&CMPI_local_mem.get()->onembed,
					CMPI_local_mem.get()->ostride,CMPI_local_mem.get()->odist,
					CUFFT_Z2Z,plansize));
			CMPI_local_mem.get()->curPlanSize = plansize;
		}
		if(CMPI_local_mem.get()->plansize < plansize) {
			if(CMPI_local_mem.get()->plansize > 0){
				cudaFree(CMPI_local_mem.get()->dev_ptr);
			}
			gpuErrchk(cudaMalloc((void**)&(CMPI_local_mem.get()->dev_ptr), 
				sizeof(cufftDoubleComplex)*CMPI_local_mem.get()->mem_size*plansize));
			CMPI_local_mem.get()->plansize = plansize;
		}
		gpuErrchk(cudaMemcpy(CMPI_local_mem.get()->dev_ptr,in,sizeof(cufftDoubleComplex)*CMPI_local_mem.get()->mem_size*plansize, cudaMemcpyHostToDevice));
		gpuErrchk((cudaError_t)cufftExecZ2Z(CMPI_local_mem.get()->plan, CMPI_local_mem.get()->dev_ptr,CMPI_local_mem.get()->dev_ptr,CUFFT_FORWARD));
		gpuErrchk(cudaMemcpy(in,CMPI_local_mem.get()->dev_ptr,sizeof(cufftDoubleComplex)*CMPI_local_mem.get()->mem_size*plansize, cudaMemcpyDeviceToHost));

	}

	void CMPI_RUN_TRANSFORM(fftw_plan plan, fftw_complex * in, fftw_complex * out) {
		if (CMPI_local_mem.get() == NULL) {
			// Allocate Arrays:
			assert("WE SHOULD NEVER BE HERE" == 0);
			fprintf(stderr, "%s\n", "Allocating global structure");
			CMPI_local_mem.reset(new DevMemory[1]);
		}
		assert(sizeof(fftw_complex) == sizeof(cufftDoubleComplex));
		//fprintf(stderr, "%s %llx %llx\n", "launching with ", (uint64_t) in, (uint64_t) out);
		fftStreamHandle handle = GetNewStream();

		// Copy the FFTW
		gpuErrchk(cudaMemcpyAsync(std::get<2>(handle), in, sizeof(fftw_complex)*CMPI_local_mem.get()->mem_size, 
			cudaMemcpyHostToDevice, std::get<0>(handle)));

		// Set the return index
		std::get<3>(handle) = in;

		// Launch the plan
		gpuErrchk((cudaError_t)cufftExecZ2Z(std::get<1>(handle), std::get<2>(handle), std::get<2>(handle), CUFFT_FORWARD));

		AddUsedStream(handle);
	}

	void CMPI_SYNC_TRANSFORMS() {
		// Sync and resetup
		gpuErrchk(cudaDeviceSynchronize());

		for (int i = 0; i < CMPI_local_mem.get()->avail_streams.size(); i++) {
			gpuErrchk(cudaMemcpyAsync(std::get<3>(CMPI_local_mem.get()->avail_streams[i]), 
						std::get<2>(CMPI_local_mem.get()->avail_streams[i]), 
						sizeof(fftw_complex)*1, 
					  	cudaMemcpyDeviceToHost, std::get<0>(CMPI_local_mem.get()->avail_streams[i])));			
		}
		gpuErrchk(cudaDeviceSynchronize());
		if (CMPI_local_mem.get()->avail_streams.size() == 0)
			CMPI_local_mem.get()->avail_streams = CMPI_local_mem.get()->used_streams;
		else
			CMPI_local_mem.get()->avail_streams.insert(CMPI_local_mem.get()->avail_streams.end(), 
				CMPI_local_mem.get()->used_streams.begin(), CMPI_local_mem.get()->used_streams.end());
		CMPI_local_mem.get()->used_streams.clear();		
	}

}
