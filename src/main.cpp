#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thread>
#include <iostream>
#include <vector>

using namespace std::chrono_literals;
using std::chrono::seconds;
using std::chrono::nanoseconds;

#define RUNTIME_API_CALL(apiFuncCall) \
	do { \
		cudaError_t _status = apiFuncCall; \
		if (_status != cudaSuccess) { \
			std::cerr << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error: " << cudaGetErrorString(_status); \
			exit(-1); \
		} \
	} while (0)


class MemoryBatch {
public:
	explicit MemoryBatch(size_t size) {
		RUNTIME_API_CALL(cudaMalloc(&handle, size));
	}

	~MemoryBatch() {
		if (handle) {
			RUNTIME_API_CALL(cudaFree(handle));
		}
	}

	void* get() const { return handle; }

	MemoryBatch(MemoryBatch&& other) {
		*this = std::move(other);
	}
	MemoryBatch& operator=(MemoryBatch&& other) {
		handle = std::exchange(other.handle, nullptr);
		return *this;
	}

	MemoryBatch(const MemoryBatch&) = delete;
	MemoryBatch& operator=(const MemoryBatch&) = delete;

private:
	void* handle = nullptr;
};

void printDeviceInfo() {
	int deviceNum = 0;
	cudaDeviceProp prop;
	RUNTIME_API_CALL(cudaGetDevice(&deviceNum));
	RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, deviceNum));
	std::cout << "Selected device nr. " << deviceNum << ": " << prop.name << "\n";
	std::cout << "Device compute capability: " << prop.major << "." << prop.minor << "\n";
}

int main(int argc, char* argv[]) {
	if (argc < 4) {
		std::cerr << "Insufficient number of parameters. <total alloc size [MB]> <alloc step size [MB]> <alloc duration [s]>\n";
		return 1;
	}

	printDeviceInfo();

	const auto totalAllocSize = std::atoi(argv[1]);
	const auto allocStepSize = std::atoi(argv[2]);
	const auto duration = seconds{ std::atoi(argv[3]) };
	
	const auto steps = totalAllocSize / allocStepSize;
	std::cout << "Allocating " << totalAllocSize << " MB of global device memory over " << duration.count() << " seconds (" << steps << " steps)...\n";

	std::vector<unsigned char> dummy;
	for (int i = 0; i < allocStepSize; ++i) { dummy.emplace_back(std::rand()); }

	std::vector<MemoryBatch> batches;
	for (int i = 0; i < steps; ++i) {
		constexpr auto MB = 1000 * 1000;
		batches.emplace_back(allocStepSize * MB);
		std::cout << "Allocated " << allocStepSize << " MB\n";
		std::this_thread::sleep_for(nanoseconds{ duration } / steps);
	}

	std::cout << "Waiting for CTRL-C...\n";
	while (true) {
		//TODO: touch memory periodically
		std::this_thread::sleep_for(1s);
	}

	//TODO: copy back the results to simulate that they're being used

	return 0;
}
