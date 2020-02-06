//
// Created by Computer on 23/01/2020.
//

// License Summary: MIT see LICENSE file

#include "al2o3_platform/platform.h"
#include "al2o3_cmath/scalar.h"
#include "al2o3_memory/memory.h"
#include "worldcylinder.hpp"
#include <random>

static int const SuperSampleRate = 8;

WorldCylinder::WorldCylinder(uint32_t width_, uint32_t height_) :
		width{width_},
		height{height_},
		doubleBufferIndex{0},
		hostIntensity{(float *) MEMORY_CALLOC(width_ * height_, sizeof(float))},
		hostNewData{(float *) MEMORY_CALLOC(height_ * SuperSampleRate, sizeof(float))},
		dataRange{height_ * SuperSampleRate, width_ * SuperSampleRate},
		downSampleRange{height_, width_} {
	using namespace cl::sycl;

	LOGINFO("World size = %d x %d", dataRange[1], dataRange[0]);

	intensity[0] = buffer<float, 2>{dataRange};
	intensity[1] = buffer<float, 2>{dataRange};
	newData = buffer<float, 1>(cl::sycl::range<1>(dataRange[0]));
	downSample = buffer<float, 2>{downSampleRange};
}

WorldCylinder::~WorldCylinder() {
	MEMORY_FREE(hostNewData);
	MEMORY_FREE(hostIntensity);
}

namespace {
struct UpdateTag1;
struct UpdateTag2;
struct DownSamplerTag1;
}

void WorldCylinder::init(cl::sycl::queue &q) {
	using namespace cl::sycl;

	try {
		q.submit([&](handler &cgh) {
			auto ptr1 = intensity[0].get_access<access::mode::discard_write>(cgh);
			cgh.fill<float>(ptr1, 0.0f);
		});
	} catch (sycl::exception const &e) {
		LOGERROR("Caught synchronous SYCL exception: %s", e.what());
	}

}
/* Attempts to determine a good local size. The OpenCL implementation can
 * do the same, but the best way to *control* performance is to choose the
 * sizes. The method here is to choose the largest number, leq 64, which is
 * a power-of-two, and divides the global work size evenly. In this code,
 * it might prove most optimal to pad the image along one dimension so that
 * the local size could be 64, but this introduces other complexities. */
cl::sycl::range<2> get_optimal_local_range(cl::sycl::range<2> globalSize,
																					 cl::sycl::device d) {
	using namespace cl::sycl;
	range<2> optimalLocalSize;
	/* 64 is a good local size on GPU-like devices, as each compute unit is
	 * made of many smaller processors. On non-GPU devices, 4 is a common vector
	 * width. */
	if (d.is_gpu()) {
		optimalLocalSize = range<2>(64, 1);
	} else {
		optimalLocalSize = range<2>(4, 1);
	}
	/* Here, for each dimension, we make sure that it divides the global size
	 * evenly. If it doesn't, we try the next lowest power of two. Eventually
	 * it will reach one, if the global size has no power of two component. */
	for (int i = 0; i < 2; ++i) {
		while (globalSize[i] % optimalLocalSize[i]) {
			optimalLocalSize[i] = optimalLocalSize[i] >> 1;
		}
	}
	return optimalLocalSize;
}

void WorldCylinder::update(cl::sycl::queue &q) {
	using namespace cl::sycl;
	auto const ndr = nd_range<2>{dataRange, range<2>(32, 32)};
	auto const dsndr = nd_range<2>{downSampleRange, range<2>(16, 32)};

	try {
		q.submit([&](handler &cgh) {
			auto newDataPtr = newData.get_access<access::mode::discard_write>(cgh);
			std::random_device r;
			std::default_random_engine e1(r());
			std::uniform_real_distribution<float> uniform_dist;
			for (uint32_t i = 0u; i < dataRange[0]; ++i) {
				hostNewData[i] = uniform_dist(e1);
			}
			cgh.copy(hostNewData, newDataPtr);
		});

		q.submit([&](handler &cgh) {
			auto inp = intensity[doubleBufferIndex].get_access<access::mode::read>(cgh);
			auto outp = intensity[doubleBufferIndex ^ 1].get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<UpdateTag1>(ndr, [=](nd_item<2> item) {
				id<2> const gid = item.get_global_id();
				float val = inp[gid];// - 0.01f;
				val = Math_MaxF(val, 0.0f);
				outp[gid] = val;
			});
		});

		q.submit([&](handler &cgh) {
			auto outp = intensity[doubleBufferIndex ^ 1].get_access<access::mode::read_write>(cgh);
			auto newDataPtr = newData.get_access<access::mode::read>(cgh);

			float const ht = (float) dataRange[0];
			float const wd = (float) dataRange[1];

			cgh.parallel_for<UpdateTag2>(range<1>(dataRange[0]/2), [=](item<1> item) {
				id<2> const dst{ (size_t)(newDataPtr[item.get_linear_id() * 2] * ht),
										 			(size_t)(newDataPtr[item.get_linear_id() * 2 + 1] * wd) };
				outp[dst] = 64.0f;
			});
		});

		q.submit([&](handler &cgh) {
			auto src = intensity[doubleBufferIndex ^ 1].get_access<access::mode::read>(cgh);
			auto dst = downSample.get_access<access::mode::discard_write>(cgh);
			cgh.parallel_for<DownSamplerTag1>(dsndr, [=](nd_item<2> item) {
				id<2> gid = item.get_global_id();
				float accum = 0;
				for (int i = 0; i < SuperSampleRate; ++i) {
					id<2> agid;
					agid[0] = (gid[0] * SuperSampleRate) + i;
					agid[1] = (gid[1] * SuperSampleRate);
					for (int j = 0; j < SuperSampleRate; ++j) {
						accum += src[agid];
						agid[1] += 1;
					}
				}
				dst[gid] = accum * (1.0f / (SuperSampleRate*SuperSampleRate));
			});
			doubleBufferIndex ^= 1;
		});


		updateDoneEvent = q.submit([&](handler &cgh) {
			auto src = downSample.get_access<access::mode::read>(cgh);
			cgh.copy(src, hostIntensity);
		});

	} catch( cl::sycl::exception const e) {
		LOGERROR("%s", e.what());
	} catch( std::exception const e) {
		LOGERROR("%s", e.what());
	}
}

void WorldCylinder::flushToHost() {
	using namespace cl::sycl;

	try {
		updateDoneEvent.wait_and_throw();
	} catch( cl::sycl::exception const e) {
		LOGERROR("%s", e.what());
	} catch( std::exception const e) {
		LOGERROR("%s", e.what());
	}
}
