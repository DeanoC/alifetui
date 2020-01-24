//
// Created by Computer on 23/01/2020.
//

// License Summary: MIT see LICENSE file

#include "al2o3_platform/platform.h"
#include "al2o3_cmath/scalar.h"
#include "al2o3_memory/memory.h"
#include "worldcylinder.hpp"

WorldCylinder::WorldCylinder(uint32_t width_, uint32_t height_) :
	width{ width_ },
	height{ height_ },
	doubleBufferIndex{ 0 },
	hostIntensity { (float*)MEMORY_CALLOC(width_*height_, sizeof(float)) },
	dataRange{ height_, width_  }
{

	for (uint32_t y = 0u; y < height; ++y) {
		for (uint32_t x = 0u; x < width; ++x) {
			float inten = (float)x / (float)width;

			//			inten += cos(b1 * ((float)item.get_id(0) / lwidth) + phi1);
			//			inten += cos(b2 * ((float)item.get_id(1) / lheight)+ phi2);
			hostIntensity[y * width + x] = inten * 128;
		}
	}

	intensity[0] = cl::sycl::buffer<float,2>{ hostIntensity, dataRange };
	intensity[1] = cl::sycl::buffer<float,2>{ hostIntensity, dataRange };

}

WorldCylinder::~WorldCylinder() {
	MEMORY_FREE(hostIntensity);
}

namespace {
struct UpdateTag1;
struct InitTag1;
}

void WorldCylinder::init(cl::sycl::queue& q) {
	using namespace cl::sycl;

	float const b1 = 1;
	float const phi1 = 0;
	float const b2 = 1;
	float const phi2 = 0;
	float const c1 = 1;
	float const c2 = 0;

	float const lwidth = (float)width;
	float const lheight = (float)height;
	//			inten += cos(b1 * ((float)item.get_id(0) / lwidth) + phi1);
	//			inten += cos(b2 * ((float)item.get_id(1) / lheight)+ phi2);

	q.submit([&](handler &cgh) {

		auto ptr = intensity[doubleBufferIndex].get_access<access::mode::read_write>(cgh);
		cgh.copy(hostIntensity, ptr);

/*		cgh.parallel_for<InitTag1>(dataRange, [=](item<2> item) {
			float inten = (float)item.get_id(0) / lwidth;

//			inten += cos(b1 * ((float)item.get_id(0) / lwidth) + phi1);
//			inten += cos(b2 * ((float)item.get_id(1) / lheight)+ phi2);

			ptr[item.get_id()] = inten * 128;
		});*/
	});

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

void WorldCylinder::update(cl::sycl::queue& q) {
	using namespace cl::sycl;

	auto const ndr = nd_range<2>{ dataRange, get_optimal_local_range(dataRange, q.get_device()) };

	q.submit([&](handler &cgh) {
		auto imp = intensity[doubleBufferIndex].get_access<access::mode::read>(cgh);
		auto outp = intensity[doubleBufferIndex^1].get_access<access::mode::read_write>(cgh);

		cgh.parallel_for<UpdateTag1>(ndr, [=](nd_item<2> item) {
					id<2> const gid = item.get_global_id();

					if(gid[1] == 10 ) {
						outp[item.get_global_id()] = '$';//imp[id];
					} else {
						outp[item.get_global_id()] = '@';
					}
		});

		doubleBufferIndex ^= 1;
	});
}

void WorldCylinder::flushToHost() {
	using namespace cl::sycl;
	accessor<float, 2, access::mode::read, access::target::host_buffer>
			hostPtr(intensity[doubleBufferIndex]);

/*	for (uint32_t y = 0u; y < height; ++y) {
		for (uint32_t x = 0u; x < width; ++x) {
			hostIntensity[y * width + x] = hostPtr[id<2>(x,y)];
		}
	}*/
	memcpy(hostIntensity, hostPtr.get_pointer(), sizeof(float) * width * height);
}
