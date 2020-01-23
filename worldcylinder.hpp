// License Summary: MIT see LICENSE file
#pragma once

#include "al2o3_platform/platform.h"
#include <CL/sycl.hpp>

struct WorldCylinder {
	WorldCylinder(uint32_t width, uint32_t height);
	~WorldCylinder();

	void FlushToHost();

	uint32_t width;
	uint32_t height;

	uint32_t doubleBufferIndex;
	float* hostIntensity;
	cl::sycl::range<2> dataRange;
	cl::sycl::buffer<float, 2> intensity[2];
};


