//
// Created by Computer on 23/01/2020.
//

// License Summary: MIT see LICENSE file

#include "al2o3_platform/platform.h"
#include "al2o3_memory/memory.h"
#include "worldcylinder.hpp"

WorldCylinder::WorldCylinder(uint32_t width_, uint32_t height_) :
	width{ width_ },
	height{ height_ },
	doubleBufferIndex{ 0 },
	hostIntensity { (float*)MEMORY_CALLOC(width_*height_, sizeof(float)) },
	dataRange{ width_, height_ }
{
	intensity[0] = cl::sycl::buffer<float,2>{ hostIntensity, dataRange };
	intensity[1] = cl::sycl::buffer<float,2>{ hostIntensity, dataRange };
}

WorldCylinder::~WorldCylinder() {
	MEMORY_FREE(hostIntensity);
}

void WorldCylinder::FlushToHost() {
	intensity[doubleBufferIndex].get_access<cl::sycl::access::mode::read>();
}
