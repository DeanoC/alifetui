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

namespace {
struct UpdateTag1;
}

void WorldCylinder::update(cl::sycl::queue& q) {
	using namespace cl::sycl;

	q.submit([&](handler &cgh) {
		auto ptr = intensity[doubleBufferIndex].get_access<access::mode::read_write>(cgh);

		cgh.parallel_for<UpdateTag1>(dataRange, [=](item<2> item) {
			if(ptr[item.get_id()] > 255.0f) {
				ptr[item.get_id()] = 0.0f;
			} else {
				ptr[item.get_id()] += 1.0f;
			}
		});
	});

}

void WorldCylinder::flushToHost() {
	using namespace cl::sycl;
	accessor<float, 2, access::mode::read, access::target::host_buffer>
			hostPtr(intensity[doubleBufferIndex]);

	memcpy(hostIntensity, hostPtr.get_pointer(), sizeof(float) * width * height);
}
