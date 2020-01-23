#include "al2o3_platform/platform.h"
#include "al2o3_os/file.hpp"
#include "accel_sycl.hpp"

int main() {
	LOGINFO("Hello, World!");
	using namespace Accel;
	Sycl* sycl = Sycl::Create();

	sycl->Destroy();

	return 0;
}
