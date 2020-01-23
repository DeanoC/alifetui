// License Summary: MIT see LICENSE file
#pragma once

typedef struct SyclCore *SyclHandle;

AL2O3_EXTERN_C SyclHandle AccelSycl_Create();
AL2O3_EXTERN_C void AccelSycl_Destroy(SyclHandle handle);

#if defined(__cplusplus)
namespace Accel {

struct Sycl {
public:
	static Sycl* Create() {
		return (Sycl*)AccelSycl_Create();
	}

	void Destroy() {
		AccelSycl_Destroy((SyclHandle)this);
	}

	~Sycl() {
		Destroy();
	}

	Sycl() = delete;
};

}
#endif