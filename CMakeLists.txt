cmake_minimum_required(VERSION 3.12.4)
get_directory_property(hasParent PARENT_DIRECTORY)
if(NOT hasParent)
	option(unittests "unittests" OFF)
	get_filename_component(_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
	set_property(GLOBAL PROPERTY GLOBAL_FETCHDEPS_BASE ${CMAKE_CURRENT_SOURCE_DIR}/al2o3 )
	include(FetchContent)
	FetchContent_Declare( al2o3 GIT_REPOSITORY https://github.com/DeanoC/al2o3 GIT_TAG master )
	FetchContent_GetProperties(al2o3)
	if(NOT al2o3_POPULATED)
		FetchContent_Populate(al2o3)
		add_subdirectory(${al2o3_SOURCE_DIR} ${al2o3_BINARY_DIR})
	endif()
	INIT_AL2O3(${CMAKE_CURRENT_SOURCE_DIR})
endif ()

SET_MIN_VERSIONS()
set(ProjectName alifetui)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(ComputeCpp_DIR "C:/Program Files/codeplay/computecpp")
set(COMPUTECPP_BITCODE "ptx64" CACHE STRING "Bitcode type to use as SYCL target in compute++")
#set(COMPUTECPP_BITCODE "spir64" CACHE STRING "Bitcode type to use as SYCL target in compute++")

project(${ProjectName} C CXX)
find_package(ComputeCpp REQUIRED)

set(Src
		main.cpp
		accel_sycl.cpp
		accel_sycl.hpp
		)
set(Deps
		al2o3_platform
		al2o3_os
		)
ADD_CONSOLE_APP( ${ProjectName} "${Src}" "${Deps}")
add_sycl_to_target(TARGET ${ProjectName} SOURCES accel_sycl.cpp)