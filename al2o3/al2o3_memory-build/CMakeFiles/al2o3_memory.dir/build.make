# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\Computer\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\193.6015.37\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\Computer\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\193.6015.37\bin\cmake\win\bin\cmake.exe -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Computer\Documents\Code\alifetui

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Computer\Documents\Code\alifetui\cmake-build-debug

# Include any dependencies generated for this target.
include ..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\depend.make

# Include the progress variables for this target.
include ..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\progress.make

# Include the compile flags for this target's objects.
include ..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\flags.make

..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\src\memory.c.obj: ..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\flags.make
..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\src\memory.c.obj: ..\al2o3\al2o3_memory-src\src\memory.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Computer\Documents\Code\alifetui\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object ../al2o3/al2o3_memory-build/CMakeFiles/al2o3_memory.dir/src/memory.c.obj"
	cd C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-build
	C:\PROGRA~2\MICROS~1\2017\COMMUN~1\VC\Tools\MSVC\1416~1.270\bin\Hostx64\x64\cl.exe @<<
 /nologo $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) /FoCMakeFiles\al2o3_memory.dir\src\memory.c.obj /FdCMakeFiles\al2o3_memory.dir\al2o3_memory.pdb /FS -c C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-src\src\memory.c
<<
	cd C:\Users\Computer\Documents\Code\alifetui\cmake-build-debug

..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\src\memory.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/al2o3_memory.dir/src/memory.c.i"
	cd C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-build
	C:\PROGRA~2\MICROS~1\2017\COMMUN~1\VC\Tools\MSVC\1416~1.270\bin\Hostx64\x64\cl.exe > CMakeFiles\al2o3_memory.dir\src\memory.c.i @<<
 /nologo $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-src\src\memory.c
<<
	cd C:\Users\Computer\Documents\Code\alifetui\cmake-build-debug

..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\src\memory.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/al2o3_memory.dir/src/memory.c.s"
	cd C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-build
	C:\PROGRA~2\MICROS~1\2017\COMMUN~1\VC\Tools\MSVC\1416~1.270\bin\Hostx64\x64\cl.exe @<<
 /nologo $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) /FoNUL /FAs /FaCMakeFiles\al2o3_memory.dir\src\memory.c.s /c C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-src\src\memory.c
<<
	cd C:\Users\Computer\Documents\Code\alifetui\cmake-build-debug

# Object files for target al2o3_memory
al2o3_memory_OBJECTS = \
"CMakeFiles\al2o3_memory.dir\src\memory.c.obj"

# External object files for target al2o3_memory
al2o3_memory_EXTERNAL_OBJECTS =

..\out_libs\al2o3_memory.lib: ..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\src\memory.c.obj
..\out_libs\al2o3_memory.lib: ..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\build.make
..\out_libs\al2o3_memory.lib: ..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Computer\Documents\Code\alifetui\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library ..\..\out_libs\al2o3_memory.lib"
	cd C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-build
	$(CMAKE_COMMAND) -P CMakeFiles\al2o3_memory.dir\cmake_clean_target.cmake
	cd C:\Users\Computer\Documents\Code\alifetui\cmake-build-debug
	cd C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-build
	C:\PROGRA~2\MICROS~1\2017\COMMUN~1\VC\Tools\MSVC\1416~1.270\bin\Hostx64\x64\link.exe /lib /nologo /machine:x64 /out:..\..\out_libs\al2o3_memory.lib @CMakeFiles\al2o3_memory.dir\objects1.rsp 
	cd C:\Users\Computer\Documents\Code\alifetui\cmake-build-debug

# Rule to build all files generated by this target.
..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\build: ..\out_libs\al2o3_memory.lib

.PHONY : ..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\build

..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\clean:
	cd C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-build
	$(CMAKE_COMMAND) -P CMakeFiles\al2o3_memory.dir\cmake_clean.cmake
	cd C:\Users\Computer\Documents\Code\alifetui\cmake-build-debug
.PHONY : ..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\clean

..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\Computer\Documents\Code\alifetui C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-src C:\Users\Computer\Documents\Code\alifetui\cmake-build-debug C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-build C:\Users\Computer\Documents\Code\alifetui\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : ..\al2o3\al2o3_memory-build\CMakeFiles\al2o3_memory.dir\depend
