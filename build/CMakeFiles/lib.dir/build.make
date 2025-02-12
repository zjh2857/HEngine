# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /environment/miniconda3/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /environment/miniconda3/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/featurize/work/HE/Homomorphic-Encryption-on-GPU

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/featurize/work/HE/Homomorphic-Encryption-on-GPU/build

# Include any dependencies generated for this target.
include CMakeFiles/lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/lib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lib.dir/flags.make

CMakeFiles/lib.dir/library.cu.o: CMakeFiles/lib.dir/flags.make
CMakeFiles/lib.dir/library.cu.o: /home/featurize/work/HE/Homomorphic-Encryption-on-GPU/library.cu
CMakeFiles/lib.dir/library.cu.o: CMakeFiles/lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/featurize/work/HE/Homomorphic-Encryption-on-GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/lib.dir/library.cu.o"
	/environment/miniconda3/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/lib.dir/library.cu.o -MF CMakeFiles/lib.dir/library.cu.o.d -x cu -rdc=true -c /home/featurize/work/HE/Homomorphic-Encryption-on-GPU/library.cu -o CMakeFiles/lib.dir/library.cu.o

CMakeFiles/lib.dir/library.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/lib.dir/library.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/lib.dir/library.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/lib.dir/library.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target lib
lib_OBJECTS = \
"CMakeFiles/lib.dir/library.cu.o"

# External object files for target lib
lib_EXTERNAL_OBJECTS =

CMakeFiles/lib.dir/cmake_device_link.o: CMakeFiles/lib.dir/library.cu.o
CMakeFiles/lib.dir/cmake_device_link.o: CMakeFiles/lib.dir/build.make
CMakeFiles/lib.dir/cmake_device_link.o: CMakeFiles/lib.dir/deviceLinkLibs.rsp
CMakeFiles/lib.dir/cmake_device_link.o: CMakeFiles/lib.dir/deviceObjects1
CMakeFiles/lib.dir/cmake_device_link.o: CMakeFiles/lib.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/featurize/work/HE/Homomorphic-Encryption-on-GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/lib.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lib.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lib.dir/build: CMakeFiles/lib.dir/cmake_device_link.o
.PHONY : CMakeFiles/lib.dir/build

# Object files for target lib
lib_OBJECTS = \
"CMakeFiles/lib.dir/library.cu.o"

# External object files for target lib
lib_EXTERNAL_OBJECTS =

lib/liblib.so: CMakeFiles/lib.dir/library.cu.o
lib/liblib.so: CMakeFiles/lib.dir/build.make
lib/liblib.so: CMakeFiles/lib.dir/cmake_device_link.o
lib/liblib.so: CMakeFiles/lib.dir/linkLibs.rsp
lib/liblib.so: CMakeFiles/lib.dir/objects1
lib/liblib.so: CMakeFiles/lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/featurize/work/HE/Homomorphic-Encryption-on-GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA shared library lib/liblib.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lib.dir/build: lib/liblib.so
.PHONY : CMakeFiles/lib.dir/build

CMakeFiles/lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lib.dir/clean

CMakeFiles/lib.dir/depend:
	cd /home/featurize/work/HE/Homomorphic-Encryption-on-GPU/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/featurize/work/HE/Homomorphic-Encryption-on-GPU /home/featurize/work/HE/Homomorphic-Encryption-on-GPU /home/featurize/work/HE/Homomorphic-Encryption-on-GPU/build /home/featurize/work/HE/Homomorphic-Encryption-on-GPU/build /home/featurize/work/HE/Homomorphic-Encryption-on-GPU/build/CMakeFiles/lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lib.dir/depend

