# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build

# Include any dependencies generated for this target.
include src/CMakeFiles/matrix-multi.fpga_profile.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/matrix-multi.fpga_profile.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/matrix-multi.fpga_profile.dir/flags.make

src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o: src/CMakeFiles/matrix-multi.fpga_profile.dir/flags.make
src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o: ../src/matrix-multi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o"
	cd /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build/src && /glob/development-tools/versions/oneapi/2023.1.2/oneapi/compiler/2023.1.0/linux/bin/dpcpp  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o -c /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/src/matrix-multi.cpp

src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.i"
	cd /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build/src && /glob/development-tools/versions/oneapi/2023.1.2/oneapi/compiler/2023.1.0/linux/bin/dpcpp $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/src/matrix-multi.cpp > CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.i

src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.s"
	cd /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build/src && /glob/development-tools/versions/oneapi/2023.1.2/oneapi/compiler/2023.1.0/linux/bin/dpcpp $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/src/matrix-multi.cpp -o CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.s

src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o.requires:

.PHONY : src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o.requires

src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o.provides: src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/matrix-multi.fpga_profile.dir/build.make src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o.provides.build
.PHONY : src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o.provides

src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o.provides.build: src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o


# Object files for target matrix-multi.fpga_profile
matrix__multi_fpga_profile_OBJECTS = \
"CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o"

# External object files for target matrix-multi.fpga_profile
matrix__multi_fpga_profile_EXTERNAL_OBJECTS =

matrix-multi.fpga_profile: src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o
matrix-multi.fpga_profile: src/CMakeFiles/matrix-multi.fpga_profile.dir/build.make
matrix-multi.fpga_profile: src/CMakeFiles/matrix-multi.fpga_profile.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../matrix-multi.fpga_profile"
	cd /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix-multi.fpga_profile.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/matrix-multi.fpga_profile.dir/build: matrix-multi.fpga_profile

.PHONY : src/CMakeFiles/matrix-multi.fpga_profile.dir/build

src/CMakeFiles/matrix-multi.fpga_profile.dir/requires: src/CMakeFiles/matrix-multi.fpga_profile.dir/matrix-multi.cpp.o.requires

.PHONY : src/CMakeFiles/matrix-multi.fpga_profile.dir/requires

src/CMakeFiles/matrix-multi.fpga_profile.dir/clean:
	cd /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build/src && $(CMAKE_COMMAND) -P CMakeFiles/matrix-multi.fpga_profile.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/matrix-multi.fpga_profile.dir/clean

src/CMakeFiles/matrix-multi.fpga_profile.dir/depend:
	cd /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/src /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build/src /home/u192974/eece-6540-labs/Labs/lab2/matrix-multi/build/src/CMakeFiles/matrix-multi.fpga_profile.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/matrix-multi.fpga_profile.dir/depend

