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
CMAKE_SOURCE_DIR = /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/build

# Utility rule file for report.

# Include the progress variables for this target.
include src/CMakeFiles/report.dir/progress.make

src/CMakeFiles/report: src/image-rotation_report.a


src/image-rotation_report.a: ../src/image-rotation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating image-rotation_report.a"
	cd /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/build/src && /glob/development-tools/versions/oneapi/2023.1.2/oneapi/compiler/2023.1.0/linux/bin/dpcpp -I/home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/src/Utils -fintelfpga -Xshardware -Xsboard=intel_a10gx_pac:pac_a10 -fsycl-link image-rotation.cpp -o /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/build/image-rotation_report.a

report: src/CMakeFiles/report
report: src/image-rotation_report.a
report: src/CMakeFiles/report.dir/build.make

.PHONY : report

# Rule to build all files generated by this target.
src/CMakeFiles/report.dir/build: report

.PHONY : src/CMakeFiles/report.dir/build

src/CMakeFiles/report.dir/clean:
	cd /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/build/src && $(CMAKE_COMMAND) -P CMakeFiles/report.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/report.dir/clean

src/CMakeFiles/report.dir/depend:
	cd /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/src /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/build /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/build/src /home/u192974/eece-6540-labs/Labs/lab2/image-rotation-skeleton/build/src/CMakeFiles/report.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/report.dir/depend
