# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /home/santosh/Programs/CLion-2019.3.5/clion-2019.3.5/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/santosh/Programs/CLion-2019.3.5/clion-2019.3.5/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/santosh/Projects/LearningVisualOdometry

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/santosh/Projects/LearningVisualOdometry/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/LearningVisualOdometry.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LearningVisualOdometry.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LearningVisualOdometry.dir/flags.make

CMakeFiles/LearningVisualOdometry.dir/main.cpp.o: CMakeFiles/LearningVisualOdometry.dir/flags.make
CMakeFiles/LearningVisualOdometry.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/santosh/Projects/LearningVisualOdometry/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LearningVisualOdometry.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LearningVisualOdometry.dir/main.cpp.o -c /home/santosh/Projects/LearningVisualOdometry/main.cpp

CMakeFiles/LearningVisualOdometry.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LearningVisualOdometry.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/santosh/Projects/LearningVisualOdometry/main.cpp > CMakeFiles/LearningVisualOdometry.dir/main.cpp.i

CMakeFiles/LearningVisualOdometry.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LearningVisualOdometry.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/santosh/Projects/LearningVisualOdometry/main.cpp -o CMakeFiles/LearningVisualOdometry.dir/main.cpp.s

# Object files for target LearningVisualOdometry
LearningVisualOdometry_OBJECTS = \
"CMakeFiles/LearningVisualOdometry.dir/main.cpp.o"

# External object files for target LearningVisualOdometry
LearningVisualOdometry_EXTERNAL_OBJECTS =

LearningVisualOdometry: CMakeFiles/LearningVisualOdometry.dir/main.cpp.o
LearningVisualOdometry: CMakeFiles/LearningVisualOdometry.dir/build.make
LearningVisualOdometry: CMakeFiles/LearningVisualOdometry.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/santosh/Projects/LearningVisualOdometry/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable LearningVisualOdometry"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LearningVisualOdometry.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LearningVisualOdometry.dir/build: LearningVisualOdometry

.PHONY : CMakeFiles/LearningVisualOdometry.dir/build

CMakeFiles/LearningVisualOdometry.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LearningVisualOdometry.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LearningVisualOdometry.dir/clean

CMakeFiles/LearningVisualOdometry.dir/depend:
	cd /home/santosh/Projects/LearningVisualOdometry/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/santosh/Projects/LearningVisualOdometry /home/santosh/Projects/LearningVisualOdometry /home/santosh/Projects/LearningVisualOdometry/cmake-build-debug /home/santosh/Projects/LearningVisualOdometry/cmake-build-debug /home/santosh/Projects/LearningVisualOdometry/cmake-build-debug/CMakeFiles/LearningVisualOdometry.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LearningVisualOdometry.dir/depend

