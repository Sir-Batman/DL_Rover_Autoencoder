# include
C++ libraries

Repository includes all header and source files for setting up project code.
Dependencies:
  - eigen3

To use, include the following lines in the CMakeLists.txt in your project folder:

include_directories(include)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -I /usr/local/include/eigen3/")

add_subdirectory(include)

set( LIBS_TO_LINK Utilities Learning Domains Agents POMDPs)
