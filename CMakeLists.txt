cmake_minimum_required(VERSION 2.8)
project(multirover)

#There are lots of scripts with cmake
#for finding external libraries. 
#see /usr/local/share/cmake-2.6/Modules/Find*.cmake for more examples
#find_package(Boost COMPONENTS system filesystem REQUIRED)
#include_directories( ${Boost_INCLUDE_DIRS} )
include_directories(include)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -I ~/eigen3/")
add_subdirectory(include)

set( LIBS_TO_LINK Utilities Learning Domains Agents)

add_executable(generateMultiRoverExperts generateMultiRoverExperts.cpp)
target_link_libraries( generateMultiRoverExperts ${LIBS_TO_LINK} )
