set( SRCS SingleRover.cpp MultiRover.cpp)
add_library( Domains SHARED ${SRCS} )
target_link_libraries(Domains Learning Utilities Agents Target)
