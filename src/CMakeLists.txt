#add_subdirectory(ast)

#-----------------------------------------------------------------------------------------
# Collect Source Files
#-----------------------------------------------------------------------------------------
file( GLOB SOURCES *.cpp)

#-----------------------------------------------------------------------------------------
# Boost Settings
#-----------------------------------------------------------------------------------------
#set(Boost_USE_STATIC_LIBS OFF)
#set(Boost_USE_MULTITHREADED ON)
#set(Boost_USE_STATIC_RUNTIME OFF)
#find_package( Boost REQUIRED COMPONENTS regex )

#-----------------------------------------------------------------------------------------
# Executable Definition
#-----------------------------------------------------------------------------------------
#if(Boost_FOUND)
#    include_directories(${Boost_INCLUDE_DIRS})
     add_executable(neural_net ${SOURCES})
#    target_link_libraries(cion_compiler ${Boost_LIBRARIES})
#endif()
