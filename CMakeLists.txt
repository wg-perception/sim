cmake_minimum_required(VERSION 2.8)
project(sim)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

find_package(VTK REQUIRED)
find_package(OpenCV REQUIRED)
if(NOT VTK_USE_RENDERING)
  message(FATAL_ERROR "Example ${PROJECT_NAME} requires VTK_USE_RENDERING.")
endif()

include(${VTK_USE_FILE})

set( VTK_REQUIRED_LIBS vtkRendering  vtkGraphics vtkHybrid  vtkWidgets 
                     vtkImaging vtkIO vtkFiltering vtkCommon vtkVolumeRendering)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BOOST_INCLUDE_DIRS}
    )

add_subdirectory(src)
add_subdirectory(cells)

add_executable(plane_render main.cpp)
target_link_libraries(plane_render sim)
