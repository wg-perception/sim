find_package(ecto)
if(ecto_FOUND)
  #set the default path for built libraries to the "ecto_opencv" directory
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  ecto_python_env_gen(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  ectomodule(sim
     module.cpp
     plane_render.cpp
  )
  link_ecto(sim
    ${VTK_REQUIRED_LIBS} ${OpenCV_LIBS} sim
  )
endif()
