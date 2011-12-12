add_library(sim SHARED sim.cpp)
target_link_libraries(sim ${VTK_REQUIRED_LIBS} ${OpenCV_LIBS} boost_thread)
