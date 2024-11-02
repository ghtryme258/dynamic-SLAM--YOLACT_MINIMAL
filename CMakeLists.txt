cmake_minimum_required(VERSION 2.8)
project(DynaSLAM)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
  # SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 ")

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O0 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O0 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

message("PROJECT_SOURCE_DIR: " ${OpenCV_DIR})
find_package(OpenCV 2.4.11 QUIET)
if(NOT OpenCV_FOUND)
    message("OpenCV > 2.4.11 not found.")
    find_package(OpenCV 3.0 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 3.0 not found.")
    endif()
endif()

#find_package(OpenCV 3.0 QUIET)
#if(NOT OpenCV_FOUND)
#   find_package(OpenCV 2.4.3 QUIET)
#   if(NOT OpenCV_FOUND)
#      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#   endif()
#endif()




find_package(Qt5Widgets REQUIRED)
find_package(Qt5Concurrent REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Test REQUIRED)

#find_package(PythonLibs REQUIRED)
#if (NOT PythonLibs_FOUND)
#    message(FATAL_ERROR "PYTHON LIBS not found.")
#else()
#    message("PYTHON LIBS were found!")
#    message("PYTHON LIBS DIRECTORY: " ${PYTHON_LIBRARY})
#endif()




find_package(Boost REQUIRED COMPONENTS thread)
if(Boost_FOUND)
    message("Boost was found!")
    message("Boost Headers DIRECTORY: " ${Boost_INCLUDE_DIRS})
    message("Boost LIBS DIRECTORY: " ${Boost_LIBRARY_DIRS})
    message("Found Libraries: " ${Boost_LIBRARIES})
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package( G2O REQUIRED )
find_package( PCL 1.8 REQUIRED )


#set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy)

#set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /home/python/.conda/envs/dySLAM/bin/python2.7/dist-packages/numpy/core/include/numpy)


find_package( PythonLibs 3.6 REQUIRED )  
include_directories( ${PYTHON_INCLUDE_DIRS} ) 


include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
/usr/include/python3.6m/
/home/python/anaconda3/lib/python3.7/site-packages/numpy/core/include/numpy/

#${Pangolin_INCLUDE_DIRS}
#/usr/include/python2.7/
#/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/

#/home/python/.conda/envs/dySLAM/include/python2.7/
#/home/python/.conda/envs/dySLAM/bin/python2.7/dist-packages/numpy/core/include/numpy/
${PCL_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
)

add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )



message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/Conversion.cc
src/MaskNet.cc
src/Geometry.cc
src/pointcloudmapping.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sim3 g2o_types_sba


#/usr/lib/x86_64-linux-gnu/libpython2.7.so
/usr/lib/x86_64-linux-gnu/libpython3.6m.so
/usr/lib/x86_64-linux-gnu/libboost_python-py36.so
/usr/lib/x86_64-linux-gnu/libstdc++.so.6



${PCL_LIBRARIES}
${Boost_LIBRARIES}
)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

#add_executable(mono_carla
#Examples/Monocular/mono_carla.cc)
#target_link_libraries(mono_carla ${PROJECT_NAME})

