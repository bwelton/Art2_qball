cmake_minimum_required(VERSION 2.8.11)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE} CACHE STRING "RelWithDebInfo" FORCE)
project(Art2_qball)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/wrappers)

cmake_host_system_information(RESULT BUILD_HOST QUERY HOSTNAME)
## If we are on titan, set FFTW paths
if ("${BUILD_HOST}" MATCHES "(titan-ext)[1-7]")
	set(FFTW_DIR /opt/cray/fftw/3.3.4.11/interlagos)
	set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CRAY_CUDATOOLKIT_DIR}")
	set(LIBCUDA_DIR "/opt/cray/nvidia/default/lib64")
endif("${BUILD_HOST}" MATCHES "(titan-ext)[1-7]")

if (NOT DEFINED FFTW_DIR)
	ExternalProject_Add(
	  FFTW_INSTALL
	  EXCLUDE_FROM_ALL 1
	  URL "ftp://ftp.fftw.org/pub/fftw/fftw-3.3.6-pl2.tar.gz"
	  URL_MD5 "927e481edbb32575397eb3d62535a856"
	  SOURCE_DIR "${PROJECT_BINARY_DIR}/3rdparty/fftw3"
	  CONFIGURE_COMMAND ${PROJECT_BINARY_DIR}/3rdparty/fftw3/configure --prefix=${CMAKE_INSTALL_PREFIX} --enable-shared CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
	  BUILD_COMMAND $(MAKE) -j 4
	)
	set(FFTW_DIR ${CMAKE_INSTALL_PREFIX})
else(NOT DEFINED FFTW_DIR)
	add_custom_target(FFTW_INSTALL)
endif(NOT DEFINED FFTW_DIR)
find_package(CUDA REQUIRED)

if (NOT DEFINED LIBCUDA_DIR)
	message(ERROR " Set -DLIBCUDA_DIR to the path containing libcuda.so")
endif(NOT DEFINED LIBCUDA_DIR)

add_subdirectory(${CMAKE_SOURCE_DIR}/src/fftw_intercept)

configure_file (
  "${PROJECT_SOURCE_DIR}/src/wrappers/qball_configure.sh.in"
  "${PROJECT_BINARY_DIR}/wrappers/qball_configure.sh"
  )

configure_file (
  "${PROJECT_SOURCE_DIR}/src/wrappers/build_qball_fixed.sh.in"
  "${PROJECT_BINARY_DIR}/wrappers/build_qball_fixed.sh"
  )

configure_file (
  "${PROJECT_SOURCE_DIR}/src/wrappers/configure.ac.in"
  "${PROJECT_BINARY_DIR}/wrappers/qball-configure.ac"
  @ONLY
)
configure_file (
  "${PROJECT_SOURCE_DIR}/src/wrappers/configure.ac.fixed.in"
  "${PROJECT_BINARY_DIR}/wrappers/qball-configure.fixed.ac"
  @ONLY
)
ExternalProject_Add(
  XERCES
  EXCLUDE_FROM_ALL 1
  URL "http://pages.cs.wisc.edu/~welton/xerces-c-3.1.4.tar.bz2"
  URL_MD5 "d04ae9d8b2dee2157c6db95fa908abfd"
  SOURCE_DIR "${PROJECT_BINARY_DIR}/3rdparty/xerces"
  CONFIGURE_COMMAND ${PROJECT_BINARY_DIR}/3rdparty/xerces/configure --prefix=${CMAKE_INSTALL_PREFIX}
  BUILD_COMMAND $(MAKE) -j 4
)

ExternalProject_Add(
  QBALL
  DEPENDS XERCES FFTW_INSTALL
  GIT_REPOSITORY "https://github.com/LLNL/qball.git"
  GIT_TAG "3e151fbdc598493085b071c2fe995a0d6845b223"
  SOURCE_DIR "${PROJECT_BINARY_DIR}/3rdparty/qball"
  CONFIGURE_COMMAND cp ${PROJECT_BINARY_DIR}/wrappers/qball_configure.sh ${PROJECT_BINARY_DIR}/3rdparty/qball && bash ${PROJECT_BINARY_DIR}/3rdparty/qball/qball_configure.sh 
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) install
  UPDATE_COMMAND ""
)

ExternalProject_Add(
  QBALL_FIXED
  DEPENDS XERCES FFTW_INSTALL fftw_intercept
  GIT_REPOSITORY "https://github.com/LLNL/qball.git"
  GIT_TAG "3e151fbdc598493085b071c2fe995a0d6845b223"
  SOURCE_DIR "${PROJECT_BINARY_DIR}/3rdparty/qball_fixed"
  CONFIGURE_COMMAND chmod 777 ${PROJECT_BINARY_DIR}/wrappers/build_qball_fixed.sh && bash ${PROJECT_BINARY_DIR}/wrappers/build_qball_fixed.sh
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) install && cp ${PROJECT_BINARY_DIR}/3rdparty/qball_fixed/finalbin/bin/qball ${CMAKE_INSTALL_PREFIX}/bin/qball_fixed
  UPDATE_COMMAND ""
)