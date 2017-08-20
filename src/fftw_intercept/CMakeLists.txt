add_library(fftw_intercept SHARED "cufftw_conversions.cpp")
add_dependencies(fftw_intercept FFTW_INSTALL)
INCLUDE_DIRECTORIES(fftw_intercept ${FFTW_DIR}/include)
INCLUDE_DIRECTORIES(fftw_intercept ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(fftw_intercept "-L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -lcudart -lcufft -lcufftw")
target_link_libraries(fftw_intercept "-L${LIBCUDA_DIR} -lcuda")
install(TARGETS fftw_intercept
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)

set (INTERCEPT_PATH ${CMAKE_INSTALL_PREFIX}/lib)