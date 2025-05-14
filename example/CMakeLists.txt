cmake_minimum_required(VERSION 3.27)

set(PROJECT_NAME SmollExample)
project(${PROJECT_NAME} LANGUAGES CXX)
find_package(SmollNet REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE SmollNet::SmollNet)
