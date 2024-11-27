#!/bin/sh
c++ ./asad_mp_ros.cpp -o asad_ros \
    $(pkg-config --libs --cflags opencv4) \
    -Wall \
    -Wextra \
    -g  \
    -fopenmp \
    -Iinclude \
    -I/opt/ros/noetic/include \
    -L/opt/ros/noetic/lib \
    -lroscpp \
    -lrostime \
    -lrosconsole \
    -limage_transport \
    -lcv_bridge \
    -Ofast \
    -flto \
    -pg