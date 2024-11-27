project('PoseEstimation', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++14'])

opencv_dep = dependency('opencv4')
incdir = include_directories('include')

executable('PoseEstimation',
           'asadmp.cpp',
           install : true, dependencies: [ opencv_dep ], include_directories: [ incdir ])
