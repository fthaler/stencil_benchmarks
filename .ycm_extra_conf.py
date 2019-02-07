def FlagsForFile( filename, **kwargs ):
  return {
    'flags': ['-x', 'c++', '-std=c++14', '-I', 'src', '-fopenmp', '-DSBENCH_FLOAT_TYPE=float'],
  }
