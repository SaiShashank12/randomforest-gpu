project('randomforest_gpu', 'c', 'fortran',
    version: '0.1.0',
    license: 'GPL-2.0-or-later',
    meson_version: '>=1.0.0',
    default_options: [
        'warning_level=2',
        'buildtype=release',
    ]
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

# Get NumPy include directories
incdir_numpy = run_command(py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# Fortran compiler options
fc = meson.get_compiler('fortran')
fortran_args = []

# Debug: Print compiler ID
message('Fortran compiler ID: ' + fc.get_id())
message('Fortran compiler version: ' + fc.version())

# Try to detect GPU compiler capabilities
if fc.get_id() == 'nvidia_hpc' or fc.get_id() == 'nvfortran' or fc.get_id() == 'pgi'
    # NVIDIA HPC SDK compiler - Enable OpenACC for GPU acceleration
    message('Detected NVIDIA compiler - Enabling OpenACC GPU support')
    fortran_args += ['-acc', '-gpu=managed', '-Minfo=accel']
    # Define _OPENACC preprocessor macro
    fortran_args += ['-D_OPENACC']
elif fc.get_id() == 'gcc'
    # GNU Fortran - CPU with OpenMP (OpenACC directives ignored)
    message('Using GNU Fortran - CPU with OpenMP parallelization')
    fortran_args += ['-fopenmp']
    # Add static linking for portability (gfortran only)
    if host_machine.system() == 'linux'
        fortran_args += ['-static-libgfortran', '-static-libgcc']
    endif
else
    message('Unknown Fortran compiler - using default settings')
endif

# Fortran source files
fortran_sources = files(
    'src/fortran/rf_gpu_wrapper.f90',
    'src/fortran/rf_core.f90',
)

# Build Fortran library and install it
rf_lib = shared_library('rf_gpu_core',
    fortran_sources,
    fortran_args: fortran_args,
    install: true,
    install_dir: py.get_install_dir() / 'randomforest_gpu',
)

# Generate ctypes wrapper info
subdir('src/randomforest_gpu')

# Install Python package
py.install_sources(
    [
        'src/randomforest_gpu/__init__.py',
        'src/randomforest_gpu/ensemble.py',
        'src/randomforest_gpu/backend.py',
    ],
    subdir: 'randomforest_gpu'
)
