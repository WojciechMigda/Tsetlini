# -*- coding: utf-8 -*-
#
"""setuptools-based setup.py template for Cython projects.

Main setup for the library.

Supports Python 3.4.

Usage as usual with setuptools:
    python setup.py build_ext
    python setup.py build
    python setup.py install
    python setup.py sdist

For details, see
    http://setuptools.readthedocs.io/en/latest/setuptools.html#command-reference
or
    python setup.py --help
    python setup.py --help-commands
    python setup.py --help bdist_wheel  # or any command
"""

import os
import sys
import numpy as np

from setuptools.extension import Extension










def declare_cython_extension(extName, build_type, use_math=False, use_openmp=False, include_dirs=None):
    """Declare a Cython extension module for setuptools.

Parameters:
    extName : str
        Absolute module name, e.g. use `tsetlin_tk.mypackage.mymodule`
        for the Cython source file `tsetlin_tk/mypackage/mymodule.pyx`.

    use_math : bool
        If True, set math flags and link with ``libm``.

    use_openmp : bool
        If True, compile and link with OpenMP.

Return value:
    Extension object
        that can be passed to ``setuptools.setup``.
"""

    # Define our base set of compiler and linker flags.
    #
    # This is geared toward x86_64, see
    #    https://gcc.gnu.org/onlinedocs/gcc-4.6.4/gcc/i386-and-x86_002d64-Options.html
    #
    # Customize these as needed.
    #
    # Note that -O3 may sometimes cause mysterious problems, so we limit ourselves to -O2.

    # Modules involving numerical computations
    #
    extra_compile_args_math_optimized    = ['-march=native', '-O2', '-msse', '-msse2', '-mfma', '-mfpmath=sse']
    extra_compile_args_math_debug        = ['-march=native', '-O0', '-g']
    extra_link_args_math_optimized       = []
    extra_link_args_math_debug           = []

    # Modules that do not involve numerical computations
    #
    extra_compile_args_nonmath_optimized = ['-O2']
    extra_compile_args_nonmath_debug     = ['-O0', '-g']
    extra_link_args_nonmath_optimized    = []
    extra_link_args_nonmath_debug        = []

    # Additional flags to compile/link with OpenMP
    #
    openmp_compile_args = ['-fopenmp']
    openmp_link_args    = ['-fopenmp']

    gdb_debug = False
    # Choose the base set of compiler and linker flags.
    #
    if build_type == 'optimized':
        my_extra_compile_args_math    = extra_compile_args_math_optimized
        my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
        my_extra_link_args_math       = extra_link_args_math_optimized
        my_extra_link_args_nonmath    = extra_link_args_nonmath_optimized
        gdb_debug = False
        print( "build configuration selected: optimized" )
    elif build_type == 'debug':
        my_extra_compile_args_math    = extra_compile_args_math_debug
        my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
        my_extra_link_args_math       = extra_link_args_math_debug
        my_extra_link_args_nonmath    = extra_link_args_nonmath_debug
        gdb_debug = True
        print( "build configuration selected: debug" )
    else:
        raise ValueError("Unknown build configuration '%s'; valid: 'optimized', 'debug'" % (build_type))


    extPath = extName.replace(".", os.path.sep)+".pyx"

    if use_math:
        compile_args = list(my_extra_compile_args_math) # copy
        link_args    = list(my_extra_link_args_math)
        libraries    = ["m"]  # link libm; this is a list of library names without the "lib" prefix
    else:
        compile_args = list(my_extra_compile_args_nonmath)
        link_args    = list(my_extra_link_args_nonmath)
        libraries    = []  # value if no libraries, see setuptools.extension._Extension

    # OpenMP
    if use_openmp:
        compile_args.insert( 0, *openmp_compile_args )
        link_args.insert( 0, *openmp_link_args )

    compile_args += ['-std=c++17', '-Wall']
    link_args += ['../../.build/lib/libtsetlini_static.a']

    # See
    #    http://docs.cython.org/src/tutorial/external.html
    #
    # on linking libraries to your Cython extensions.
    #
    return Extension( extName,
                      [extPath],
                      extra_compile_args=compile_args,
                      extra_link_args=link_args,
                      include_dirs=include_dirs,
                      libraries=libraries
                    ), gdb_debug


def collect_datafiles(datadirs, dataexts):
    # Gather user-defined data files
    #
    # http://stackoverflow.com/questions/13628979/setuptools-how-to-make-package-contain-extra-data-folder-and-all-folders-inside
    #
    datafiles = []
    getext = lambda filename: os.path.splitext(filename)[1]
    for datadir in datadirs:
        datafiles.extend( [(root, [os.path.join(root, f) for f in files if getext(f) in dataexts])
                           for root, dirs, files in os.walk(datadir)] )

    # Add standard documentation (README et al.), if any, to data files
    #
    # Standard documentation to detect (and package if it exists).
    #
    standard_docs     = ["README", "LICENSE", "TODO", "CHANGELOG", "AUTHORS"]  # just the basename without file extension
    standard_doc_exts = [".md", ".rst", ".txt", ""]  # commonly .md for GitHub projects, but other projects may use .rst or .txt (or even blank).

    detected_docs = []
    for docname in standard_docs:
        for ext in standard_doc_exts:
            filename = "".join( (docname, ext) )  # relative to the directory in which setup.py resides
            if os.path.isfile(filename):
                detected_docs.append(filename)
    datafiles.append( ('.', detected_docs) )

    return datafiles


def extract_version(libname):
    # Extract __version__ from the package __init__.py
    # (since it's not a good idea to actually run __init__.py during the build process).
    #
    # http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
    #
    import ast
    init_py_path = os.path.join(libname, '__init__.py')
    version = '0.0.unknown'
    try:
        with open(init_py_path) as f:
            for line in f:
                if line.startswith('__version__'):
                    version = ast.parse(line).body[0].value.s
                    break
            else:
                print( "WARNING: Version information not found in '%s', using placeholder '%s'" % (init_py_path, version), file=sys.stderr )
    except FileNotFoundError:
        print( "WARNING: Could not find file '%s', using placeholder version information '%s'" % (init_py_path, version), file=sys.stderr )

    return version


def setup_package():

    #########################################################
    # General config
    #########################################################

    # Name of the top-level package of your library.
    #
    # This is also the top level of its source tree, relative to the top-level project directory setup.py resides in.
    #
    libname="tsetlin_tk"

    # Short description for package list on PyPI
    #
    SHORTDESC="Tsetlin Machine Toolkit"

    # Long description for package homepage on PyPI
    #
    DESC="""Tsetlin Machine Toolkit

    Scikit-learn compatible implementation.
    """

    # Set up data files for packaging.
    #
    # Directories (relative to the top-level directory where setup.py resides) in which to look for data files.
    datadirs  = ("tests",)

    # File extensions to be considered as data files. (Literal, no wildcards.)
    dataexts  = (".py",  ".pyx", ".pxd",  ".c", ".cpp", ".h",  ".sh",  ".lyx", ".tex", ".txt", ".pdf")

    #########################################################
    # Init
    #########################################################

    # check for Python 3.4 or later
    # http://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
    if sys.version_info < (3,4):
        sys.exit('Sorry, Python < 3.4 is not supported')

    # Make absolute cimports work.
    #
    # See
    #     https://github.com/cython/cython/wiki/PackageHierarchy
    #
    # For example: my_include_dirs = [np.get_include()]
    my_include_dirs = [".", "../../lib/include", "../../lib/src", np.get_include()]


    #########################################################
    # Set up modules
    #########################################################

    # declare Cython extension modules here
    #
    ext_module_libtsetlin, gdb_debug = declare_cython_extension( "tsetlin_tk.libtsetlin", "optimized", use_math=False, use_openmp=True, include_dirs=my_include_dirs)

    # this is mainly to allow a manual logical ordering of the declared modules
    #
    cython_ext_modules = [ext_module_libtsetlin]

    # Call cythonize() explicitly, as recommended in the Cython documentation. See
    #     http://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiling-with-distutils
    #
    # This will favor Cython's own handling of '.pyx' sources over that provided by setuptools.
    #
    # Note that my_ext_modules is just a list of Extension objects. We could add any C sources (not coming from Cython modules) here if needed.
    # cythonize() just performs the Cython-level processing, and returns a list of Extension objects.
    #
    try:
        from Cython.Build import cythonize
    except ImportError:
        sys.exit("Cython not found. Cython is needed to build the extension modules.")
    my_ext_modules = cythonize( cython_ext_modules, include_path=my_include_dirs, gdb_debug=gdb_debug, compiler_directives={"language_level": 3} )

    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner'] if needs_pytest else []

    #########################################################
    # Call setup()
    #########################################################

    from setuptools import setup

    setup(
        name="Tsetlin Machine Toolkit",
        version=extract_version(libname),
        author="Wojciech Migda",
        author_email="wojciech.migda@interia.pl",
        url="https://github.com/WojciechMigda/TsetlinMachineToolkit",

        description=SHORTDESC,
        long_description=DESC,

        license= "MIT",

        # free-form text field; http://stackoverflow.com/questions/34994130/what-platforms-argument-to-setup-in-setup-py-does
        platforms=["Linux"],

        # See
        #    https://pypi.python.org/pypi?%3Aaction=list_classifiers
        #
        # for the standard classifiers.
        #
        # Remember to configure these appropriately for your project, especially license!
        #
        classifiers=["Development Status :: 4 - Beta",
                     "Environment :: Console",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Science/Research",
                     "License :: MIT",
                     "Operating System :: POSIX :: Linux",
                     "Programming Language :: Cython",
                     "Programming Language :: Python",
                     "Programming Language :: Python :: 3",
                     "Programming Language :: Python :: 3.4",
                     "Topic :: Scientific/Engineering",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     "Topic :: Software Development :: Libraries",
                     "Topic :: Software Development :: Libraries :: Python Modules"
                    ],

        # See
        #    http://setuptools.readthedocs.io/en/latest/setuptools.html
        #
        setup_requires = ["cython", "numpy", "scikit-learn"] + pytest_runner,
        tests_require=["pytest"],
        install_requires = ["numpy", "scikit-learn"],
        provides = ["setup_template_cython"],

        # keywords for PyPI (in case you upload your project)
        #
        # e.g. the keywords your project uses as topics on GitHub, minus "python" (if there)
        #
        keywords = ["tsetlin machine learning"],

        # All extension modules (list of Extension objects)
        #
        ext_modules=my_ext_modules,

        # Declare packages so that  python -m setup build  will copy .py files (especially __init__.py).
        #
        # This **does not** automatically recurse into subpackages, so they must also be declared.
        #
        packages=["tsetlin_tk"],

        # Install also Cython headers so that other Cython modules can cimport ours
        #
        # Fileglobs relative to each package, **does not** automatically recurse into subpackages.
        #
        # FIXME: force sdist, but sdist only, to keep the .pyx files (this puts them also in the bdist)
        package_data={'tsetlin_tk': ['*.pxd', '*.pyx']},

        # Disable zip_safe, because:
        #   - Cython won't find .pxd files inside installed .egg, hard to compile libs depending on this one
        #   - dynamic loader may need to have the library unzipped to a temporary directory anyway (at import time)
        #
        zip_safe=False,

        # Custom data files not inside a Python package
        data_files=collect_datafiles(datadirs, dataexts)
    )


if __name__ == "__main__":
    setup_package()
