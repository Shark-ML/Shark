
This file provides basic assistance for getting the Shark documentation installed.
Note that there is a somewhat more detailed tutorial as part of the online documentation.


DOCUMENTATION INSTALLATION GUIDE
--------------------------------

  What you need to have: 
  
    - A working Doxygen installation
    - EITHER a working GraphViz installation (more specifically, the dot tool),
      OR manually edited doc/shark.dox(.in) to use a different plotting tool by
      setting HAVE_DOT = NO
    - A working python installation, ideally set up not to easily conflict with
      different versions of itself.
    - Either an internet connection, or a completely functional dependency
      tree for the Sphinx and Doxylink documentation tools.
    
    
  What you need to do:

    - Make sure that both Sphinx and Doxylink are installed to your system (if
      you install doxylink via some python package manager, e.g., via
      "pip install --install-option="--prefix=/home/user/path/to/your/pip_python_packages" sphinxcontrib_doxylink"
      or similar, Sphinx will be automatically installed as a dependency). Then
      make sure that $PATH is set to find the Sphinx executable "sphinx-build"
      and that the doxylink package is in the $PYTHONPATH  (e.g. from the example
      above, /home/user/path/to/your/pip_python_packages/lib/python3.3/site-packages)
      
    - Configure/populate the CMake build system for the documentation via "ccmake ."
      in <SHARK_SRC_DIR>/doc. Note that you can build the documentation both in- and
      out-of-source. We recommend not to build the documentation together with the
      possible several versions of the Shark library, but rather separately either
      in- or out-of-source.
       
    - Issue "make doc" in the doc/ subdirectory. You know that you are done when
      "make doc" exits with "build succeeded. Built target doc",
      and when you can successfully view the page $SHARKHOME/doc/index.html.
    
	  Common problems can be:
	  
	    - You have to configure the build with "ccmake ." first
	    
	    - The sphinx-build executable is not found. In such a case, consider 
    
		   * installing Sphinx through your distribution again.
		   
		   * examining PATH; maybe adding a manual alias
		     from "sphinx-build" to the correct executable of your Sphinx installation;
		     and/or examining your PYTHONPATH
		     
		   * it can also help to issue "make doc" with a user-controlled pythonpath, e.g. as in
			 " PYTHONPATH=/usr/lib/python2.7/site-packages/:/path/to/your/Shark3/contrib/doxylink/build/lib/python make doc "
			 
           * in extreme cases, you might want to look into the www.virtualenv.org tool for managing concurrent python installations. ]
    
      
  Further troubleshooting:

	- In general, if you run into troubles, you should try to make sure all dependencies are installed and accessible.
	  The most relevant dependencies are Sphinx and Doxylink, which in turn rely on a number of tools, e.g., Docutils,
	  Jinja2, Pygments, and Pyparsing, which however should be taken care of automatically by the parent packages.
	  Then it usually boils down to either installing what's missing or making the path known in the correct manner.
	  Good luck!
