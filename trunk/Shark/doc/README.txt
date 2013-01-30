
This file provides basic assistance for getting the Shark documentation installed.
Note that there is a somewhat more detailed tutorial as part of the online documentation.


DOCUMENTATION INSTALLATION GUIDE
--------------------------------

  What you need to have: 
  
    - A working Doxygen installation
    - EITHER a working GraphViz installation (more specifically, the dot tool),
      OR manually edited doc/shark.dox(.in) to use a different plotting tool by
      setting HAVE_DOT = NO
    - A working python installation, ideally not conflicting with different
      versions of itself.
    - The python setuptools package.
    - Either an internet connection, or a completely functional dependency
      tree for the Sphinx and Doxylink documentation tools.
    
    
  What you need to do:

    - In the shark home directory, go to the sub-directory contrib/doxylink/sources
    
    - There, try to issue the command
      PYTHONPATH=../build/lib/python python setup.py install --home=../build --record file_list_for_uninstall.txt
      
      [Note: on a mac, add a colon after the python path, like:
       PYTHONPATH=../build/lib/python: python setup.py install --home=../build --record file_list_for_uninstall.txt ]
       
      [Also note: on some systems, the python command might invoke python 3, and python 2 be invoked by python2.
       In such a case, make sure that the above command calls the appropriate version.]
       
      [Further note: this is the most common place for the installation process to fail.
       Reasons can either be that one of the python prerequesites is not met. This will
       usually boil down to python setuptools, because that is what pulls in the other
       missing tools. So, if there is an error, then check if you have python setuptools
       installed. Other types of error messages may stem from conflicts between different
       python versions, or between tools installed for different python versions. ]
       
    - Issue "make doc" in the doc/ subdirectory.
    
	  [Note: Common problems can be:
	  
	    - You have to configure the build with "ccmake ." first
	    
	    - The sphinx-build executable is not found. In such a case, consider 
    
		   * installing Sphinx through your distribution again.
		   
		   * examining, relative to the $SHARKHOME directory, both "contrib/doxylink/build/bin" as well as e.g.
		     "contrib/doxylink/build/lib/python/Sphinx-1.0.7-py2.7.egg/sphinx" and maybe adding a manual alias
		     from "sphinx-build" to the correct executable "contrib/doxylink/build/bin/sphinx-build", or including
		     the former directory to your python and/or system path.
		     
		   * it can also help to issue "make doc" with a user-controlled pythonpath, e.g. as in
			 " PYTHONPATH=/usr/lib/python2.7/site-packages/:/path/to/your/Shark3/contrib/doxylink/build/lib/python make doc "
			 
           * in extreme cases, you might want to look into the www.virtualenv.org tool for managing concurrent python installations. ]
    
    - You know that you are done when make doc exits with "build succeeded. Built target doc",
      and when you can successfully view the page $SHARKHOME/doc/index.html.
      
      
  Further troubleshooting:

	- In general, if you run into troubles, you should try to make sure all dependencies are installed and accessible.
	  The most relevant dependencies are Sphinx and Doxylink, which in turn rely on a number of tools, e.g., Docutils,
	  Jinja2, Pygments, and Pyparsing, which however should be taken care of automatically by setuptools. You can check
	  upon the added installations by examining the directory tree under ``contrib/doxylink/`` with e.g. ``ls -R``.
	  Then it usually boils down to either installing what's missing or making the path known in the correct manner.
	  Good luck!
