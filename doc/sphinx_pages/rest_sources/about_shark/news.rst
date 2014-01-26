News
====

Shark goes LGPL
^^^^^^^^^^^^^^^

As of January 2014, Shark is distributed under the permissive
`GNU Lesser General Public License <http://http://www.gnu.org/copyleft/lesser.html>`_.

.. image:: images/lgplv3-147x51.png


Repository upgraded
^^^^^^^^^^^^^^^^^^^

We upgraded our Sourceforge repository to the newest version
as recommended by Sourceforge. The new path to Shark is now
(a ``Shark`` directory will be created as a subfolder -- if you
want the tree contents directly in the current directory, add a
space and period ``.`` to the end of the command):

.. code-block:: none

    svn co https://svn.code.sf.net/p/shark-project/code/trunk/Shark



Shark 3.0 alpha moved to svn trunk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are happy to announce that the new version 3.0 of Shark has now moved to the trunk of our
svn repository, replacing the last stable version of Shark 2.0.

Shark's high-level interface is regarded as stable. But until
a group of lower-level changes is included, Shark 3.0 is considered
in beta stage. Note that Shark 3.0 is already actively used for
everyday research by several machine learning groups.

Also note that some tutorials still need to be
updated to reflect recent interface changes, so code from
some tutorials may not compile. Especially the Data tutorials
are known to be outdated.

Feel free to try and test Shark, we are happy about any feedback!

You can download Shark from our svn repository:

.. code-block:: none

    svn co https://svn.code.sf.net/p/shark-project/code/trunk/Shark


There is currently a known problem for MacOs users that they cannot compile using gcc 4.2.1. In this
case you have to update to a newer version of the gcc or use clang as described in our faq.


Gold Prize for Shark alpha-release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**October 22, 2011:**
We are happy to announce that our alpha-release of Shark 3.0 has won
the Gold Prize at this year's `Open Source Software World Challenge 2011 <http://www.ossaward.org/>`_.
