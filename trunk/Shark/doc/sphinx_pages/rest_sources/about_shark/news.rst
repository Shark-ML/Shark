News
====

Repository upgraded
^^^^^^^^^^^^^^^^^^^

We upgraded our Sourceforge repository to the newest version as
recommended by Sourceforge. The new path to Shark is now:

    svn co https://svn.code.sf.net/p/shark-project/code/trunk/Shark



Shark 3.0beta moved to svn trunk
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

You can download Shark from our svn repository::

    svn co https://svn.code.sf.net/p/shark-project/code/trunk/Shark


There is currently a known problem for MacOs users that they cannot compile using gcc 4.2.1. In this
case you have to update to a newer version of the gcc or use clang as described in our faq.


Gold Prize for Shark alpha-release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**October 22, 2011:**
We are happy to announce that our alpha-release of Shark 3.0 has won
the Gold Prize at this year's `Open Source Software World Challenge 2011 <http://www.ossaward.org/>`_.

Alpha release of Shark 3.0 submitted to OSS  World Challenge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**September 18 2011:** The first pre-release version of Shark has
been finalized and uploaded on occasion of the
`Open Source Software World Challenge 2011 <http://www.ossaward.org/>`_. We
thank the organizers, hosts, and sponsors of the competition for the opportunity
to participate in such a challenge.

We for convenience again provide the
:download:`Shark project development plan <downloads/development_plan_shark_project_igel.pdf>`
as submitted to the first (registration) stage of the competition.

We also provide direct entry points to all deliverables for the competition:

* **Source code**: please find a complete source code package on the main
  :doc:`Shark download page <../downloads/downloads>`. There, you will also
  find pre-compiled binary packages for all major platforms, and other
  download links.

* **License information**: please find all license information on the
  :doc:`Credits and Copyright page <../about_shark/copyright>`.

* **Installation instructions**: Shark can be installed either using
  pre-compiled binary packages or by compiling from source code. Some
  attention has to be paid to installation/availability of the Boost
  C++ library as an external dependency. The
  :doc:`Shark installation page <../getting_started/installation>`
  will cover all aspects of the installation in detail.

  We recognize that installation practices, systems, and experience may
  vary between different companies, labs, and universities. If there are any
  gaps or ambiguities in the installation instructions, or documentation in
  general, please do not hesitate to point them out to us and we will fix any
  remaining issues immediately.

* **Other documentation**: We provide class and member documentation extracted
  from in-code documentation, as well as a vast number of tutorials, test sets,
  and examples. Please look at our guide to the different entry points into these
  documentation options :doc:`here <../getting_started/using_the_documentation>`.

* **Introductory videos**: As part of the submission, we created a
  video giving a short overview over the library:

  .. raw:: html

     <iframe width="420" height="315" src="http://www.youtube.com/embed/zxvApdNZVgA" frameborder="0" allowfullscreen></iframe>


  We further provide an introductory video on Shark installation
  under the Windows operating system on the :doc:`Shark installation
  page <../getting_started/installation>`.
