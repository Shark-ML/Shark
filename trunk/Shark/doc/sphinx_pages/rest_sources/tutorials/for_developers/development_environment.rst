The Shark Development Environment
=================================

There is a number of tools the Shark developer team uses to ease
contributing code and to ensure the code quality. The most important
tools are described below in order to facilitate new members 
working with Shark, or external developers contributing to Shark.


Software Quality Management
---------------------------

The Shark library deploys a range of sophisticated methods and technologies
to ensure a high level of software quality and to prevent from regressions.
First and foremost, an extensive and ever-growing set of unit tests is available
that reflects our approach of test-driven development. Moreover, a 
`pre-commit review system <https://nisys.dyndns.biz:8081>`_ is in place,
and every check-in to the central subversion repository triggers a built-test-package
cycle for all of our supported platforms on our `continuous integration system
<https://nisys.dyndns.biz/shark>`_. In addition, we continuously execute static
code analysises to check for errors and to enforce our coding conventions.


More on the review system (ReviewBoard)
---------------------------------------

We use the `ReviewBoard <http://www.reviewboard.org/>`_ code review
software. After getting an account `on the Shark ReviewBoard
<https://nisys.dyndns.biz:8081>`_, you are ready to post your
files for review to the Shark developer team. We recommend using
the post-review tool with the ReviewBoard system, since it greatly
simplifies the whole process:

After editing and/or creating the Shark source files, you will need
the list of all files you want to post for review, with their paths
relative to the Shark main directory. You will pass this
list to the ``post-review`` command (issued from the Shark main
directory) together with additional options:

* Either only use the ``-o`` option to ``post-review``. This should
  open a web browser window showing a menu which allows you to fill
  in important details for your review commit. Thus, your final command
  for posting your files for review will look like this::
  
      post-review -o path/to/your/file/one/relative/to/shark/main/directory path/to/your/file/two/relative/to/shark/main/directory
  
* Alternatively, and/or if the above fails to open a menu in the browser,
  pass all this important information to the ``post-review`` command
  directly. A good template is the following::

      post-review -p --summary='very brief summary' --description='a few lines of description as appropriate' --target-groups='SharkCoreDevelopers' path/to/your/file/one/relative/to/shark/main/directory path/to/your/file/two/relative/to/shark/main/directory

You can also make changes to your own review, a link to which will be
mailed to all developers in the target group. For this reason, it is
important to include the information on the target group because no
mail will be sent if the group is wrong or missing.



