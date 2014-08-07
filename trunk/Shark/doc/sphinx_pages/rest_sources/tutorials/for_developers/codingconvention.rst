Coding Convention
=================

This document specifies the coding conventions for the Shark library.

Most of the conventions are considered mandatory.
A few optional rules are marked as such.


Naming Conventions
------------------

Class/Interface Names
.....................

All type names (classes, interfaces, enumerations) should use the
InfixCaps style. Start with an upper-case letter, and capitalize the
first letter of any subsequent word in the name, as well as any letters
that are part of an acronym. All other characters in the name are
lower-case. Do not use underscores to separate words. Class names
should be nouns or noun phrases. Interface names depend on the salient
purpose of the interface and are prefixed with a capitalized I. If the
purpose is primarily to endow an object with a particular capability,
then the name should be an adjective (ending in -able or -ible if
possible) that describes the capability; e.g., ISearchable, ISortable,
INetworkAccessible. Otherwise use nouns or noun phrases.


Member Names
............

Names of non-constant, non-pointer fields are prefixed with ``m_``
and use the camelCase-style. Pointers that are owned by the class,
i.e. they are freed on destruction, are prefixed with ``mp_`` and
use the camelCase-style. Pointers that are not owned by the class are
prefixed with ``mep_``.

* ``m_``: Member

* ``mp_``: Member Pointer (optional rule)

* ``mep_``: Member External Pointer (optional rule)

Names of fields being used as constants should be all upper-case, with
underscores separating words. The following are considered to be
constants:

* ``MIN_VALUE``

* ``MAX_BUFFER_SIZE``

* ``OPTIONS_FILE_NAME``

One-character field names should be avoided except for temporary and
looping variables. In these cases use (optional rule):

* ``c`` for a char
* ``d`` d for a double
* ``e`` for an Exception object
* ``f`` for a float
* ``i``, ``j``, ``k``, ``m``, ``n`` for integers
* ``p``, ``q``, ``r``, ``s`` for strings

An exception is where a strong convention for the one-character name
exists, such as ``x``, ``y``, ``z`` for coordinates.

Method Names
............

Method names should use the infixCaps style. Start with a lower-case
letter, and capitalize the first letter of any subsequent word in the
name, as well as any letters that are part of an acronym. All other
characters in the name are lower-case. Do not use underscores to
separate words. Method names should be imperative verbs or verb
phrases, for example:

* ``showStatus()``

* ``drawCircle()``

* ``addLayoutComponent()``

A method to get or set some property or member of the class should be
called ``property()`` or ``setProperty()`` respectively, where
"property" is the name of the property.

A method to test some boolean property of the class should be called
``isProperty()``, where "property" is the name of the property.

Formatting and White Space Usage
--------------------------------

Blank Lines
...........

Blank lines can improve readability by grouping sections of the code
that are logically related. Blank lines should also be used in the
following places:

* After the copyright block comment.

* Between class declarations.

* Between method declarations.

* Before a block or single-line comment, unless it is the first line in a block.

Indentation
...........

Only tabulators should be used to indent lines, i.e., no space character
should be part of the white space at the beginning of a line.
The reasoning behind this convention is: (1) convenient tabulator
characters are used for indentation, (2) code is well-aligned in any
text editor, independent of the interpretation of tabulator characters,
and (3) text further on in a line remains aliged as long as it does not
spread beyond a single level of indentation.
Example:

.. code-block:: c++

	.___.___{
	.___.___.___int a = 42;.......................................//.magic.answer
	.___.___.___char q[] = "The.ultimate.question.of.life,."
	.___.___.___..........."the.universe,.and.everything";........//.corresponding.question
	.___.___}

Here, ".___" is used to indicate a tabulator, while a single dot "."
indicates a space character. Obviously, the code block will look pretty
with different tabulator sizes in place, and the comments at the end of
lines two and four remain aligned.

A single tabulator should be added at each indentation level. Usually
curly braces indicate the next level of indentation, with the exception
of namespaces.

Braces and space character placement, other details
...................................................

Details such as where to place spaces around brackets etc. are not
specified, since the **astyle** tool makes such conventions superfluous.


Namespaces
----------

All code in the Shark library is placed into the namespace ``shark``.
Currently, the only other namespace in ``shark::detail``. Content within
the detail namespace is considered 'protected', and it is often found in
files places in 'impl' sub-directories. This code may be less
well-documented than the 'public' code base, since it is not intended to
be used directly from outside the library.


Class Layout
------------

A class definition should be structured as follows:

.. code-block:: c++

	/// Documentation of the role of the class as a whole
	class ClassName : public BaseClass {
	public:
		/// Documentation for the constructor, if necessary
		ClassName();

		~ClassName();

		/// Documentation for property 1
		PropertyType1 property1() const;
		void setProperty1( const PropertyType1 & property );

		/// Documentation for property 2
		PropertyType2 & property2();
		const PropertyType2 & property2() const;
		void setProperty2();

		/// Documentation for property 3
		PropertyType3 * property3();
		const PropertyType3 * property3() const;
		void setProperty3();

		/// Documentation for property 4
		bool isProperty4() const;
		void setProperty4( bool value );

	protected:
		/// Documentation for member m_property1
		PropertyType1 m_property1;

		/// Documentation for member m_property2
		PropertyType2 m_property2;

		/// Documentation for member m_property3
		PropertyType3 * mp_property3();

		/// Documentation for member m_property4
		bool m_property4;
	};


Interface Layout
----------------

An interface should only contain pure virtual methods and a virtual,
empty destructor. No members should be put within an interface
declaration. To reduce the effort to implement an interface, a general
purpose default implementation can be provided in an abstract class that
inherits the respective interface.


Header and Source Files
-----------------------

The general rule is that declarations should be put into header
files and implementations should go into source files. There may be
exceptions for declarations that are used only locally within one
source file, such as in example files or unit test.

A declaration in the above sense is everything that does not directly
generate code, while everything that has a direct imprint as executable
code or data in the library is an implementation. Examples of
declarations are:

* class declarations

* inline functions, including their implementations

* template classes and functions, including their implementations

Examples of implementations are

* bodies of non-template functions, free or members of a class

* static variables

All header files have to be protected against multiple inclusion by
the following sequence of pre-processor statements:

.. code-block:: c++

	#ifndef SHARK_<MODULE>_<FILENAME>_H
	#define SHARK_<MODULE>_<FILENAME>_H

	[...declarations...]

	#endif

For example, the file **Exception.h** in the module **Core** is
protected by the name ``SHARK_CORE_EXCEPTION_H``.

Shark makes extensive use of templates. Therefore large parts of the
code base are found in header files. To maintain a clean structure some
headers are 'hidden' in sub-directories with name *impl*.

Statements with strong side effects should be avoided in header files.
``using`` statements must not be used at global scope of the scope of
the *shark* namespace in header files. Definition of names by means of
``#define`` statements should be avoided where possible.



Unit Tests
----------


When adding functionality to Shark it is **mandatory** to also add
meaningful test cases.



Other tasks
-----------

It is one feature of the Shark tutorials that they list all Models,
Kernels, Losses, Optimizers, StoppingCriteria, and Trainers implemented
in Shark. These lists are one of the few components that do not update
automatically via Sphinx-Doxygen-Code magic. Thus, if you add a new
class implementing any of the above, please make this known in the
corresponding list. Thank you!

