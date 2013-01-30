1.3 (Sep 13, 2012)
====================

- Add fix from Matthias Tuma from Shark3 to allow friend declarations inside classes.

1.2 (Nov 3, 2011)
====================

- Add Python 3 support

1.1 (Feb 19, 2011)
====================

- Add support for linking directly to struct definitions.
- Allow to link to functions etc. which are in a header/source file but not a member of a class.

1.0 (Dec 14, 2010)
====================

- New Dependency: PyParsing (http://pyparsing.wikispaces.com/)
- Completely new tag file parsing system. Allows for function overloading.
  The parsed results are cached to speed things up.
- Full usage documentation. Build with `sphinx-build -W -b html doc html`.
- Fix problem with mixed slashes when building on Windows.

0.4 (Aug 15, 2010)
====================

- Allow URLs as base paths for the HTML links.
- Don't append parentheses if the user has provided them already in their query.

0.3 (Aug 10, 2010)
====================

- Only parse the tag file once per run. This should increase the speed.
- Automatically add parentheses to functions if the add_function_parentheses config variable is set.

0.2 (Jul 31, 2010)
====================

- When a target cannot be found, make the node an `inline` node so there's no link created.
- No longer require a trailing slash on the `doxylink` config variable HTML link path.
- Allow doxylinks to work correctly when created from a documentation subdirectory.

0.1 (Jul 22, 2010)
==================

- Initial release
