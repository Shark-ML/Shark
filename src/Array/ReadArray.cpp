/*!
*  \file ReadArray.cpp
*
*  \brief Contains functions for reading the content of an array from a
*         given input stream.
*
*  \author  M. Kreutz
*
*  \par
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
*
*  \par Project:
*      Array
*
*
*  <BR><HR>
*  This file is part of Array. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 2, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, write to the Free Software
*  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#include <cmath>
#include <iostream>
#include <list>
#include <algorithm>
#include <SharkDefs.h>
#include <Array/ArrayOp.h>
#include <Array/ArrayIo.h>

using namespace std;


//===========================================================================
/*!
 *  \brief Reads from input stream "is" until "token" is found.
 *
 *  Starting at the current position of the get pointer, it is searched
 *  for the next occurence of \em token. If the token is found,
 *  the get pointer is set to the first position after the token, otherwise
 *  the pointer is set to the end of the file.
 *
 *      \param  is    the input stream
 *      \param  token a string for which is searched in \em is.
 *                    If the string is empty, the function returns
 *                    immediately
 *      \return none
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
static void skipuntil(istream& is, const string& token)
{
	if (token.length() == 0) {
		return;
	}

	char   c;
	string cmp;

	//
	// read first chunk of characters
	//
	while (cmp.length() < token.length() && is && ! is.eof()) {
		is.get(c);
		cmp += c;
	}

	//
	// loop until token found
	//
	while (!(cmp == token) && is && ! is.eof()) {
		is.get(c);
		cmp.erase(0, 1);
		cmp += c;
	}
}


//===========================================================================
/*!
 *  \brief Reads the next token "token" form the input stream "is".
 *
 *  A token here is a decimal number, that can be introduced by some
 *  ignored characters, listed in \em ignore or a string of characters. <br>
 *  If the characters read from the input stream are not a decimal
 *  number and no prefix of the string is listed in \em tokenlist, then
 *  all characters until the end of the stream are saved as
 *  a single token, even if this string of characters contains a decimal
 *  number or newline characters. <br>
 *  If a prefix of the string is in \em tokenlist, then the reading will stop
 *  after the prefix and this prefix will be taken as token. <br>
 *  If the characters read from the input stream begins with a digit or
 *  a decimal point, then the function automatically will recognize
 *  the end of the decimal number and store it as token, when no
 *  prefix of the number is in \em tokenlist. <br>
 *  Otherwise again only the prefix will be stored as token.
 *
 *      \param  is        the input stream from which the next token is read
 *      \param  token     the next token read from \em is
 *      \param  tokenlist list of tokens, after which the reading from
 *                        \em is will stop and that are stored in \em token
 *                        then
 *      \param  ignore    string of single characters, that are ignored,
 *                        when introducing the next token. By default
 *                        whitespaces, tabulator characters, newline
 *                        characters, form feed characters, vertical
 *                        tabulators and carriage returns are ignored
 *      \return none
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
static void nexttoken(istream& is, string& token,
					  const list< string >& tokenlist,
					  const string& ignore = " \t\r\n\v\f")
{
	char c;          // the current character
	bool sign;       // +/-
	bool point;      // decimal point
	bool digit;      // digit greater than zero
	bool exp;        // exponential part, introduced by "e" after the decimal
	// point
	bool expsign;    // sign in the exponential part
	bool expdigit;   // digit in the exponential part
	bool number;     // digit greater than zero/+/-/decimal point

	// Skip leading "ignore" characters:
	while (is.good() && ! is.eof()) {
		is.get(c);
		if (ignore.find(c) == string::npos) break;
	}

	if (is.good() && ! is.eof()) {
		token    = c;
		sign     = c == '-' || c == '+';
		point    = c == '.';
		digit    = isdigit(c) != 0;
		number   = digit || sign || point;
		exp      = false;
		expsign  = false;
		expdigit = false;

		// Get next character:
		while (is.good() && ! is.eof()) {
			is.get(c);

			// End of number reached?
			if (number) {
				if (! isdigit(c) &&
						c != '.' &&
						c != '-' &&
						c != '+' &&
						c != 'e' &&
						c != 'E') {
					if (digit)
						break;
					else
						number = false;
				}
			}

			// End of number reached?
			if (number) {
				if ((c == '.' && point) ||
						((c == 'e' || c == 'E') && exp) ||
						((c == '-' || c == '+') &&
						 (! exp || expsign || expdigit))) {
					if (digit)
						break;
					else
						number = false;
				}
			}

			// Check for exponential part of number:
			if (number) {
				digit    |= ! exp && isdigit(c);
				exp      |= c == 'e' || c == 'E';
				expsign  |= exp && (c == '-' || c == '+');
				expdigit |= exp && isdigit(c);
			}
			else {
				if (count(tokenlist.begin(), tokenlist.end(), token) > 0)
					break;
			}

			token += c;
		}

		is.putback(c);
	}
}


//===================================================================
/*!
 *  \brief Reads the content for array "arr" from input stream "is".
 *
 *  You have several possibilities of reading data from \em is. <br>
 *  For all possibilities holds, that you can also use the "values"
 *  \f$NaN\f$ (for not a number), \f$+Inf\f$ or \f$Inf\f$
 *  (for positive infinity)
 *  and \f$-Inf\f$ (for negative infinity) in your input data.
 *  It is also possible to
 *  use comments (enclosed in the strings \em beginComment
 *  and \em endComment) at any position of your data and several
 *  separators for the single array values at once (each character
 *  in the \em separator string represents one single separator).<br>
 *
 *  If the preprocessor flag #__ARRAY_NO_GENERIC_IOSTREAM in file
 *  "Array.h" is not defined and you have set \em seek to "true",
 *  you can read data in the following format: <br>
 *
 *  Array<\em type>( \em list \em of \em dimension \em sizes ) <br>
 *  \em list \em of \em all \em array \em elements <br>
 *  Please notice that there are no whitespaces allowed in the
 *  list of dimension sizes and between the parentheses and the
 *  list, otherwise the method will exit with an exception. <br>
 *
 *  \par Example
 *  From \em is the following data is read: <br>
 *  Array<double>(3,2) <br>
 *  1.     2.     3.     4.     5.     6. <br>
 *
 *  So the array \em arr will be a \f$3 \times 2\f$
 *  array of type "double" with content
 *  \f$
 *      \left(\begin{array}{ll}
 *          1. & 2.\\
 *          3. & 4.\\
 *          5. & 6.
 *      \end{array}\right)
 *  \f$
 *
 *  You can also explicitly give the dimension sizes, by using parameter
 *  \em dimensions. In this case you only have to list the single
 *  array values: <br>
 *
 *  \par Example
 *  \code
 *  #include "Array/ArrayIo.h"
 *
 *  void main()
 *  {
 *    vector< unsigned > dim_vec;
 *    Array< double > arr;
 *
 *    dim_vec.push_back(3);
 *    dim_vec.push_back(2);
 *
 *    readArray( arr, cin, true, "", "\n", ";", "\n", " ,", dim_vec );
 *  }
 *  \endcode
 *
 *  Given the code above, the data read from the input stream "cin",
 *  must be something like: <br>
 *
 *  \f$
 *  \mbox{\ }\\ \noindent
 *  \mbox{1.,\ 2.,\ 3.,\ 4.,\ 5.,\ 6.}
 *  \f$
 *
 *  And you will get the same array again. <br>
 *  If you set the first value of \em dimension to zero, this functions
 *  as wildcard. After the reading of the data, the function will
 *  determine the final size of the array by itself. Guess, we will
 *  replace the first "push_back"-instruction in the code above
 *  by "dim_vec.push_back(0)". Then given
 *  \f$\mbox{1.,\ 2.,\ 3.,\ 4.,\ 5.,\ 6.}\f$ as input again,
 *  we will get the same array as in the two previous examples.<br>
 *
 *  If you use neither the specification of the array type and dimensions
 *  in the input data nor the specification of the dimensions in the
 *  function call, all input is interpreted as unformatted. <br>
 *  You must then use the \em beginRecord and \em endRecord strings
 *  to mark the several dimensions of the array data. If you use the
 *  default strings (no \em beginRecord string and a newline as
 *  \em endRecord string), it is obvious that you can only read in
 *  data for one-dimensional arrays (one line of values ended by a newline
 *  character) or 2-dimensional arrays (each line contains the values
 *  for one array row). <br>
 *  If you want to read in data for arrays with 3 or more dimensions,
 *  then you have to use a nonempty \em beginRecord string. <br>
 *  Guess, we have set the \em beginRecord string to "\f$(\f$" and the
 *  \em endRecord string to "\f$)\f$". To read in data for a 3-dimensional
 *  array, that will match the one defined in the following code
 *
 *  \par Example
 *  \code
 *  Array< double > test( 2, 2, 4 );
 *
 *  test( 0, 0, 0 ) = 1.;
 *  test( 0, 0, 1 ) = 2.;
 *  test( 0, 0, 2 ) = 3.;
 *  test( 0, 0, 3 ) = 4.;
 *  test( 0, 1, 0 ) = 5.;
 *  test( 0, 1, 1 ) = 6.;
 *  test( 0, 1, 2 ) = 7.;
 *  test( 0, 1, 3 ) = 8.;
 *  test( 1, 0, 0 ) = 9.;
 *  test( 1, 0, 1 ) = 10.;
 *  test( 1, 0, 2 ) = 11.;
 *  test( 1, 0, 3 ) = 12.;
 *  test( 1, 1, 0 ) = 13.;
 *  test( 1, 1, 1 ) = 14.;
 *  test( 1, 1, 2 ) = 15.;
 *  test( 1, 1, 3 ) = 16.;
 *  \endcode
 *
 *  the data read in must have the following format: <br>
 *
 *  \f$
 *  \mbox{\ }\\ \noindent
 *  (((1.\mbox{\ }2.\mbox{\ }3.\mbox{\ }4.)(5.\mbox{\ }6.\mbox{\ }7.\mbox{\ }8.))((9.\mbox{\ }10.\mbox{\ }11.\mbox{\ }12.)(13.\mbox{\ }14.\mbox{\ }15.\mbox{\ }16.)))
 *  \f$
 *
 *  The function is error tolerant, i.e. if you use a separator that is not
 *  the whitespace character, then two separators without any value
 *  between them will cause the function to insert the infinity value.
 *  The same occurs if a nonempty \em beginRecord string is directly
 *  followed by a separator (not the whitespace character). <br>
 *  So using the \em beginRecord and the \em endRecord strings from
 *  above and the comma as separator, the input "\f$(,,,)\f$"
 *  will result in an one-dimensional array with the values
 *  "\f$inf\mbox{\ }inf\mbox{\ }inf\mbox{\ }inf\f$". <br>
 *  Different numbers of values in the same dimension and \em beginRecord
 *  strings not ended by \em endRecord strings will be recognized
 *  and bring the function to exit with a corresponding error message.
 *
 *  \param arr             the array whose content will be read from \em is
 *  \param seek            must be set to "true" if you want to
 *                         read data with the prefix
 *                         "Array< type >(dimensions)"
 *  \param is              the input stream from which the arrays content is
 *                         read
 *  \param beginRecord     the string that marks the beginning of a new
 *                         record (= subdimension of the array). This string
 *                         is empty by default (this means you can only
 *                         read in one- or 2-dimensional arrays when
 *                         using unformatted input data)
 *  \param endRecord       the string that marks the end
 *                         of the data of each subdimension, the default
 *                         is the newline character
 *  \param beginComment    marks the beginning of a comment, the default is
 *                         the semicolon
 *  \param endComment      marks the end of a comment, the default is a
 *                         newline
 *  \param separator       contains all single characters that will separate
 *                         single array values. By default the whitspace
 *                         character and the comma can be used as
 *                         separators
 *  \param dimensions      can be used to specify the structure of the
 *                         array data. Each position indicates the
 *                         size of one array dimension. By default this
 *                         vector is set empty and not used
 *  \return "-2" - the parameter strings are not valid,
 *          "-1" - the data read has a wrong format,
 *          "0"  - data for the array was successfully read
 *  \throw check_exception the type of the exception is "range check error"
 *         and indicates that the given dimension sizes in \em dimensions
 *         or the dimensions implicitly given by the data prefix
 *         "Array<...>(...)" do not correspond to the number of
 *         array values read from \em is
 *
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      2002-03-20, ra: method readArray() didn't work for
 *      seek = true, when using explicit specification
 *      of array type and dimensions - fixed!
 *
 *  \par Status
 *      stable
 *
 */
int readArray(Array< double >& arr, istream& is, bool seek,
			  const string beginRecord,
			  const string endRecord,
			  const string beginComment,
			  const string endComment,
			  const string separator,
			  const vector< unsigned > dimensions)
{
#ifdef _WIN32
	const double NotANumber  = numeric_limits< double >::quiet_NaN();
	const double Infinity    = numeric_limits< double >::infinity();
	const double NegInfinity = -Infinity;
#elif __SUNOS__
	const double NotANumber  = sqrt(-1.);
	const double Infinity    = 1. / 0.;
	const double NegInfinity = -1. / 0.;
#else
	// <awd, 2000-03-13> assume ix86 architecture where HUGE_VAL is
	// +infinity (thus, getting rid of annoying warnings about
	// division by zero)
	const double NotANumber  = sqrt(-1.);
	const double Infinity    = HUGE_VAL;
	const double NegInfinity = -HUGE_VAL;
#endif

	string token;
	string ignore;
	string delim;
	list< string > tokens;

	//=======================================================================
	//
	// token must not contain white space characters
	//
	if (beginRecord .find(' ') != string::npos ||
			endRecord   .find(' ') != string::npos ||
			beginComment.find(' ') != string::npos ||
			endComment  .find(' ') != string::npos) {
		cerr << "record and comment tokens must not contain white space chars\n";
		return -2;
	}

	//=======================================================================
	//
	// if no dimensions are defined we need at least the end of record token
	//
	if (dimensions.size() == 0 && endRecord.length() == 0) {
		cerr << "end of record token is empty\n";
		return -2;
	}

	//=======================================================================
	//
	// if begin of comment is defined we need the end of comment token
	//
	if (beginComment.length() > 0 && endComment.length() == 0) {
		cerr << "end of comment token is empty\n";
		return -2;
	}

	//=======================================================================
	//
	// ignore the following characters
	// only if they don't occur in any token)
	// (except end of token which is handled separately)
	//
	ignore = ' ';
	token = beginRecord + endRecord + beginComment + separator;
	if (token.find('\t') == string::npos) ignore += '\t';
	if (token.find('\r') == string::npos) ignore += '\r';
	if (token.find('\n') == string::npos) ignore += '\n';
	if (token.find('\v') == string::npos) ignore += '\v';
	if (token.find('\f') == string::npos) ignore += '\f';

	//=======================================================================
	//
	// build list of used tokens (leave end of comment token out)
	//
	if (beginRecord .length() > 0) tokens.push_back(beginRecord);
	if (endRecord   .length() > 0) tokens.push_back(endRecord);
	if (beginComment.length() > 0) tokens.push_back(beginComment);
	for (unsigned i = 0; i < separator.size(); i++)
		if (separator[ i ] != ' ')
			tokens.push_back(string(1, separator[ i ]));

	//=======================================================================
	//
	// add special tokens for NaN and Inf
	//
	if (count(tokens.begin(), tokens.end(), "NaN") +
			count(tokens.begin(), tokens.end(), "Inf") +
			count(tokens.begin(), tokens.end(), "+Inf") +
			count(tokens.begin(), tokens.end(), "-Inf") > 0) {
		cerr << "tokens 'NaN' and 'Inf' are not allowed\n";
		return -2;
	}
	tokens.push_back("NaN");
	tokens.push_back("Inf");
	tokens.push_back("+Inf");
	tokens.push_back("-Inf");

	//=======================================================================
	//
	// if seeking is allowed read the first line of the stream
	//
	if (seek) {
		char prefix[100];
		streampos pos = is.tellg();
		is.getline(prefix, sizeof(prefix));
		token = (string) prefix;
		is.seekg(pos, ios::beg);
	}
	else
		token = "";

	//=======================================================================
	//
	// formatted input ? line begins with "Array< T >( ... )"
	//
	if (token.find("Array<") == 0) {
		is >> arr;
	}
	//=======================================================================
	//
	// dimension vector defined ?
	//
	else if (dimensions.size() > 0) {
		//===================================================================
		//
		// dimension totally fixed (0 serves as wildcard)
		//
		if (dimensions[ 0 ] > 0) {
			arr.resize(dimensions);

			for (unsigned i = 0;
					i < arr.nelem() && is.good() && ! is.eof();
					++i) {
				//
				// dimensions are fixed now,
				// ignore all begin/end of record and separator tokens
				//
				do {
					nexttoken(is, token, tokens, ignore);

					//
					// comment found ?
					//
					if (token == beginComment) {
						skipuntil(is, endComment);
						continue;
					}
				}
				while ((token == beginRecord || token == endRecord ||
						separator.find(token) != string::npos) &&
						is.good() && ! is.eof());

				//
				// handle 'not a number's and infinite values
				//
				if (token == "NaN")
					arr.elem(i) = NotANumber;
				else if (token == "Inf" || token == "+Inf")
					arr.elem(i) = Infinity;
				else if (token == "-Inf")
					arr.elem(i) = NegInfinity;
				else
					arr.elem(i) = atof(token.c_str());
			}
		}
		//===================================================================
		//
		// dimension only partially fixed (dimension 0 undefined)
		//
		else {
			unsigned i, lastdim;
			list< double > value;
			list< double >::iterator li;
			vector< unsigned > dim;

			while (is.good() && ! is.eof()) {
				//
				// ignore all begin/end of record and separator tokens
				//
				do {
					nexttoken(is, token, tokens, ignore);

					//
					// comment found ?
					//
					if (token == beginComment) {
						skipuntil(is, endComment);
						continue;
					}
				}
				while ((token == beginRecord || token == endRecord ||
						separator.find(token) != string::npos) &&
						is.good() && ! is.eof());

				if (is.good() && ! is.eof()) {
					//
					// handle 'not a number's and infinite values
					//
					if (token == "NaN")
						value.push_back(NotANumber);
					else if (token == "Inf" || token == "+Inf")
						value.push_back(Infinity);
					else if (token == "-Inf")
						value.push_back(NegInfinity);
					else
						value.push_back(atof(token.c_str()));
				}
			}

			lastdim = value.size();
			for (i = 1; i < dimensions.size(); ++i)
				lastdim /= dimensions[ i ];

			dim = dimensions;
			dim[ 0 ] = lastdim;

			//
			// copy found values to the Array
			//
			arr.resize(dim);
			for (i = 0, li = value.begin();
					i < arr.nelem() && li != value.end();
					++i, ++li)
				arr.elem(i) = *li;
		}
	}
	//=======================================================================
	//
	// else unformatted input
	//
	else {
		bool lastsep;
		bool expectval;
		bool blanksep;
		unsigned i, depth, end, lastdim;
		double missing = HUGE_VAL;
		list< double > value;
		list< double >::iterator li;
		vector< unsigned > dim;
		// contains the number of values per dimension:
		vector< unsigned > stack;

		lastsep   = true;
		blanksep  = separator.length() == 0 ||
					separator.find(' ') != string::npos;
		expectval = blanksep || beginRecord.length() == 0;
		// no. of begin records that were not ended yet:
		depth     = 0;
		// end of stack:
		end       = 0;
		stack.push_back(0);

		while (is.good() && ! is.eof()) {
			nexttoken(is, token, tokens, ignore);

			//
			// comment found ?
			//
			if (token == beginComment) {
				skipuntil(is, endComment);
				continue;
			}

			if (is.good() && ! is.eof()) {
				// begin of new record
				if (beginRecord.length() > 0 && token == beginRecord) {
					depth++;
					lastsep = expectval = true;
					// end of current record
				}
				else if (token == endRecord) {
					if (lastsep && stack[ 0 ] > 0) {
						value.push_back(missing);
						stack[ end = 0 ]++;
					}
					lastsep = false;
					if (beginRecord.length() > 0) {
						// end of record with missing begin of record?
						if (depth == 0) {
							cerr << "unbalanced end of record token\n";
							return -1;
						}
						// end of current record
						depth--;
					}
					// new dimension found?
					if (end >= dim.size())
						dim.push_back(stack[ end ]);
					else {
						if (stack[ end ] != dim[ end ]) {
							cerr << "dimensions do not match\n";
							return -1;
						}
					}
					stack[ end++ ] = 0;
					if (end >= stack.size())
						stack.push_back(0);
					stack[ end ]++;
					expectval = blanksep;
					// separator found
				}
				else if (separator.find(token) != string::npos) {
					if (lastsep) {
						value.push_back(missing);
						stack[ end = 0 ]++;
					}
					lastsep = expectval = true;
					// array value
				}
				else {
					if (! expectval) {
						cerr << "unexpected value\n";
						return -1;
					}

					//
					// handle 'not a number's and infinite values
					//
					if (token == "NaN")
						value.push_back(NotANumber);
					else if (token == "Inf" || token == "+Inf")
						value.push_back(Infinity);
					else if (token == "-Inf")
						value.push_back(NegInfinity);
					else
						value.push_back(atof(token.c_str()));

					stack[ end = 0 ]++;
					lastsep = false;
					expectval = blanksep;
				}
			}
		}

		//
		// append missing sizes
		//
		if (stack.size() > dim.size())
			dim.insert(dim.end(),
					   stack.begin() + dim.size(),
					   stack.end());

		//
		// check if dimensions are consistent
		//
		if (beginRecord.length() > 0 && depth > 0) {
			cerr << "missing end of record token\n";
			return -1;
		}
		lastdim = value.size();
		for (i = 0; i < dim.size() - 1; ++i)
			lastdim /= dim[ i ];

		if (dim[ dim.size() - 1 ] != lastdim) {
			//
			// if end of record token is newline append missing dimensions
			// silently
			//
			if (endRecord == "\n")
				dim[ dim.size() - 1 ] = lastdim;
			else {
				cerr << "missing end of record token\n";
				return -1;
			}
		}

		//
		// copy found values to the Array
		//
		reverse(dim.begin(), dim.end());
		arr.resize(dim);
		if (value.size() != arr.nelem()) {
			cerr << "Array dimensions are corrupted\n";
			return -1;
		}
		for (i = 0, li = value.begin(); li != value.end(); ++i, ++li)
			arr.elem(i) = *li;
	}
	//=======================================================================

	if (! is.eof() && ! is.good()) {
		cerr << "invalid stream operation\n";
		return -1;
	}

	return 0;
}


