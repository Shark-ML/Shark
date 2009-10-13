//===========================================================================
/*!
 *  \file FileUtil.h
 *
 *  \brief Offers various input- and output-functions especially
 *         used for dealing with configuration files.
 *
 *  Most of the functions in this file are created for the writing
 *  and reading configuration files, i.e. files where the names of
 *  variables (named here in this file "tokens" or "token names")
 *  are listed and each variable name is followed by a value
 *  (named "token value" here), that can be used to initialize this value.
 *  Using this functions here you can automatize the initial
 *  assignment of variables
 *
 *  \author  M. Kreutz, R. Alberts
 *  \date    1998-10-06
 *
 *  \par Copyright (c) 1995-2002:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
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
//===========================================================================


#ifndef __FILEUTIL_H
#define __FILEUTIL_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <SharkDefs.h>


namespace FileUtil
{


//===========================================================================
/*!
 *  \brief The get pointer is set to the beginning of stream "is".
 *
 *      \param  is The input stream.
 *      \return None.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
inline void rewind(std::istream& is)
{
	is.seekg(0, std::ios::beg);
}

//===========================================================================
/*!
 *  \brief The put pointer is set to the beginning of stream "os".
 *
 *      \param  os The output stream.
 *      \return None.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
inline void rewind(std::ostream& os)
{
	os.seekp(0, std::ios::beg);
}

//! Returns the number of characters in input stream "is".
unsigned long filesize(std::istream& is);

//! Returns the number of characters in output stream "os".
unsigned long filesize(std::ostream& os);

//! Returns the number of characters in file "name".
unsigned long filesize(const std::string& name);

//! The whole content of input stream "is" is stored in "buf".
unsigned long readfile(std::istream& is, std::string& buf);

//! The whole content of file "name" is stored in "buf".
unsigned long readfile(const std::string& name, std::string& buf);

//! Reads from input stream "is" until "token" is found.
void skipuntil(std::istream& is, const std::string& token);

//! Reads the first token from input stream "is".
char gettoken(std::istream&      is,
			  std::string&       token,
			  const std::string& delim = " \t\r\n\v\f");


//===========================================================================
/*!
 *  \brief Reads the single value of a token from input stream "is".
 *         Very simple version.
 *
 *  This is the opposite function of
 *  #printTo(std::ostream& os, const std::string& token, const T& t).
 *  and is used for files that contain initialization values
 *  for variables. The name of the variable is used as search token
 *  and the value following these token is stored and can be used
 *  as new value for the variable with name \em token.  <br>
 *  In contrast to the other "scanFrom" functions you can not
 *  enclose the token value in quotation marks or inverted commas
 *  for a better identification, nor can the token value include
 *  escape sequences. The disadvantage of the other "scanFrom"
 *  functions is, that the token values are always stored in strings,
 *  independend from their real type. Here you can use a buffer
 *  \em t with the right type. <br>
 *  Use this function with caution, because a string read from the input
 *  stream is identified as the searched token when this searched token
 *  is a prefix of this string, e.g. if you are searching for a token "a"
 *  and one line of the input stream begins with "ab". <br>
 *  This is especially dangerous when using short token names. <br>
 *  If you are using token names of this kind or are not sure, that
 *  the scenario above won't occur, use function
 *  #scanFrom_strict(std::istream& is, const std::string& token, T& t, bool rew = false) instead.
 *
 *      \param  is    The input stream the token value is read from.
 *      \param  token The token for which the value is read.
 *      \param  t     Used to store the token value found.
 *      \param  rew   If set to "true", the search will start at
 *                    the beginning of the stream; if set to "false"
 *                    (default value) the search will start at the
 *                    current position of the get pointer.
 *      \return "true", if no error occured while reading from
 *              \em is, "false" otherwise.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
inline bool scanFrom(std::istream& is, const std::string& token,
					 T& t, bool rew = false)
{


	is.clear();    // no one knows ...
	if (rew) rewind(is);
	skipuntil(is, token);
	if (is)
		is >> t;
	else
		is.clear();

	return is.good();
}



//===========================================================================
/*!
 *  \brief Writes a token and its single value "t" to output stream "os".
 *         Very simple version.
 *
 *  If you want to create files that can be used to initialize
 *  variables, you can use this function. The value \em t for a variable
 *  with name \em token will be written to output stream \em os.
 *  The data is written in a formatted manner, i.e.
 *  the width of a token-value line is always 40 characters and
 *  the token and its value are separated by "_" characters. <br>
 *  Please notice that the formatted output is ignored by older
 *  compilers. <br>
 *  In contrast to the other "printTo" functions the token value can not
 *  include escape sequences. The disadvantage of the other "printTo"
 *  functions is, that the token values must always be stored in strings,
 *  independend from their real type. Here you can use a copy of
 *  the original value without casting it to string before. <br>
 *  This is the opposite function of
 *  #scanFrom(std::istream& is, const std::string& token, T& t, bool rew = false ).
 *
 *      \param  os    The input stream the token and its value are
 *                    written to.
 *      \param  token The token that will be written.
 *      \param  t     The token value that will be written.
 *      \return "true", if no error occured while writing to
 *              \em os, "false" otherwise.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
inline bool printTo(std::ostream& os, const std::string& token, const T& t)
{
	os << token << std::setw(40) << std::setfill('_') << t << std::endl;
	return os.good();
}

//! Reads the single value of a token from input stream "is".
template < >
bool scanFrom(std::istream& is, const std::string& token,
			  std::string& t, bool rew);


//! Writes a token and its single value "t" to output stream "os".
template < >
bool printTo(std::ostream& os, const std::string& token,
			 const std::string& t);

//! Reads one or several values of a token.
template < >
bool scanFrom(std::istream& is, const std::string& token,
			  std::vector< std::string >& t, bool rew);

//! Writes a token and its value(s) "t" to output stream "os".
template < >
bool printTo(std::ostream& os, const std::string& token,
			 const std::vector< std::string >& t);

//! Checks, whether the given character "ch" introduces a
//! newline or tabulator.
bool special_character(std::istream& is, char *ch);

//! Skips one line while reading from the input stream "is".
bool skipLine(std::istream& is, char *ch);


//===========================================================================
/*!
 *  \brief Used for the functions "io" and "io_strict" to define the kind
 *         of action, that should be performed.
 *
 * <ul>
 * <li>\em SetDefault - The given token value will be initialized by the given
 * default value.</li>
 * <li>\em ScanFrom - The value for the given token will be read from the
 * given input stream</li>
 * <li>\em PrintTo - The given token and its value are written to the
 * given output stream</li>
 * </ul>
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
enum iotype
{
	SetDefault, ScanFrom, PrintTo
};


//===========================================================================
/*!
 *  \brief An easy to use interface to perform the three actions needed
 *         for a token.
 *
 *  You can use this function to specify all information for a token
 *  once and then initialize it with a default value or read a new
 *  value for it from an input stream or write the token and its
 *  already given value to an output stream. This can be done by using
 *  different values for the action indicator \em type. <br>
 *  Use this function with caution, if you are using it for reading
 *  the value of a token from input stream \em is, because a string read
 *  from the input
 *  stream is identified as the searched token when this searched token
 *  is a prefix of this string, e.g. if you are searching for a token "a"
 *  and one line of the input stream begins with "ab". <br>
 *  This is especially dangerous when using short token names. <br>
 *  If you are using token names of this kind or are not sure, that
 *  the scenario above won't occur, use function
 *  #io_strict instead.
 *
 *      \param  is     The input stream used, when reading a token value.
 *      \param  os     The output stream used, when writing the token and
 *                     its value.
 *      \param  token  The token used for initializing/reading/writing.
 *      \param  val    Used to store the current value of the token.
 *      \param  defval The default value that is used for initialization.
 *      \param  type   The action indicator. See #iotype definition
 *                     for the different values and their meanings.
 *      \return None.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
inline void io(std::istream& is, std::ostream& os, const std::string& token,
			   T& val, const T& defval, iotype type)
{
	switch (type)
	{
	case SetDefault :
		val = defval;
		break;
	case ScanFrom :
		scanFrom(is, token, val, true);
		break;
	case PrintTo :
		printTo(os, token, val);
		break;
	}
}



//===========================================================================
/*!
 *  \brief An easy to use interface to perform the three actions needed
 *         for a token. A very strict version, used for a special format.
 *
 *  You can use this function to specify all information for a token
 *  once and then initialize it with a default value or read a new
 *  value for it from an input stream or write the token and its
 *  already given value to an output stream. This can be done by using
 *  different values for the action indicator \em type.  <br>
 *  This function here is used to prevent
 *  a disadvantage of #io, when reading a token value from an input
 *  stream. There a string read from the input stream is
 *  identified as the searched token when this searched token is a prefix of
 *  this string, e.g.
 *  if you are searching for a token "a" and one line
 *  of the input stream begins with "ab".  <br>
 *  This is especially dangerous when using short token names.
 *
 *      \param  is     The input stream used, when reading a token value.
 *      \param  os     The output stream used, when writing the token and
 *                     its value.
 *      \param  token  The token used for initializing/reading/writing.
 *      \param  val    Used to store the current value of the token.
 *      \param  defval The default value that is used for initialization.
 *      \param  type   The action indicator. See #iotype definition
 *                     for the different values and their meanings.
 *      \return None.
 *
 *  \author  R. Alberts
 *  \date    1999
 *
 *  \par Changes
 *      2002-01-03, ra <br>
 *      Renamed to from "io2" to "io_strict" for unification,
 *      new function "printTo_strict" added.
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
inline void io_strict(std::istream& is, std::ostream& os,
					  const std::string& token,
					  T& val, const T& defval, iotype type)
{
	switch (type)
	{
	case SetDefault :
		val = defval;
		break;
	case ScanFrom :
		scanFrom_strict(is, token, val, true);
		break;
	case PrintTo :
		printTo_strict(os, token, val);
		break;
	}
}


//===========================================================================
/*!
 *  \brief Reads the single value of a token from input stream "is".
 *         A very strict version, used for a special format.
 *
 *  This function is used for files that contain initialization values
 *  for variables. The name of the variable is used as search token
 *  and the value following these token is stored and can be used
 *  as new value for the variable with name \em token. <br>
 *  In contrast to the other "scanFrom" functions the files used
 *  for this function must have a very strict format: <br>
 *  Each line contains the token name AND its corresponding value,
 *  it is not allowed to store the token name in one line and its
 *  value in the next line. Furthermore, each token name stands at
 *  the beginning of a line, no leading whitespaces or tabulators
 *  are allowed. The first character of a token name must be a
 *  letter or an underscore, all further characters must be a
 *  letter, a digit or an underscore. The token name and the token
 *  value are separated by whitespaces and/or tabulators, no other
 *  delimiters are allowed. The token name read from the stream
 *  must match the token name given exactly.
 *
 *      \param  is     The input stream the token value is read from.
 *      \param  token  The token for which the value is read from \em is.
 *      \param  t      The value of the token.
 *      \return "true", if the token and its value are found;
 *              "false", if the token or its value are not found or
 *              if an error occured while reading from \em is.
 *
 *  \author  R. Alberts
 *  \date    1999
 *
 *  \par Changes
 *      2002-01-03, ra: <br>
 *      Renamed from "scanLine" to "scanLine_strict" for unification.
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
bool scanLine_strict(std::istream& is, const std::string& token,
					 T& t)
{
	std::string  var_name;
	char         ch;


	// Get first character from stream but put it back immediately:
	if (!is.eof()) is.get(ch); else return false;
	is.putback(ch);

	// Each valid token must begin with a letter or an underscore:
	if (isalpha(ch) || ch == '_')
	{
		// Each token name continues with a letter, a digit or an
		// underscore:
		do
		{
			if (!is.eof()) is.get(ch); else return false;
			special_character(is, &ch);

			if (!isspace(ch) && ch != '\t' &&
					(isalpha(ch) || isdigit(ch) || ch == '_'))
			{
				var_name += ch;
			}
		}
		while (!is.eof() && !isspace(ch) && ch != '\t' && ch != '\n');

		// Token name and token value are separated by whitespaces or
		// tabulators. Otherwise there's something wrong:
		if (is.eof() || ch == '\n') return false;

		// Skip all further whitespaces and tabulators that separate
		// Token name and value:
		do
		{
			if (!is.eof()) is.get(ch); else return false;
			special_character(is, &ch);
		}
		while (!is.eof() && (isspace(ch) || ch == '\t'));

		// Token name and token value are separated by whitespaces or
		// tabulators. Otherwise there's something wrong:
		if (is.eof() || ch == '\n') return false; else is.putback(ch);

		// Are the token name read and the token name searched for equal?
		// Yes => Read token value. No => Maybe the token and its value
		// can be found in one of the lines following, so set the get pointer
		// to the beginning of the next line and exit:
		if (var_name == token)
		{
			is >> t;
			return true;
		}
	}

	// No valid token, so set get pointer to the beginning of
	// the next line and return:
	skipLine(is, &ch);
	return false;
}



//===========================================================================
/*!
 *  \brief Reads the single value of a token from input stream "is".
 *         A very strict version, used for a special format. This
 *         function is used to offer the same parameter format
 *         than #scanFrom( std::istream& is, const std::string& token, T& t, bool rew = false ).
 *
 *  See function
 * #scanFrom_strict(std::istream& is, const std::string& token, T& t) for a
 * description of the functioning. This function here is used to prevent
 * a disadvantage of
 * #scanFrom( std::istream& is, const std::string& token, T& t, bool rew = false ), where a string read from the input stream is identified as the
 * searched token when this searched token is a prefix of this string, e.g.
 * if you are searching for a token "a" and one line
 * of the input stream begins with "ab". <br>
 * This is especially dangerous when using short token names. <br>
 * This function here is the opposite function of
 * #printTo_strict.
 *
 *      \param  is    The input stream the token value is read from.
 *      \param  token The token for which the value is read.
 *      \param  t     Used to store the token value found.
 *      \param  rew   If set to "true", the search will start at
 *                    the beginning of the stream; if set to "false"
 *                    (default value) the search will start at the
 *                    current position of the get pointer.
 *      \return "true", if no error occured while reading from
 *              \em is, "false" otherwise.
 *
 *  \author  R. Alberts
 *  \date    1999
 *
 *  \par Changes
 *      Renamed from "scanFrom2" to "scanFrom_strict" for unification.
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
inline bool scanFrom_strict(std::istream& is, const std::string& token,
							T& t, bool rew = false)
{
	bool status(true);


	is.clear();    // No one knows ...
	if (rew) rewind(is);
	do
	{
		status = scanLine_strict(is, token, t);
	}
	while (status == false && is);

	return status;
}


//===========================================================================
/*!
 *  \brief Writes a token and its single value "t" to output stream "os".
 *         A very strict version, used for a special format.
 *
 *  If you want to create files that can be used to initialize
 *  variables, you can use this function. The value \em t for a variable
 *  with name \em token will be written to output stream \em os. <br>
 *  Before writing the token name and its value, the token name
 *  is checked for the right format: <br>
 *  The first character of a token name must be a
 *  letter or an underscore, all further characters must be a
 *  letter, a digit or an underscore. If this format is not
 *  satisfied, the function will exit with an error message. <br>
 *  This function is the opposite function of
 *  #scanFrom_strict(std::istream&, const std::string&, T&, bool).
 *
 *      \param  os    The input stream the token and its value are
 *                    written to.
 *      \param  token The token that will be written.
 *      \param  t     The token value that will be written.
 *      \return "true", if no error occured while writing to
 *              \em os, "false" otherwise.
 *
 *  \author  R. Alberts
 *  \date    2002-01-03
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
inline bool printTo_strict(std::ostream& os,
						   const std::string& token,
						   const T& t)
{
	unsigned i;


	// Check format of token name:
	if (!isalpha(token[ 0 ]) && token[ 0 ] != '_')
	{
		std::cerr << "Error! Token \"" << token << "\" is not valid, "
		<< "because it must begin with a letter or an underscore!"
		<< std::endl;
		return false;
	}
	for (i = 1; i < token.size(); i++)
	{
		if (!isalpha(token[ i ]) &&
				!isdigit(token[ i ]) &&
				token[ i ] != '_')
		{
			std::cerr << "Error! Token \"" << token << "\" is not valid, "
			<< "because the characters after the first one must "
			<< "be a letter, a digit or an underscore!"
			<< std::endl;
			return false;
		}
	}

	// Token name has the right format, so write it and its value
	// to the output stream:
	os << token << "\t\t" << t << std::endl;
	return os.good();
}


}


#endif

