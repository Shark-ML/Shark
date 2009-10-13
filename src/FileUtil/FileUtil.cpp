//===========================================================================
/*!
 *  \file FileUtil.cpp
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
 *  \date    1998-06-10
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
 *  \par Project:
 *      FileUtil
 * 
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of FileUtil. This library is free software;
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


#include <SharkDefs.h>
#include <FileUtil/FileUtil.h>


using namespace std;

//===========================================================================


namespace FileUtil
{


//===========================================================================
/*!
 *  \brief Checks, whether the given character "ch" introduces a
 *         newline or tabulator.
 *
 *  Escape sequences like the tabulator or the newline consist
 *  of two characters. The first one is the escape character "\\" that
 *  marks the beginning of an escape sequence, the second one
 *  is the type of sequence ("t" for tabulator, "n" for newline).
 *  This function takes the given character \em ch, checks,
 *  whether this character is an escape character and when this is true,
 *  reads in the next character from the given input stream \em is.
 *  If the next character is a "t" or "n", then "true" is returned.
 *
 *      \param  is The input stream from which the first character
 *                 was taken and the next character can be read.
 *      \param  ch The first character that was read from the
 *                 input stream and is checked for the escape
 *                 character.
 *      \return "true", when \em ch introduces a newline or
 *              tabulator, "false" otherwise.
 *
 *  \author  R. Alberts
 *  \date    1999
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
bool special_character(std::istream& is, char *ch)
{
	if (*ch == '\\')
	{
		if (!is.eof()) is.get(*ch); else return false;
		switch (*ch)
		{
		case 't' : *ch = '\t'; return true; break;
		case 'n' : *ch = '\n'; return true; break;
		default:   return false;
		}
	}

	return false;
}


//===========================================================================
/*!
 *  \brief Skips one line while reading from the input stream "is".
 *
 *  This function reads from the given input stream \em is, starting
 *  at the current position of the get pointer and searches for the
 *  next newline character if available. When a newline exists, the
 *  whole line was read in and so the position of the get pointer
 *  set to the beginning of the next line.
 *
 *      \param  is The input stream from which is read.
 *      \param  ch At the end of the function you can find
 *                 the last character that was read here
 *                 (a newline in normal case).
 *      \return "true", when a newline was found, "false" when
 *              the end of file was reached before or a character
 *              couldn't be read.
 *
 *  \author  R. Alberts
 *  \date    1999
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
bool skipLine(std::istream& is, char *ch)
{
	do
	{
		if (!is.eof()) is.get(*ch); else return false;
		special_character(is, ch);
	}
	while (!is.eof() && *ch != '\n');

	if (!is.eof()) return true;
	else return false;
}


//===========================================================================
/*!
 *  \brief Returns the number of characters in input stream "is".
 *
 *  Given the input stream "is", the number of characters in this
 *  file is returned. The position of the get pointer is not
 *  finally changed by this function.
 *
 *      \param  is The input stream to count the characters in.
 *      \return The number of characters in the stream.
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
unsigned long filesize(std::istream& is)
{
#ifdef __GNUC__
	size_t pos, end;
#elif defined( __CRAY_T3E__ )
	streampos pos, end;
#else
	std::istream::pos_type pos, end;
#endif
	pos = is.tellg();
	is.seekg(0, ios::end);
	end = is.tellg();
	is.seekg(pos, ios::beg);
	return end;
}


//===========================================================================
/*!
 *  \brief Returns the number of characters in output stream "os".
 *
 *  Given the output stream "os", the number of characters in this
 *  file is returned. The position of the put pointer is not
 *  finally changed by this function.
 *
 *      \param  os The output stream to count the characters in.
 *      \return The number of characters in the stream.
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
unsigned long filesize(std::ostream& os)
{
#ifdef __GNUC__
	size_t pos, end;
#elif defined( __CRAY_T3E__ )
	streampos pos, end;
#else
	std::istream::pos_type pos, end;
#endif
	pos = os.tellp();
	os.seekp(0, ios::end);
	end = os.tellp();
	os.seekp(pos, ios::beg);
	return end;
}


//===========================================================================
/*!
 *  \brief Returns the number of characters in file "name".
 *
 *      \param  name Name of the file.
 *      \return The number of characters in the file or zero, when
 *              the file can't be opened.
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
unsigned long filesize(const std::string& name)
{
	std::ifstream is(name.c_str());

	if (is)
		return filesize(is);
	else
		return 0;
}


//===========================================================================
/*!
 *  \brief The whole content of input stream "is" is stored in "buf".
 *
 *  All characters in input stream \em is are read and stored
 *  in buffer \em buf. The current position of the get pointer
 *  is not finally changed.
 *
 *      \param  is Input stream from which the content is read.
 *      \param  buf Buffer where the content of \em is is stored in.
 *      \return The number of characters read from \em is.
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
unsigned long readfile(std::istream& is, std::string& buf)
{
#ifdef __GNUC__
	size_t pos, end;
#elif defined( __CRAY_T3E__ )
	streampos pos, end;
#else
	std::istream::pos_type pos, end;
#endif
	pos = is.tellg();
	is.seekg(0, ios::end);
	end = is.tellg();
	char *b = new char[ static_cast< unsigned >(end) + 1 ];
	is.seekg(0, ios::beg);
	is.read(b, end);
	b[ end ] = '\0';
	buf = b;
	delete[ ] b;
	is.seekg(pos, ios::beg);
	return end;
}


//===========================================================================
/*!
 *  \brief The whole content of file "name" is stored in "buf".
 *
 *  All characters in the file named \em name are read and stored
 *  in buffer \em buf. The current position of the get pointer
 *  is not finally changed.
 *
 *      \param  name Name of file from which the content is read.
 *      \param  buf Buffer where the content of file \em name is stored in.
 *                  If \em name doesn't exist, the buffer is empty.
 *      \return The number of characters read from \em name,
 *              or "0", when the file doesn't exist.
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
unsigned long readfile(const std::string& name, std::string& buf)
{
	std::ifstream is(name.c_str());

	if (is)
		return readfile(is, buf);
	else
	{
		buf = "";
		return 0;
	}
}


//===========================================================================
/*!
 *  \brief Reads from input stream "is" until "token" is found.
 *
 *  Starting at the current position of the get pointer, it is searched
 *  for the next occurence of \em token. If the token is found,
 *  the get pointer is set to the first position after the token, otherwise
 *  the pointer is set to the end of the file.
 *
 *      \param  is    The input stream.
 *      \param  token A string for which is searched in \em is.
 *                    If the string is empty, the function returns
 *                    immediately.
 *      \return None.
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
void skipuntil(std::istream& is, const std::string& token)
{
	if (token.length() == 0)
		return;

	char   c;
	std::string cmp;

	//
	// read first chunk of characters
	//
	while (cmp.length() < token.length() && is && ! is.eof())
	{
		is.get(c);
		cmp += c;
	}

	//
	// loop until token found
	//
	while (!(cmp == token) && is && ! is.eof())
	{
		is.get(c);
		cmp.erase(0, 1);
		cmp += c;
	}
}


//===========================================================================
/*!
 *  \brief Reads the first token from input stream "is".
 *
 *  Tokens are separated by delimiters. The string \em delim is
 *  used to enumerate all characters used as delimiters.
 *  The delimiters are used then to identify the first token.
 *  Any leading delimiters are ignored. An iterative usage
 *  of this function will read token after token, because the
 *  get pointer is set to the first delimiter after the next token.
 *
 *      \param  is    The input stream the token is read from.
 *      \param  token Used to store the token that is read from \em is.
 *      \param  delim String of characters that are used as delimiters.
 *                    By default the whitespace character, the
 *                    tabulator, the carriage return, the newline
 *                    character, the vertical tabulator and the
 *                    form feed character are used as delimiters.
 *      \return The first delimiter after the token.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      2002-01-03, ra:
 *      The first character of a token was ignored, when no whitespace
 *      followed the leading delimiters - fixed.
 *
 *  \par Status
 *      stable
 *
 */
char gettoken(std::istream& is, std::string& token, const std::string& delim)
{
	char   c;

	//
	// skip leading delimiters
	//
	while (is && ! is.eof())
	{
		is.get(c);
		if (delim.find(c) == std::string::npos)
		{
			is.putback(c);
			break;
		}
	}

	//
	// read characters until the first occurence of a delimiter
	//
	while (is && ! is.eof())
	{
		is.get(c);
		if (delim.find(c) != std::string::npos)
			break;
		token += c;
	}

	return c;
}

#ifndef __GNUC__
}
#endif

//===========================================================================

/*
 * \t = 0x09 tab
 * \r = 0x0d carriage return
 * \n = 0x0a newline
 * \v = 0x0b vertical tab
 * \a = 0x07 alert bell
 * \f = 0x0c form feed
 * \b = 0x08 backspace
 * \e = 0x1b escape
 */

#ifndef __GNUC__
namespace FileUtil
{
#endif

//===========================================================================
/*!
 *  \brief Reads the first string from "is".
 *
 *  This function can be used to read a token from
 *  an input stream. To identify a token it can be enclosed in quotation
 *  marks or inverted commas. The token name can include tabulator
 *  characters, carriage returns, newlines, vertical tabulators,
 *  alert bells, form feeds, backspaces and escape characters, but no
 *  whitespaces. <br>
 *  This is the opposite function of #printTo(std::ostream& os, const std::string& t).
 *
 *      \param  is The input stream the token is read from.
 *      \param  t  Used to store the token found.
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
static bool scanFrom(std::istream& is, std::string& t)
{
	char c, delim = ' ';
	std::string tt = "";

	if (is)
	{
		//
		// skip leading white space characters
		//
		do
			is.get(c);
		while (is && ! is.eof() && isspace(c));
		if (c == '\"' || c == '\'')
		{
			delim = c;
			is.get(c);
		}

		while (is && ! is.eof() && c != delim && (delim != ' ' || ! isspace(c)))
		{
			if (c == '\\')
			{
				is.get(c);
				switch (c)
				{
				case 't' : c = '\t'; break;
				case 'r' : c = '\r'; break;
				case 'n' : c = '\n'; break;
				case 'v' : c = '\v'; break;
				case 'a' : c = '\a'; break;
				case 'f' : c = '\f'; break;
				case 'b' : c = '\b'; break;
				}
			}
			tt += c;
			is.get(c);
		}

		t = tt;
	}
	else
		is.clear();

	return is.good();
}

//===========================================================================
/*!
 *  \brief Reads the single value of a token from input stream "is".
 *
 *  This function is used for files that contain initialization values
 *  for variables. The name of the variable is used as search token
 *  and the value following these token is stored and can be used
 *  as new value for the variable with name \em token.
 *  To identify the token value (and to differentiate
 *  it from the token itself) it can be enclosed in quotation
 *  marks or inverted commas. The value can include tabulator
 *  characters, carriage returns, newlines, vertical tabulators,
 *  alert bells, form feeds, backspaces and escape characters,
 *  but no whitespaces. <br>
 *  This is the opposite function of
 *  #printTo(std::ostream& os, const std::string& token, const std::string& t).
 *
 *      \param  is    The input stream the token value is read from.
 *      \param  token The token for which the value is read.
 *      \param  t     Used to store the token value found. Note, that the
 *                    value is stored as string, independend from its
 *                    original type.
 *      \param  rew   If set to "true", the search will start at
 *                    the beginning of the stream; if set to "false"
 *                    the search will start at the current position
 *                    of the get pointer.
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
template < >
bool scanFrom(std::istream& is, const std::string& token, std::string& t, bool rew)
{
	if (rew) rewind(is);
	skipuntil(is, token);
	return scanFrom(is, t);
}


//===========================================================================
/*!
 *  \brief Reads one or several values of a token from input stream "is".
 *
 *  This function is used for files that contain initialization values
 *  for variables. The name of the variable is used as search token
 *  and the value(s) following these token are stored and can be used
 *  as new value(s) for the variable with name \em token.
 *  To identify the token value(s) (and to differentiate
 *  it/them from the token itself), each single value can be enclosed in
 *  quotation marks or inverted commas. If more than one value is used,
 *  the list of all values must be enclosed in parentheses.
 *  Values can include tabulator
 *  characters, carriage returns, newlines, vertical tabulators,
 *  alert bells, form feeds, backspaces and escape characters,
 *  but no whitespaces. <br>
 *  This is the opposite function of
 *  #printTo(std::ostream& os, const std::string& token, const std::vector< std::string >& t).
 *
 *      \param  is    The input stream the token value(s) is/are read from.
 *      \param  token The token for which the value(s) is/are read.
 *      \param  t     All token values found are stored in this vector.
 *                    Note, that the value(s) is/are stored as string(s),
 *                    independend from its/their original type(s).
 *      \param  rew   If set to "true", the search will start at
 *                    the beginning of the stream; if set to "false"
 *                    the search will start at the current position
 *                    of the get pointer.
 *      \return "true", if no error occured while reading from
 *              \em is, "false" otherwise.
 *
 *  \par Example
 *
 *  \f$
 *  \begin{array}{ll}
 *  var1 & ("0.25"\ "0.32"\ "0.99")\\
 *  var2 & "testfile.dat"\\
 *  var3 & 10\\
 *  \end{array}
 *  \f$
 *
 *  If you call the function three times for a file with the content shown
 *  above it first will read the values \f$0.25\f$, \f$0.32\f$ and \f$0.99\f$
 *  for a double variable with the name \f$var1\f$, then the single value
 *  \f$testfile.dat\f$ for a variable named \f$var2\f$ and finally
 *  the value \f$10\f$ (note, that this value is not enclosed in
 *  quotation marks or inverted commas) for an integer variable named
 *  \f$var3\f$.
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
template < >
bool scanFrom(std::istream& is, const std::string& token, std::vector< std::string >& t,
			  bool rew)
{
	char   c;
	bool   ok;
	std::string tt;

	if (rew) rewind(is);

	skipuntil(is, token);

	if (is)
	{
		t.erase(t.begin(), t.end());
		is >> c;
		if (c != '(')
		{
			is.putback(c);
			ok = scanFrom(is, tt);
			t.push_back(tt);
			return ok;
		}
		else while (is)
			{
				ok = scanFrom(is, tt);
				t.push_back(tt);
				if (! ok)
					break;
				is >> c;
				if (c == ')')
					break;
				is.putback(c);
			}
	}
	else
		is.clear();

	return is.good();
}



//===========================================================================
/*!
 *  \brief Writes string "t" to output stream "os".
 *
 *  This function can be used to write a string \em t to an output
 *  stream, when this string also contains some escape sequences
 *  \em t can include tabulator characters, carriage returns, newlines,
 *  vertical tabulators, alert bells, form feeds, backspaces and escape
 *  characters. <br>
 *  This is the opposite function of #scanFrom(std::istream& is, std::string& t).
 *
 *      \param  os    The input stream the string is written to.
 *      \param  t     The string that will be written.
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
static bool printTo(std::ostream& os, const std::string& t)
{
	os << '"';
	for (unsigned i = 0; i < t.length(); ++i)
	{
		char c = t[ i ];
		if (c == '\\')
		{
			switch (t[ ++i ])
			{
			case 't' : c = '\t'; break;
			case 'r' : c = '\r'; break;
			case 'n' : c = '\n'; break;
			case 'v' : c = '\v'; break;
			case 'a' : c = '\a'; break;
			case 'f' : c = '\f'; break;
			case 'b' : c = '\b'; break;
			default  : c = t[ i-1 ];
			}
		}
		else if (c == '"')
			os << '\\';
		os << c;
	}
	os << '"' << endl;
	return os.good();
}


//===========================================================================
/*!
 *  \brief Writes a token and its single value "t" to output stream "os".
 *
 *  If you want to create files that can be used to initialize
 *  variables, you can use this function. The value \em t for a variable
 *  with name \em token will be written to output stream \em os.
 *  \em t can include tabulator characters, carriage returns, newlines,
 *  vertical tabulators, alert bells, form feeds, backspaces and escape
 *  characters. <br>
 *  This is the opposite function of
 *  #scanFrom(std::istream& is, const std::string& token, std::string& t, bool rew).
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
template < >
bool printTo(std::ostream& os, const std::string& token, const std::string& t)
{
	os << token << '\t';
	return printTo(os, t);
}



//===========================================================================
/*!
 *  \brief Writes a token and its value(s) "t" to output stream "os".
 *
 *  If you want to create files that can be used to initialize
 *  variables, you can use this function. The value(s) \em t for a variable
 *  with name \em token will be written to output stream \em os.
 *  Each single value in \em t can include tabulator characters,
 *  carriage returns, newlines,
 *  vertical tabulators, alert bells, form feeds, backspaces and escape
 *  characters. <br>
 *  This is the opposite function of
 *  #scanFrom(std::istream& is, const std::string& token, std::vector< std::string >& t, bool rew).
 *
 *      \param  os    The input stream the token and its value(s) are
 *                    written to.
 *      \param  token The token that will be written.
 *      \param  t     The value(s) of the token that will be written.
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
template < >
bool printTo(std::ostream& os, const std::string& token, const std::vector< std::string >& t)
{
	os << token << "\t(";
	for (unsigned i = 0; i < t.size() && os; ++i)
	{
		os << ' ';
		printTo(os, t[ i ]);
	}
	os << " )" << endl;
	return os.good();
}


}

