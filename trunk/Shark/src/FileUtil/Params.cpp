//===========================================================================
/*!
 *  \file Params.cpp
 *
 *  \brief This file offers a class for dealing with configuration files.
 *
 *  The class in this file is used for configuration files, i.e. files
 *  that include values for variables used in your programs. See the
 *  description of class Params for further information.
 *
 *  \author  M. Kreutz
 *  \date    1998-10-18
 *
 *  \par Copyright (c) 1998-2000:
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


#include <FileUtil/FileUtil.h>
#include <FileUtil/Params.h>


//===========================================================================
/*!
 *  \brief Checks whether a configuration file is given.
 *
 *  The main program you use to include your own class, derived from
 *  class Params must take at least two additional parameters, when called.
 *  One of these parameter is "-conf", a flag that denotes, that the
 *  following parameter will be the name of a configuration file.
 *  The additional parameters are then used by this constructor
 *  to check for the occurrence of the mentioned flag and the
 *  name of the configuration file. If both parameters are found,
 *  the name of the configuration file is saved internally.
 *
 *  \param argc Number of arguments stored in \em argv.
 *  \param argv The parameters (arguments) taken from the main program.
 *  \return An instance of class Params.
 *
 *  \par Example
 *  \code
 *  #include "FileUtil/Params.h"
 *
 *  // My own class, derived from the Params class:
 *  //
 *  class MyParams : public Params
 *  {
 *    public:
 *
 *        // The derived constructor, that will display the
 *        // name of the configuration file or an error
 *        // message, if no filename was found:
 *        MyParams( int argc, char **argv ) : Params( argc, argv )
 *        {
 *            if ( confFile.size( ) > 0 )
 *            {
 *                cout << "Name of the configuration file: " << confFile << endl;
 *            }
 *            else
 *            {
 *                cerr << "No configuration file given!" << endl;
 *            }
 *        }
 *
 *        // The derived destructor:
 *        ~MyParams( ) { }
 *
 *        // This function must be derived, when using class Params:
 *       void io( std::istream& is, std::ostream& os, FileUtil::iotype type )
 *       {
 *           ...
 *       }
 *
 *  }; // End of derived class.
 *
 *  // Call the main program with parameters "-conf [filename]":
 *  void main( int argc, char **argv )
 *  {
 *      // Pass the parameters of the main program to the
 *      // constructor:
 *      MyParams param( argc, argv );
 *  }
 *  \endcode
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
Params::Params(int argc, char **argv)
{
	for (int i = 0; i < argc; ++i)
		if (strcmp(argv[ i ], "-conf") == 0 && i + 1 < argc)
		{
			confFile = argv[ i+1 ];
			break;
		}
}


//===========================================================================
/*!
 *  \brief Set the default values for parameters.
 *
 *  Depending on your overloaded version of #io, parameters will be
 *  initialized by default values. See the example program
 *  for #io, for a further description.
 *
 *  \return
 *      none
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
void Params::setDefault()
{
	io(std::cin, std::cout, FileUtil::SetDefault);
}

//===========================================================================
/*!
 *  \brief Read values for parameters from the input stream "is".
 *
 *  Depending on your overloaded version of #io, the values of
 *  parameters will be read from the input stream \em is.
 *  See the example program for #io, for a further description.
 *
 *  \param is The input stream.
 *  \return "false", if an error occured while reading from \em is, "true"
 *          otherwise.
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
bool Params::scanFrom(std::istream& is)
{
	io(is, std::cout, FileUtil::ScanFrom);

	return is.good();
}

//===========================================================================
/*!
 *  \brief Read values for parameters from the configuration file
 *         "name".
 *
 *  Depending on your overloaded version of #io, the values of
 *  parameters will be read from the configuration file \em name.
 *  See the example program for #io, for a further description.
 *
 *  \param name The name of the configuration file.
 *  \return "false", if an error occured while reading from \em name, "true"
 *          otherwise.
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
bool Params::scanFrom(const std::string& name)
{
	std::string buf;
	if (FileUtil::readfile(name, buf) > 0)
	{
		std::istringstream istr(buf.c_str());
		return scanFrom(istr);
	}

	return false;
}

//===========================================================================
/*!
 *  \brief Write values for parameters to the output stream "os".
 *
 *  Depending on your overloaded version of #io, the values of
 *  parameters together with their identifying token names will be
 *  written to the output stream \em os.
 *  See the example program for #io, for a further description.
 *
 *  \param os The output stream.
 *  \return "false", if an error occured while writing to \em os, "true"
 *          otherwise.
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
bool Params::printTo(std::ostream& os)
{
	io(std::cin, os, FileUtil::PrintTo);

	return os.good();
}

//===========================================================================
/*!
 *  \brief Write values for parameters to the configuration file "name".
 *
 *  Depending on your overloaded version of #io, the values of
 *  parameters together with their identifying token names will be
 *  written to the configuration file \em name.
 *  See the example program for #io, for a further description.
 *
 *  \param name The name of the configuration file.
 *  \return "false", if an error occured while writing to \em name, "true"
 *          otherwise.
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
bool Params::printTo(const std::string& name)
{
	std::ofstream os(name.c_str());

	return os && printTo(os);
}

