//===========================================================================
/*!
 *  \file Params.h
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
 *  \par Copyright (c) 1995-2000:
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


#ifndef __PARAMS_H
#define __PARAMS_H

#include <SharkDefs.h>
#include <FileUtil/FileUtil.h>


//===========================================================================
/*!
 *  \brief This class offers methods for easily using configuration
 *         files to change the values of variables in your programs.
 *
 *  If you have written a program that can be used for different
 *  situations you often have the problem of changing the content
 *  of the variables in your program that will define the situation
 *  the program is used for. <br>
 *  You must change the content of the variables to adapt it to a
 *  special situation and recompile your program. <br>
 *  More easily is the usage of configuration files, that will
 *  include predefined values for your program for different situations.
 *  This class here offers methods for easily using such configuration
 *  files. For flexibility (i.e. different formats of configuration
 *  files), this class can only be used by deriving your own class
 *  from it. Class Params has the role of an interface between your
 *  derived class and the functions offered by the file FileUtil.h
 *  (there you will also find a more detailed description of the
 *  general format of configuration files).
 *  For a simple example of deriving your own class from Params, see
 *  method #io. Or take a look at a complete derivation (class NetParams)
 *  that is used for the creation of neural networks in package ReClaM.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class Params
{
public:

	//! Set the default values for parameters.
	void setDefault();

	//! Read values for parameters from the input stream "is".
	bool scanFrom(std::istream& is);

	//! Read values for parameters from the configuration file "name".
	bool scanFrom(const std::string& name);

	//! Write values for parameters to the output stream "os".
	bool printTo(std::ostream& os);

	//! Write values for parameters to the configuration file "name".
	bool printTo(const std::string& name);

	//========================================================================
	/*!
	 *  \brief Default destructor, that must be overloaded by
	 *         the derived class.
	 *
	 *  Overload this virtual destructor in your class derived from
	 *  class Params. If you don't overload the destructor, the default
	 *  destructor will be used.
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
	virtual ~Params()
	{ }

protected:

	//! Name of the configuration file.
	std::string confFile;

	//! Checks whether a configuration file is given.
	Params(int argc = 0, char **argv = NULL);

	//========================================================================
	/*!
	 * 
	 *  \brief Interface between the methods of this class and
	 *         the function FileUtil::io.
	 *
	 *  This method must be overloaded by the derived class and will
	 *  function as interface between the methods of this class and
	 *  the io-function offered by FileUtil.h. <br>
	 *  The implementation of the derived class method will then
	 *  consist of several commands, each command is responsible
	 *  for one parameter saved in the used configuration file. <br>
	 *  Each of these commands will have the format: <br>
	 *  FileUtil::io[_strict]( \em is, \em os, \em token, \em variable, \em default, \em type ); <br>
	 *  <br>
	 *  You can use FileUtil::io for configuration files with a more
	 *  liberate format or FileUtil::io_strict for configuration files
	 *  with a very strict format. <br>
	 *  <ul>
	 *  <li> \em is, \em os and \em type will have the same name as
	 *  the parameters used in the derived class' io-method. </li>
	 *  <li> \em token is the string used in the configuration file to
	 *  identify the current parameter. </li>
	 *  <li> \em variable is the name of the parameter used in your code </li>
	 *  <li> \em default is the value that will be assigned to the parameter
	 *       as default value. </li>
	 *  </ul>
	 *  <br>
	 *
	 *  \par Example
	 *  \code 
	 *  #include "FileUtil/Params.h"
	 *
	 *  // My own derived class for managing my configuration files:
	 *  //
	 *  class MyParams : public Params 
	 *  {
	 *    public:
	 *
	 *        // Overload default constructor:
	 *        MyParams( int argc, char **argv ) : Params( argc, argv ) { }
	 *
	 *        // Guess, my configuration file includes the definition
	 *        // for only one parameter, named "MyParam1" in the
	 *        // file, but marking the value for the class variable
	 *        // "my_param_1" (see below). The default value of this
	 *        // class variable is "10.0":
	 *        void io( std::istream& is, std::ostream& os, FileUtil::iotype type )
	 *        {
	 *            FileUtil::io( is, os, "MyParam1", my_param_1, 10.0, type );
	 *        }
	 *
	 *        // Method to show the current content of the class variable: 
	 *        void monitor( )
	 *        {
	 *            cout << "Value of \"my_param_1\" = " << my_param_1 << endl;
	 *        }
	 *
	 *
	 *    private:
	 *
	 *        // The variable with its value defined in the configuration
	 *        // file:
	 *        double my_param_1;
	 *
	 *  };
	 *
	 *
	 *  void main( int argc, char **argv )
	 *  {
	 *      MyParams param( argc, argv );
	 *      param.setDefault( );
	 *      param.printTo( "test.conf" );
	 *      param.monitor( );
	 *  }
	 *  \endcode
	 *
	 *  Here we use the program only to assign the given default value
	 *  to the class variable and then write this default value together
	 *  with its identifying token name to the configuration file 
	 *  "test.conf". <br>
	 *  The principle is transferable to other configuration files. 
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
	virtual void io(std::istream&, std::ostream&, FileUtil::iotype) = 0;

};


#endif

