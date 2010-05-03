//===========================================================================
/*!
 *  \file NetParams.cpp
 *
 *  \brief Offers functions for easily reading information about
 *         a network, an error measure and an optimization algorithm
 *         from a configuration file.
 *
 *  A structure of predefined variables can be used to store information read
 *  from a configuration file. These information can be used to create
 *  a network, define an error measure and initialize an optimization
 *  algorithm, to automize the usage of the ReClaM library for a personal
 *  network.
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Copyright (c) 1999-2000
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
 *      ReClaM
 *
 *
 *  This file is part of ReClaM. This library is free software;
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
 *
 *
 */
//===========================================================================


#include "ReClaM/NetParams.h"

using namespace std;


//========================================================================
/*!
 *
 *  \brief Reads the values for all parameters defined in #io
 *         from the configuration file given in "argv".
 *
 *  The configuration file, where the name is given in \em argv
 *  is used to read the values of all parameters defined in
 *  #io. If the configuration file is not found or a parameter
 *  is not listed in the configuration file, the parameter
 *  will have its default value. <br>
 *  For a more detailed description about calling this constructor,
 *  refer to Params::Params.
 *
 *  \param argc The number of arguments stored in \em argv.
 *  \param argv Parameters taken from the main program,
 *              including the name of the configuration file.
 *  \return none
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
NetParams::NetParams(int argc, char **argv)
		: Params(argc, argv)
{
	set_format_liberate();
	setDefault();

	//
	// check whether a configuration file is given
	//
	if (confFile.length() > 0)
	{
		if (! scanFrom(confFile))
			cerr << "error: read failed on configuration file "
			<< confFile << endl;
	}
}


//========================================================================
/*!
 *
 *  \brief Defines FileUtil::io to be used, when dealing
 *         with the configuration file.
 *
 *  Flag #strict is set to "false". This will cause NetParams
 *  to call FileUtil::io every time #io is called. <br>
 *  FileUtil::io is used for configuration files with a
 *  more liberate format.
 *
 *  \return none
 *
 *  \author  R. Alberts
 *  \date    2002-01-11
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void NetParams::set_format_liberate()
{
	strict = false;
}


//========================================================================
/*!
 *
 *  \brief Defines FileUtil::io_strict to be used, when dealing
 *         with the configuration file.
 *
 *  Flag #strict is set to "true". This will cause NetParams
 *  to call FileUtil::io_strict every time #io is called. <br>
 *  FileUtil::io_strict is used for configuration files with a
 *  strict format.
 *
 *  \return none
 *
 *  \author  R. Alberts
 *  \date    2002-01-11
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void NetParams::set_format_strict()
{
	strict = true;
}



//========================================================================
/*!
 *
 *  \brief This method is called by the constructor and defines
 *         the default values token names and variables for
 *         all parameters.
 *
 *  Used as interface between NetParams and FileUtil, this method
 *  delivers all information necessary to read all parameters
 *  (class variables) from a configuration file. <br>
 *  For a more detailed description of the general working, refer
 *  to class Params. <br>
 *  The reading of the configuration file is depending on the
 *  value of strict.
 *
 *  \return none
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes
 *      2002-01-11, ra <br>
 *      Integrated the optional usage of FileUtil::io_strict for
 *      reading from configuration files.
 *
 *  \par Status
 *      stable
 *
 */
void NetParams::io(istream& is, ostream& os, FileUtil::iotype type)
{
	if (strict == false)
	{
		FileUtil::io(is, os, "netFilename"         , netFilename         , string(""), type);
		FileUtil::io(is, os, "prefix"              , prefix              , string(""), type);
		FileUtil::io(is, os, "trainDataFilename"   , trainDataFilename   , string(""), type);
		FileUtil::io(is, os, "testDataFilename"    , testDataFilename    , string(""), type);
		FileUtil::io(is, os, "validateDataFilename", validateDataFilename, string(""), type);
		FileUtil::io(is, os, "useRprop"            , useRprop            , true , type);
		FileUtil::io(is, os, "norm"                , norm                , true , type);
		FileUtil::io(is, os, "normByVariance"      , normByVariance      , true , type);
		FileUtil::io(is, os, "init"                , init                , true , type);
		FileUtil::io(is, os, "linearOutput"        , linearOutput        , true , type);
		FileUtil::io(is, os, "plot"                , plot                , true , type);


		FileUtil::io(is, os, "seed"                , seed                , 42   , type);
		FileUtil::io(is, os, "cycles"              , cycles              , 1000U, type);
		FileUtil::io(is, os, "interval"            , interval            , 1U   , type);
		FileUtil::io(is, os, "runs"                , runs                , 1U   , type);
		FileUtil::io(is, os, "lr"                  , lr                  , .001 , type);
		FileUtil::io(is, os, "weightDecay"         , weightDecay         , .001 , type);
		FileUtil::io(is, os, "momentum"            , momentum            , .001 , type);
		FileUtil::io(is, os, "low"                 , low                 , .2  , type);
		FileUtil::io(is, os, "high"                , high                , .8  , type);
		FileUtil::io(is, os, "function"            , function            , string("sigmoid") , type);

		FileUtil::io(is, os, "delta0"              , delta0              , 0.1, type);
		FileUtil::io(is, os, "np"                  , np                  , 1.2, type);
		FileUtil::io(is, os, "nm"                  , nm                  , .5, type);
		FileUtil::io(is, os, "dMin"                , dMin                , 1e-6, type);
		FileUtil::io(is, os, "dMax"                , dMax                , 50.0, type);

		FileUtil::io(is, os, "lineSearch"          , lineSearch          , 0u, type);
		FileUtil::io(is, os, "ax"                  , ax                  , 0., type);
		FileUtil::io(is, os, "bx"                  , bx                  , 1., type);
		FileUtil::io(is, os, "lambda"              , lambda              , .25, type);
	}
	else
	{
		FileUtil::io_strict(is, os, "netFilename"         , netFilename         , string(""), type);
		FileUtil::io_strict(is, os, "prefix"              , prefix              , string(""), type);
		FileUtil::io_strict(is, os, "trainDataFilename"   , trainDataFilename   , string(""), type);
		FileUtil::io_strict(is, os, "testDataFilename"    , testDataFilename    , string(""), type);
		FileUtil::io_strict(is, os, "validateDataFilename", validateDataFilename, string(""), type);
		FileUtil::io_strict(is, os, "useRprop"            , useRprop            , true , type);
		FileUtil::io_strict(is, os, "norm"                , norm                , true , type);
		FileUtil::io_strict(is, os, "normByVariance"      , normByVariance      , true , type);
		FileUtil::io_strict(is, os, "init"                , init                , true , type);
		FileUtil::io_strict(is, os, "linearOutput"        , linearOutput        , true , type);
		FileUtil::io_strict(is, os, "plot"                , plot                , true , type);


		FileUtil::io_strict(is, os, "seed"                , seed                , 42   , type);
		FileUtil::io_strict(is, os, "cycles"              , cycles              , 1000U, type);
		FileUtil::io_strict(is, os, "interval"            , interval            , 1U   , type);
		FileUtil::io_strict(is, os, "runs"                , runs                , 1U   , type);
		FileUtil::io_strict(is, os, "lr"                  , lr                  , .001 , type);
		FileUtil::io_strict(is, os, "weightDecay"         , weightDecay         , .001 , type);
		FileUtil::io_strict(is, os, "momentum"            , momentum            , .001 , type);
		FileUtil::io_strict(is, os, "low"                 , low                 , .2  , type);
		FileUtil::io_strict(is, os, "high"                , high                , .8  , type);
		FileUtil::io_strict(is, os, "function"            , function            , string("sigmoid") , type);

		FileUtil::io_strict(is, os, "delta0"              , delta0              , 0.1, type);
		FileUtil::io_strict(is, os, "np"                  , np                  , 1.2, type);
		FileUtil::io_strict(is, os, "nm"                  , nm                  , .5, type);
		FileUtil::io_strict(is, os, "dMin"                , dMin                , 1e-6, type);
		FileUtil::io_strict(is, os, "dMax"                , dMax                , 50.0, type);

		FileUtil::io_strict(is, os, "lineSearch"          , lineSearch          , 0u, type);
		FileUtil::io_strict(is, os, "ax"                  , ax                  , 0., type);
		FileUtil::io_strict(is, os, "bx"                  , bx                  , 1., type);
		FileUtil::io_strict(is, os, "lambda"              , lambda              , .25, type);
	}
}





