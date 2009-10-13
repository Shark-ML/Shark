//===========================================================================
/*!
 *  \file FileUtilSimple.h
 *
 *  \brief simple example for reading in parameters from a configuration file
 *
 *
 *  \author  M. Kreutz, C. Igel
 *
 *  \par Copyright (c) 1995-2007:
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


#include <stdlib.h>
#include <fstream>
#include <FileUtil/FileUtil.h>
#include <cmath>

//
// call ./FileUtilSimple parameter.conf
//
int main(int argc, char **argv)
{
	// default values
	unsigned n = 4;
	double   d = 3.14;
	double   f = 3.14;		// this value should not change

	if (argc > 1)
	{
		std::ifstream ifs(argv[1]);
		if (ifs)
		{
			FileUtil::scanFrom_strict(ifs, "n", n, true); // "true" indicates that the scan always starts from the first line of the configuration file
			FileUtil::scanFrom_strict(ifs, "d", d, true);
		}
		ifs.close();

		std::cout << "n=" << n << " \td=" << d << " \tf=" << f << std::endl;
	}
	else
	{
		std::cout << "filename needed --- call ./FileUtilSimple parameter.conf" << std::endl;
	}

	// lines below are for self-testing this example, please ignore
       	if(  fabs(d-42.42)<1.e-6) exit(EXIT_SUCCESS);
       	else exit(EXIT_FAILURE);
}
