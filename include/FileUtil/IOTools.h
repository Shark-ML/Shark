//===========================================================================
/*!
 *  \file IOTools.h
 *
 *  \brief Offers a function to read in input patterns
 *         and the corresponding target values for a network from a file.
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Copyright (c) 1999-2000:
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


#ifndef IO_TOOLS_H
#define IO_TOOLS_H

#include <SharkDefs.h>
#include <Array/ArrayIo.h>
#include <LinAlg/LinAlg.h>
#include <iostream>
#include <fstream>
#include <string>
#include <FileUtil/FileUtil.h>


//===========================================================================
/*!
 *  \brief Used to read input patterns and their corresponding
 *         target values from a file.
 *
 *  The data for input patterns and the corresponding target values
 *  is read from a file and stored in two arrays, that can be
 *  used as direct input for networks. <br>
 *  Each line of the file will contain the values for one input
 *  pattern, followed by the corresponding target values for this
 *  pattern. The single values must be separated by whitespaces. <br>
 *
 *  \param  filename The name of the file, from which the data is read.
 *  \param  id       The input dimension, i.e. the no. of input neurons.
 *  \param  od       The output dimension, i.e. the no. of output neurons.
 *  \param  in       The input patterns will be stored here.
 *  \param  out      The target patterns will be stored here.
 *  \return none.
 *
 *  \par Example
 *  \code
 *  #include "Array/ArrayIo.h"
 *  #include "FileUtil/IOTools.h"
 *
 *  void main()
 *  {
 *      Array< double > input, target;
 *
 *      loadOrganizedData( "test.dat", 4, 2, input, target );
 *
 *      cout << "input patterns:" << endl;
 *      writeArray( input, cout );
 *      cout << "\ntarget values:" << endl;
 *      writeArray( target, cout );
 *  }
 *  \endcode
 *
 *  If "test.dat" has the following content: <br>
 *
 *  \f$
 *  \begin{array}{rrrrrr}
 *  0.5 & 0.3 & 1.2 & 4.3 & 1.0 & 0.0\\
 *  -2.3 & 3.4 & 5.0 & 3.1 & 1.0 & 1.0\\
 *  1.9 & -2.3 & -4.4 & 9.2 & 0.0 & 1.0\\
 *  \end{array}
 *  \f$
 *
 *  then the example program will produce the output: <br>
 *
 *  \f$
 *  \mbox{input patterns:}\\
 *  \begin{array}{rrrr}
 *  0.5 & 0.3 & 1.2 & 4.3\\
 *  -2.3 & 3.4& 5 & 3.1\\
 *  1.9 & -2.3 & -4.4 & 9.2\\
 *  \end{array}\\[1cm]
 *  \mbox{target values:}\\
 *  \begin{array}{ll}
 *  1 & 0\\
 *  1 & 1\\
 *  0 & 1\\
 *  \end{array}
 *
 *  \f$
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
void loadOrganizedData(const std::string &filename, const unsigned id, const unsigned od, Array<double> &in, Array<double> &out)
{
	Array<double> raw;
	unsigned i, c;

	std::ifstream dataFile(filename.c_str());
	if (!dataFile)
	{
		std::cerr << "cannot open data file " << filename << std::endl;
		exit(EXIT_FAILURE);
	}
	readArray(raw, dataFile);
	dataFile.close();

	// if someone wants to learn just a single pattern...
	if (raw.ndim() == 1)
	{
		Array<double> dummy = raw;
		raw.resize(1, dummy.nelem());
		raw[0] = dummy;
	}

	out.resize(raw.dim(0), od);
	in.resize(raw.dim(0), id);
	for (i = 0; i < raw.dim(0); i++)
	{
		for (c = 0; c < id; c++)
			in(i, c) = raw(i, c);
		for (c = 0; c < od; c++)
			out(i, c) = raw(i, id + c);
	}
}


#endif

