//===========================================================================
/*!
 *  \file FileUtilClass.h
 *
 *  \brief  example for reading in parameters from a configuration file
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


#include <FileUtil/Params.h>


class MyParams : public Params
{
public:
	MyParams(int argc, char **argv) : Params(argc, argv)
	{
		setDefault();
		scanFrom(confFile);
	}

	void io(std::istream& is, std::ostream& os, FileUtil::iotype type)
	{
		FileUtil::io_strict(is, os, "n"        , n     , 4u     , type);
		FileUtil::io_strict(is, os, "d"        , d     , 3.14   , type);
		FileUtil::io_strict(is, os, "f"        , f     , 3.14   , type);
	}

	// Method to show the current content of the class variable:
	void monitor(std::ostream &os) const
	{
		os << "n\t"       << n        << "\t(the n)\n"
		<< "d\t"       << d        << "\t(the d)\n"
		<< "f\t"       << f        << "\t(the f)\n" << std::endl;
	}

	friend std::ostream& operator <<(std::ostream &os, const MyParams &obj);

	unsigned n;
	double   d;
	double   f;
};

std::ostream& operator <<(std::ostream &os, const MyParams &obj)
{
	obj.monitor(os);
	return os;
}

//
// call ./fileUtilClass -conf parameter.conf
//
int main(int argc, char **argv)
{
	MyParams p(argc, argv);


	std::cout << p << std::endl;
}

