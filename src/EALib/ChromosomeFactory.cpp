/*!
*  \file ChromosomeFactory.cpp
*
*  \brief Creates Chromosome objects of given type.
*
*  \author  C. Igel
* 
*  \date    2007
*
*  \par
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR> 
*
*  \par Project:
*      EALib
*
*
*  <BR><HR>
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
*/

#include <EALib/ChromosomeFactory.h>
#include <EALib/ChromosomeT.h>


Chromosome* CreateChromosome(const char* type)
{
	if (strcmp(type, typeid(double).name()) == 0) {
		return new ChromosomeT<double>();
	}
	else if (strcmp(type, typeid(int).name()) == 0) {
		return new ChromosomeT<int>();
	}
	else if (strcmp(type, typeid(bool).name()) == 0) {
		return new ChromosomeT<bool>();
	}
	else if (strcmp(type, typeid(char).name()) == 0) {
		return new ChromosomeT<char>();
	}

	throw SHARKEXCEPTION("[CreateChromosome] unknown type");
	return NULL;
}

