/*!
*  \file Chromosome.cpp
* 
*  \author M. Kreutz, C. Igel
*
*  \brief Instances of Chromosome make up the genetic information of
*  individuals.
* 
*  \date 1998
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
*  This file is part of the EALib. This library is free software;
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


#include <SharkDefs.h>
#include <EALib/Chromosome.h>


bool Chromosome::operator == (const Chromosome& c) const
{
	UNDEFINED
	return false;
}

bool Chromosome::operator < (const Chromosome& c) const
{
	UNDEFINED
	return false;
}


//===========================================================================

//
// added by Marc Toussaint and Stefan Wiegand at 20.11.2002
//

/*! inteface methods for more externally from the EALib defined chromosomes */
void Chromosome::init()
{}
void Chromosome::init(const char* filename)
{}
void Chromosome::mutate()
{}
#ifdef EALIB_REGISTER_INDIVIDUAL	
void Chromosome::registerIndividual(const Individual& i, uint you) {}
#endif
void Chromosome::appendToIndividual(Individual& i)
{}

/*! PVM routines */
int  Chromosome::pvm_pkchrom()
{
	std::cerr << "EALib/Chromosome.cpp: default dummy routine for pvm_pkchrom() implemented." << std::endl;
	return -1 ;
}
int  Chromosome::pvm_upkchrom()
{
	std::cerr << "EALib/Chromosome.cpp: default dummy routine for pvm_upkchrom() implemented." << std::endl;
	return -1;
}

//===========================================================================
//

