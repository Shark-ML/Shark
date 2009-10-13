/*!
*  \file ArchiveMOO.cpp
*
*  \brief External Archive of candidates for pareto optimality
*
*  \author Tatsuya Okabe <tatsuya.okabe@honda-ri.de>
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
*      MOO-EALib
*  <BR>
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

/* ====================================================================== */
//
// 	Authors message
//======================================================================
/*	Thank you very much for your interest to MOO-EALib.

Since our company's name was changed on 1st, January, 2003,
my E-mail address in the source codes were also changed.
The current E-mail address (6th,Feb.,2004) is as follows:

tatsuya.okabe@honda-ri.de.

If you cannot contact me with the above E-mail address,
you can also use the following E-mail address:

t_okabe_de@hotmail.com.

If you have any questions, please don't hesitate to
ask me. It's my pleasure.

Best Regards,
Tatsuya Okabe

*********************************************************
Tatsuya Okabe
Honda Research Institute Europe GmbH
Carl-Legien-Strasse 30, 63073 Offenbach/Main, Germany
Tel: +49-69-89011-745
Fax: +49-69-89011-749
**********************************************************/


#include <SharkDefs.h>
#include <MOO-EALib/PopulationMOO.h>
#include <MOO-EALib/IndividualMOO.h>
#include <MOO-EALib/ArchiveMOO.h>
#include <Array/Array.h>


using namespace std;

//*******************************************
//** Constructor
//*******************************************

//**** TO-AM-001
ArchiveMOO::ArchiveMOO() : std::vector< IndividualMOO * >()
{ }

//***** TO-AM-002
ArchiveMOO::ArchiveMOO(bool strategy)
{
	setStrategy(strategy);
}

//***** TO-AM-003
ArchiveMOO::ArchiveMOO(unsigned max) : std::vector< IndividualMOO * >()
{
	setMaxArchive(max);
}

//***** TO-AM-004
ArchiveMOO::ArchiveMOO(unsigned max, bool strategy) : std::vector< IndividualMOO * >()
{
	setMaxArchive(max);
	setStrategy(strategy);
}

//*******************************************
//** Destructor
//*******************************************

//***** TO-AM-005
ArchiveMOO::~ArchiveMOO()
{
	unsigned n = size();
	for (unsigned i = 0; i < n; i++)
	{
		delete(*this)[ i ];
	}
}

//*******************************************
//** Internal variables I/O
//*******************************************

//***** TO-AM-010
unsigned ArchiveMOO::getMaxArchive()
{
	return MaxArchive;
}

//***** TO-AM-011
void ArchiveMOO::setMaxArchive(unsigned max)
{
	MaxArchive = max;
}

//***** TO-AM-012
unsigned ArchiveMOO::getCapacity()
{
	return getMaxArchive() - size();
}

//***** TO-AM-013
unsigned ArchiveMOO::size()
{
	//return std::vector< IndividualMOO * >::size( );
	return vector< IndividualMOO * >::size();
}

//***** TO-AM-014
bool ArchiveMOO::getStrategy()
{
	return Strategy;
}

//***** TO-AM-015
void ArchiveMOO::setStrategy(bool strategy)
{
	Strategy = strategy;
}

//***** TO-AM-016
void ArchiveMOO::minimize()
{
	Strategy = true;
}

//***** TO-AM-017
void ArchiveMOO::maximize()
{
	Strategy = false;
}

//*******************************************
//** Archive I/O
//*******************************************
//***** TO-AM-050
void ArchiveMOO::addArchive(IndividualMOO& indmoo)
{
	if (getMaxArchive() <= size())
	{
		cerr << getMaxArchive() << " " << size() << endl;
		throw SHARKEXCEPTION("Full capacity in TO-AM-050");
	}
	else
	{
		//std::vector< IndividualMOO * >::push_back( new IndividualMOO( indmoo ) );
		vector< IndividualMOO * >::push_back(new IndividualMOO(indmoo));
	}
}

//***** TO-AM-051
IndividualMOO& ArchiveMOO::readArchive(unsigned i)
{
	if (i < size())
	{
		return *(*(begin() + i));
	}
	else
	{
	  throw SHARKEXCEPTION("Access an unused archive in TO-AM-051");
	}
}

//***** TO-AM-052
void ArchiveMOO::delArchive(unsigned i)
{
	if (i < size())
	{
		delete(*this)[i];
		//std::vector< IndividualMOO * >::erase( begin( ) + i );
		vector< IndividualMOO * >::erase(begin() + i);
	}
	else
	{
	  throw SHARKEXCEPTION("Access an unused archive in TO-AM-052");
	}
}

//***** TO-AM-053
void ArchiveMOO::delArchive(std::vector< unsigned >& v)
{
	for (unsigned i = v.size(); i--;)
	{
		if (v[i] >= size())
		{
		  throw SHARKEXCEPTION("Access an unused archive in TO-AM-053");
		}
		else
		{
			delArchive(v[i]);
		}
	}
}

//***** TO-AM-054
void ArchiveMOO::delArchive(Array< unsigned >& a)
{
	for (unsigned i = a.nelem(); i--;)
	{
		if (a(i) >= size())
		{
		  throw SHARKEXCEPTION("Access an unused archive in TO-AM-054");
		}
		else
		{
			delArchive(a(i));
		}
	}
}

//***** TO-AM-055
void ArchiveMOO::cleanArchive()
{
	unsigned n = size();
	for (unsigned i = n; i--;)
	{
		delArchive(i);
	}
}

//***** TO-AM-056
void ArchiveMOO::delSharingWorst()
{
	unsigned i;
	i = sharingWorst();
	delArchive(i);
}

//***** TO-AM-057
void ArchiveMOO::delSharingWorst(double div)
{
	unsigned i;
	i = sharingWorst(div);
	delArchive(i);
}

//***** TO-AM-058
IndividualMOO& ArchiveMOO::readBestArchive()
{
	unsigned i = sharingBest();
	if (i < size())
	{
		return *(*(begin() + i));
	}
	else
	{
	  throw SHARKEXCEPTION("Access an unused archive in TO-AM-051");
	}
}

//***** TO-AM-059
void ArchiveMOO::nonDominatedSolutions()
{
	unsigned i;
	const unsigned no = size();
	int domination;
	PopulationMOO temp(no);
	for (i = no; i--;)
	{
		delete *(temp.begin() + i);
		*(temp.begin() + i) = new IndividualMOO(readArchive(i));
	}
	cleanArchive();
	addArchive(temp[ 0 ]);
	for (i = 0; i < no; i++)
	{
		domination = Dominate(temp[ i ]);
		if (domination > 2)
		{
			delDominateArchive(temp[ i ]);
			addArchive(temp[ i ]);
		}
		else if (domination == 2)
		{
			addArchive(temp[ i ]);
		}
	}
}

//*******************************************
//** Dominate
//*******************************************
//***** TO-AM-100
int ArchiveMOO::Dominate(IndividualMOO& im1, IndividualMOO& im2)
{
	// Strategy
	bool minimize = getStrategy();
	// Number of objective functions
	unsigned No1 = im1.getNoOfObj();
	unsigned No2 = im2.getNoOfObj();
	// Check each number of objective functions
	if (No1 != No2)
	{
		return 0; // abnormal
	}
	// Set flags for calculation
	unsigned flag1 = 0;
	unsigned flag2 = 0;
	unsigned flag3 = 0;
	// Calculation
	for (unsigned i = No1; i--;)
	{
		if (im1.getMOOFitness(i) > im2.getMOOFitness(i))
		{
			flag1++;
		}
		else if (im1.getMOOFitness(i) < im2.getMOOFitness(i))
		{
			flag3++;
		}
		else
		{
			flag2++;
		}
	}
	// Strategy
	if (minimize == false)
	{
		unsigned temp = flag1;
		flag1 = flag3;
		flag3 = temp;
	}
	// Relationship
	if (flag1 + flag2 + flag3 != No1)
	{
		return 0;  // abnormal
	}
	else if (flag3 == No1)
	{
		return 3;  // im1 dominates im2 strongly
	}
	else if (flag3 != 0 && flag1 == 0)
	{
		return 2;  // im1 dominates im2 weakly
	}
	else if (flag2 == No1)
	{
		return 1;  // im1 equals im2
	}
	else if (flag1 == No1)
	{
		return -3; // im2 dominates im1 strongly
	}
	else if (flag1 != 0 && flag3 == 0)
	{
		return -2; // im2 dominates im1 weakly
	}
	else
	{
		return -1; // trade off
	}
}

//***** TO-AM-101
int ArchiveMOO::Dominate(IndividualMOO& im1)
{
	if (size() == 0)
	{
		return 5;
	}
	// set flags for calculation
	unsigned flagA = 0; // for  3 ( dominate strongly )
	unsigned flagB = 0; // for  2 ( dominate weakly )
	unsigned flagC = 0; // for  1 ( equal )
	unsigned flagD = 0; // for  0 ( abnormal )
	unsigned flagE = 0; // for -1 ( trade off )
	unsigned flagF = 0; // for -2 ( dominated weakly )
	unsigned flagG = 0; // for -3 ( dominated strongly )
	// calculation
	unsigned No = size();
	int data;
	for (unsigned i = 0; i < No; i++)
	{
		data = Dominate(im1, readArchive(i));
		switch (data)
		{
		case 3:
			flagA++;
			break;
		case 2:
			flagB++;
			break;
		case 1:
			flagC++;
			break;
		case 0:
			flagD++;
			break;
		case - 1:
			flagE++;
			break;
		case - 2:
			flagF++;
			break;
		case - 3:
			flagG++;
			break;
		}
	}
	// evaluation
	if (flagD != 0)
	{
		return 0;  // Error
	}
	else if (flagF != 0 || flagG != 0)
	{
		return -1; // Dominated by archive
	}
	else if (flagC != 0)
	{
		return 1;  // Equal
	}
	else if (flagA == No)
	{
		return 5;  // Dominate all archive strongly
	}
	else if (flagA + flagB == No)
	{
		return 4;  // Dominate all archive weakly
	}
	else if (flagA != 0 || flagB != 0)
	{
		return 3;  // Dominate some of archive
	}
	else
	{
		return 2;  // Trade off
	}
}

//***** TO-AM-102
void ArchiveMOO::delDominateArchive(IndividualMOO& im1)
{
	unsigned No = size();
	int result;
	Array< unsigned > v(0);
	for (unsigned i = 0; i < No; i++)
	{
		result = Dominate(im1, readArchive(i));
		if (result > 1)
		{
			v.append_elem(i);
		}
	}
	delArchive(v);
}

//*******************************************
//** Distance on Fitness Space
//*******************************************
//***** TO-AM-150
double ArchiveMOO::distanceOnFitness(unsigned i1, unsigned i2)
{
	if (i1 >= size() || i2 >= size())
	{
	  throw SHARKEXCEPTION("Access an unused archive in TO-AM-150");
	}
	double distance = 0.0;
	unsigned NoOfObj = readArchive(i1).getNoOfObj();
	for (unsigned i = 0; i < NoOfObj; i++)
	{
		distance += pow(readArchive(i1).getMOOFitness(i)
						- readArchive(i2).getMOOFitness(i), 2);
	}
	distance = pow(distance, 0.5);
	return distance;
}

//***** TO-AM-151
double ArchiveMOO::distanceOnFitness(IndividualMOO& im1)
{
	unsigned No = size();
	unsigned NoOfObj = im1.getNoOfObj();
	double mindistance;
	double distance = 0.0;
	if (No == 0)
	{
		cerr << "Archive is empty in TO-AM-151" << endl;
		return MAXDOUBLE;
	}
	for (unsigned j = 0; j < NoOfObj; j++)
	{
		distance += pow(im1.getMOOFitness(j)
						- readArchive(0).getMOOFitness(j), 2);
	}
	distance = pow(distance, 0.5);
	mindistance = distance;
	for (unsigned i = 1; i < No; i++)
	{
		distance = 0.0;
		for (unsigned j = 0; j < NoOfObj; j++)
		{
			distance += pow(im1.getMOOFitness(j)
							- readArchive(i).getMOOFitness(j), 2);
		}
		distance = pow(distance, 0.5);
		if (distance < mindistance)
		{
			mindistance = distance;
		}
	}
	return mindistance;
}

//***** TO-AM-152
Array< double > ArchiveMOO::distanceDataOnFitness()
{
	const unsigned No = size();
	// calculate distance table
	Array< double > distance(No, No);
	for (unsigned i = 0; i < No; i++)
	{
		distance(i, i) = -1.0;
		for (unsigned j = i + 1; j < No; j++)
		{
			distance(i, j) = distanceOnFitness(i, j);
			distance(j, i) = distance(i, j);
		}
	}
	return distance;
}

//*****TO-AM-153
unsigned ArchiveMOO::sharingWorst()
{
	unsigned i, j, k;
	//Calculate the distance table
	const unsigned No = size();
	Array< double > distance(No, No);
	distance = distanceDataOnFitness();
	//buble sorting in the same line
	double temp;
	for (i = 0; i < No; i++)
	{
		for (j = 0; j < No; j++)
		{
			for (k = 0; k < j; k++)
			{
				if (distance(i, j) < distance(i, k))
				{
					temp = distance(i, j);
					distance(i, j) = distance(i, k);
					distance(i, k) = temp;
				}
			}
		}
	}
	//Decide the worst individual(s)
	Array< bool > check(No);
	for (i = 0; i < No; i++)
	{
		check(i) = true;
	}
	double min = -1.0;
	unsigned noind = 0;
	unsigned left = No;

	for (j = 1; j < No; j++)
	{
		min = -1.0;
		for (i = 0; i < No; i++)
		{
			if (check(i) == true)
			{
				if (min < 0.0)
				{
					min = distance(i, j);
				}
				else if (min < distance(i, j))
				{
					left--;
					check(i) = false;
				}
				else if (min > distance(i, j))
				{
					for (k = 0; k < i; k++)
					{
						if (check(k) == true)
						{
							left--;
							check(k) = false;
						}
					}
					min = distance(i, j);
				}
			}
		}
	}

	//=====================================================================
	// If ( left > 1 ), the index of the last individual will be returned.
	//=====================================================================
	if (left != 1)
	{
		cout << "More than 2 individuals may be the same from "
		<< "the point of sharing"
		<< endl;
	}

	for (i = 0; i < No; i++)
	{
		if (check(i) == true)
		{
			noind = i;
		}
	}

	return noind;

}

//*****TO-AM-154
unsigned ArchiveMOO::sharingWorst(double div)
{
	double f;
	unsigned counter;
	bool difference;
	int max;
	unsigned select;
	unsigned i, j, k;

	if (size() == 0)
	{
	  throw SHARKEXCEPTION("Archive has no individual in TO-AM-154");
	}
	const unsigned DIM = readArchive(0).getNoOfObj();
	Array< int > clustering(size(), DIM + 1);
	for (i = 0; i < size(); i++)
	{
		for (j = 0; j < DIM; j++)
		{
			f = readArchive(i).getMOOFitness(j);
			clustering(i, j) = (int) floor(f / div);
		}
	}
	for (i = 0; i < size(); i++)
	{
		counter = 0;
		for (j = 0; j < size(); j++)
		{
			difference = false;
			for (k = 0; k < DIM; k++)
			{
				if (clustering(i, k) != clustering(j, k))
				{
					difference = true;
				}
			}
			if (!difference)
			{
				counter++;
			}
		}
		clustering(i, DIM) = counter;
	}
	max = 0;
	for (i = 0; i < size(); i++)
	{
		if (clustering(i, DIM) > max)
		{
			max = clustering(i, DIM);
		}
	}
	counter = 0;
	for (i = 0; i < size(); i++)
	{
		if (clustering(i, DIM) == max)
		{
			counter++;
		}
	}
	DiscreteUniform du(0, counter - 1);
	select = du();
	counter = 0;
	for (i = 0; i < size(); i++)
	{
		if (clustering(i, DIM) == max)
		{
			if (counter == select)
			{
				return i;
			}
			else
			{
				counter++;
			}
		}
	}
	return 0;
}

//*****TO-AM-155
unsigned ArchiveMOO::sharingBest()
{
	unsigned i, j, k;
	//Calculate the distance table
	const unsigned No = size();
	Array< double > distance(No, No);
	distance = distanceDataOnFitness();
	//buble sorting in the same line
	double temp;
	for (i = 0; i < No; i++)
	{
		for (j = 0; j < No; j++)
		{
			for (k = 0; k < j; k++)
			{
				if (distance(i, j) > distance(i, k))
				{
					temp = distance(i, j);
					distance(i, j) = distance(i, k);
					distance(i, k) = temp;
				}
			}
		}
	}
	//Decide the worst individual(s)
	Array< bool > check(No);
	for (i = 0; i < No; i++)
	{
		check(i) = true;
	}
	double max = -1.0;
	unsigned noind = 0;
	unsigned left = No;

	for (j = 1; j < No; j++)
	{
		max = -1.0;
		for (i = 0; i < No; i++)
		{
			if (check(i) == true)
			{
				if (max < 0.0)
				{
					max = distance(i, j);
				}
				else if (max > distance(i, j))
				{
					left--;
					check(i) = false;
				}
				else if (max < distance(i, j))
				{
					for (k = 0; k < i; k++)
					{
						if (check(k) == true)
						{
							left--;
							check(k) = false;
						}
					}
					max = distance(i, j);
				}
			}
		}
	}

	//=====================================================================
	// If ( left > 1 ), the index of the last individual will be returned.
	//=====================================================================
	if (left != 1)
	{
		cout << "More than 2 individuals may be the same from "
		<< "the point of sharing" << endl;
	}

	for (i = 0; i < No; i++)
	{
		if (check(i) == true)
		{
			noind = i;
		}
	}
	return noind;
}





//***** TO-AM-156
double ArchiveMOO::minDistanceOnFitness()
{
	const unsigned No = size();
	Array< double > distance(No, No);
	distance = distanceDataOnFitness();
	double min = -1.0;
	for (unsigned i = 0; i < No; i++)
	{
		for (unsigned j = 0; j < No; j++)
		{
			if (distance(i, j) >= 0.0)
			{
				if (min < 0.0)
				{
					min = distance(i, j);
				}
				if (min > distance(i, j))
				{
					min = distance(i, j);
				}
			}
		}
	}
	return min;
}

//***** TO-AM-157
double ArchiveMOO::minDistanceOnFitness(unsigned i1)
{
	const unsigned No = size();
	Array< double > distance(No, No);
	distance = distanceDataOnFitness();
	double min = -1.0;
	for (unsigned j = 0; j < No; j++)
	{
		if (distance(i1, j) >= 0.0)
		{
			if (min < 0.0)
			{
				min = distance(i1, j);
			}
			if (min > distance(i1, j))
			{
				min = distance(i1, j);
			}
		}
	}
	return min;
}


//***** TO-AM-300
unsigned ArchiveMOO::crowded(IndividualMOO& im, double div)
{
	const unsigned DIM = im.getNoOfObj();
	double f;
	unsigned i, j;
	Array< double > Start(DIM);
	for (i = 0; i < DIM; i++)
	{
		Start(i) = (double)(floor(im.getMOOFitness(i) / div)) * div;
	}
	unsigned counter = 0;
	bool inside;
	for (i = 0; i < size(); i++)
	{
		inside = true;
		for (j = 0; j < DIM; j++)
		{
			f = readArchive(i).getMOOFitness(j);
			if (Start(j) > f || f > Start(j) + div)
			{
				inside = false;
				break;
			}
		}
		if (inside)
		{
			counter++;
		}
	}
	return counter;
}


//***** TO-AM-1000
void ArchiveMOO::saveArchive(char *filename)
{
	unsigned No = size();
	unsigned NoOfObj;
	if (No > 0)
	{
		NoOfObj = readArchive(0).getNoOfObj();
	}
	else
	{
		NoOfObj = 0;
	}
	double   f;
	FILE *fp;
	fp = fopen(filename, "w");
	fprintf(fp, "%i\n", No);
	fprintf(fp, "%i\n", NoOfObj);
	for (unsigned i = 0; i < No; i++)
	{
		for (unsigned j = 0; j < NoOfObj; j++)
		{
			f = readArchive(i).getMOOFitness(j);
			fprintf(fp, "%f\n", f);
		}
	}
	fclose(fp);
}

//***** SW-AM-1001
void ArchiveMOO::saveArchiveGPT(char *filename)
{
	unsigned No = size();
	unsigned NoOfObj;
	if (No > 0)
	{
		NoOfObj = readArchive(0).getNoOfObj();
	}
	else
	{
		NoOfObj = 0;
	}
	double   f;
	ofstream ofs(filename);
	for (unsigned i = 0; i < No; i++)
	{
		for (unsigned j = 0; j < NoOfObj; j++)
		{
			f = readArchive(i).getMOOFitness(j);
			ofs << f << " " << std::flush;
		}
		ofs << std::endl;
	}
	ofs.close();
}

