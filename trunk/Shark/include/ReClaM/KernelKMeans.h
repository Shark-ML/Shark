//===========================================================================
/*!
 *  \file KernelKMeans.h
 *
 *  \author  T. Suttorp
 *
 *  \brief Kernel k-means clustering
 *
 *  \date    2006
 *
 *  \par Copyright (c) 2006:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      ReClaM
 *
 *
 *  <BR>
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
//===========================================================================


#ifndef _KernelKMeans_H_
#define _KernelKMeans_H_


#include <vector>
#include <ReClaM/KernelFunction.h>


//! \brief Kernel k-means clustering
class KernelKMeans
{
public:
	KernelKMeans(KernelFunction* pKernel, const Array<double>& input, std::vector<int>& indexListForClustering, unsigned noClusters);
	~KernelKMeans();

	void updateFandG();

	std::vector<int>* clusterVectors();

	inline const Array<double>& getPoints()
	{
		return *x;
	}

private:
	std::vector<int>* mpIndexListOfClusterCenters;
	std::vector<int> mIndexListForClustering;

	unsigned mNoVectorsForClustering;

	const Array<double>* x;
	KernelFunction* mpKernel;
	unsigned mExamples, mDimension;
	unsigned mK;

	Array<bool>* mpD;
	Array<double>* mpF;

	std::vector<double> mAbsC;
	std::vector<double> mgC;
	std::vector<unsigned int> mNoMembers;
};


#endif

