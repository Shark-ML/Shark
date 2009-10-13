//===========================================================================
/*!
 *  \file KernelKMeans.cpp
 *
 *  \brief Kernel k-means clustering
 *
 *  \author  T. Suttorp
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


#include <vector>
#include <algorithm>
#include <ReClaM/KernelKMeans.h>


KernelKMeans::KernelKMeans(KernelFunction* pKernel, const Array<double>& input, std::vector<int>& indexListForClustering, unsigned noClusters)
{
	mpKernel = pKernel;

	x          = &input;
	mExamples  = input.dim(0);
	mDimension = input.dim(1);

	mIndexListForClustering = indexListForClustering;
	mNoVectorsForClustering = mIndexListForClustering.size();

	this->mK = noClusters;

	mpD = new Array<bool>(mNoVectorsForClustering, mK);
	mpF = new Array<double>(mNoVectorsForClustering, mK);

	mAbsC.resize(mK);
	mgC.resize(mK);
	mNoMembers.resize(mK);
}

KernelKMeans::~KernelKMeans()
{
	delete mpD;
	delete mpF;
}




void KernelKMeans::updateFandG()
{
	const Array<double> &xi = this->getPoints();;
	unsigned int i, j, l, kiT;

	// for each cluster
	for (i = 0; i < mK; ++i)
	{
		// compute noMembers in cluster i (absC)
		mAbsC[i] = 0;
		for (j = 0; j < mNoVectorsForClustering; ++j)
			if ((*mpD)(j, i) == 1)
				mAbsC[i] = mAbsC[i] + 1;

		// compute g(C_k)
		mgC[i] = 0;

		for (j = 0; j < mNoVectorsForClustering; ++j)
			for (l = 0; l < mNoVectorsForClustering; ++l)
				if ((*mpD)(j, i) == 1 && (*mpD)(l, i) == 1)
					mgC[i] += mpKernel->eval(xi[mIndexListForClustering[j]], xi[mIndexListForClustering[l]]);

		if (mgC[i] > 1e-10)
			mgC[i] /= (mAbsC[i] * mAbsC[i]);
		else // should not happen
			std::cout << "Empty cluster; no: " << i << "\n";
	}

	for (i = 0; i < mNoVectorsForClustering; ++i)
		for (kiT = 0; kiT < mK; ++kiT)
		{
			(*mpF)(i, kiT) = 0;
			for (j = 0; j < mNoVectorsForClustering; ++j)
			{
				if ((*mpD)(j, kiT) == 1)
				{
					(*mpF)(i, kiT) += mpKernel->eval(xi[mIndexListForClustering[i]], xi[mIndexListForClustering[j]]);
				}
			}

			if ((*mpF)(i, kiT) > 1e-10)
				(*mpF)(i, kiT) *= (-2 / mAbsC[kiT]);
		}
}


std::vector<int>* KernelKMeans::clusterVectors()
{
	double J; // function to be minimized
	const Array<double> &xi = this->getPoints();
	unsigned int i, kiT, q, r;

	// assign initial classes to vectors
	*mpD = 0;

	// create permutation vector
	std::vector<int> permVec(mNoVectorsForClustering);
	for (i = 0; i < mNoVectorsForClustering ; ++i)
		permVec[i] = i;

	std::random_shuffle(permVec.begin(), permVec.end());

	for (i = 0; i < mNoVectorsForClustering; ++i)
		(*mpD)(i, (int)(permVec[i] % mK)) = 1;


	bool bTerminationCriterionFulfilled = false;
	double objectiveFunctionValueOfLastIteration = 1e20;

	while (!bTerminationCriterionFulfilled)
	{
		// notation like in
		// Zhang, Rudnicky
		// A Large Scale Clustering Scheme for Kernel K-Means

		this->updateFandG();

		// assign to clusters
		int indexOfBestCluster = 0;

		J = 0; // target function for minimization
		for (i = 0; i < mNoVectorsForClustering; ++i)
		{
			for (kiT = 0; kiT < mK; ++kiT)
				if ((*mpD)(i, kiT) == 1)
					J += (*mpF)(i, kiT) + mgC[kiT];
		}

		(*mpD) = 0;
		for (i = 0; i < mNoVectorsForClustering; ++i)
		{
			double minDist = 1e20;
			for (kiT = 0; kiT < mK; ++kiT)
			{
				if (mgC[kiT] == 0)
					continue;

				if ((*mpF)(i, kiT) + mgC[kiT] < minDist)
				{
					minDist = (*mpF)(i, kiT) + mgC[kiT];
					indexOfBestCluster = kiT;
				}
			}
			(*mpD)(i, indexOfBestCluster) = 1;
		}

		if (objectiveFunctionValueOfLastIteration == J)
			bTerminationCriterionFulfilled = true;

		objectiveFunctionValueOfLastIteration = J;

		// deal with empty clusters
		bool bClusterEmpty;
		for (kiT = 0; kiT < mK; ++kiT)
		{

			bClusterEmpty = true;
			for (i = 0; i < mNoVectorsForClustering; ++i)
				if ((*mpD)(i, kiT) == 1)
					bClusterEmpty = false;


			if (bClusterEmpty)
			{
				// choose vector that contributes most to SSE
				double distance;
				unsigned indexOfMaxContributingVector = 0;
				double distOfMaxContributingVector = 0;

				// for every vector: determine distance to closest centre
				for (q = 0; q < mNoVectorsForClustering; ++q)
				{
					double minDist = 1e10;

					for (r = 0; r < mK; ++r)
					{
						distance  =  mpKernel->eval(xi[q], xi[q]) + (*mpF)(q, r) +  mgC[r];

						if (distance  < minDist)
							minDist = distance;
					}

					if (minDist > distOfMaxContributingVector)
					{
						distOfMaxContributingVector = minDist;
						indexOfMaxContributingVector = q;
					}
				}

				for (r = 0; r < mK; ++r)
					(*mpD)(indexOfMaxContributingVector, r) = 0;

				(*mpD)(indexOfMaxContributingVector, kiT) = 1;

				this->updateFandG();
			}
		}
	}

	// determine pseudo cluster centers
	mpIndexListOfClusterCenters = new std::vector<int>;
	mpIndexListOfClusterCenters->clear();
	int indexOfBestVector;
	double minDist;
	for (kiT = 0; kiT < mK; ++kiT)
	{
		minDist = 1e10;
		indexOfBestVector = -1;

		for (i = 0; i < mNoVectorsForClustering; ++i)
		{
			if (mgC[kiT] == 0)	continue;

			(*mpF)(i, kiT) += mpKernel->eval(xi[mIndexListForClustering[i]], xi[mIndexListForClustering[i]]);


			if ((*mpF)(i, kiT) + mgC[kiT] < minDist)
			{
				// if the vector is already in the list use another one
				bool bAlreadyInList = false;
				for (q = 0; q < kiT; ++q)
					if ((*mpIndexListOfClusterCenters)[q] == mIndexListForClustering[i]) bAlreadyInList = true;

				if (!bAlreadyInList)
				{
					minDist = (*mpF)(i, kiT) + mgC[kiT];
					indexOfBestVector = i;
				}
			}
		}

		mpIndexListOfClusterCenters->push_back(mIndexListForClustering[indexOfBestVector]);
	}

	return mpIndexListOfClusterCenters;
}

