/*! \file	GenModRM.cpp
	
 * \brief	Evolutionary Aglorithm Generator with Local PCA (RM-MEDA)
 * \brief  "RM-MEDA: A Regularity Model-Based Multiobjective Estimation of Distribution Algorithm". IEEE Transaction on Evolutionary Computation
	
 * \author Aimin ZHOU
 * \author Department of Computer Science,
 * \author University of Essex, 
 * \author Colchester, CO4 3SQ, U.K
 * \author azhou@essex.ac.uk
 *
 * Copyright (c) 2005, 2006, 2007, Aimin ZHOU
 *
 * \date	Nov.29 2005 make great changes: noise, border checking
 * \date	Apr.10 2006 redesign
 * \date	Jul.18 2006 add quadratic models
 * \date	Nov.12 2006 modify to uniform version
 * \date	Jun.26 2006 rename and change Generate()
 * \date	Sep.03 2007 modify the boundary checking procedure, it plays an important role in the algorithm
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
*/
#include <ctime>
#include <list>
#include <vector>
#include <cmath>
#include <float.h>
//#include "LocalPCA.h"
//#include "HCSampler.h"
#include <Array/Array.h>
#include <Rng/GlobalRng.h>
#include <MOO-EALib/RMMEDA.h>

//#if defined(WIN32)
//   #define wxFinite(n) _finite(n)
//#el
#if defined(_LINUX)
#define wxFinite(n) finite(n)
#else
#define wxFinite(n) ((n)==(n))
#endif


// HCSampler.cpp

#include <algorithm>
//#include "HCSampler.h"
//#include "Random.h"

namespace az
{

namespace alg
{
	void LHC(std::vector< std::vector<double> >& rand, std::vector<double>& low, std::vector<double>& upp)
	{
		unsigned int i,j;
		for(i=0; i<rand.size(); i++) 
		{
		  for(j=0; j<rand[i].size(); j++) rand[i][j] = low[i] + (upp[i]-low[i])*(j+Rng::uni(0.0,1.0))/double(rand[i].size());
			std::random_shuffle(rand[i].begin(), rand[i].end());
		}
	}

	void Uniform(std::vector< std::vector<double> >& rand, std::vector<double>& low, std::vector<double>& upp)
	{
		unsigned int i,j;
		for(i=0; i<rand.size(); i++) 
		{
		  for(j=0; j<rand[i].size(); j++) rand[i][j] = Rng::uni(low[i],upp[i]);
		}
	}

} //namespace alg

} //namespace az


namespace az
{
namespace mea
{
namespace gen
{
namespace mod
{

//Local PCA based EDA generator
//constructor
RM::RM()
{
	mDataSize	= mDataDim	= 0;
	pData		= 0;
	mExtension	= 0.2;
	mLatentDim	= 0;
	mMaxCluster	= 0;
}

RM::~RM()
{
	Clear();
}

//clear data pool
void RM::Clear()
{
	unsigned int i;
	if( pData != 0 )
	{
		for( i=0; i<mDataSize; i++ ) delete []pData[i];
		delete []pData;
		pData	  = 0;
		mDataSize = 0;
	}
}


void RM::Generate(	unsigned int    latent, 
			unsigned int    cluster, 
			unsigned int    trainsteps, 
			double 		extension, 
			Array<double>&	xlow,
			Array<double>&	xupp,
			unsigned int    sizenew, 
			PopulationMOO&  offspring, 
			PopulationMOO&  parent) {
  std::vector<std::vector<double> >PopX,OffX;
  std::vector<double> blow, bupp;
  unsigned i,j,s;

  OffX.resize(offspring.size());
  PopX.resize(parent.size());
  blow.resize(xlow.dim(0)); bupp.resize(xupp.dim(0));
  for(s=0;s<offspring[0][0].size();s++){blow[s] = xlow(s); bupp[s] = xupp(s);}
  for(i=0; i<offspring.size(); i++) {
    OffX[i].resize(offspring[0][0].size());
    for (j=0;j<offspring[0][0].size();j++){
      OffX[i][j] = dynamic_cast<ChromosomeT<double>&>(offspring[i][0])[j];
    }  
  }
  for(i=0; i<parent.size(); i++) {
    PopX[i].resize(parent[0][0].size());
    for (j=0;j<parent[0][0].size();j++){
      PopX[i][j] = dynamic_cast<ChromosomeT<double>&>(parent[i][0])[j];
    }
  }
  Generate(latent, cluster, trainsteps,extension, blow, bupp, sizenew, OffX,PopX);

  for(i=0; i<offspring.size(); i++) {
    for (j=0;j<offspring[0][0].size();j++)
      dynamic_cast<ChromosomeT<double>&>(offspring[i][0])[j]=OffX[i][j];
  }
  for(i=0; i<parent.size(); i++) {
    for (j=0;j<parent[0][0].size();j++)
      dynamic_cast<ChromosomeT<double>&>(parent[i][0])[j]=PopX[i][j];
  }

}


void RM::Generate(	unsigned int			latent, 
					unsigned int			cluster, 
					unsigned int			trainsteps, 
					double					extension, 
					std::vector<double>&	xlow,
					std::vector<double>&	xupp,
					unsigned int			sizenew, 
					std::vector< std::vector<double> >& popnew, 
					std::vector< std::vector<double> >& popref)
{
	unsigned int c,i,j,k,noNew;
	if(sizenew<1) return;

	mLatentDim	= latent;
	mMaxCluster	= cluster;
	mTrainSteps	= trainsteps;
	mExtension	= extension;

	//Step 1: assign new data
	Clear();	
	mDataSize	= (unsigned int)popref.size();
	mDataDim	= (unsigned int)popref[0].size();
	pData		= new double*[mDataSize];
	for( i=0; i<mDataSize; i++ ) pData[i] = new double[mDataDim];
	for( i=0; i<mDataSize; i++ ) for( j=0; j<mDataDim; j++ ) pData[i][j] = popref[i][j];

	popnew.resize(sizenew);
	for(i=0; i<sizenew; i++) popnew[i].resize(mDataDim);
	noNew = sizenew;
	
	//Step 2: train with Local PCA
	alg::LocalPCA lpca;
	//alg::Kmeans lpca;
	lpca.Set(mMaxCluster, mDataSize, mDataDim, mLatentDim, mTrainSteps);
	for( i=0; i<mDataSize; i++ ) for( j=0; j<mDataDim; j++ ) lpca.mvX[i][j] = popref[i][j];
	lpca.Train();
	//Step 3: calculate the probability of each cluster, i.e the size to create in each cluster
	// 2 at least in each cluster
	unsigned int nt = 0;
	std::vector<unsigned int> nc(mMaxCluster);
	for(i=0; i<mMaxCluster; i++) { nc[i] = (lpca.mvNo[i] >= 1) ? 2:0; nt += nc[i];}
	double vt = 0.0;
	std::vector<double> vc(mMaxCluster);
	for(i=0; i<mMaxCluster; i++) 
	{ 
		vc[i] = 0.0;
		if(lpca.mvNo[i] >  1)
		{
			vc[i] = 1.0;
			for(j=0; j<mLatentDim; j++) vc[i] *= lpca.mvProMax[i][j] - lpca.mvProMin[i][j];
 		}
		vt += vc[i];
	}
	double ns = noNew - nt + 0.0;
	for(i=0; i<mMaxCluster; i++) { c = (unsigned int)(ns*vc[i]/vt); nc[i] += c; nt += c;}
	while(nt<noNew)
	{
	  i = Rng::discrete(0, mMaxCluster-1);
		if(lpca.mvNo[i]>1) {nc[i]++; nt++;}
	}
#ifdef AZ_MODEL_OUT
	std::ofstream fhand("model.set");
	fhand<<"PCA"<<std::endl;
	fhand<<mMaxCluster<<std::endl;
#endif
	//Step 4: create new trial solutions in each cluster
	unsigned int np=0; 
	double sd;
	for(c=0; c<mMaxCluster; c++)
	{
		std::vector<unsigned int> dataindex;
		if(lpca.mvNo[c] > 0)
		{
			dataindex.resize(lpca.mvNo[c]);j=0;
			for(i=0; i<(unsigned int)lpca.mvIndex.size(); i++) if(lpca.mvIndex[i] == c) dataindex[j++] = i;
		}

		// only one point in the cluster
		if(lpca.mvNo[c] == 1)
		{
			sd  = 0.0;
			for(i=0; i<mDataDim; i++) sd += 0.1*(xupp[i]-xlow[i]);
			sd /= mDataDim;
			for(i=0; i<nc[c]; i++) 
			{
				for( j=0; j<mDataDim; j++ ) 
				{
				  popnew[np+i][j] = lpca.mvMean[c][j] + sd*Rng::gauss(0,1);

					if( wxFinite(popnew[np+i][j]) == 0 )popnew[np+i][j] = Rng::uni(xlow[j],xupp[j]);
					else if(popnew[np+i][j] < xlow[j])	popnew[np+i][j] = 0.5*(xlow[j]+popref[dataindex[0]][j]);
					else if(popnew[np+i][j] > xupp[j])	popnew[np+i][j] = 0.5*(xupp[j]+popref[dataindex[0]][j]);
				}
			}
		}
		// more than one point in the cluster
		else if(lpca.mvNo[c] > 1)
		{
			std::vector< std::vector<double> > t(mLatentDim);
			for(i=0; i<mLatentDim; i++) t[i].resize(nc[c]);
			std::vector<double> low(mLatentDim),upp(mLatentDim);
			for(i=0; i<mLatentDim; i++) 
			{
				low[i] = lpca.mvProMin[c][i]-mExtension*(lpca.mvProMax[c][i]-lpca.mvProMin[c][i]);
				upp[i] = lpca.mvProMax[c][i]+mExtension*(lpca.mvProMax[c][i]-lpca.mvProMin[c][i]);
			}
			alg::LHC(t, low, upp);
			sd = 0.0;
			for(i=mLatentDim; i<mDataDim; i++) sd += fabs(lpca.mvEigenvalue[c][i]);

			sd = sqrt(sd / double(mDataDim-mLatentDim));

			//			if(sd<0.05*fabs(lpca.mvEigenvalue[c][mLatentDim-1])) sd=0.05*fabs(lpca.mvEigenvalue[c][mLatentDim-1]);
			for(i=0; i<nc[c]; i++) for( j=0; j<mDataDim; j++ ) 
			{
			  popnew[np+i][j] = lpca.mvMean[c][j] +sd* Rng::gauss(0,1);
			  for(k=0; k<mLatentDim; k++) popnew[np+i][j] += t[k][i]*lpca.mvEigenvector[c][k][j];
			  if( wxFinite(popnew[np+i][j]) == 0 )  popnew[np+i][j] = Rng::uni(xlow[j],xupp[j]);
			  else if(popnew[np+i][j] < xlow[j])	popnew[np+i][j] = 0.5*(xlow[j]+popref[dataindex[Rng::discrete(0,dataindex.size()-1)]][j]);
			  else if(popnew[np+i][j] > xupp[j])	popnew[np+i][j] = 0.5*(xupp[j]+popref[dataindex[Rng::discrete(0,dataindex.size()-1)]][j]);
			}
		}
		np += nc[c];
#ifdef AZ_MODEL_OUT
		fhand<<lpca.mvProMin[c][0] * lpca.mvEigenvector[c][0][0] + lpca.mvMean[c][0] <<"\t"
			 <<lpca.mvProMin[c][0] * lpca.mvEigenvector[c][0][1] + lpca.mvMean[c][1] <<"\t"
			 <<lpca.mvProMax[c][0] * lpca.mvEigenvector[c][0][0] + lpca.mvMean[c][0] <<"\t"
			 <<lpca.mvProMax[c][0] * lpca.mvEigenvector[c][0][1] + lpca.mvMean[c][1] <<"\t"
			 <<sd<<std::endl;
#endif
	}

#ifdef AZ_MODEL_OUT
	fhand.close();
#endif	
}

} //namespace mod
} //namespace gen
} //namespace mea
} //namespace az


#include <cmath>
//#include "NDSelector.h"
#include <MOO-EALib/PopulationMOO.h>

namespace az
{
namespace mea
{
namespace sel
{
// Dominate
// point1 dominates point2	: 1
// point2 dominates point1	: -1
// non-dominates each other	: 0
int NDS::Dominate(int iA, int iB)
{
	int strictBetter = 0;
	int strictWorse  = 0;
	int better		 = 0;
	int worse		 = 0;
	double tolF		 = 1.0E-3;
	int i;

	for(i=0; i<FDim; i++)
	{
		if(pF[iA][i]<=pF[iB][i]+tolF)
		{
			better++;
			strictBetter += pF[iA][i]<(pF[iB][i]-tolF) ? 1:0;
		}
		if(pF[iA][i]>=pF[iB][i]-tolF)
		{
			worse++;
			strictWorse += pF[iA][i]>(pF[iB][i]+tolF) ? 1:0;
		}
	}

	if(better == FDim && strictBetter > 0) return 1;
	if(worse  == FDim && strictWorse  > 0) return -1;
	return 0;
}

// Sort population by rank value
void NDS::SortRank()
{
	int i,j,dom;
	std::vector< std::vector<int> > domM;	//dominate matrix
	std::vector<int> domV;					//dominate vector

	domM.resize(NData);
	domV.resize(NData);
	rankV.resize(NData);
	for(i=0; i<NData; i++)
	{
		domM[i].resize(NData);
		domV[i] 	= 0;
		rankV[i]	= -1;
	}

	// Set dominate matrix
	for(i=0; i<NData; i++)
	{
		domM[i][i] = 0;
		for(j=i+1; j<NData; j++)
		{
			dom = Dominate(i,j);
			if(dom>0) {domM[j][i] = 1; domV[j]++;}
			else if(dom<0) {domM[i][j] = 1; domV[i]++;}
		}
	}

	// Assign rank
	int NAssign = 0;
	int CRank   = 0;
	int MRank;
	while(NAssign < NData)
	{
		CRank   = CRank + 1;
		MRank	= NData + 10;
		for(i=0; i<NData; i++) if(rankV[i]<0 && domV[i]<MRank ) MRank = domV[i];
		for(i=0; i<NData; i++) if(rankV[i]<0 && domV[i]==MRank) {rankV[i]= CRank; NAssign++;}
		for(i=0; i<NData; i++)
			if(rankV[i] == CRank)
				for(j=0; j<NData; j++) if(rankV[j]<0 && domM[j][i]>0) domV[j]--;
	}

	// Sort by rank
	double *p; int r;
	for(i=0; i<NData; i++)
		for(j=i+1; j<NData; j++) if(rankV[i]>rankV[j])
		{
			r = rankV[i]; 	rankV[i]= rankV[j]; rankV[j]= r;
			p = pF[i]; 		pF[i] 	= pF[j]; 	pF[j] 	= p;
			p = pX[i]; 		pX[i] 	= pX[j]; 	pX[j] 	= p;
			r = id[i];		id[i]   = id[j];	id[j]	= r;
		}
}

// Contribution to the density
double NDS::FDen(double dis)
{
	if(dis<1.0E-10) return 1.0E10;
	else return 1.0/dis;
}

// Sort population by density
void NDS::SortDensity(int iS, int iE)
{
	if(iE == SData || iS == SData-1) return;

	std::vector< double > denV;
	std::vector< int > denI;
	int i,j,k,N,tI,index;

	N = iE-iS+1;
	denV.resize(N);
	denI.resize(N);
	
	while(iE >= SData)
	{
	  N = iE-iS+1;
	  for(i=0; i<N; i++) denV[i] = 0.0;
	   
	  for(k=0; k<FDim; k++)
	  {
		for(i=0; i<N; i++) denI[i] = i;
		for(i=0; i<N; i++) for(j=i+1; j<N; j++) if(pF[denI[j]][k] < pF[denI[i]][k]) 
		{
		  tI = denI[i]; denI[i] = denI[j]; denI[j] = tI; 
		}
		denV[denI[0]] 	+= 1.0E50;
		denV[denI[N-1]] += 1.0E50;
		double range = (pF[denI[N-1]][k] - pF[denI[0]][k] < 1.0E-50) ? 1.0 : (pF[denI[N-1]][k] - pF[denI[0]][k]);
		for(i=1; i<N-1; i++) denV[denI[i]] += (pF[denI[i+1]][k] - pF[denI[i-1]][k])/range;
	  }
	  
	  index = 0;
	  for(i=0; i<N; i++) if(denV[index] > denV[i]) index = i;

	  double *p;
      p = pF[index+iS]; pF[index+iS] = pF[iE]; pF[iE] = p;
      p = pX[index+iS]; pX[index+iS] = pX[iE]; pX[iE] = p;
	  int r;
	  r = id[index+iS];	id[index+iS] = id[iE]; id[iE] = r;
	  
	  iE--;
	}	
}

// Select operator
void NDS::Select(unsigned int size, std::vector< std::vector<double> >& of, std::vector< std::vector<double> >& ox, std::vector< std::vector<double> >& pf, std::vector< std::vector<double> >& px)
{
	int i,j;
	// Check for data number
	FDim 	= (int)pf[0].size();
	XDim 	= (int)px[0].size();
	NData	= (int)pf.size();
	SData	= size;
	id.resize(NData);

	// Copy data.
	pF = new double*[NData];
	pX = new double*[NData];
	for(i=0; i<NData; i++)
	{
		pF[i] = new double[FDim];
		pX[i] = new double[XDim];
		for(j=0; j<FDim; j++) pF[i][j] = pf[i][j];
		for(j=0; j<XDim; j++) pX[i][j] = px[i][j];
		id[i] = i;
	}

	if(SData<NData)
	{
		// Sort population by rank value
		SortRank();

		// Find the subpopulation to sort by density
		int iS, iE;
		iS = iE = 0;
		while(iE < NData && rankV[iE] == rankV[iS]) iE++;
		iE--;
		while(iE < SData-1)
		{
    		iS = iE + 1; iE = iE + 1;
    		while(iE < NData && rankV[iE] == rankV[iS]) iE++;
    		iE--;
		}

		// Sort by density
		SortDensity(iS, iE);
	}
	else
	{
		SData = NData;
	}

	// Create matrix for the return arguments.
	of.resize(SData); for(i=0; i<SData; i++) of[i].resize(FDim);
	ox.resize(SData); for(i=0; i<SData; i++) ox[i].resize(XDim);
	// Copy data to return arguments.
	for(i=0; i<SData; i++)
	{
		for(j=0; j<FDim; j++) of[i][j] = pF[i][j];
		for(j=0; j<XDim; j++) ox[i][j] = pX[i][j];
	}

	// Free space
	for(i=0; i<NData; i++)
	{
		delete []pF[i];
		delete []pX[i];
	}
	delete []pF;
	delete []pX;
}

  void NDS::Select(unsigned int size, unsigned int numberOfObjectives, PopulationMOO &offspring,PopulationMOO &total){
    std::vector<std::vector<double> >PopF,PopX, tmpF,tmpX;
    unsigned i,j;
    tmpX.resize(size+size);
    tmpF.resize(size+size);
    PopX.resize(size);
    PopF.resize(size);
    for(i=0; i<total.size(); i++) {
      tmpX[i].resize(total[0][0].size());
      tmpF[i].resize(numberOfObjectives);
      for (j=0;j<total[0][0].size();j++)
	tmpX[i][j] = dynamic_cast<ChromosomeT<double>&>(total[i][0])[j];
      for (j=0;j<numberOfObjectives;j++)
	tmpF[i][j] = total[i].getMOOFitness(j);
    }
    for(i=0; i<offspring.size(); i++) {
      PopX[i].resize(offspring[0][0].size());
      PopF[i].resize(numberOfObjectives);
      for (j=0;j<offspring[0][0].size();j++)
	PopX[i][j] = dynamic_cast<ChromosomeT<double>&>(offspring[i][0])[j];
      for (j=0;j<numberOfObjectives;j++)
	PopF[i][j] = offspring[i].getMOOFitness(j);
    }
    Select(size, PopF, PopX, tmpF, tmpX);
    for(i=0; i<offspring.size(); i++) {
      for (j=0;j<offspring[0][0].size();j++)
	dynamic_cast<ChromosomeT<double>&>(offspring[i][0])[j]=PopX[i][j];
      for (j=0;j<numberOfObjectives;j++)
	offspring[i].setMOOFitness(j,PopF[i][j]);
    }
  }

void NDS::Select(unsigned int size, std::vector< unsigned int >& ids, std::vector< std::vector<double> >& pf, std::vector< std::vector<double> >& px)
{
	int i,j;
	// Check for data number
	FDim 	= (int)pf[0].size();
	XDim 	= (int)px[0].size();
	NData	= (int)pf.size();
	SData	= size;
	id.resize(NData);

	// Copy data.
	pF = new double*[NData];
	pX = new double*[NData];
	for(i=0; i<NData; i++)
	{
		pF[i] = new double[FDim];
		pX[i] = new double[XDim];
		for(j=0; j<FDim; j++) pF[i][j] = pf[i][j];
		for(j=0; j<XDim; j++) pX[i][j] = px[i][j];
		id[i] = i;
	}

	if(SData<NData)
	{
		// Sort population by rank value
		SortRank();

		// Find the subpopulation to sort by density
		int iS, iE;
		iS = iE = 0;
		while(iE < NData && rankV[iE] == rankV[iS]) iE++;
		iE--;
		while(iE < SData-1)
		{
    		iS = iE + 1; iE = iE + 1;
    		while(iE < NData && rankV[iE] == rankV[iS]) iE++;
    		iE--;
		}

		// Sort by density
		SortDensity(iS, iE);
	}
	else
	{
		SData = NData;
	}
	
	ids.resize(SData);
	for(i=0; i<SData; i++) ids[i] = id[i];

	// Free space
	for(i=0; i<NData; i++)
	{
		delete []pF[i];
		delete []pX[i];
	}
	delete []pF;
	delete []pX;

}
}
}
}




// Matrix.cpp

#include <float.h>
#include <math.h>
//#include "Matrix.h"

//#if defined(WIN32)
//    #define wxFinite(n) _finite(n)
//#elif defined(_LINUX)
//    #define wxFinite(n) finite(n)
//#else
    #define wxFinite(n) ((n)==(n))
//#endif

namespace az
{

namespace alg
{

//constructor
Matrix::Matrix(unsigned int row, unsigned int col)
	:mRow(row), mCol(col)
{
	if(mRow* mCol > 0)
	{
		pData = new double[mRow * mCol];
		for(unsigned int i=0; i<mRow*mCol; i++) pData[i]=0.0;
	}
	else 
		pData = 0;
}

//constructor
Matrix::Matrix(const Matrix& mat)
	:mRow(mat.mRow), mCol(mat.mCol)
{
	if(mRow* mCol > 0)
	{
		pData = new double[mRow * mCol];
		memcpy(pData, mat.pData, mRow*mCol*sizeof(double));
	}
	else 
		pData = 0;
}

//destructor
Matrix::~Matrix()
{
	if(pData) delete []pData;
}

//reset the matrix size
Matrix& Matrix::Resize(unsigned int row, unsigned int col)
{
	if(row != mRow || col != mCol)
	{
		if(pData) delete []pData;
		mRow	= row;
		mCol	= col;
		pData = new double[mRow*mCol];
		for(unsigned int i=0; i<mRow*mCol; i++)	pData[i] = 0.0;
	}
	return *this;
}

//create an identity matrix
Matrix& Matrix::Identity(unsigned int size)
{
	(*this).Resize(size, size);
	for(unsigned int i=0; i<size; i++)	(*this)(i, i) = 1.0;
	return *this;
}

//get an element
double& Matrix::operator()(unsigned int row, unsigned int col)
{
	CHECK(row < mRow && col < mCol, "Matrix::(row,col)");
	return pData[col*mRow + row];
}

//reset to another matrix
Matrix& Matrix::operator= (const Matrix& mat)
{
	Resize(mat.mRow, mat.mCol);
	for(unsigned int i=0; i<mRow*mCol; i++) pData[i] = mat.pData[i];
	return *this;
}

//get a row
FVECTOR& Matrix::Row(unsigned int row, FVECTOR& value)
{
	CHECK(row < mRow, "Matrix::Row()");
	value.resize(mCol);
	for(unsigned int i=0; i<mCol; i++) value[i] = (*this)(row, i);
	return value;
}

//get a column
FVECTOR& Matrix::Column(unsigned int col, FVECTOR& value)
{
	CHECK(col < mCol, "Matrix::Column()");
	value.resize(mRow);
	for(unsigned int i=0; i<mRow; i++) value[i] = (*this)(i, col);
	return value;
}

//get a sub-matrix except a row and a column
Matrix& Matrix::Sub(unsigned int row, unsigned int col, Matrix& mat)
{
	CHECK(row < mRow && mRow > 1 && col < mCol && mCol > 1, "Matrix::Sub()");

	mat.Resize(mRow-1, mCol-1);

	unsigned int i,j;

	std::vector<unsigned int> ir(mRow-1), jc(mCol-1);
	for(i=0; i<mRow-1; i++) ir[i] = i<row ? i : i+1;
	for(j=0; j<mCol-1; j++) jc[j] = j<col ? j : j+1;

	for(i=0; i<mat.RowSize(); i++)
		for(j=0; j<mat.ColSize(); j++)
			mat(i,j) = (*this)(ir[i],jc[j]);
	return mat;
}

//calculate the determinant of a square matrix
double Matrix::Det()
{
	CHECK(mRow == mCol, "Matrix::Det()");

	// compute the determinant by definition
	//if(mRow == 1) return (*this)(0,0);

	//if(mRow == 2) return (*this)(0,0)*(*this)(1,1) - (*this)(0,1)*(*this)(1,0);
	//
	//if(mRow == 3) return  (*this)(0,0) * (*this)(1,1) * (*this)(2,2) 
	//					 +(*this)(0,1) * (*this)(1,2) * (*this)(2,0) 
	//					 +(*this)(0,2) * (*this)(1,0) * (*this)(2,1)
	//					 -(*this)(0,0) * (*this)(1,2) * (*this)(2,1) 
	//					 -(*this)(0,1) * (*this)(1,0) * (*this)(2,2) 
	//					 -(*this)(0,2) * (*this)(1,1) * (*this)(2,0);
	//
	//double tmp = 0.0;
	//Matrix submatrix;
	//for(unsigned int i=0; i<mCol; i++)
	//	tmp += (i % 2 == 0 ? 1:-1) * (*this)(0,i) * (*this).Sub(0, i, submatrix).Det();
	//return tmp;

	// modified Nov.19 2006
	Matrix mat = *this;
	std::vector<unsigned int> indx(mCol); 
	double d;
	LUdcmp(mat, indx, d);
	for(unsigned int i=0; i<mCol; i++) d *= mat(i,i);
	return d;
}

//translate the matrix
Matrix& Matrix::Trans()
{
	if(mRow<2 || mCol<2) 
	{
		std::swap((*this).mCol, (*this).mRow);
		return *this;
	}
	Matrix mat(*this);
	std::swap((*this).mCol, (*this).mRow);
	for(unsigned int i=0; i<mRow; i++)
		for(unsigned int j=0; j<mCol; j++)
			(*this)(i, j) = mat(j, i);				
	return *this;
}

//inverse the matrix
Matrix& Matrix::Inv()
{
	CHECK(mRow == mCol, "Matrix::Inv()");

	//Adjoint method
	//unsigned int i,j;
	//double det=0.0;
	//Matrix adjmat(mRow,mCol), submatrix;
	//for(i=0; i<mRow; i++)
	//	for(j=0; j<mCol; j++)
	//		adjmat(i, j) = ((i + j)%2 == 0 ? 1 : -1)*(*this).Sub(j, i, submatrix).Det();
	//for(i=0; i<mRow; i++) det += (*this)(0,i)*adjmat(i,0);
	//*this = adjmat; this->Divide(det);

	//LU decomposition version
	Matrix lumat(*this);
	std::vector<unsigned int> indx(mRow);
	std::vector<double> col(mRow);
	double d;
	unsigned i,j;
	LUdcmp(lumat,indx,d);
	for(j=0; j<mRow; j++)
	{
		for(i=0; i<mRow; i++) col[i]=0.0;
		col[j]=1.0;
		LUbksb(lumat,indx,col);
		for(i=0; i<mRow; i++) (*this)(i,j)=col[i];
	}
	
	return *this;
}

//calculate the eigenvalue and egienvectors
void Matrix::Eig(FVECTOR& eigvalue, Matrix& eigvector)
{
	CHECK(mRow == mCol, "Matrix::Eig()");
	eigvalue.resize(mRow);
	FVECTOR interm;
	interm.resize(mRow);
	eigvector = *this;
	
	tred2(eigvalue, interm, eigvector);
	tqli(eigvalue, interm, eigvector);
	//Eigenvector in columns
	Sort(eigvalue, eigvector);
}

//multiply a matrix
Matrix& Matrix::Multiply(Matrix& mat, Matrix& result)
{
	unsigned int i,j,k;

	CHECK(mCol == mat.RowSize(), "Matrix::Myltiply()");
	
	result.Resize((*this).RowSize(), mat.ColSize());
	
	for(i=0; i<result.RowSize(); i++)
		for(j=0; j<result.ColSize(); j++)
		{
			result(i, j) = 0;
			for(k=0; k<mat.RowSize(); k++)
				result(i, j) += (*this)(i, k) * mat(k, j);
		}
	return result;
}

//left multiply a vector
FVECTOR& Matrix::LeftMultiply(FVECTOR& vec, FVECTOR& result)
{
	CHECK(mRow == vec.size(), "Matrix::LeftMultiply()");
	result.resize((*this).ColSize());
	for(unsigned int i=0; i<(*this).ColSize(); i++)
	{
		result[i] = 0;
		for(unsigned int j=0; j<(*this).RowSize(); j++)
			result[i] += (*this)(j, i) * vec[j];
	}
	return result;
}

//right multiply a vector
FVECTOR& Matrix::RightMultiply(FVECTOR& vec, FVECTOR& result)
{
	CHECK(mCol == vec.size(), "Matrix::RightMultiply()");
	result.resize((*this).RowSize());
	for(unsigned int i=0; i<(*this).RowSize(); i++)
	{
		result[i] = 0;
		for(unsigned int j=0; j<(*this).ColSize(); j++)
			result[i] += (*this)(i, j) * vec[j];
	}
	return result;
}

//divide a scalar
Matrix& Matrix::Divide(double sca)
{
	unsigned int i,j;
	for(i=0; i<(*this).RowSize(); i++)
		for(j=0; j<(*this).ColSize(); j++)
			(*this)(i, j) /= sca;
	return *this;
}

//get the mean of all columns
FVECTOR& Matrix::ColMean(FVECTOR& mean)
{
	mean.resize((*this).RowSize());
	unsigned int i,j;
	for(i=0; i<mean.size(); i++)
	{
		mean[i] = 0.0;
		for(j=0; j<(*this).ColSize(); j++)
			mean[i] += (*this)(i, j);
		mean[i] /= (double) (*this).ColSize() ;
	}
	return mean;
}

//get the mean of all rows
FVECTOR& Matrix::RowMean(FVECTOR& mean)
{
	mean.resize((*this).ColSize());
	unsigned int i,j;
	for(i=0; i<mean.size(); i++)
	{
		mean[i] = 0;
		for(j=0; j<(*this).RowSize(); j++)
			mean[i] += (*this)(j, i);
		mean[i] /= (double) (*this).RowSize() ;
	}	
	return mean;
}

//get standard variation of all columns
FVECTOR& Matrix::ColStd(FVECTOR& std)
{
	std.resize((*this).RowSize());
	FVECTOR mean;
	ColMean(mean);
	unsigned int i,j;
	for(i=0; i<std.size(); i++)
	{
		std[i] = 0.0;
		for(j=0; j<(*this).ColSize(); j++)
			std[i] += ((*this)(i,j) - mean[i])*((*this)(i,j) - mean[i]);
		std[i] /= double((*this).ColSize() -1);
		std[i] = sqrt(std[i]);
	}
	return std;
}

//get standard variation of all rows
FVECTOR& Matrix::RowStd(FVECTOR& std)
{
	std.resize((*this).ColSize());
	FVECTOR mean;
	RowMean(mean);
	unsigned int i,j;
	for(i=0; i<std.size(); i++)
	{
		std[i] = 0.0;
		for(j=0; j<(*this).RowSize(); j++)
			std[i] += ((*this)(j,i) - mean[i])*((*this)(j,i) - mean[i]);
		std[i] /= double((*this).RowSize() -1);
		std[i] = sqrt(std[i]);
	}
	return std;
}

//subtract a row vector
Matrix& Matrix::RowSub(FVECTOR& value)
{
	unsigned int i,j;
	for(i=0; i<(*this).RowSize(); i++)
		for(j=0; j<(*this).ColSize(); j++)
			(*this)(i, j) -= value[j];
	return *this;
}

//subtract a column vector
Matrix& Matrix::ColSub(FVECTOR& value)
{
	unsigned int i,j;
	for(i=0; i<(*this).RowSize(); i++)
		for(j=0; j<(*this).ColSize(); j++)
			(*this)(i, j) -= value[i];
	return *this;
}

//read a matrix
std::istream& operator>>(std::istream& is, Matrix& mat)
{
	unsigned int row,col;
	is>>row>>col;
	mat.Resize(row,col);
	for(unsigned int i=0; i<mat.mRow; i++)
		for(unsigned int j=0; j<mat.mCol; j++)
			is>>mat(i, j);
	return is;
}

//write a matrix
std::ostream& operator<< (std::ostream& os, Matrix& mat)
{
	os<<mat.RowSize()<<"\t"<<mat.ColSize()<<std::endl;
	os.setf(std::ios::right|std::ios::scientific);
	os.precision(5);   
	for(unsigned int i=0; i<mat.mRow; i++)
	{
		for(unsigned int j=0; j<mat.mCol; j++)
			os<<mat(i, j)<<"\t";
		os<<std::endl;
	}
	return os;
}

//solve A X = b, mat if the decomposition of A
void LUbksb(Matrix& mat, std::vector<unsigned int>& indx, std::vector<double>& b)
{
	int i,ii=0,ip,j, n=mat.RowSize();
	double sum;

	for(i=0;i<n;i++) 
	{
		ip=indx[i];
		sum=b[ip];
		b[ip]=b[i];
		if(ii!=0)
			for(j=ii-1;j<=i;j++) sum -= mat(i,j)*b[j];
		else if(sum!=0.0) ii=i+1;
		b[i]=sum;
	}

	for(i=n-1;i>=0;i--) 
	{
		sum=b[i];
		for(j=i+1;j<n;j++) sum -= mat(i,j)*b[j];
		b[i]=sum/mat(i,i);
	}
}

//LU decomposition 
void LUdcmp(Matrix& mat, std::vector<unsigned int>& indx, double& d)
{
	const double TINY = 1.0e-20;
	unsigned int i,imax=0,j,k;
	double big,dum,sum,temp;
	unsigned int n = mat.RowSize();
	std::vector<double> vv(n);

	d=1.0;
	for(i=0;i<n;i++) 
	{
		big=0.0;
		for(j=0;j<n;j++)
			if((temp=fabs(mat(i,j)))>big) big=temp;
		//if(big == 0.0) std::cout<<"error"<<std::endl;
		vv[i]=1.0/big;
	}

	for(j=0;j<n;j++) 
	{
		for(i=0;i<j;i++) 
		{
			sum = mat(i,j);
			for(k=0;k<i;k++) sum -= mat(i,k)*mat(k,j);
			mat(i,j) = sum;
		}
		big=0.0;
		for (i=j;i<n;i++) 
		{
			sum=mat(i,j);
			for(k=0;k<j;k++)
				sum -= mat(i,k)*mat(k,j);
			mat(i,j) = sum;
			if((dum=vv[i]*fabs(sum)) >= big) 
			{
				big=dum;
				imax=i;
			}
		}
		if(j != imax) 
		{
			for(k=0;k<n;k++) 
				std::swap(mat(imax,k), mat(j,k));
			d = -d;
			vv[imax]=vv[j];
		}
		indx[j]=imax;
		if (mat(j,j) == 0.0) mat(j,j)=TINY;
		if (j != n-1) 
		{
			dum=1.0/mat(j,j);
			for (i=j+1;i<n;i++) mat(i,j) *= dum;
		}
	}
}

//Householder reduction of Matrix a to tridiagonal form.
//
// Algorithm: Martin et al., Num. Math. 11, 181-195, 1968.
// Ref: Smith et al., Matrix Eigensystem Routines -- EISPACK Guide
// Springer-Verlag, 1976, pp. 489-494.
// W H Press et al., Numerical Recipes in C, Cambridge U P,
// 1988, pp. 373-374. 
void Matrix::tred2(FVECTOR& eigenvalue, FVECTOR& interm, Matrix& eigenvector)
{
	double scale, hh, h, g, f;
	int k, j, i,l;

	for(i = mRow-1; i >= 1; i--)
	{
		l = i - 1;
		h = scale = 0.0;
		if(l > 0)
		{
			for (k = 0; k <= l; k++)
				scale += fabs(eigenvector(i, k));
			if (scale == 0.0)
				interm[i] = eigenvector(i, l);
			else
			{
				for (k = 0; k <= l; k++)
				{
					eigenvector(i, k) /= scale;
					h += eigenvector(i, k) * eigenvector(i, k);
				}
				f = eigenvector(i, l);
				g = f>0 ? -sqrt(h) : sqrt(h);
				interm[i] = scale * g;
				h -= f * g;
				eigenvector(i, l) = f - g;
				f = 0.0;
				for (j = 0; j <= l; j++)
				{
					eigenvector(j, i) = eigenvector(i, j)/h;
					g = 0.0;
					for (k = 0; k <= j; k++)
						g += eigenvector(j, k) * eigenvector(i, k);
					for (k = j+1; k <= l; k++)
						g += eigenvector(k, j) * eigenvector(i, k);
					interm[j] = g / h;
					f += interm[j] * eigenvector(i, j);
				}
				hh = f / (h + h);
				for (j = 0; j <= l; j++)
				{
					f = eigenvector(i, j);
					interm[j] = g = interm[j] - hh * f;
					for (k = 0; k <= j; k++)
						eigenvector(j, k) -= (f * interm[k] + g * eigenvector(i, k));
				}
			}
		}
		else
			interm[i] = eigenvector(i, l);
		eigenvalue[i] = h;
	}
	eigenvalue[0] = 0.0	;	
	interm[0] = 0.0;
	for(i = 0; i < (int)mRow; i++)
	{
		l = i - 1;
		if(eigenvalue[i])
		{
			for (j = 0; j <= l; j++)
			{
				g = 0.0;
				for (k = 0; k <= l; k++)
					g += eigenvector(i, k) * eigenvector(k, j);
				for (k = 0; k <= l; k++)
					eigenvector(k, j) -= g * eigenvector(k, i);
			}
		}
		eigenvalue[i] = eigenvector(i, i);
		eigenvector(i, i) = 1.0;
		for (j = 0; j <= l; j++)
			eigenvector(j, i) = eigenvector(i, j) = 0.0;
	}
}

#define SIGN(a, b) ((b) < 0 ? -fabs(a) : fabs(a))
//Tridiagonal QL algorithm -- Implicit 
void Matrix::tqli(FVECTOR& eigenvalue, FVECTOR& interm, Matrix& eigenvector)
{
	int m, l, iter, i, k;
	double s, r, p, g, f, dd, c, b;
	
	for (i = 1; i < (int)mRow; i++)
		interm[i-1] = interm[i];
	interm[mRow-1] = 0.0;
	for (l = 0; l < (int)mRow; l++)
	{
		iter = 0;
		do
		{
			if (iter++ > 100) break;

			for (m = l; m < (int)mRow-1; m++)
			{
				dd = fabs(eigenvalue[m]) + fabs(eigenvalue[m+1]);
				if (fabs(interm[m]) + dd == dd || !wxFinite(dd)) break;
			}
			if (m != l)
			{
				//if (iter++ == 30) erhand("No convergence in TLQI.");
				g = (eigenvalue[l+1] - eigenvalue[l]) / (2.0 * interm[l]);
				r = sqrt((g * g) + 1.0);
				g = eigenvalue[m] - eigenvalue[l] + interm[l] / (g + SIGN(r, g));
				s = c = 1.0;
				p = 0.0;
				for (i = m-1; i >= l; i--)
				{
					f = s * interm[i];
					b = c * interm[i];
					if (fabs(f) >= fabs(g))
					{
						c = g / f;
						r = sqrt((c * c) + 1.0);
						interm[i+1] = f * r;
						c *= (s = 1.0/r);
					}
					else
					{
						s = f / g;
						r = sqrt((s * s) + 1.0);
						interm[i+1] = g * r;
						s *= (c = 1.0/r);
					}
					g = eigenvalue[i+1] - p;
					r = (eigenvalue[i] - g) * s + 2.0 * c * b;
					p = s * r;
					eigenvalue[i+1] = g + p;
					g = c * r - b;
					for (k = 0; k < (int)mRow; k++)
					{
						f = eigenvector(k, i+1);
						eigenvector(k, i+1)= s * eigenvector(k, i) + c * f;
						eigenvector(k, i) = c * eigenvector(k, i) - s * f;
					}
				}
				eigenvalue[l] = eigenvalue[l] - p;
				interm[l] = g;
				interm[m] = 0.0;
			}
		}  while (m  != l);
	}
}

//sort the eigenvalue by decreasing order
void Matrix::Sort(FVECTOR& eigenvalue, Matrix& eigenvector)
{
	unsigned int i,j,k;
	unsigned int m = (unsigned int)eigenvalue.size();

	//repair 
	for(i=0; i<m; i++) if(!wxFinite(eigenvalue[i])) eigenvalue[i] = 1.0E-100;
	for(i=0; i<m; i++) 
	{
		j=0;
		for(k=0; k<m; k++) if(!wxFinite(eigenvector(k, i))) j = 1;
		if(j>0) for(k=0; k<m; k++) eigenvector(k, i) = 1.0/sqrt(m+0.0);
	}

	for(i=0; i<m; i++) if(eigenvalue[i]<0)
	{
		eigenvalue[i] *= -1;
		for(j=0; j<m; j++) eigenvector(j,i) *= -1;
	}
	for(i=0; i<m - 1; i++)
		for(j=i+1; j<m; j++)
			if(eigenvalue[j] > eigenvalue[i])
			{
				std::swap(eigenvalue[j], eigenvalue[i]);
				for(k=0; k<m; k++)
					std::swap(eigenvector(k, j), eigenvector(k, i));
			}
}

} //namespace alg

} //namespace az

// PCA.cpp

#include <cmath>
#include <fstream>
//#include "PCA.h"

namespace az
{

namespace alg
{

//constructor
PCA::PCA()
{
	pData = 0;
}

//constructor
PCA::PCA(alg::Matrix& data)
{
	pData = &data;
}

//destructor
PCA::~PCA()
{
	pData = 0;
}

//initialize colvariance matrix to be an identity
void PCA::Initialize(unsigned int dim)
{
	mCov.Identity(dim);
	mEigenvector.Identity(dim);
	mEigenvalue.resize(dim);
	for(unsigned int i=0; i<dim; i++) mEigenvalue[i] = 1.0;
}

//get the range of the projections in a dimension 
void PCA::PrimaryRange(unsigned int dim, double& min, double& max)
{
	unsigned int i, j;
	double tmp;
	min = 1.0e20; max = -1.0e20;
	for(i=0; i<(*pData).ColSize(); i++)
	{
		tmp = 0.0;
		for(j=0; j<mMean.size(); j++)
			tmp += ((*pData)(j, i) - mMean[j])*mEigenvector(j, dim);
		if(tmp < min) min = tmp;
		if(tmp > max) max = tmp;
	}
}	
//calculate the mean, eigenvalue and eigenvectors 
void PCA::Train()
{ 
	//Step 1: calculate the mean
	(*pData).ColMean(mMean);

	//Step 2: get the covariance matrix
	alg::Matrix datatmp = (*pData);
	datatmp.ColSub(mMean);
	alg::Matrix datatmp1 = datatmp;
	datatmp1.Trans();
	datatmp.Multiply(datatmp1, mCov);
	mCov.Divide(double( (*pData).ColSize()>1 ? (*pData).ColSize()-1.0 : (*pData).ColSize() ));

	//Step 3: calculate the eigenvalue and eigenvector
	mCov.Eig(mEigenvalue, mEigenvector);
}

//write results into a stream
std::ostream& operator<<(std::ostream& os, PCA& pca)
{
	pca.Write(os);
	return os;
}

//write results into a stream
void PCA::Write( std::ostream& os )
{
	unsigned int i;
	//datas
	os<<(*pData).RowSize()<<"\t"<<(*pData).ColSize()<<std::endl;
	os<<(*pData)<<std::endl<<std::endl;
	//cov
	os<<"covariance"<<std::endl;
	os<<mCov<<std::endl<<std::endl;
	//mean
	os<<"mean"<<std::endl;
	for(i=0; i<mMean.size(); i++)
		os<<mMean[i]<<"\t";
	os<<std::endl<<std::endl;
	//eigenvalue
	os<<"eigenvalue"<<std::endl;
	for(i=0; i<mEigenvalue.size(); i++)
		os<<mEigenvalue[i]<<"\t";
	os<<std::endl<<std::endl;
	//eigenvector
	os<<"eigenvector"<<std::endl;
	os<<mEigenvector<<std::endl;
	//range
	os<<"range in primary dimension"<<std::endl;
	double min, max;
	PrimaryRange(0, min, max);
	os<<min<<"\t"<<max<<std::endl;
}

//write results into a file
void PCA::Write(std::string& name)
{
	std::ofstream os;
	os.open(name.c_str());
	
	Write(os);

	os.close();
}

//write results into a file
void PCA::Write( const char* name )
{
	std::string str = std::string(name);
	Write(str);
}

} //namespace alg

} //namespace az


// LocalPCA.cpp

#include <algorithm>
#include <cmath>
//#include "Random.h"
//#include "LocalPCA.h"

namespace az
{

namespace alg
{

void LocalPCA::Train()
{
	unsigned int	c,	// cluster index
					m,	// row index
					n;	// col index
	unsigned int i;

	// initialize the PI matrix
	mvPI.resize(mNClu);
	for(c=0; c<mNClu; c++)
	{
		mvPI[c].resize(mDX);
		for(m=0; m<mDX; m++) mvPI[c][m].resize(mDX);
		for(m=0; m<mDX; m++) for(n=m; n<mDX; n++) mvPI[c][m][n] = mvPI[c][m][n] = (m==n) ? 1.0 : 0.0;
	}

	// initialize the center points
	std::vector<unsigned int> index(mNX);
	for(i=0; i<mNX; i++) index[i] = i;
	for(c=0; c<mNClu; c++)
	{
	  std::swap(index[c], index[Rng::discrete(c, mNX-1)]);
		mvMean[c] = mvX[index[c]];
	}

	// trainning
	unsigned int failupdate = 0;
	double	dis, mindis, pro, err;
	std::vector<double> meanold(mDX);
	while( mIter++ < mMaxIter && failupdate<mNClu )
	{
		// partition trainning data
		for(c=0; c<mNClu; c++) mvNo[c] = 0;	// number point in each cluster
		for(m=0; m<mNX; m++)
		{
			mindis		= 1.0E100;
			mvIndex[m]	= 0;				// to which cluster it belongs
			for(c=0; c<mNClu; c++)
				if( (dis = Distance(m,c))< mindis)
				{
					mindis		= dis;
					mvIndex[m]	= c;
				}
			mvNo[mvIndex[m]]++;
		}

		// update parameters
		failupdate	= 0;
		for(c=0; c<mNClu; c++) 
		{
			// save old mean
			meanold = mvMean[c];

			// no data assigned to cluster c
			if(mvNo[c] < 1)
			{
			  mvMean[c] = mvX[Rng::discrete(0, mNX-1)];
				for(m=0; m<mDX; m++) 
				{
					mvPI[c][m][m] = 1.0;
					for(n=m+1; n<mDX; n++) mvPI[c][m][n] = mvPI[c][n][m] = 0.0;
				}
			}
			// only one data assigned to cluster c
			else if(mvNo[c] == 1)
			{
				for(m=0; m<mNX; m++) if(mvIndex[m] == c) break;
				mvMean[c] = mvX[m];
				for(m=0; m<mDX; m++) 
				{
					mvPI[c][m][m] = 1.0;
					for(n=m+1; n<mDX; n++) mvPI[c][m][n] = mvPI[c][n][m] = 0.0;
				}
			}
			// more than one data assigned to cluster c
			else if(mvNo[c] > 1)
			{
				// find data index
				index.resize(mvNo[c]); n=0;
				for(m=0; m<mNX; m++) if(mvIndex[m] == c) index[n++] = m;
				
				// calculate the eigenvalues and eigenvectors
				Eigen(mvMean[c], mvEigenvalue[c], mvEigenvector[c], index);

				// calculate pi
				for(m=0; m<mDX; m++) for(n=0; n<mDX; n++)
				{
					mvPI[c][m][n] = 0.0;
					for(i=mDLat; i<mDX; i++) mvPI[c][m][n] += mvEigenvector[c][i][m]*mvEigenvector[c][i][n];
				}
			}// else if(mvNo[c] > 1)

			// calculate the error
			err = 0.0;
			for(m=0; m<mDX; m++) err += (meanold[m] - mvMean[c][m])*(meanold[m] - mvMean[c][m]);
			failupdate += (sqrt(err)<mErrTol) ? 1:0;
		}// for(c=0; c<mNClu; c++)
	}// while( mIter++ < mMaxIter && failupdate<mNClu )

	// calculate the projection
	for(c=0; c<mNClu; c++) if(mvNo[c] > 1)
	{
		// copy data
		index.resize(mvNo[c]); n=0;
		for(m=0; m<mNX;   m++) if(mvIndex[m] == c) index[n++] = m;
		for(m=0; m<mDLat; m++) {mvProMin[c][m] = 1.0E100; mvProMax[c][m] = -1.0E100;}
		for(m=0; m<index.size(); m++) for(n=0; n<mDLat; n++)
		{
			pro = 0.0;
			for(i=0; i<mDX; i++) pro += (mvX[index[m]][i]-mvMean[c][i])*mvEigenvector[c][n][i];
			if(pro>mvProMax[c][n]) mvProMax[c][n] = pro;
			if(pro<mvProMin[c][n]) mvProMin[c][n] = pro;
		}
	}
}

double LocalPCA::Distance(unsigned int m, unsigned int c)
{
	unsigned int i,j;
	double dis = 0.0;
	std::vector<double> tmpdis(mDX);
	for(i=0; i<mDX; i++)
	{
		tmpdis[i] = 0.0;
		for(j=0; j<mDX; j++) tmpdis[i] += (mvX[m][j]-mvMean[c][j])*mvPI[c][j][i];
	}
	for(i=0; i<mDX; i++) dis += tmpdis[i]*(mvX[m][i]-mvMean[c][i]);
	return dis;
}

} //namespace alg

} //namespace az

// Model.cpp
#ifdef AZ_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#else
//#include "Matrix.h"
#endif
#include <iomanip>
//#include "Model.h"

namespace az
{

//!\brief	alg namespace, contains algorithms
namespace alg
{

void ModelRM::Set(unsigned int nclu, unsigned int nx, unsigned int dx, unsigned int dlat, unsigned int maxiter, double errtol)
{
	unsigned int i,j;
	
	mNClu	= nclu;
	mNX		= nx;
	mDX		= dx; 
	mDLat	= dlat;
	mMaxIter= maxiter;
	mErrTol	= errtol;
	mIter	= 0;

	mvIndex.resize(mNX);
	mvNo.resize(mNClu);
	
	mvX.resize(mNX);
	for(i=0; i<mNX; i++) mvX[i].resize(mDX);

	mvMean.resize(mNClu);
	for(i=0; i<mNClu; i++) mvMean[i].resize(mDX);

	mvEigenvalue.resize(mNClu);
	for(i=0; i<mNClu; i++) mvEigenvalue[i].resize(mDX);

	mvProMin.resize(mNClu);
	mvProMax.resize(mNClu);
	for(i=0; i<mNClu; i++)
	{
		mvProMin[i].resize(mDLat);
		mvProMax[i].resize(mDLat);
	}

	mvEigenvector.resize(mNClu);
	for(i=0; i<mNClu; i++) 
	{
		mvEigenvector[i].resize(mDX);
		for(j=0; j<mDX; j++) mvEigenvector[i][j].resize(mDX);
	}
}

void ModelRM::Write(std::string file)
{
	unsigned int i,j,k;
	std::ofstream out(file.c_str());
	out<<std::scientific<<std::setprecision(5);
	
	out<<"Train Steps "<<mIter<<std::endl;

	for(i=0; i<mNClu; i++)
	{
		out<<std::endl<<"===========cluster "<<i<<"==========="<<std::endl;
		out<<"data"<<std::endl;
		for(j=0;j<mNX; j++) if(mvIndex[j]==i)
		{
			for(k=0; k<mDX; k++) out<<mvX[j][k]<<"\t";
			out<<std::endl;
		}
		out<<"mean"<<std::endl;	for(j=0; j<mDX; j++) out<<mvMean[i][j]<<"\t";out<<std::endl;
		out<<"eigenvalue"<<std::endl;	for(j=0; j<mDX; j++) out<<mvEigenvalue[i][j]<<"\t";out<<std::endl;
		out<<"eigenvector"<<std::endl;
		for(j=0; j<mDX; j++)
		{
			for(k=0; k<mDX; k++) out<<mvEigenvector[i][j][k]<<"\t"; out<<std::endl;
		}
		//out<<"PI"<<std::endl;
		//for(j=0; j<mDX; j++)
		//{
		//	for(k=0; k<mDX; k++) out<<mvPI[i][j][k]<<"\t"; out<<std::endl;
		//}
	}

	out.close();
}

void ModelRM::Eigen(std::vector<double>& mean, std::vector<double>& eva, std::vector< std::vector<double> >& eve, std::vector< unsigned int >& index)
{
	unsigned int i,j,k;
	//calculate the mean
	for(i=0; i<mDX; i++)
	{
		mean[i] = 0.0;
		for(j=0; j<index.size(); j++) mean[i] += mvX[index[j]][i];
		mean[i] /= double(index.size());
	}

	//calulate the covariance
	std::vector< std::vector<double> > cov(mDX); for(i=0; i<mDX; i++) cov[i].resize(mDX);
	for(i=0; i<mDX; i++)
	{
		for(j=i; j<mDX; j++)
		{
			cov[i][j]  = 0.0;
			for(k=0; k<index.size(); k++) cov[i][j] += (mvX[index[k]][i] - mean[i])*(mvX[index[k]][j] - mean[j]);
			cov[i][j] /= double(index.size()-0.0);
			cov[j][i]  = cov[i][j];
		}
	}
	alg::Eigen(eva, eve, mDX, cov);
	cov.clear();
}

void Eigen(std::vector<double>& eva, std::vector< std::vector<double> >& eve, unsigned int no, std::vector< std::vector<double> >& cov)
{
	unsigned int i,j,mDX = (unsigned int)cov.size();
#ifdef AZ_GSL
//============================================================================================================
	gsl_matrix* cov1 	= gsl_matrix_alloc(mDX,mDX);
	gsl_vector* eval 	= gsl_vector_alloc(mDX);
	gsl_matrix* evec	= gsl_matrix_alloc(mDX,mDX);
	
	for(i=0; i<mDX; i++)
		for(j=0; j<mDX; j++)
			gsl_matrix_set(cov1, i, j, cov[i][j]);

	gsl_set_error_handler_off();
	gsl_eigen_symmv_workspace* w = gsl_eigen_symmv_alloc(mDX);
	gsl_eigen_symmv(cov1, eval, evec, w);
	gsl_eigen_symmv_free(w);
	gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_DESC);
	
	for(i=0; i<no; i++)
	{
		eva[i] = fabs(gsl_vector_get(eval,i));
		for(j=0; j<mDX; j++) eve[i][j] = gsl_matrix_get(evec,j,i);
	}
	gsl_vector_free(eval);
	gsl_matrix_free(evec);
	gsl_matrix_free(cov1);
#else
//============================================================================================================
	alg::Matrix cov1(mDX,mDX), eve1(mDX,mDX);
	std::vector<double> eva1(mDX);
	for(i=0; i<mDX; i++)
		for(j=0; j<mDX; j++)
			cov1(i,j)  = cov[i][j];
	cov1.Eig(eva1, eve1);
	for(i=0; i<no; i++)
	{
		eva[i] = eva1[i];
		for(j=0; j<mDX; j++) eve[i][j] = eve1(j,i);
	}
	eva1.clear();
//============================================================================================================
#endif
}

} //namespace alg
} //namespace az

// MatrixAlgebra.cpp

#include <cmath>
#include <vector>
//#include "Matrix.h"

namespace az
{

namespace alg
{

	//!\brief	cholesky factorization of A1: L*L'
	// this algorithm comes from the version of TNT(jama_cholesky.h)
	bool Cholesky(Matrix&L, Matrix&A1)
	{
		int i,j,k;
		double d,s;

		if(A1.RowSize() != A1.ColSize()) return false;

		bool isspd=true;

		int n	= A1.RowSize();
		L.Resize(n,n);

		//Main loop.
		for(j=0; j<n; j++) 
		{
			d=0.0;
			for(k=0; k<j; k++) 
			{
				s=0.0;
				for(i=0; i<k; i++) s += L(k,i)*L(j,i);
				L(j,k)=s =(A1(j,k)-s)/L(k,k);
				d=d + s*s;
				isspd=isspd &&(A1(k,j) == A1(j,k)); 
			}
			d=A1(j,j) - d;
			isspd=isspd &&(d > 0.0);
			L(j,j)=sqrt(d > 0.0 ? d : 0.0);
			for(k=j+1; k<n; k++) L(j,k)=0.0;
		}
		return isspd;
	}

	//!\brief 	Solve a linear system A1*X=B, using cholesky factorization of A1: L*L'
	// this algorithm comes from the version of TNT(jama_cholesky.h)
	bool CholeskySolve(Matrix& X, Matrix& A1, Matrix& B)
	{
		int i,j,k;
		Matrix L_;

		X.Resize(0,0);
		if(A1.RowSize() != A1.ColSize() || A1.RowSize() != B.RowSize() || !Cholesky(L_,A1)) return false;

		int n	= A1.RowSize();
		int nx	= B.ColSize();

		//Step 1: Solve L*Y=B;
		X=B;
		for(j=0; j<nx; j++)
		{
			for(k=0; k<n; k++) 
			{
				for(i=0; i<k; i++) 
					X(k,j) -= X(i,j)*L_(k,i);
				X(k,j) /= L_(k,k);
			}
		}

		//Step 2: Solve L'*X=Y;
		for(j=0; j<nx; j++)
		{
			for(k=n-1; k>=0; k--) 
			{
				for(i=k+1; i<n; i++) 
					X(k,j) -= X(i,j)*L_(i,k);
				X(k,j) /= L_(k,k);
			}
		}

		return true;
	}

	#define MAX(a,b) ((a)<(b) ? (b):(a))
	#define MIN(a,b) ((a)<(b) ? (a):(b))
	double hypot(const double &a, const double &b)
	{
		if(a== 0) return fabs(b);
		else
		{
			double c=b/a;
			return fabs(a) * sqrt(1 + c*c);
		}
	}

	// an m-by-n matrix A1 with m>=n => A1=U*S*V'.
	// this algorithm comes from the version of TNT(jama_svd.h)
	void SVD(Matrix& U, Matrix&S, Matrix&V, Matrix& A)
	{
		int m =A.RowSize();
		int	n =A.ColSize();
		int nu=MIN(m,n);
		U.Resize(m, nu);
		V.Resize(n, n);
		S.Resize(MIN(m+1,n),MIN(m+1,n));
		std::vector<double> s(MIN(m+1,n)), e(n), work(m);
		Matrix A1(A);
		int wantu=1;		/* boolean */
		int wantv=1;		/* boolean */
		int i=0, j=0, k=0;

		// Reduce A1 to bidiagonal form, storing the diagonal elements
		// in s and the super-diagonal elements in e.

		int nct=MIN(m-1,n);
		int nrt=MAX(0,MIN(n-2,m));
		for(k=0; k < MAX(nct,nrt); k++) 
		{
			if(k<nct) 
			{
				// Compute the transformation for the k-th column and
				// place the k-th diagonal in s[k].
				// Compute 2-norm of k-th column without under/overflow.
				s[k]=0;
				for(i=k; i<m; i++) 
				{
					s[k]=hypot(s[k],A1(i,k));
				}
				if(s[k] != 0.0)
				{
					if(A1(k,k)<0.0) 
					{
						s[k]=-s[k];
					}
					for(i=k; i<m; i++) 
					{
						A1(i,k) /= s[k];
					}
					A1(k,k) += 1.0;
				}
				s[k]=-s[k];
			}

			for(j=k+1; j<n; j++) 
			{
				if((k<nct) &&(s[k] != 0.0))  
				{
					// Apply the transformation.
					double t=0;
					for(i=k; i<m; i++) 
					{
						t += A1(i,k)*A1(i,j);
					}
					t=-t/A1(k,k);
					for(i=k; i<m; i++) 
					{
						A1(i,j) += t*A1(i,k);
					}
				}
				// Place the k-th row of A1 into e for the
				// subsequent calculation of the row transformation.
				e[j]=A1(k,j);
			}
			if(wantu &(k<nct)) 
			{
				// Place the transformation in U for subsequent back
				// multiplication.
				for(i=k; i<m; i++) 
				{
					U(i,k)=A1(i,k);
				}
			}
			if(k<nrt) 
			{
				// Compute the k-th row transformation and place the
				// k-th super-diagonal in e[k].
				// Compute 2-norm without under/overflow.
				e[k]=0;
				for(i=k+1; i<n; i++) 
				{
					e[k]=hypot(e[k],e[i]);
				}
				if(e[k] != 0.0) 
				{
					if(e[k+1]<0.0) 
					{
						e[k]=-e[k];
					}
					for(i=k+1; i<n; i++) 
					{
						e[i] /= e[k];
					}
					e[k+1] += 1.0;
				}
				e[k]=-e[k];
				if((k+1<m) &(e[k] != 0.0)) 
				{
					// Apply the transformation.
					for(i=k+1; i<m; i++) 
					{
						work[i]=0.0;
					}
					for(j=k+1; j<n; j++) 
					{
						for(i=k+1; i<m; i++) 
						{
							work[i] += e[j]*A1(i,j);
						}
					}
					for(j=k+1; j<n; j++) 
					{
						double t=-e[j]/e[k+1];
						for(i=k+1; i<m; i++) 
						{
							A1(i,j) += t*work[i];
						}
					}
				}
				if(wantv) 
				{
					// Place the transformation in V for subsequent
					// back multiplication.
					for(i=k+1; i<n; i++) 
					{
						V(i,k)=e[i];
					}
				}
			}
		}

		// Set up the final bidiagonal matrix or order p.
		int p=MIN(n,m+1);
		if(nct<n) 
		{
			s[nct]=A1(nct,nct);
		}
		if(m<p) 
		{
			s[p-1]=0.0;
		}
		if(nrt+1<p) 
		{
			e[nrt]=A1(nrt,p-1);
		}
		e[p-1]=0.0;

		// If required, generate U.
		if(wantu) 
		{
			for(j=nct; j<nu; j++) 
			{
				for(i=0; i<m; i++) 
				{
					U(i,j)=0.0;
				}
				U(j,j)=1.0;
			}
			for(k=nct-1; k>=0; k--) 
			{
				if(s[k] != 0.0) 
				{
					for(j=k+1; j<nu; j++) 
					{
						double t=0;
						for(i=k; i<m; i++) 
						{
							t += U(i,k)*U(i,j);
						}
						t=-t/U(k,k);
						for(i=k; i<m; i++) 
						{
							U(i,j) += t*U(i,k);
						}
					}
					for(i=k; i<m; i++ ) 
					{
						U(i,k)=-U(i,k);
					}
					U(k,k)=1.0 + U(k,k);
					for(i=0; i<k-1; i++) 
					{
						U(i,k)=0.0;
					}
				} 
				else 
				{
					for(i=0; i<m; i++) 
					{
						U(i,k)=0.0;
					}
					U(k,k)=1.0;
				}
			}
		}

		// If required, generate V.
		if(wantv) 
		{
			for(k=n-1; k>=0; k--) 
			{
				if((k<nrt) &(e[k] != 0.0)) 
				{
					for(j=k+1; j<nu; j++) 
					{
						double t=0;
						for(i=k+1; i<n; i++) 
						{
							t += V(i,k)*V(i,j);
						}
						t=-t/V(k+1,k);
						for(i=k+1; i<n; i++) 
						{
							V(i,j) += t*V(i,k);
						}
					}
				}
				for(i=0; i<n; i++) 
				{
					V(i,k)=0.0;
				}
				V(k,k)=1.0;
			}
		}

		// Main iteration loop for the singular values.
		int pp=p-1;
		int iter=0;
		double eps=pow(2.0,-52.0);
		while(p > 0) 
		{
			int k=0;
			int kase=0;
			// Here is where a test for too many iterations would go.

			// This section of the program inspects for
			// negligible elements in the s and e arrays.  On
			// completion the variables kase and k are set as follows.

			// kase=1     if s(p) and e[k-1] are negligible and k<p
			// kase=2     if s(k) is negligible and k<p
			// kase=3     if e[k-1] is negligible, k<p, and
			//              s(k), ..., s(p) are not negligible(qr step).
			// kase=4     if e(p-1) is negligible(convergence).

			for(k=p-2; k>=-1; k--) 
			{
				if(k == -1) 
				{
					break;
				}
				if(fabs(e[k]) <= eps*(fabs(s[k]) + fabs(s[k+1]))) 
				{
					e[k]=0.0;
					break;
				}
			}
			if(k == p-2) 
			{
				kase=4;
			}
			else 
			{
				int ks;
				for(ks=p-1; ks>=k; ks--) 
				{
					if(ks == k) 
					{
						break;
					}
					double t =(ks != p ? fabs(e[ks]) : 0.) + 
						(ks != k+1 ? fabs(e[ks-1]) : 0.);
					if(fabs(s[ks]) <= eps*t)  
					{
						s[ks]=0.0;
						break;
					}
				}
				if(ks == k) 
				{
					kase=3;
				} 
				else if(ks == p-1) 
				{
					kase=1;
				} 
				else 
				{
					kase=2;
					k=ks;
				}
			}
			k++;

			// Perform the task indicated by kase.

			switch(kase) 
			{
				// Deflate negligible s(p).
			case 1: 
				{
					double f=e[p-2];
					e[p-2]=0.0;
					for(j=p-2; j>=k; j--) 
					{
						double t=hypot(s[j],f);
						double cs=s[j]/t;
						double sn=f/t;
						s[j]=t;
						if(j != k) 
						{
							f=-sn*e[j-1];
							e[j-1]=cs*e[j-1];
						}
						if(wantv) 
						{
							for(i=0; i<n; i++) 
							{
								t=cs*V(i,j) + sn*V(i,p-1);
								V(i,p-1)=-sn*V(i,j) + cs*V(i,p-1);
								V(i,j)=t;
							}
						}
					}
				}
				break;

				// Split at negligible s(k).
			case 2: 
				{
					double f=e[k-1];
					e[k-1]=0.0;
					for(j=k; j<p; j++)
					{
						double t=hypot(s[j],f);
						double cs=s[j]/t;
						double sn=f/t;
						s[j]=t;
						f=-sn*e[j];
						e[j]=cs*e[j];
						if(wantu) 
						{
							for(i=0; i<m; i++) 
							{
								t=cs*U(i,j) + sn*U(i,k-1);
								U(i,k-1)=-sn*U(i,j) + cs*U(i,k-1);
								U(i,j)=t;
							}
						}
					}
				}
				break;
				
				// Perform one qr step.
			case 3: 
				{
					// Calculate the shift.
					double scale=MAX(MAX(MAX(MAX(
						fabs(s[p-1]),fabs(s[p-2])),fabs(e[p-2])), 
						fabs(s[k])),fabs(e[k]));
					double sp=s[p-1]/scale;
					double spm1=s[p-2]/scale;
					double epm1=e[p-2]/scale;
					double sk=s[k]/scale;
					double ek=e[k]/scale;
					double b =((spm1 + sp)*(spm1 - sp) + epm1*epm1)/2.0;
					double c =(sp*epm1)*(sp*epm1);
					double shift=0.0;
					if((b != 0.0) ||(c != 0.0)) 
					{
						shift=sqrt(b*b + c);
						if(b<0.0) 
						{
							shift=-shift;
						}
						shift=c/(b + shift);
					}
					double f =(sk + sp)*(sk - sp) + shift;
					double g=sk*ek;

					// Chase zeros.

					for(j=k; j<p-1; j++) 
					{
						double t=hypot(f,g);
						double cs=f/t;
						double sn=g/t;
						if(j != k) 
						{
							e[j-1]=t;
						}
						f=cs*s[j] + sn*e[j];
						e[j]=cs*e[j] - sn*s[j];
						g=sn*s[j+1];
						s[j+1]=cs*s[j+1];
						if(wantv) 
						{
							for(i=0; i<n; i++) 
							{
								t=cs*V(i,j) + sn*V(i,j+1);
								V(i,j+1)=-sn*V(i,j) + cs*V(i,j+1);
								V(i,j)=t;
							}
						}
						t=hypot(f,g);
						cs=f/t;
						sn=g/t;
						s[j]=t;
						f=cs*e[j] + sn*s[j+1];
						s[j+1]=-sn*e[j] + cs*s[j+1];
						g=sn*e[j+1];
						e[j+1]=cs*e[j+1];
						if(wantu &&(j<m-1)) 
						{
							for(i=0; i<m; i++) 
							{
								t=cs*U(i,j) + sn*U(i,j+1);
								U(i,j+1)=-sn*U(i,j) + cs*U(i,j+1);
								U(i,j)=t;
							}
						}
					}
					e[p-2]=f;
					iter=iter + 1;
				}
				break;

				// Convergence.
			case 4: 
				{
					// Make the singular values positive.
					if(s[k] <= 0.0) 
					{
						s[k] =(s[k]<0.0 ? -s[k] : 0.0);
						if(wantv) 
						{
							for(i=0; i <= pp; i++) 
							{
								V(i,k)=-V(i,k);
							}
						}
					}
					// Order the singular values.
					while(k<pp) 
					{
						if(s[k]>=s[k+1]) 
						{
							break;
						}
						double t=s[k];
						s[k]=s[k+1];
						s[k+1]=t;
						if(wantv &&(k<n-1)) 
						{
							for(i=0; i<n; i++) 
							{
								t=V(i,k+1); V(i,k+1)=V(i,k); V(i,k)=t;
							}
						}
						if(wantu &&(k<m-1)) 
						{
							for(i=0; i<m; i++) 
							{
								t=U(i,k+1); U(i,k+1)=U(i,k); U(i,k)=t;
							}
						}
						k++;
					}
					iter=0;
					p--;
				}
				break;
			}
		}

		//set S
		for(i=0; i<int(s.size()); i++) 
		{
			S(i,i) = s[i];
		}
	}

	// find Pseudo inverse matrix by SVD
	void pinv(Matrix& inA, Matrix& A)
	{
		unsigned int i;
		Matrix S,U,V;
		
		SVD(U,S,V,A);
		
		Matrix S1(S);
		for(i=0; i<S1.ColSize(); i++) S1(i,i)=1.0/S1(i,i);
		
		Matrix trU(U); trU.Trans();

		Matrix tmp;
		V.Multiply(S1,tmp);
		tmp.Multiply(trU,inA);
	}

} //namespace alg

} //namespace az
