//===========================================================================
/*!
 *  \file Utilities.h
 *
 *  \brief Different utilities of generic use, or especially for SVM solvers
 *
 *  \author  M.Tuma
 *  \date	2010
 *
 *  \par Copyright (c) 1999-2007:
 *	  Institut f&uuml;r Neuroinformatik<BR>
 *	  Ruhr-Universit&auml;t Bochum<BR>
 *	  D-44780 Bochum, Germany<BR>
 *	  Phone: +49-234-32-25558<BR>
 *	  Fax:   +49-234-32-14209<BR>
 *	  eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *	  www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
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
//===========================================================================


#ifndef _Utilities_H_
#define _Utilities_H_

#include <SharkDefs.h>
#include <ReClaM/KernelFunction.h>
#include <Array/Array.h>
#include <sys/time.h>
#include <vector>
#include <ctime>

////////////////////////////////////////////////////////////////////////////////

//! Convenience structure for cpu-time keeping
struct sCpuTimer 
{
	std::clock_t start_time;

	void tic()
	{
		start_time = std::clock();
	}

	double toc() 
	{
		return ( std::clock() - start_time ) / (double)CLOCKS_PER_SEC;
	}

	double tocAndTic() 
	{
		double tmp = ( std::clock() - start_time ) / (double)CLOCKS_PER_SEC;
		start_time = std::clock();
		return tmp;
	}
};

////////////////////////////////////////////////////////////////////////////////

//! Convenience structure for wall-time keeping
struct sWallTimer
{
	timeval tv;
	double seconds;
	void tic()
	{
		gettimeofday(&tv, NULL);
		seconds = tv.tv_sec+(tv.tv_usec/1000000.0);
	}
	//return time difference in microseconds (but w/ precision of only 0.01 s)
	double toc()
	{
		gettimeofday(&tv, NULL);
		return ( tv.tv_sec+(tv.tv_usec/1000000.0) - seconds );
	}
	//return time difference and restart the timer
	double tocAndTic()
	{
		gettimeofday(&tv, NULL);
		double tmp = ( tv.tv_sec+(tv.tv_usec/1000000.0) - seconds );
		seconds = tv.tv_sec+(tv.tv_usec/1000000.0);
		return tmp;
	}
};

////////////////////////////////////////////////////////////////////////////////

enum eSvmMode
{
	eC1,				// C-SVM with 1-norm penalty for binary classification	  //  0
	eC2,				// C-SVM with 2-norm penalty for binary classification
	eEpsilon,			// \varepsilon-SVM for regression
	eNu,				// \nu-SVM, not implemented yet							 //  3
	eRegularizationNetwork,
	eGaussianProcess,	// Gaussian Process
	e1Class,			// One-Class SVM for density estimation XXX				 //  6
	eAllInOne,			// standard multi class SVM
	eCrammerSinger,		// MC-SVM by Crammer and Singer
	eLLW,				// MC-SVM by Lee, Lin and Wahba							 //  9
	eDGI,				// MC-SVM by Dogan, Glasmachers and Igel
	eOVA,				// one-versus-all multi class SVM
	eOCC,				// one-class-cost multi-class SVM						   // 12
	eEBCS				// epoch-based Crammer-Singer
};

////////////////////////////////////////////////////////////////////////////////

typedef std::pair < unsigned int, double > tIndexedSupClass;
typedef std::vector < tIndexedSupClass > tSupClassCollection;
typedef std::pair < unsigned int, tSupClassCollection > tIndexedSupPat; //= one support Pat
typedef std::vector < tIndexedSupPat > tSupPatCollection; //one supPatColl is one SVM state
typedef std::vector <tSupPatCollection> tStateCollection; //one stateColl is a list of SVMs

////////////////////////////////////////////////////////////////////////////////

class SvmStatesCollection
{
public:

	SvmStatesCollection( unsigned int classes, KernelFunction *in_kernel, const Array<double>& train_inputs );
	~SvmStatesCollection();
	
	//! for resizing of the internal arrays: how many shots do i want to take, what sizes of arrays will
	//! i push at each snapshot, how many performance indicators will i later want to derive for each shot
	void declareIntentions( unsigned int noof_shots, unsigned int raw_param_length, 
							unsigned int raw_closure_length_double, unsigned int raw_closure_length_unsigned, 
							unsigned int noof_measures );
	
	//! true svm training should be influenced as little as possible, and the influence should be constant.
	//! hence, here push a direct snapshot of the parameter vector only,
	//! along with necessary closure information to reconstruct the
	//! pushed sparse solution later
	void pushSnapShot( const Array<double>& param, const Array<double>& closure_double, 
					   const Array<unsigned int>& closure_unsigned );
	
	//! true svm training should be influenced as little as possible, and the influence should be constant.
	//! hence, here construct sparse versions of all snapshots recorded
	void makeHistorySparse( unsigned int svm_variant );
	
	//! using the sparsified history, calculate and store the primal of each snapshot
	void storePrimal( unsigned int svm_variant, unsigned int target_index, 
					  const Array<double>& train_targets, double regC );
					  
	//! using the sparsified history, calculate and store the test error of each snapshot
	void storeTestErr( unsigned int svm_variant, unsigned int target_index, 
					   const Array<double>& test_inputs, const Array<double>& test_targets );
					   
	//! auxiliary methods for direct access to #m_performanceMeasures
	void storeDirectly( double what, unsigned int target_index, unsigned int shot_number );
	double accessDirectly( unsigned int target_index, unsigned int shot_number );
	
	void printPerformanceMeasureAsNumPyArray( unsigned int target_index );

private:

	bool m_wasInitialized;
	bool m_wasMadeSparse;
	unsigned int m_noofShots;
	unsigned int m_noofClasses;
	unsigned int m_noofTrainExamples;
	KernelFunction * mep_kernel;
	const Array<double> * mep_trainData;
	
	tStateCollection m_snapshots; //collection of sparse (i.e. already re-looked at) svm models
	
	unsigned int m_nextShot;
	unsigned int m_paramLength;
	unsigned int m_closureLengthDouble;
	unsigned int m_closureLengthUnsigned;
	Array < double > m_rawParameterShots; //main data structures for raw state variables
	Array < double > m_rawClosureShotsDouble;
	Array < unsigned > m_rawClosureShotsUnsigned;
	
	unsigned int m_noofMeasures;
	Array < double > m_performanceMeasures; //main data structure for results
};

////////////////////////////////////////////////////////////////////////////////



#endif
