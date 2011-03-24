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
#include <vector>

#include <ctime>
//cross-platform wall-clock timing:
#ifdef WIN32
	#include <windows.h>
#elif linux
	#include <time.h>
#elif __MACH__
	#include <stdint.h>
	#include <mach/mach_time.h>
#endif

////////////////////////////////////////////////////////////////////////////////
//! Convenience structure for cpu-time keeping, cross-platform

class sCpuTimer 
{
public:
	//start the timer
	void tic()
	{
		m_start_time = std::clock();
	}
	//return time difference in seconds
	double toc() 
	{
		return ( std::clock() - m_start_time ) / (double)CLOCKS_PER_SEC;
	}
	//return time difference and restart the timer
	double tocAndTic() 
	{
		double tmp = ( std::clock() - m_start_time ) / (double)CLOCKS_PER_SEC;
		m_start_time = std::clock();
		return tmp;
	}
	
private:
	std::clock_t m_start_time;
};

////////////////////////////////////////////////////////////////////////////////
//! Convenience structure for wall-clock time keeping, platform-specific

class sWallTimer
{
public:
	//start the timer (i.e., set m_start_time)
	void tic()
	{
		m_start_time = getCurTimeInSecs();
	}
	//return time difference in seconds
	double toc() 
	{
		return getCurTimeInSecs() - m_start_time;
	}
	//return time difference and restart the timer
	double tocAndTic() 
	{
		m_temp = getCurTimeInSecs() - m_start_time;
		m_start_time = getCurTimeInSecs();
		return m_temp;
	}
	
protected:
	double m_temp;
	double m_start_time;
	#ifdef WIN32
		LARGE_INTEGER li1;
		LARGE_INTEGER li2;
	#else
		timespec m_tv;
	#endif

	double getCurTimeInSecs()
	{
		#ifdef WIN32
			QueryPerformanceCounter( &li1 );
			QueryPerformanceFrequency( &li2 );
			return li1.QuadPart / li2.QuadPart;
		#elif __linux
			clock_gettime( CLOCK_REALTIME, &m_tv );
			return m_tv.tv_sec + (m_tv.tv_nsec/1E9);
		#elif __MACH__
			static mach_timebase_info_data_t info = {0,0};
			mach_timebase_info(&info);
			uint64_t now = mach_absolute_time();
			m_tv.tv_sec = now * 1E-9;  
			m_tv.tv_nsec = now - (m_tv.tv_sec * 1E9);
		#endif
	}
};

////////////////////////////////////////////////////////////////////////////////

enum eSvmMode
{
	eC1,				// C-SVM with 1-norm penalty for binary classification	//  0
	eC2,				// C-SVM with 2-norm penalty for binary classification
	eEpsilon,			// \varepsilon-SVM for regression
	eNu,				// \nu-SVM, not implemented yet				//  3
	eRegularizationNetwork,
	eGaussianProcess,		// Gaussian Process
	e1Class,			// One-Class SVM for density estimation XXX		//  6
	eAllInOne,			// standard multi class SVM
	eCrammerSinger,			// MC-SVM by Crammer and Singer
	eLLW,				// MC-SVM by Lee, Lin and Wahba				//  9
	eDGI,				// MC-SVM by Dogan, Glasmachers and Igel
	eOVA,				// one-versus-all multi class SVM
	eOCC,				// one-class-cost multi-class SVM			// 12
	eEBCS				// epoch-based Crammer-Singer
};

////////////////////////////////////////////////////////////////////////////////

typedef std::pair < unsigned int, double > tIndexedSupClass;
typedef std::vector < tIndexedSupClass > tSupClassCollection;
typedef std::pair < unsigned int, tSupClassCollection > tIndexedSupPat; //= one support Pat
typedef std::vector < tIndexedSupPat > tSupPatCollection; //one supPatColl is one SVM state
typedef std::vector <tSupPatCollection> tStateCollection; //one stateColl is a list of SVMs

////////////////////////////////////////////////////////////////////////////////


//!
//! \brief Utility class for storing snapshots of SVM states
//!
//! \par
//! When training an SVM (solving its QpProgram), it can be
//! interesting to analyze the intermittant solver states 
//! and thereby its learning curve. An #SvmStatesCollection
//! instance provides storage to which the relevant parameters
//! can be pushed during training. The parameter copy procedure is non-sparse,
//! so that copy time is constant during model buildup. All 
//! stored model snapshots can be sparsified after the optimization
//! run. The storeXXX methods allow triggering calculations of
//! more resource-intensive measures after training and sparsification,
//! while the XXXdirectly methods are for quick direct storage
//! of cheaply obtainable measures from within the solver.
//!

class SvmStatesCollection
{
public:

	//! Constructor
	//! \param  classes			number of classes
	//! \param  in_kernel 		pointer to kernel function
	//! \param  train_inputs 	training data
	SvmStatesCollection( unsigned int classes, KernelFunction *in_kernel, const Array<double>& train_inputs );
	~SvmStatesCollection();
	
	//! initialization method for resizing of the internal arrays. 
	//! \param noof_shots 					how many snapshots do i want to take?
	//! \param raw_param_length				of what size will the parameter array be?
	//! \param raw_closure_length_double 	how many doubles will accompany the parameter array as closure-information?
	//! \param raw_closure_length_unsigned  how many unsigned ints as closure-information?
	//! \param noof_measures				how many performance indicators will i later want to derive for each shot
	void declareIntentions( unsigned int noof_shots, unsigned int raw_param_length, 
							unsigned int raw_closure_length_double, unsigned int raw_closure_length_unsigned, 
							unsigned int noof_measures );
	
	//! true svm training should be influenced as little as possible, and the influence should be constant.
	//! hence, here push a direct snapshot of the parameter vector only,
	//! along with necessary closure information to reconstruct the corresponding true sparse solution later
	//! \param param 			the model parameter array
	//! \param closure_double 	necessary closure information of type double
	//! \param closure_unsigned	necessary closure information of type unsigned (e.g., the permutation of param)
	void pushSnapShot( const Array<double>& param, const Array<double>& closure_double, 
					   const Array<unsigned int>& closure_unsigned );
	
	//! true svm training should be influenced as little as possible, and the influence should be constant.
	//! hence, here construct sparse versions of all snapshots recorded
	//! \param svm_variant to which SVM variant do my stored snapshots correspond?
	void makeHistorySparse( unsigned int svm_variant );
	
	//! using the sparsified history, calculate and store the primal of each snapshot
	//! \param svm_variant 		to which SVM variant do my stored snapshots correspond?
	//! \param target_index 	store in what row of the performance-measuring array?
	//! \param train_targets 	label array of the training data
	//! \param regC 			regularization constant
	void storePrimal( unsigned int svm_variant, unsigned int target_index, 
					  const Array<double>& train_targets, double regC );
					  
	//! using the sparsified history, calculate and store the test error of each snapshot
	//! \param svm_variant 		to which SVM variant do my stored snapshots correspond?
	//! \param target_index 	store in what row of the performance-measuring array?
	//! \param test_inputs 		test data
	//! \param test_targets 	test labels
	void storeTestErr( unsigned int svm_variant, unsigned int target_index, 
					   const Array<double>& test_inputs, const Array<double>& test_targets );
					   
	//! auxiliary methods for direct access to #m_performanceMeasures
	//! \param what 			the value to store
	//! \param target_index 	store in what row of the performance-measuring array?
	//! \param shot_number  	store in what column of the performance-measuring array?
	void storeDirectly( double what, unsigned int target_index, unsigned int shot_number );
	
	//! auxiliary methods for direct access to #m_performanceMeasures
	//! \param target_index 	query what row of the performance-measuring array?
	//! \param shot_number  	query what column of the performance-measuring array?
	double accessDirectly( unsigned int target_index, unsigned int shot_number );
	
	//! pretty-print the performance measuring array
	//! \param target_index 	query what row of the performance-measuring array?
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
	Array < double > m_performanceMeasures; //main data structure for performance indicators
};

////////////////////////////////////////////////////////////////////////////////



#endif
