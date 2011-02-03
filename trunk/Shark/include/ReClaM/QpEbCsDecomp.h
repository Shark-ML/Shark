//===========================================================================
/*!
 *  \file QpEbCsDecomp.h
 *
 *  \brief Quadratic programming for Epoch-based Crammer-Singer multi-class machines
 
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

#ifndef _QpEbCsDecomp_H_
#define _QpEbCsDecomp_H_

#include <ReClaM/Utilities.h>
//#include <tr1/unordered_set>
#include <set>

// even if the stopping criterion is the duality gap, terminate after how many epochs?
#define MAX_EPOCHS_HARD_BOUNDARY 5 

#if 0 //mt_count_kernel_lookups
	#define MT_COUNT_KERNEL_LOOKUPS( mt_token ) { mt_token }
#else
	#define MT_COUNT_KERNEL_LOOKUPS( mt_token ) {  }
#endif

//! 
//! \brief Epoch-based quadratic program approximator for Crammer-Singer type multi-class machines 
//! 
//! \par
//! This algorithm implements an epoch-based approach 
//! to solving the optimization problem occuring in a
//! Crammer-Singer multi-class machine.
//! 
//! \par
//! One epoch is understood as the solver in a perceptron-like
//! manner looking at all training examples once, possibly  
//! carrying out intermediate optimization steps on variables
//! corresponding to known examples in-between. 
//! 
//! \par
//! Note the following conventions that are used in the documentation and underlie the code: 
//! A destinction is made between support vectors (SVs), support classes (SCs), and support patterns (SPs).
//! An SV is a combined pair of input pattern i and potential class label c \f$ (x_i, c) \f$ 
//! for which the corresponding coefficient \f$ \alpha^c_i \f$  is non-zero.
//! An SC is any label c that makes an input pattern \f$ x_i \f$ an SV.
//! An SP is any pattern \f$ x_i \f$ that has at least one SC.
//! 
class QpEbCsDecomp : public QPSolver
{
public:
	
	//! Constructor
	//! \param  kernel   kernel matrix cache
	//! \param  y		 classification targets
	//! \param  classes  number of classes
	//! \param  w  		 working set selection preference. 0=mvpOnOneSample 1=mvpAmongLargestGrad-Samples 2=mvpAmongAllSamples. see documentation below
	//! \param  r  		 reprocess alternation rule preference. see #eReprocessRules for documentation
	//! \param  states   pointer to instance of record-keeping class #SvmStatesCollection, for storing intermediate states of the solver
	QpEbCsDecomp(CachedMatrix& kernel, const Array<double>& y, unsigned int classes, 
				 unsigned int w = 0, unsigned int r = 0, SvmStatesCollection * states = NULL );
	
	//! Destructor
	~QpEbCsDecomp();
	
	//! set the stopping conditions for the solver. The last one greater zero is considered active in addition to accuracy.
	//! \param 	a  accuracy, this is always active (as in exact solvers)
	//! \param  e  maximal number of epochs, negative to set inactive
	//! \param  d  desired duality gap, negative to set inactive
	void setStoppingConditions(double a, int e = 1, double d = -1.0);
	
	//! \brief solve the quadratic program. 
	//! unlike the solve method of other solvers, we here do not accept a custom initialization
	//! for the soluction vector, instead set it to zero no matter what is passed in
	//! (being an online algorithm, we rely on the concept of "unseen" samples
	//!  for which the coefficents are supposed do be zero, i.e. wich are non-SPs)
	//! \param  solutionVector  input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	//! \param  regC	  		regularization constant C
	void Solve(Array<double>& solutionVector, double regC);
	
	//! get number of kernel evaluations
	inline long getKernelEvals()
	{
		return kernelMatrix.getBaseMatrixKernelEvals();
	}
	
	//! set kernel evaluation counter to zero
	inline void resetKernelEvals()
	{
		kernelMatrix.resetBaseMatrixKernelEvals();
	}
	
	//! set verbosity mode. 0=quiet, 1=python-compatible arrays (one row per epoch)
	inline void setVerbose(bool verbose = false)
	{
		introsp.verbosity = verbose;
	}
	
protected:
	
// PARTIES TO THE QP EQUATION

	//! solution candidate
	Array<double> alpha;
	
	//! box constraint upper bound, that is, maximal variable value.
	Array<double> boxMax;
	
	//! quadratic part, kernel matrix cache
	CachedMatrix& kernelMatrix;
	
// ADDITIONAL ARRAYS
	
	//! for each example, store the class label (between 0 and classes-1)
	Array<unsigned int> label;
	
	//! diagonal matrix entries
	Array<double> diagonal;
	
	//! gradient
	Array<double> gradient;
	
	//! i-th element gives the original index of the current i-th sample
	Array<unsigned int> origIndex;
	
	//! i-th element gives the current index of the originally i-th sample
	Array<unsigned int> curIndex; //introduced b/c shuffling would be cumbersome otherwise
	
	//! In each epoch, pre-define how to traverse through the samples 
	Array<unsigned int> lottery;

	//! Typedef for convenience
	// WHEN DOING BOOST CONVERSION: CONVERT ALL CALLS TO .erase() TO quick_erase()
	// OR erase_return_void() BECAUSE OF POTENTIAL WORST-CASE COMPLEXITY BUG AT
	// https://svn.boost.org/trac/boost/ticket/3966
//	typedef std::tr1::unordered_set<unsigned int> tActiveClasses;
	typedef std::set<unsigned int> tActiveClasses;

	//! For each pattern, maintain a set of the corresponding support vectors. 
	//! An element of the set of value i corresponds to the i-th class
	Array<tActiveClasses> ninaClasses; //not-inactive classes, i.e., active SPs and untouched non-SCs

	Array<tActiveClasses> asClasses; //active SCs
		
	//! For each class, maintain a set of the corresponding support patterns.
	//! value i in the j-th set denotes that alpha(i) is an SC for class j 
	//! (not that class j is an SC for the i-th sample).
	// in practice this means that the standard access pattern will be redundant,
	// like this: activeGlobalSCs(j).doSomething(someSampleIndex*cardi.classes + j)
	Array<tActiveClasses> activeGlobalSCs;
	
	// same principle, but only used for computing new gradients,
	// which also need the contributions from the inactive samples. 
	Array<tActiveClasses> dormantGlobalSCs;
	
//ENVELOPES FOR STATE/STRATEGY/TMP VARIABLES, ETC

	//! List of possible configurations that can occur when flipping two examples
	enum eFlipVariants
	{
		fDeactivate,
		fRemove_first,
		fRemove_second,
		fInsert_first,
		fInsert_second
	};

	//! List of variants of the WSS algorithm, i.e., among which set of samples is MVP-WSS carried out
	enum eWsVariants
	{
		wRandom,		//wss only within the selected (random old) sample
		wClassInversion, //use maxClippedGain between both updated and one random class during SMO gradient-update
		wSS_MAX		//counter
	};
	
	//! List of possible actions for the next processing step
	enum eNextProc
	{
		nNew,	//process a new (non-support pattern) example
		nOld,	//process a known support pattern, consider all classes for WSS
		nOpt,	//process a known support pattern, only consider SCs for WSS
		nMAX	//counter
	};

	//! List of possible rules how many and which reprocess steps (nOld or nOpt) to be done after one nNew
	enum eReprocessRules
	{
		rOrig,	  	//(nNew,nOld,nOpt) = (1,b,10*c), b and c evolving on-line and according to last gain rate
		rMAX		//counter
	};
	
	//! List of possible shrinking modes.
	enum eShrinkingModes
	{
		sNever,	 //do not use shrinking
		sBook,	  //shrink by the book, i.e., don't deactivate samples with active non-SCs
		sAgressive  //agressive sample-deactivation, i.e., deactivate samples with active non-SCs
	};
	
	//! envelope for current overall strategy
	struct sStrategy
	{
		//FIXED throughout entire optimization run:
		double creg;				//the regularization parameter C
		eWsVariants wss;			//specifies the variant of the WSS strategy
		eShrinkingModes shrinkMode;	//0: never, 1: by the book, 2: more agressive sample-deactivation
		double shrinkCritFraction;  //deactivate a variable if gVar > gLargestFeasibleUp * shrinkCritFraction
		double shrinkEqualFraction; //deactivate an example if |gVar - gFirst| < |gFirst| * shrinkEqFraction
		bool useClippedAsWitness;   //use clipped or normal gain in wClassInv
		bool useClippedInReproc;	//use clipped or normal gain in reproc
		eReprocessRules reRu;		//how many and which reprocess steps to do after having processed a new sample
		unsigned int numMultipl[nMAX]; //how many successful iterations of itself should each processing step strive for
		unsigned int tryMultipl[nMAX]; //how many times is it allowed to try to reach numMultipl successful smo-steps
		double probAdaptationRate;  //adaptation rate for probabilites of processing steps (mu in original paper)  //rOrig only
		double guaranteeFraction; 	//probability[i] won't sink below this fraction of probSum (eta in original paper)  //rOrig only
		//VARIABLE:
		eNextProc proc;				//next processing step
		double probability[nMAX]; 	//relative probabilities for next processing step //rOrig only
		double probSum;			 	//for convenience: sum over tStrategy.probability //rOrig only
		unsigned int nextPat;		//next pattern to work on
		unsigned int nextI;			//next variable to work on, to increase
		unsigned int nextJ;			//next variable to work on, to decrease
		unsigned int thirdClass;	//for the wClassInversion wss, a random third class not equal to nextI, nextJ
	};
	
	//! envelope for convenience cardinalities of important sets
	struct sCardinalities
	{
		unsigned int examples;		//number of examples in the problem (size of the kernel matrix)
		unsigned int classes;		//number of classes in the problem
		unsigned int variables;		//number of variables in the problem = examples times classes
		unsigned int activePatterns; //current number of active (i.e., unshrinked) patterns
		unsigned int sPatterns;		//current number of support patterns
		unsigned int seenEx;		//index into lottery (how many samples seen in this epoch)
		unsigned int epochs;		//number of epochs undertaken
		long planned_steps[nMAX]; //number of processing steps planned for each processing type
	};
	
	//! envelope for variables about the current state of the solver
	struct sIntrospection
	{
		double dual;			//current value of the dual
		double primal;			//value of the primal, needs a call to calcPrimal()
		double curGain;			//how useful the last processing step was
		double curGainRate;		//dito, but normalized to the time elapsed
		sWallTimer rolex;			//for keeping track of dual gain per processing time unit
		unsigned int verbosity; //0=quiet, 1=output results in python-compatible arrays
	};
	
	//! envelope for caching intermediate results
	struct sTmpData
	{
		double gPlus;   //upward gradient
		double gMinus;  //downward gradient
		float* q;		//kernel row for current example. needed in WSS (sometimes) and SMO
		Array<double> reprocGradients; //intermediate storage for gradients in reproc-MVP
	};
	
	//! helper enum: list of possible stopping conditions
	enum eStopReason
	{
		sNone,				//keep running
		sEndEpochSoon,		//keep running, but start new epoch before next procNew
		sEND_MARKER,		//marks the border between keep-running-non-new-steps and end-epoch
		sEndEpochNow,		//keep running, but start new epoch immediately
		sQUIT_MARKER,		//marks border between epoch and overall stopping conditions
		sEpochs,			//number of epochs has reached stopCrit.maxEpochs
		sDualAim			//dual gap has fallen below stopCrit.dualAim
	};
	
	//! envelope for stopping criterion.
	struct sStopCrit		
	{
		double accuracy;		//Not a stopping crit., but for choosing samples for SMO by their KKT violation
		unsigned int maxEpochs;	//number of epochs: how many iterations through the training set to complete
		double dualAim;			//desired duality gap as stopping criterion. Alternative to #maxEpochs
		eStopReason stop;   	//flag that summarizes the status of the different stopping conditions
	};
	
	struct sLogVars
	{
		long kernel_lookups;
		sWallTimer overall_timer;
		unsigned int writeDualEvery;
		unsigned int writeDualModuland;
		unsigned int measurements_per_epoch;
		unsigned int overall_noof_measurements;
	};
	
	sLogVars log;
	
	//! envelope for overall strategy
	sStrategy strat;
	
	//! envelope for convenience cardinalities of important sets
	sCardinalities cardi;
	
	//! envelope for variables about the current state of the solver
	sIntrospection introsp;
	
	//! envelope for caching intermediate results
	sTmpData temp;
	
	//! Stopping criterion. The last one greater zero is considered active in addition to accuracy.
	sStopCrit stopCrit;
	
	
// WORKING SET AND PROCESSING STEP SELECTION
	
	//! select next processing step
	void selectNextProcessingStep();
	
	//! find the next pattern to work on. return true if there are now unseen samples left.
	bool selectNextPattern();
	
	//! Choose working set according to current strategy. return corresponding KKT violation.
	double selectWorkingSet();

	struct tClassInversionCandidateList
	{
		//core variables
		unsigned int i;
		unsigned int j;
		double cur_witness;
		double max_witness;
		unsigned int candidate;
		unsigned int cur_priority;
		unsigned int max_priority; //deactivate 2 vars => deactivate sample => highest. deactivate 1 => deact. var => medium
		
		//strategic variables
		bool fallback_mode;
		unsigned int desired_noof_hits;
		
		//helper variables
		double g, mu_test, delta_g;
		unsigned int e, t;
		unsigned int asc_size_third;    //two tmp helpers
		unsigned int asc_size_current;
		unsigned int start_index;    	  //current index
		unsigned int counter_index;   //how many have we looked at?
		unsigned int hits_so_far;   //how many have we looked at?
//		tActiveClasses::size_type b; //these four for tr1::unordered_set variant only
//		tActiveClasses::size_type x;
//		tActiveClasses::size_type bucket_count;
//		tActiveClasses::const_local_iterator lit;
		tActiveClasses::iterator lit;
	};
	
	tClassInversionCandidateList cicl;
	
	//! do the actual update on the working set, return the gain achieved
	double performSmoStep();
	
	
// STOPPING CRITERION / INTROSPECTION
	
	//! return the current value of the primal, and also store in introsp.primal
	double calcPrimal();
	
// MISC

	//! reset all variables that are valid for one epoch only 
	void init();

	//! randomly reorder the lottery vector defining the iteration through the dataset
	void shuffleSamples();
	
	//! exclude a variable from getting its gradient updated and from being considered in wss
	// return the iterator to next in activeGlopalSPs
	// *first* determine #wasSC based on context or the old alpha value (wasSC <=> e ATM is in active SC sets)
	// *then* update the alpha value (only when shrinking from within #performSmoStep)
	// *only then* call this function (and don't forget to pass the correct #wasSC)
//////	tActiveClasses::iterator deactivateVariable( unsigned int v, unsigned int e, unsigned int c );
	void deactivateVariable( unsigned int v, unsigned int e, unsigned int c );
	
	//! freeze an entire example: exclude it from wss and all its gradients from being updated 
	// returns the new location of the deactivated sample, so vars can be kept consistent
	// *first* delete all variables, then call this!
	unsigned int deactivateExample( unsigned int e );
	
	//! exchange all data corresponding to two samples, both in internal arrays and in the cachedMatrix
	//! Important note: flipAll inherits the assumption that i < j, i.e. firstArgument < secondArgument,
	//! from #CachedMatrix::FlipColumnsAndRows
	void flipAll( unsigned int i, unsigned int j, eFlipVariants f );

	//! one central place to account for numerical inaccuracies when determining upper-boundedness
	//! this defines the parameter kappa of the original paper
	//! param index which component of alpha should be tested for boundedness via corresponding component of boxMax
	//! param extra for determining whether \f$ \alpha(i) + \f$ extra is bounded
	inline bool canIncrease( unsigned int index, double extra = 0.0 )
	{
		return ( (alpha(index) + extra + 1e-12) <= boxMax(index) );
	}
	
	//! With respect to a fixed pattern, determine whether a class is a support class
	//! This should be identical to #inSCs, unless in the middle of insertion, etc.
	//! Accounts for numerical inaccuracies in one fixed location
	//! Attention: for convenience, i is index directly into alpha, i.e. sample*cardi.classes + (relative class number)
	//! param i index directly into alpha
	inline bool isSC( unsigned int i )
	{
		return ( ( alpha(i) < -1e-12 ) || ( alpha(i) > 1e-12 )  );
	}
	
	inline unsigned int safe_discrete( unsigned int upper, unsigned int subtract_upper, unsigned int lower = 0 )
	{
		if ( subtract_upper > upper )
			upper = 0;
		else
			upper = upper - subtract_upper;
		return Rng::discrete( lower, upper );
	}
	
	SvmStatesCollection * mep_historian;
	
	Array< double > m_dummyClosureDouble;
	
};


#endif
