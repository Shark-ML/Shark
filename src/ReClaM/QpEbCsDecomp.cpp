//===========================================================================
/*!
 *  \file QpEbCsDecomp.cpp
 *
 *  \brief Quadratic programming for Epoch-based Crammer-Singer multi-class machines
 *
 *  \author  M.Tuma
 *  \date	2010
 *
 *  \par Copyright (c) 1999-2010:
 *	  Institut f&uuml;r Neuroinformatik<BR>
 *	  Ruhr-Universit&auml;t Bochum<BR>
 *	  D-44780 Bochum, Germany<BR>
 *	  Phone: +49-234-32-25558<BR>
 *	  Fax:   +49-234-32-14209<BR>
 *	  eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *	  www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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


#include <SharkDefs.h>
#include <Rng/GlobalRng.h>
#include <Array/ArrayIo.h>
#include <Array/ArrayOp.h>
#include <LinAlg/LinAlg.h>
#include <ReClaM/QuadraticProgram.h>

#include <math.h>
#include <iostream>
#include <iomanip>


using namespace std;

// useful exchange macros for Array<T> and std::vector<T>
#define XCHG_A(t, a, i, j) {t temp; temp = a(i); a(i) = a(j); a(j) = temp;}
#define XCHG_V(t, a, i, j) {t temp; temp = a[i]; a[i] = a[j]; a[j] = temp;}


QpEbCsDecomp::QpEbCsDecomp(CachedMatrix& kernel, const Array<double>& y, 
							unsigned int classes, unsigned int w, unsigned int r,
							SvmStatesCollection * states)
: kernelMatrix(kernel)
{
	//experimental/recently added vars
	unsigned int e, i; //examples and misc.
	strat.shrinkMode = static_cast < eShrinkingModes > ( 0 );
	if ( w == 1 ) //combine cliv with clipped-gain
		strat.useClippedInReproc = true;
	else
		strat.useClippedInReproc = false;
	strat.useClippedAsWitness = true;
	strat.shrinkCritFraction = 1.0;
	strat.shrinkEqualFraction = 0.0;
	temp.reprocGradients.resize(classes);
	
	// init cardinalities (seenEx is done in init)
	cardi.examples = kernelMatrix.getMatrixSize();
	cardi.classes = classes;
	cardi.variables = cardi.examples * cardi.classes;
	cardi.sPatterns = 0;
	cardi.activePatterns = 0;
	cardi.epochs = 0;
	cardi.seenEx = 0;
	for (i = 0; i < nMAX; i++) cardi.planned_steps[i] = 0;
	cardi.planned_steps[0] = 1;	 //account for pre-defined first procNew step
	
	SIZE_CHECK(y.ndim() == 2);
	SIZE_CHECK(y.dim(0) == cardi.examples);
	SIZE_CHECK(y.dim(1) == 1);
	RANGE_CHECK( classes > 1 );
	
	// size lists; alpha, boxMax, and gradient have length of number of variables, rest of number of examples
	alpha.resize(cardi.variables, false);
	boxMax.resize(cardi.variables, false);
	label.resize(cardi.examples, false);
	diagonal.resize(cardi.examples, false);
	gradient.resize(cardi.variables, false);
	origIndex.resize(cardi.examples, false);
	curIndex.resize(cardi.examples, false);
	lottery.resize(cardi.examples, false);
	ninaClasses.resize(cardi.examples, false);
	asClasses.resize(cardi.examples, false);
	activeGlobalSCs.resize(cardi.classes, false);
	dormantGlobalSCs.resize(cardi.classes, false);
	
	double peek; //tmp helper
	// fill lists, part 1. Rest done later, because boxMax depends on C, etc.
	for (e = 0; e < cardi.examples; e++) 
	{
		// fill in label
		peek = y(e,0);
		ASSERT ( peek == (int)peek ) //no regression-type datasets
		RANGE_CHECK ( peek >= 0 ) // no +-1 encoded binary datasets
		RANGE_CHECK( peek < cardi.classes );
		label(e) = peek;
		// fill in rest
		diagonal(e) = kernelMatrix.Entry(e, e);
		origIndex(e) = e;
		curIndex(e) = e;
		lottery(e) = e;
		// initially, ninaClasses contain all classes:
		for (unsigned c=0; c<cardi.classes; c++)
			ninaClasses(e).insert(c);
	}
	log.kernel_lookups = 0;
	MT_COUNT_KERNEL_LOOKUPS( log.kernel_lookups = cardi.examples; )
	
	// first init the strategy parameters that are fixed during the entire run:
	strat.reRu = static_cast < eReprocessRules >( r );;
	switch ( strat.reRu )
	{
		case rOrig:
		{
			strat.numMultipl[0] = 1;
			strat.numMultipl[1] = 1;
			strat.numMultipl[2] = 10;
			strat.tryMultipl[0] = 1;
			strat.tryMultipl[1] = 10;
			strat.tryMultipl[2] = 10;
			strat.probAdaptationRate = 0.05;
			strat.guaranteeFraction = 1.0 / 20.0;
			break;
		}
		default: throw SHARKEXCEPTION("[QpEbCsDecomp::QpEbCsDecomp] Invalid reprocRules designator" );
	}
	
	// set working set selection variables and related:
	// for the time being, set the additional pseudo-parameter for the class
	// inversion method by overloading the constructor's wss parameter. if 
	// w < wSS_MAX and w = wClassInv directly, we will look at all variables.
	if ( w >= wSS_MAX ) 
	{
		cicl.desired_noof_hits = w - wSS_MAX + 1;
		w = 1;
	}
	else
		cicl.desired_noof_hits = 0;
	strat.wss = static_cast < eWsVariants >( w );
	if ( strat.wss == wClassInversion && cardi.classes == 2 ) 
		throw SHARKEXCEPTION("[QpEbCsDecomp::QpEbCsDecomp] wClassInversion only applicable to true multi-class tasks.");
	if ( strat.wss >= wClassInversion )
	{
		cicl.candidate = 0;
		cicl.max_witness = 0;
		cicl.max_priority = 0;
		cicl.i = 0;
		cicl.j = 1;
		cicl.fallback_mode = true;
	}

	// init strategy vars: variable parameters (rest initialized in course of program)
	strat.proc = nNew; //first move must be to draw a new sample
	if ( strat.reRu == rOrig )
	{
		for (i = 0; i < nMAX; i++) 
			strat.probability[i] = 1.0;
		strat.probSum = 3.0;
	}
	
	// init introspection and temporary-results vars 
	introsp.dual = 0.0;
	introsp.primal = 0.0;
	introsp.curGainRate = 0.0;
	temp.q = NULL;

	// init default stopping conditions. overriden by setStoppingConditions.
	stopCrit.accuracy = 0.0001; //default is 1e-4 in original paper
	stopCrit.maxEpochs = 1;
	stopCrit.dualAim = 0.0;
	stopCrit.stop = sNone;
	
	// initialize record keeping historian
	mep_historian = states;
	log.writeDualEvery = 0;
	if ( mep_historian != NULL )
	{	
		log.measurements_per_epoch = 10;
		log.overall_noof_measurements = log.measurements_per_epoch*stopCrit.maxEpochs;
										// v-- number of shots, param length, closure lengths, noof measures
		mep_historian->declareIntentions( log.overall_noof_measurements, cardi.variables, 0, cardi.examples, 4 );
		m_dummyClosureDouble.resize(0, false);
	}
	else
	{
		log.measurements_per_epoch = 0;
		log.overall_noof_measurements = 0;
	}
	if ( log.measurements_per_epoch )
	{
		log.writeDualEvery = cardi.examples / log.measurements_per_epoch;
		log.writeDualModuland = cardi.examples % log.measurements_per_epoch;
	}
	
}

QpEbCsDecomp::~QpEbCsDecomp()
{ }

void QpEbCsDecomp::init()
{
//	ASSERT( false )
	cardi.seenEx = 0;
	stopCrit.stop = sNone; //this can only reset any former break-to-next-epoch
	shuffleSamples();
}

void QpEbCsDecomp::setStoppingConditions(double a, int e, double d)
{
	RANGE_CHECK( a > 0 );
	stopCrit.accuracy = a;
	
	if ( d > 0 ) 
	{
		stopCrit.maxEpochs = MAX_EPOCHS_HARD_BOUNDARY;
		stopCrit.dualAim = d;
	}
	else if ( e > 0 ) 
	{
		if ( e < MAX_EPOCHS_HARD_BOUNDARY )
			stopCrit.maxEpochs = e;
		else
			stopCrit.maxEpochs = MAX_EPOCHS_HARD_BOUNDARY;
		stopCrit.dualAim = 0.0;
	}
	else throw SHARKEXCEPTION("[EpochBasedCsMcSvm::setStoppingConditions]" 
							  "Invalid stopping criterion.");
							  					  
	// initialize record keeping historian
	if ( mep_historian != NULL )
	{	
		log.measurements_per_epoch = 10;
		log.overall_noof_measurements = log.measurements_per_epoch*stopCrit.maxEpochs;
										// v-- number of shots, param length, closure lengths, noof measures
		mep_historian->declareIntentions( log.overall_noof_measurements, cardi.variables, 0, cardi.examples, 4 );
		m_dummyClosureDouble.resize(0, false);
	}
	else
	{
		log.measurements_per_epoch = 0;
		log.overall_noof_measurements = 0;
	}
	if ( log.measurements_per_epoch )
	{
		log.writeDualEvery = cardi.examples / log.measurements_per_epoch;
		log.writeDualModuland = cardi.examples % log.measurements_per_epoch;
	}
}

void QpEbCsDecomp::Solve(Array<double>& solutionVector, double regC)
{
	SIZE_CHECK( solutionVector.ndim() == 1 );
	SIZE_CHECK( solutionVector.dim(0) == cardi.variables );
	RANGE_CHECK( regC > 0 );
	strat.creg = regC;
	double curGain = 0.0; //tmp helper
	unsigned int e, c, j, i = 0; //used for examples, classes, misc.
	
	//fill lists, part 2
	solutionVector = 0; //to be clear: in contrast to other solvers, no custom init allowed here
	alpha = 0; 			//dito
	for (e = 0; e < cardi.examples; e++)
		for (c = 0; c < cardi.classes; c++)
			boxMax(e*cardi.classes+c) = ( c == label(e) )*regC;
	
	log.overall_timer.tic();
	// ONE LOOP = ONE EPOCH  ( Stop condition handling ugly here, but allows for flexible, extendable structure. )
	while( stopCrit.stop < sQUIT_MARKER ) //only first stopping condition indicates immediate end-of-epoch
	{
		init();  //reset all vars that are valid for one epoch only
		while ( stopCrit.stop < sEND_MARKER ) //cycle through samples, with intermittant reprocess/optimize steps
		{
			introsp.curGainRate = 0.0;
			introsp.rolex.tic(); //start timer
			for (i=0,j=0; i < strat.tryMultipl[strat.proc]; i++) //i=trials, j=successes
			{
				if ( selectNextPattern() ) break; //end epoch if seen all non-SPs
				if ( selectWorkingSet() > stopCrit.accuracy)
				{
					curGain = performSmoStep();
					introsp.curGainRate += curGain;
					++j;
					if ( j >= strat.numMultipl[strat.proc] ) //reached foreseen amount of iterations: finish
						break;
				}
				else //potential cacheRowRelease & potential safety exit for cicl-based wss-strategies
				{
					if ( strat.proc == nNew ) //we'll never see that sample again
					{
						kernelMatrix.CacheRowRelease( strat.nextPat ); //then delete the corresponding row
						if ( stopCrit.stop == sEndEpochSoon ) //mostly for completeness: tryMultipl[nNew] is 1 for all known reru
							break;
					}
					else if ( strat.wss == wClassInversion )
					{
						// this is possible e.g. in beginning: we'll have endless opimizations on few samples,
						// hence, e.g., max_witness = max_gain may be e-10, which is below kkt limit
						cicl.fallback_mode = true; //next time, do nOpt over all SCs
					}
				}
			}
			introsp.curGainRate /= ( introsp.rolex.toc() + 0.00001 ); //stop timer
			selectNextProcessingStep(); //update probabilities and select next step based on timing and dual
		}
		
		++cardi.epochs;
		if ( cardi.epochs >= stopCrit.maxEpochs ) 
			stopCrit.stop = sEpochs;
		if ( stopCrit.dualAim > 0 && calcPrimal()-introsp.dual <= stopCrit.dualAim )
			stopCrit.stop = sDualAim;
			
		if ( log.measurements_per_epoch )
		{
			mep_historian->storeDirectly( introsp.dual, 0, cardi.epochs*log.measurements_per_epoch-1 );
			mep_historian->storeDirectly( log.overall_timer.toc(), 1, cardi.epochs*log.measurements_per_epoch-1 );
			mep_historian->pushSnapShot( alpha, m_dummyClosureDouble, origIndex );
		}

	}
	
	//print results/performance measures to screen
	if ( introsp.verbosity == 1 )
	{
		if ( !log.measurements_per_epoch )
		{
			cout << "\t\t[ " << introsp.dual << ", " << getKernelEvals() << ", "
				 << kernelMatrix.getMaxCacheSize()/1048576*sizeof(float) << ", " 
				 << cardi.activePatterns << ", " << cardi.sPatterns << ", " 
				 << log.overall_timer.toc() << ", " << cardi.planned_steps[0] << ", " << cardi.planned_steps[1] 
				 << ", " << cardi.planned_steps[2] << ", " << log.kernel_lookups << " ]" << endl;
		}
		else
		{
			cout << "\t\t[ " << introsp.dual << ", " << getKernelEvals() << ", "
				 << kernelMatrix.getMaxCacheSize()/1048576*sizeof(float) << ", " 
				 << cardi.activePatterns << ", " << cardi.sPatterns << ", " 
				 << log.overall_timer.toc() << ", " << cardi.planned_steps[0] << ", " << cardi.planned_steps[1] 
				 << ", " << cardi.planned_steps[2] << ", " << log.kernel_lookups;
			for (unsigned int i=0; i<log.overall_noof_measurements-10; i++)
				cout << ", 0 ";
			cout << " ]" << endl;
			cout << "\t\t," << endl;
			cout << "\t\t[ ";
			for (unsigned int i=0; i<log.overall_noof_measurements-1; i++)
				cout << mep_historian->accessDirectly( 0, i ) << ", ";
			cout << mep_historian->accessDirectly( 0, log.overall_noof_measurements-1 );
			cout << " ]" << endl;
			cout << "\t\t," << endl;
			cout << "\t\t[ ";
			for (unsigned int i=0; i<log.overall_noof_measurements-1; i++)
				cout << mep_historian->accessDirectly( 1, i ) << ", ";
			cout << mep_historian->accessDirectly( 1, log.overall_noof_measurements-1 );
			cout << " ]" << endl;
		}
	}
		
	// return alpha
	for (e = 0; e < cardi.examples; e++)
		for (c = 0; c < cardi.classes; c++)
			solutionVector( origIndex(e)*cardi.classes + c ) = alpha( e*cardi.classes + c );

}


void QpEbCsDecomp::selectNextProcessingStep()
{
	unsigned int i;
	if ( stopCrit.stop == sEndEpochNow )
		return; //if the last procNew in this epoch was aborted, leave everything as is
	if ( cardi.activePatterns == 0 )
	{
		strat.proc = nNew; //if currently there are no APs, procOld and procOpt don't make sense
		if ( stopCrit.stop == sEndEpochSoon )
			stopCrit.stop = sEndEpochNow;
		return;
	}
	switch ( strat.reRu )
	{
		case rOrig:
		{
			// update probabilities
			strat.probability[strat.proc] = strat.probAdaptationRate * introsp.curGainRate +
											(1.0 - strat.probAdaptationRate) * strat.probability[strat.proc];
			strat.probSum = 0.0;
			for (i = 0; i < nMAX; i++) strat.probSum += strat.probability[i]; //update the sum
			for (i = 0; i < nMAX; i++) //raise other probabilities if necessary
				if ( strat.probability[i] < strat.guaranteeFraction*strat.probSum )
					strat.probability[i] = strat.guaranteeFraction*strat.probSum;
			strat.probSum = 0.0; 
			for (i = 0; i < nMAX; i++) 
				strat.probSum += strat.probability[i]; //and update the sum again
				
			// choose next step according to new probabilites
			double draw = Rng::uni(0, strat.probSum);
			for (i = 0; i < nMAX; i++) 
			{
				if ( draw <= strat.probability[i] ) 
				{
					strat.proc = static_cast < eNextProc >( i );
					break;
				}
				else 
				{
					draw -= strat.probability[i];
				}
			}
			break;
		}
		default: throw SHARKEXCEPTION("[QpEbCsDecomp::selectNextProcessingStep] unkown reprocess rule");
	}
	
	if ( stopCrit.stop == sEndEpochSoon && strat.proc == nNew )
		stopCrit.stop = sEndEpochNow; //after endSoon, only allow continuation for nOld,nOpt, not for nNew
	else
		++cardi.planned_steps[strat.proc];
}

bool QpEbCsDecomp::selectNextPattern()
{
	switch ( strat.proc )
	{
		case nNew:
		{
			do //iterate through the lottery until a non-supportPattern is reached
			{
				strat.nextPat = curIndex( lottery(cardi.seenEx) );
				++cardi.seenEx;
				if ( log.writeDualEvery )
				{
					if (	cardi.seenEx % log.writeDualEvery == log.writeDualModuland 	//take measurements every x-th example...
						 && cardi.seenEx != log.writeDualModuland							//...but not very close to the beginning 
						 && cardi.seenEx / log.writeDualEvery != log.measurements_per_epoch ) //...and not very close to the end
					{
						int index = (cardi.epochs*log.measurements_per_epoch)+(cardi.seenEx/log.writeDualEvery)-1;
						mep_historian->storeDirectly( introsp.dual, 0, index );
						mep_historian->storeDirectly( log.overall_timer.toc(), 1, index );
						mep_historian->pushSnapShot( alpha, m_dummyClosureDouble, origIndex );
					}
				}
				if ( cardi.seenEx >= cardi.examples ) //picked this epoch's last pattern
				{
					if ( strat.nextPat < cardi.sPatterns ) //bummer, there were only SPs left, so bail out.
					{
						stopCrit.stop = sEndEpochNow; 
						return true;
					}
					else //found one SP, but that's the last one, so one more SMO
					{
						stopCrit.stop = sEndEpochSoon;
						return false;
					}
				}
			} while ( strat.nextPat < cardi.sPatterns ); //ensure that not an SP (only necessary if epochs > 1)
			break;
		}
		case nOld:
		case nOpt:
		{
			if ( strat.wss >= wClassInversion ) //retrieve sample from candidate list
			{
				if ( cicl.fallback_mode )
					strat.nextPat = safe_discrete( cardi.activePatterns, 1 );
				else if ( cicl.candidate >= cardi.activePatterns )
				{
					strat.nextPat = safe_discrete( cardi.activePatterns, 1 );
					cicl.fallback_mode = true;
				}
				else
				{
					strat.nextPat = cicl.candidate;
				}
			}
			else
				strat.nextPat = safe_discrete( cardi.activePatterns, 1 );
			break; 
		}
		default: 
		{
			cout << "received strat.proc = " << strat.proc << endl;
			throw SHARKEXCEPTION("[QpEbCsDecomp::selectNextPattern] Not a valid processing mode.");
		}
	}
	return false;
}

double QpEbCsDecomp::selectWorkingSet()
{
	temp.q = NULL;
	double g_cur;
	unsigned int i, p = strat.nextPat;
	unsigned int y = label(p);
	tActiveClasses::iterator it, jt;
	temp.gPlus = -1e100;
	temp.gMinus = 1e100;
	
	switch ( strat.proc )
	{
		case nNew: 
		{
//			if ( cardi.sPatterns > cardi.activePatterns + 1 ) throw SHARKEXCEPTION("[am ziel] .");
			p = strat.nextPat;
			if (cardi.sPatterns)
			{
				temp.q = kernelMatrix.Row(p, 0, cardi.sPatterns);
				MT_COUNT_KERNEL_LOOKUPS ( log.kernel_lookups += cardi.sPatterns; )
			}
				
			// prepare variables, set first variable
			temp.gPlus = 1.0;
			strat.nextI = label(p); //all others are at upper bound. now get gradient:
			for ( it = activeGlobalSCs(strat.nextI).begin(); //mt_costly_loop_marker
				  it != activeGlobalSCs(strat.nextI).end(); ++it )
				temp.gPlus -= temp.q[*it/cardi.classes] * alpha(*it);
			for ( it = dormantGlobalSCs(strat.nextI).begin(); //mt_costly_loop_marker
				  it != dormantGlobalSCs(strat.nextI).end(); ++it )
				temp.gPlus -= temp.q[*it/cardi.classes] * alpha(*it);
			
			
			// calc gradient for every but actual class and get minimum
			for (unsigned int c = 0; c < cardi.classes; c++) //mt_costly_loop_marker
			{
				if ( c == strat.nextI ) continue; //i should not equal j
				g_cur = 0; //get gradient:
				for ( it = activeGlobalSCs(c).begin(); //mt_costly_loop_marker
					  it != activeGlobalSCs(c).end(); ++it ) 
					g_cur -= temp.q[*it/cardi.classes] * alpha(*it);
				for ( it = dormantGlobalSCs(c).begin(); //mt_costly_loop_marker
					  it != dormantGlobalSCs(c).end(); ++it) 
					g_cur -= temp.q[*it/cardi.classes] * alpha(*it);
				if ( g_cur < temp.gMinus ) 
				{
					strat.nextJ = c;
					temp.gMinus = g_cur;
				}
			}
			break;
		}
		case nOld: 
		{
			// iterate through all classes of this sample and get extremal gradient
			for ( it = ninaClasses(p).begin(); it != ninaClasses(p).end(); ++it ) //mt_costly_loop_marker
			{
				i = p*cardi.classes + *it;
				if ( !isSC(i) ) //calc gradient
				{
					if ( temp.q == NULL && cardi.sPatterns ) //only get kernel row if it is really needed
					{
						temp.q = kernelMatrix.Row(p, 0, cardi.sPatterns);
						MT_COUNT_KERNEL_LOOKUPS ( log.kernel_lookups += cardi.sPatterns; )
					}
					g_cur = ( *it == y );
					for ( jt = activeGlobalSCs(*it).begin(); 
						  jt != activeGlobalSCs(*it).end(); ++jt ) //mt_costly_loop_marker
						g_cur -= temp.q[*jt/cardi.classes] * alpha(*jt);
					for ( jt = dormantGlobalSCs(*it).begin(); //mt_costly_loop_marker
						  jt != dormantGlobalSCs(*it).end(); ++jt )
						g_cur -= temp.q[*jt/cardi.classes] * alpha(*jt);
				}
				else //retrieve stored gradient
					g_cur = gradient(i);
				// now that we have the gradient, test for extremal one:
				if ( strat.useClippedInReproc || strat.shrinkMode != sNever) //this idea temporarily stores all gradients
					temp.reprocGradients(*it) = g_cur;
				if ( canIncrease(i) && g_cur > temp.gPlus )
				{
					strat.nextI = *it;
					temp.gPlus = g_cur;
				}
				if ( g_cur < temp.gMinus ) //both immediately identify the smallest gradient
				{
					strat.nextJ = *it;
					temp.gMinus = g_cur;
				}
			}
			if ( strat.useClippedInReproc ) //now find the largest-gaining gPlus for our gMinus
			{
				double delta_g, mu_test, cur_gain, max_gain = -1e100;
				for ( it = ninaClasses(p).begin(); it != ninaClasses(p).end(); ++it ) //mt_costly_loop_marker
				{
					if ( *it == strat.nextJ ) continue;
					g_cur = temp.reprocGradients(*it);
					i = p*cardi.classes + *it;
					if ( canIncrease(i) )
					{
						delta_g = g_cur - temp.gMinus;
						mu_test = delta_g / ( 2*diagonal(p) );
						if ( !canIncrease(i, mu_test) )
							mu_test = boxMax(i) - alpha(i);
						cur_gain = mu_test*( delta_g - mu_test*diagonal(p) );
						if ( cur_gain > max_gain )
						{
							strat.nextI = *it;
							temp.gPlus = g_cur;
							max_gain = cur_gain;
						}
					}
				}
			}
			// variable-wise shrinking here:
			if ( strat.shrinkMode != sNever )
			{
				for ( it = ninaClasses(p).begin(); it != ninaClasses(p).end(); ++it ) //mt_costly_loop_marker
				{
					g_cur = temp.reprocGradients(*it);
					if ( *it == strat.nextI || *it == strat.nextJ || g_cur < 0 ) //don't shrink WS or down-pointing
						continue; 
					i = p*cardi.classes + *it;
					// shrink if no progress with this var is possible (or only up to a certain fraction of gPlus)
					if ( !canIncrease(i) )
					{
						if ( g_cur > temp.gPlus * strat.shrinkCritFraction )
						{
							deactivateVariable( i, p, *it );
						}
					}
				}
			}
			break;
		}
		case nOpt: 
		{
			if ( strat.wss == wClassInversion && !cicl.fallback_mode )
			{
				strat.nextI = cicl.i;
				strat.nextJ = cicl.j;
				temp.gPlus = gradient( p*cardi.classes + strat.nextI );
				temp.gMinus = gradient( p*cardi.classes + strat.nextJ );
				if ( !canIncrease( p*cardi.classes + strat.nextI ) ) return 0;
			}
			else
			{
				// iterate through all support classes of this sample and get extremal gradient
				for ( it = asClasses(p).begin(); it != asClasses(p).end(); ++it ) //mt_costly_loop_marker
				{
					i = p*cardi.classes + (*it);
					g_cur = gradient(i);
					// test for extremal gradient
					if ( canIncrease(i) && g_cur > temp.gPlus )
					{
						strat.nextI = (*it);
						temp.gPlus = g_cur;
					}
					if ( g_cur < temp.gMinus )
					{
						strat.nextJ = (*it);
						temp.gMinus = g_cur;
					}
				}
				if ( strat.useClippedInReproc ) //now find the largest-gaining(!) gPlus for our gMinus
				{
					double delta_g, mu_test, cur_gain, max_gain = -1e100;
					for ( it = asClasses(p).begin(); it != asClasses(p).end(); ++it ) //mt_costly_loop_marker
					{
						if ( (*it) == strat.nextJ ) continue;
						i = p*cardi.classes + (*it);
						g_cur = gradient(i);
						if ( canIncrease(i) )
						{
							delta_g = g_cur - temp.gMinus;
							mu_test = delta_g / ( 2*diagonal(p) );
							if ( !canIncrease(i, mu_test) )
								mu_test = boxMax(i) - alpha(i);
							cur_gain = mu_test*( delta_g - mu_test*diagonal(p) );
							if ( cur_gain > max_gain )
							{
								strat.nextI = (*it);
								temp.gPlus = g_cur;
								max_gain = cur_gain;
							}
						}
					}
				}
			}
			break;
		}
		default: throw SHARKEXCEPTION("[QpEbCsDecomp::selectWorkingSet] Not a valid wss mode.");
	}
	return (temp.gPlus - temp.gMinus);
}

double QpEbCsDecomp::performSmoStep()
{
	ASSERT ( strat.nextI != strat.nextJ ) 
	unsigned int p = strat.nextPat;
	unsigned int nI = p*cardi.classes + strat.nextI; //index into arrays of length cardi.variables
	unsigned int nJ = p*cardi.classes + strat.nextJ;
	
	bool wasSCi = isSC( nI );
	bool wasSCj = isSC( nJ );
	double mu = (temp.gPlus - temp.gMinus) / ( 2*diagonal(p) );
	// clip to upper constraint and update alpha
	if ( !canIncrease( nI, mu) )  
		mu = boxMax( nI ) - alpha( nI );
	ASSERT ( mu > 0 ) //there shouldn't be a way to still get worse here
	alpha( nI ) += mu;
	alpha( nJ ) -= mu;
	
	// insert both i and j as new SC (if) or clean up non-SC (else if)
	bool isSCi = isSC( nI );
	bool isSCj = isSC( nJ );
			
	// update gradients (relies on temp.q being unchanged since selectWorkingSet), and also fill wss lists
	if ( temp.q == NULL && cardi.activePatterns ) //can be NULL if nOpt or all classes SCs for nOld
	{
		temp.q = kernelMatrix.Row(p, 0, cardi.activePatterns, false, true); //try to keep dangling patterns
		MT_COUNT_KERNEL_LOOKUPS ( log.kernel_lookups += cardi.activePatterns; )
	}
	
	if ( strat.wss == wClassInversion ) //pick a random third class, not equal to members of last WS
	{
		strat.thirdClass = Rng::discrete(0, cardi.classes-1);
		while ( strat.thirdClass == strat.nextI || strat.thirdClass == strat.nextJ )
			strat.thirdClass = Rng::discrete(0, cardi.classes-1);
	}

	//UPDATE J
	for ( tActiveClasses::iterator jt = activeGlobalSCs(strat.nextJ).begin(); 
		  jt != activeGlobalSCs(strat.nextJ).end(); ++jt )
	{
		gradient(*jt) += mu * temp.q[*jt/cardi.classes]; // e = *jt/cardi.classes
	}
	//DO CICL-WSS
	if ( strat.wss == wClassInversion )
	{
		cicl.max_priority = 0;
		cicl.max_witness = 0.0;
		cicl.fallback_mode = true;
		
		cicl.asc_size_current = activeGlobalSCs( strat.nextJ ).size();
		cicl.asc_size_third = activeGlobalSCs( strat.thirdClass ).size();
		if ( cicl.asc_size_third )
		{
			cicl.hits_so_far = 0;
			cicl.counter_index = 0;
			cicl.start_index = safe_discrete( cicl.asc_size_current, 1 ); //random index into ascs(strat.nextJ)
////// snip
//			// LOOP THROUGH SUBSET: a bit tricky because tr1::unordered set is not a random access iterator.
//			// (cf. http://stackoverflow.com/questions/124671/picking-a-random-element-from-a-set)
//			// first, find the bucket the start_index-th element is in, then its index in that bucket:
//			cicl.x = cicl.start_index;
//			cicl.bucket_count = activeGlobalSCs(strat.nextJ).bucket_count();
//			for ( cicl.b=0; cicl.b < cicl.bucket_count; cicl.b++ ) 
//			{
//				if ( cicl.x < activeGlobalSCs(strat.nextJ).bucket_size(cicl.b) )
//			        break;
//				else
//					cicl.x -= activeGlobalSCs(strat.nextJ).bucket_size(cicl.b);
//		    }
//		    cicl.lit = activeGlobalSCs(strat.nextJ).begin(cicl.b);
//			while ( cicl.x > 0 )
//			{
//				++cicl.lit;
//				ASSERT( cicl.lit != activeGlobalSCs(strat.nextJ).end(cicl.b) );
//				--cicl.x;
//			}
//			while ( cicl.hits_so_far < cicl.desired_noof_hits && cicl.counter_index < cicl.asc_size_current )
//			{
//				while ( cicl.lit == activeGlobalSCs(strat.nextJ).end(cicl.b) ) //overflow to next bucket
//				{
//					++cicl.b;
//					if ( cicl.b == cicl.bucket_count ) //overflow within the entire set
//						cicl.b = 0;
//					cicl.lit = activeGlobalSCs(strat.nextJ).begin(cicl.b);
//				}
////// snap
			cicl.lit = activeGlobalSCs(strat.nextJ).begin();
			for ( unsigned int i=0; i<cicl.start_index; i++ )
				++cicl.lit;
			while ( cicl.hits_so_far < cicl.desired_noof_hits && cicl.counter_index < cicl.asc_size_current )
			{
				if ( cicl.lit == activeGlobalSCs(strat.nextJ).end() ) //continue from beginning
				    cicl.lit = activeGlobalSCs(strat.nextJ).begin();
////// snup
				++cicl.counter_index;
				cicl.e = *cicl.lit/cardi.classes; //current example
				cicl.t = cicl.e*cardi.classes+strat.thirdClass;
				if ( !isSC(cicl.t) || 
					 ( strat.shrinkMode != sNever
					   && asClasses(cicl.e).find( strat.thirdClass ) == asClasses(cicl.e).end() ) 
				   ) //only look at samples for which both classes are active SPs
				{
					++cicl.lit;
					continue;
				}
				++cicl.hits_so_far;
				//actual stuff:
				cicl.g = gradient(*cicl.lit);
				cicl.cur_priority = 0;
				if ( !strat.useClippedAsWitness )
				{
					cicl.mu_test = (cicl.g-gradient(cicl.t)) / (2*diagonal(cicl.e)); //this is used for up-down-swap
					cicl.cur_witness = fabs(cicl.mu_test); //and this as criterion
				}
				else
				{
					cicl.delta_g = cicl.g - gradient(cicl.t);
					cicl.mu_test = cicl.delta_g / ( 2*diagonal(cicl.e) );
					if ( cicl.mu_test > 0 )
					{
						if ( !canIncrease(*cicl.lit, cicl.mu_test) ) // j would hit its bound
						{
							cicl.mu_test = boxMax(*cicl.lit) - alpha(*cicl.lit);
							if ( strat.nextJ != label(cicl.e) ) // j would be deactivated
								cicl.cur_priority = 1;
						}
					}
					else
					{
						if ( !canIncrease(cicl.t, -cicl.mu_test) ) // t would hit its bound
						{
							cicl.mu_test = -(boxMax(cicl.t)-alpha(cicl.t));
							if ( strat.thirdClass != label(cicl.e) ) // t would be deactivated
								cicl.cur_priority = 1;
						}
					}
					cicl.cur_witness = cicl.mu_test*( cicl.delta_g - cicl.mu_test*diagonal(cicl.e) ); //minus sign factors out
				}
				if (      cicl.cur_priority > cicl.max_priority 
					 || ( cicl.cur_witness > cicl.max_witness && cicl.cur_priority == cicl.max_priority ) 
				   )
				{
					cicl.candidate = cicl.e;
					cicl.max_witness = cicl.cur_witness;
					cicl.max_priority = cicl.cur_priority;
					if ( cicl.mu_test > 0 )
					{
						cicl.i = strat.nextJ;
						cicl.j = strat.thirdClass;
					}
					else
					{
						cicl.i = strat.thirdClass;
						cicl.j = strat.nextJ;
					}
				}
				// finished -- set iterator to next element
				++cicl.lit;
			}
		}
	}
		
	//UPDATE I
	for ( tActiveClasses::iterator it = activeGlobalSCs(strat.nextI).begin(); 
		  it != activeGlobalSCs(strat.nextI).end(); ++it )
	{
		gradient(*it) -= mu * temp.q[*it/cardi.classes]; // e = *it/cardi.classes
	}
	//DO CICL-WSS
	if ( strat.wss == wClassInversion )
	{
		cicl.asc_size_current = activeGlobalSCs( strat.nextI ).size();
		if ( cicl.asc_size_third )
		{
			cicl.hits_so_far = 0;
			cicl.counter_index = 0;
			cicl.start_index = safe_discrete( cicl.asc_size_current, 1 ); //random index into ascs(strat.nextI)
////// snip
//			// LOOP THROUGH SUBSET: a bit tricky because tr1::unordered set is not a random access iterator.
//			// (cf. http://stackoverflow.com/questions/124671/picking-a-random-element-from-a-set)
//			// first, find the bucket the start_index-th element is in, then its index in that bucket:
//			cicl.x = cicl.start_index;
//			cicl.bucket_count = activeGlobalSCs(strat.nextI).bucket_count();
//			for ( cicl.b=0; cicl.b < cicl.bucket_count; cicl.b++ ) 
//			{
//				if ( cicl.x < activeGlobalSCs(strat.nextI).bucket_size(cicl.b) )
//			        break;
//				else
//					cicl.x -= activeGlobalSCs(strat.nextI).bucket_size(cicl.b);
//		    }
//		    cicl.lit = activeGlobalSCs(strat.nextI).begin(cicl.b);
//			while ( cicl.x > 0 )
//			{
//				++cicl.lit;
//				ASSERT( cicl.lit != activeGlobalSCs(strat.nextI).end(cicl.b) );
//				--cicl.x;
//			}
//			while ( cicl.hits_so_far < cicl.desired_noof_hits && cicl.counter_index < cicl.asc_size_current )
//			{
//				while ( cicl.lit == activeGlobalSCs(strat.nextI).end(cicl.b) ) //overflow to next bucket
//				{
//					++cicl.b;
//					if ( cicl.b == cicl.bucket_count ) //overflow within the entire set
//						cicl.b = 0;
//					cicl.lit = activeGlobalSCs(strat.nextI).begin(cicl.b);
//				}
////// snap
			cicl.lit = activeGlobalSCs(strat.nextI).begin();
			for ( unsigned int i=0; i<cicl.start_index; i++ )
				++cicl.lit;
			while ( cicl.hits_so_far < cicl.desired_noof_hits && cicl.counter_index < cicl.asc_size_current )
			{
				if ( cicl.lit == activeGlobalSCs(strat.nextI).end() ) //continue from beginning
				    cicl.lit = activeGlobalSCs(strat.nextI).begin();
////// snup
				++cicl.counter_index;
				cicl.e = *cicl.lit/cardi.classes; //current example
				cicl.t = cicl.e*cardi.classes+strat.thirdClass;
				if ( !isSC(cicl.t) || 
					 ( strat.shrinkMode != sNever
					   && asClasses(cicl.e).find( strat.thirdClass ) == asClasses(cicl.e).end() ) 
				   ) //only look at samples for which both classes are active SPs
				{
					++cicl.lit;
					continue;
				}
				++cicl.hits_so_far;
				//actual stuff:
				cicl.g = gradient(*cicl.lit);
				cicl.cur_priority = 0;
				if ( !strat.useClippedAsWitness )
				{
					cicl.mu_test = (cicl.g-gradient(cicl.t)) / (2*diagonal(cicl.e)); //this is used for up-down-swap
					cicl.cur_witness = fabs(cicl.mu_test); //and this as criterion
				}
				else
				{
					cicl.delta_g = cicl.g - gradient(cicl.t);
					cicl.mu_test = cicl.delta_g / ( 2*diagonal(cicl.e) );
					if ( cicl.mu_test > 0 )
					{
						if ( !canIncrease(*cicl.lit, cicl.mu_test) ) // j would hit its bound
						{
							cicl.mu_test = boxMax(*cicl.lit) - alpha(*cicl.lit);
							if ( strat.nextI != label(cicl.e) ) // j would be deactivated
								cicl.cur_priority = 1;
						}
					}
					else
					{
						if ( !canIncrease(cicl.t, -cicl.mu_test) ) // t would hit its bound
						{
							cicl.mu_test = -(boxMax(cicl.t)-alpha(cicl.t));
							if ( strat.thirdClass != label(cicl.e) ) // t would be deactivated
								cicl.cur_priority = 1;
						}
					}
					cicl.cur_witness = cicl.mu_test*( cicl.delta_g - cicl.mu_test*diagonal(cicl.e) ); //minus sign factors out
				}
				if (      cicl.cur_priority > cicl.max_priority 
					 || ( cicl.cur_witness > cicl.max_witness && cicl.cur_priority == cicl.max_priority ) 
				   )
				{
					cicl.candidate = cicl.e;
					cicl.max_witness = cicl.cur_witness;
					cicl.max_priority = cicl.cur_priority;
					if ( cicl.mu_test > 0 )
					{
						cicl.i = strat.nextI;
						cicl.j = strat.thirdClass;
					}
					else
					{
						cicl.i = strat.thirdClass;
						cicl.j = strat.nextI;
					}
				}
				// finished -- set iterator to next element
				++cicl.lit;
			}
		}
		if ( cicl.max_witness > 0 )
			cicl.fallback_mode = false;
	}

	if ( !wasSCi && isSCi ) 
	{
		asClasses(p).insert( strat.nextI );
		if ( strat.proc == nOld ) //if nNew, done in flip. if nOpt, never happens
			activeGlobalSCs(strat.nextI).insert( nI ); //update active list
		gradient( nI ) = temp.gPlus - mu*diagonal(p); //if not SC, no need to assign. if formerly SC, already correct
	}
	else if ( wasSCi && !isSCi ) 
	{
		asClasses(p).erase( strat.nextI );
		activeGlobalSCs(strat.nextI).erase( nI ); //update active list
	}
	if ( !wasSCj && isSCj ) 
	{
		asClasses(p).insert( strat.nextJ );
		if ( strat.proc == nOld ) //if nNew, done in flip. if nOpt, never happens
			activeGlobalSCs(strat.nextJ).insert( nJ ); //update active list
		gradient( nJ ) = temp.gMinus + mu*diagonal(p); //if not SC, no need to assign. if formerly SC, already correct
	}
	else if ( wasSCj && !isSCj )
	{
		asClasses(p).erase( strat.nextJ );
		activeGlobalSCs(strat.nextJ).erase( nJ ); //update active list
	}
	// insert as new AP (if) or remove from SP-section (else)
	if ( !asClasses(p).empty() )
	{
		if ( p >= cardi.sPatterns ) //was a new sample: move to active section
		{
			// temporarily move it to inactive section
			flipAll( cardi.sPatterns, p, fInsert_first ); //careful here: flip inherits 1stArg < 2ndArg from cachedMatrix.flip
			p = cardi.sPatterns;
			++cardi.sPatterns;
			// then continue and move it to active section
			flipAll( cardi.activePatterns, p, fInsert_second );
			p = cardi.activePatterns;
			++cardi.activePatterns;
			strat.nextPat = p; //keep vars intact
			nI = p*cardi.classes + strat.nextI;
			nJ = p*cardi.classes + strat.nextJ;
		}
	}
	else //remove from AP section (test for p<cardi.activePatterns unnecessary)
	{
		// release all vars and the cache
		kernelMatrix.CacheRowRelease(p);
		flipAll( p, cardi.activePatterns-1, fRemove_first ); //temporarily move it to inactive region
		p = cardi.activePatterns-1;
		--cardi.activePatterns;
		flipAll( p, cardi.sPatterns-1, fRemove_second ); //then continue and move it to non-SP-section
		p = cardi.sPatterns-1;
		--cardi.sPatterns;
		strat.nextPat = p; //keep vars intact
		nI = p*cardi.classes + strat.nextI;
		nJ = p*cardi.classes + strat.nextJ;
	}
	
	// see if the last update step made nextI shrinkable
	if ( strat.shrinkMode != sNever && !asClasses(p).empty() ) //never shrink non-SPs (important)
	{
		if ( !canIncrease(nI) && gradient(nI) > 0 )
		{
			unsigned int i;
			double g_cur, g_max = -1e100;
			// over all increasables, get max gradient
			for ( tActiveClasses::iterator kt = asClasses(p).begin(); kt != asClasses(p).end(); ++kt )
			{
				i = p*cardi.classes + (*kt);
				if ( *kt == strat.nextI || !canIncrease(i) ) continue; //don't compare to self
				g_cur = gradient(i);
				if ( g_cur > g_max )
					g_max = g_cur;
			}
			
			if ( gradient(nI) > g_max * strat.shrinkCritFraction )
			{
				deactivateVariable( nI, p, strat.nextI );
				// now test if the whole sample can be deactivated
				g_cur = 1e100; //dummy marker value
				bool deactivate = false;
				for ( tActiveClasses::iterator kt = ninaClasses(p).begin(); kt != ninaClasses(p).end(); ++kt )
				{
					i = p*cardi.classes + (*kt);
					if ( !isSC(i) ) //we do not know the gradient
					{
						if ( strat.shrinkMode < sAgressive ) //and non-SC usually means no deactivation anyway
						{
							deactivate = false;
							break;
						}
					}
					else //we know the gradient and it should equal all other known gradients
					{
						if ( g_cur == 1e100 )
							g_cur = gradient(i);
						else if ( fabs( gradient(i)-g_cur ) > fabs(g_cur)*strat.shrinkEqualFraction )
						{
							deactivate = false;
							break;
						}
					}
				}
				if ( deactivate )
				{
////					cout << " DEACTIVATE " << endl;
					deactivateExample(p);
				}
			}
		}
	}
	// increment counters
	introsp.dual += mu * ((temp.gPlus - temp.gMinus) - mu*diagonal(p));
	return mu * ((temp.gPlus - temp.gMinus) - mu*diagonal(p)); //copied from original paper
}

void QpEbCsDecomp::shuffleSamples()
{
	unsigned int i, j, ic = cardi.examples;
	for (i=1; i<ic; i++)
	{
		j = Rng::discrete(0, i);
		if (i != j) XCHG_A(unsigned int, lottery, i, j);
	}
}

// requires i < j
void QpEbCsDecomp::flipAll( unsigned int i, unsigned int j, eFlipVariants f )
{
	tActiveClasses::iterator it;
	unsigned int c;
	if ( i == j )
	{
		// IV-a (insert new SCs as active)
		if ( f == fInsert_second ) //insert active SCs from lower-index sample 
			for ( it = asClasses(i).begin(); it != asClasses(i).end(); ++it )
				activeGlobalSCs(*it).insert( i*cardi.classes + (*it) );
		if ( f == fRemove_second )
			for ( c=0; c<cardi.classes; c++)
				ninaClasses(j).insert(c);
		return;
	}
	else if ( i > j ) throw SHARKEXCEPTION("[QpEbCsDecomp::flip] Invalid arguments.");
	
	ASSERT( asClasses(i).size() == 0 ) //standard safety check for all
	ASSERT( f != fDeactivate || ninaClasses(i).size() == 0 ) //treat all vars first, then the example, then flip.
	ASSERT( f != fInsert_second || ninaClasses(j).size() == cardi.classes ) //no shrinking before insertion
	
	// vars of cardinality cardi.examples
	curIndex( origIndex(i) ) = j; //flip curIndex
	curIndex( origIndex(j) ) = i;
	XCHG_A(unsigned int, origIndex, i, j); //flip OrigIndex
	XCHG_A(unsigned int, label, i, j);
	XCHG_A(double, diagonal, i, j);
	
	// vars of cardinality cardi.variables: classes always maintain original order
	unsigned int bi = i * cardi.classes;
	unsigned int bj = j * cardi.classes;
	unsigned int b = (i+1) * cardi.classes;
	unsigned int bic, bjc;
	for (bic=bi,bjc=bj; bic<b; bic++,bjc++) //two-var loop //mt_costly_loop_marker
	{
		XCHG_A(double, alpha, bic, bjc);
		XCHG_A(double, boxMax, bic, bjc);
		XCHG_A(double, gradient, bic, bjc);
	}
	
	// update all unordered_sets holding the current SCs (needs to happen after above swaps)
	
	//  I-d (remove unknown dormants from i)
	if ( f == fDeactivate || f == fInsert_second ) //we do not know the dormant SCs (and all SCs are dormant)
		for (c=0; c<cardi.classes; c++)
			if ( isSC(i*cardi.classes + c) )
				dormantGlobalSCs(c).erase( i*cardi.classes + c );
	// II-a (remove active from j)
	if ( f == fDeactivate || f == fRemove_first )
		for ( it = asClasses(j).begin(); it != asClasses(j).end(); ++it )
			activeGlobalSCs(*it).erase( j*cardi.classes + (*it) );
	// II-d-1 (remove known dormant from j) (dormant SC can only be the true-label one)
	if ( f == fDeactivate || f == fRemove_first )
		if ( isSC(j*cardi.classes+label(j) ) )
			if ( asClasses(j).find(label(j)) == asClasses(j).end() )  //maybe faster to always directly delete?
				dormantGlobalSCs( label(j) ).erase( j*cardi.classes+label(j) );
	// II-d-2 (remove unknown dormants from j)
	if ( f == fRemove_second ) //we do not know the dormant SCs (and all SCs are dormant)
		for (c=0; c<cardi.classes; c++)
			if ( isSC(j*cardi.classes + c) )
				dormantGlobalSCs(c).erase( j*cardi.classes + c );
	// III (swap nina- and asClasses)
	XCHG_A(tActiveClasses, ninaClasses, i, j); //swap sample-wise sets
	XCHG_A(tActiveClasses, asClasses, i, j); //swap sample-wise sets
	// IV-a (insert active to i from old j)
	if ( f != fRemove_second && f != fInsert_first )
		for ( it = asClasses(i).begin(); it != asClasses(i).end(); ++it )
			activeGlobalSCs(*it).insert( i*cardi.classes + (*it) );
	// IV-d-1 (insert known dormant to i from old j) (dormant SC can only be the true-label one)
	if ( f == fDeactivate || f == fRemove_first )
		if ( isSC(i*cardi.classes+label(i) ) )
			if ( asClasses(i).find(label(i)) == asClasses(i).end() ) //for insertion, this check is necessary
				dormantGlobalSCs( label(i) ).insert( i*cardi.classes+label(i) );
	// IV-d-2 (insert unknown dormants to i from old j) we do not know the dormant SCs (and all SCs are dormant)
	if ( f == fRemove_second )
		for (c=0; c<cardi.classes; c++)
			if ( isSC(i*cardi.classes + c) )
				dormantGlobalSCs(c).insert( i*cardi.classes + c );
	// V-d (insert unknown dormants to j from old i) //we do not know the dormant SCs (and all SCs are dormant)
	if ( f == fDeactivate || f == fInsert_second )
		for (c=0; c<cardi.classes; c++)
			if ( isSC(j*cardi.classes + c) )
				dormantGlobalSCs(c).insert( j*cardi.classes + c );
	
	// this relates to multi-epoch settings. however, a strategy (and testing/debugging) for that is still lacking
	if ( f == fRemove_second )
		for ( c=0; c<cardi.classes; c++)
			ninaClasses(j).insert(c);

	// notify cache
	kernelMatrix.FlipColumnsAndRows(i, j);
	
}

//////QpEbCsDecomp::tActiveClasses::iterator QpEbCsDecomp::deactivateVariable( unsigned int v, unsigned int e, unsigned int c )
void QpEbCsDecomp::deactivateVariable( unsigned int v, unsigned int e, unsigned int c )
{
	ninaClasses(e).erase(c); //always good
	if ( isSC(v) ) //also treat SC-relevant containers
	{
		asClasses(e).erase(c);
		dormantGlobalSCs(c).insert(v); //(only not necessarily implies wasSC if shrinked at the very end of smo)
		std::pair<tActiveClasses::iterator, tActiveClasses::iterator> mypair = 
			activeGlobalSCs(c).equal_range(v); 
		ASSERT( mypair.first != activeGlobalSCs(c).end() );
//////		return activeGlobalSCs(c).erase(mypair.first); //return pointer, in case we're in a loop through activeGlobalSCs
		activeGlobalSCs(c).erase(mypair.first); //return pointer, in case we're in a loop through activeGlobalSCs
	}
//////	else //don't waste time looking for erasables. for safety, wasSC defaults to true.
//////		return activeGlobalSCs(c).begin(); //pretty arbitrary
}

unsigned int QpEbCsDecomp::deactivateExample( unsigned int e )
{
	ASSERT( ninaClasses(e).empty() );
	ASSERT( asClasses(e).empty() );
	kernelMatrix.CacheRowRelease( e );
	flipAll( e, cardi.activePatterns-1, fDeactivate );
	--cardi.activePatterns;
	return cardi.activePatterns; //remember to keep vars intact - this return value will help you
}

double QpEbCsDecomp::calcPrimal()
{
	introsp.primal = 0.0;
	float* krow; //kernel row
	unsigned int e, c, i, true_label; //helper
	double sum_weights = 0.0; //norm of weight vector
	double sum_slacks = 0.0; //sum over slack variables
	double cur_score, true_score, max_other_scores;
	
	// run over all examples, utilizing all known gradients:
	for ( e=0; e<cardi.examples; e++ )
	{
		krow = NULL;
		true_label = label(e);
		max_other_scores = -1e100;
		//GET MAXIMUM SCORE
		for ( c=0; c<cardi.classes; c++ )
		{
			i = e*cardi.classes + c;
			// GET SCORE FOR THIS CLASS
			// if it is not an SC, or not listed as active SC, we have to calculate the score:
			if ( !isSC(i) || asClasses(e).find(c) == asClasses(e).end() )
			{
				if ( krow == NULL && cardi.sPatterns ) //get kernel row
				{
					krow = kernelMatrix.Row(e, 0, cardi.sPatterns);
					MT_COUNT_KERNEL_LOOKUPS ( log.kernel_lookups += cardi.sPatterns; )
				}
				cur_score = 0.0;
				for ( tActiveClasses::iterator it = activeGlobalSCs(c).begin(); 
					  it != activeGlobalSCs(c).end(); ++it )
					cur_score += krow[*it/cardi.classes] * alpha(*it);
				if ( strat.shrinkMode != sNever ) //if necessary, consider inactive contributions as well
				{
					for ( tActiveClasses::iterator it = dormantGlobalSCs(c).begin();
						  it != dormantGlobalSCs(c).end(); ++it )
						cur_score += krow[*it/cardi.classes] * alpha(*it);
				}
			}
			else //we are an sc, and we are active: simply retreive
			{
				cur_score = (c==true_label) - gradient(i);
			}
			if ( cur_score > max_other_scores && c != true_label )
				max_other_scores = cur_score;
			if ( c == true_label )
				true_score = cur_score;
			// also treat the weight vector:
			sum_weights += alpha(i)*cur_score;
		}
		//if there is some slack, add it to sum_slacks
		if ( true_score-1 < max_other_scores ) //slack var is > 0
			sum_slacks += max_other_scores - true_score + 1;
	}
	introsp.primal = 0.5*sum_weights + strat.creg*sum_slacks;
	return introsp.primal;
}


