//===========================================================================
/*!
 *  \file Utilities.cpp
 *
 *  \brief Different utilities of generic use, or especially for SVM solvers
 *
 *  \author M.Tuma
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

#include <ReClaM/Utilities.h>


SvmStatesCollection::SvmStatesCollection( unsigned int classes, KernelFunction *in_kernel, const Array<double>& train_input )
{
	m_noofClasses = classes;
	m_noofTrainExamples = train_input.dim(0);
	mep_kernel = in_kernel;
	mep_trainData = &train_input;
	m_wasInitialized = false;
	m_wasMadeSparse = false;
}

SvmStatesCollection::~SvmStatesCollection()
{ }

void SvmStatesCollection::declareIntentions( unsigned int noof_shots, unsigned int raw_param_length, 
											unsigned int raw_closure_length_double, unsigned int raw_closure_length_unsigned,
											unsigned int noof_measures )
{
	m_nextShot = 0;
	m_wasInitialized = true;
	m_noofShots = noof_shots;
	m_noofMeasures = noof_measures;
	m_performanceMeasures.resize( m_noofMeasures, m_noofShots, false );
	
	m_paramLength = raw_param_length;
	m_closureLengthDouble = raw_closure_length_double;
	m_closureLengthUnsigned = raw_closure_length_unsigned;
	m_rawParameterShots.resize( m_noofShots, m_paramLength, false );
	m_rawClosureShotsDouble.resize( m_noofShots, m_closureLengthDouble, false );
	m_rawClosureShotsUnsigned.resize( m_noofShots, m_closureLengthUnsigned, false );
	
	m_snapshots.clear();
}

void SvmStatesCollection::pushSnapShot( const Array< double >& param, const Array< double >& closure_double, 
									   const Array< unsigned int >& closure_unsigned )
{
	ASSERT( m_wasInitialized );
	ASSERT( m_nextShot < m_noofShots );
	m_rawParameterShots[m_nextShot] = param;
	if ( m_closureLengthDouble )
		m_rawClosureShotsDouble[m_nextShot] = closure_double;
	if ( m_closureLengthUnsigned )
		m_rawClosureShotsUnsigned[m_nextShot] = closure_unsigned;
	++m_nextShot;
}

void SvmStatesCollection::makeHistorySparse( unsigned int svm_variant )
{
	ASSERT( m_wasInitialized );
	eSvmMode emode = static_cast< eSvmMode > (svm_variant);	
	switch ( emode )
	{
		case eEBCS:
		{
			double cur_value;
			
			unsigned int cur_orig_index;
			Array < unsigned int > * p_orig_indices = &m_rawClosureShotsUnsigned; //alias for clarity
			for (unsigned int i=0; i<m_noofShots; i++)
			{
				tSupPatCollection cur_model;
				for (unsigned int e=0; e<m_noofTrainExamples; e++)
				{
					tSupClassCollection cur_class_list;
					cur_orig_index = (*p_orig_indices)(i, e);
					for (unsigned int c=0; c<m_noofClasses; c++)
					{
						cur_value = m_rawParameterShots(i, e*m_noofClasses+c );
						if ( cur_value != 0.0) //found a support class
							cur_class_list.push_back( std::make_pair(c, cur_value) );
					}
					if ( cur_class_list.size() ) //example has >= 1 support class -> make support example
						cur_model.push_back( std::make_pair(cur_orig_index, cur_class_list ) );
				}
				m_snapshots.push_back( cur_model );
			}
			break;
		}
		default: throw SHARKEXCEPTION("[SvmStatesCollection::makeHistorySparseEbCs]. Not yet supported (please define for your SVM).");
	}
	m_wasMadeSparse = true;
}

void SvmStatesCollection::storePrimal( unsigned int svm_variant, unsigned int target_index, 
									   const Array<double>& train_targets, double regC )
{
//    std::cout << "\tstoring primal " << std::endl;
	ASSERT( m_wasInitialized );
	ASSERT( m_wasMadeSparse );
	eSvmMode emode = static_cast< eSvmMode > (svm_variant);
	
	switch ( emode )
	{
		case eEBCS:
		{
			// helper variables
			unsigned int cur_ex;
			unsigned int cur_class;
			unsigned int true_label;
            int max_label;
			double w2;
			double cur_val;
			double sum_slacks;
			double true_score;
			double cur_kernel_entry;
			double max_other_scores;
			double scores [m_noofClasses];
			double beta_is [m_noofClasses];
			for (unsigned int i=0; i<m_noofShots; i++) //loop over all stored snapshots
			{
//                std::cout << "looking at shot " << i << std::endl;
				w2 = 0.0;
				sum_slacks = 0.0;
				for (unsigned int j=0; j<m_noofTrainExamples; j++) //loop over ALL training examples
				{
//                    std::cout << "  looking at sample " << j << std::endl;
                    max_label = -1;
					for (unsigned int k=0; k<m_noofClasses; k++) //initialize to zero
						{ scores[k] = 0.0; beta_is[k] = 0.0; }
                        
					// compute scores: loop over support patterns of snapshot i
					for ( tSupPatCollection::iterator it = m_snapshots[i].begin(); it != m_snapshots[i].end(); it++)
					{
						cur_ex = it->first;
						cur_kernel_entry = mep_kernel->eval( (*mep_trainData)[cur_ex], (*mep_trainData)[j] );
						// iterate over all support classes
						for ( tSupClassCollection::iterator jt = it->second.begin(); jt != it->second.end(); jt++)
						{
							cur_class = jt->first;
							cur_val = jt->second;
							scores[cur_class] += cur_val * cur_kernel_entry;
						}
						if ( cur_ex == j ) //get beta values of current example (can't directly assign, because indices are permuted)
						{
							for ( tSupClassCollection::iterator jt = it->second.begin(); jt != it->second.end(); jt++)
							{
								cur_class = jt->first;
								cur_val = jt->second;
								beta_is[cur_class] = cur_val;
							}
						}
					}
                    // now that we have all scores, compute contribution to primal and slack vars
					max_other_scores = -1e100;
//                    std::cout << " train_targets.ndim() " << train_targets.ndim() << std::endl;
//                    std::cout << " train_targets.nelem() " << train_targets.nelem() << std::endl;
//                    std::cout << " train_targets.dim(0) " << train_targets.dim(0) << std::endl;
//                    std::cout << " train_targets.dim(1) " << train_targets.dim(1) << std::endl;
//                    std::cout << " train_targets.dim(2) " << train_targets.dim(2) << std::endl;
//                    std::cout << "    now looking at target " << std::endl;
					true_label = train_targets(j, 0);
//                    std::cout << "    done. " << std::endl;
					true_score = scores[ true_label ];
					for (unsigned int k=0; k<m_noofClasses; k++)
					{
						w2 += beta_is[k] * scores[k];
						if ( k == true_label )
							continue;
						if ( scores[k] > max_other_scores )
                        {
							max_other_scores = scores[k];
                            max_label = k;
                        }
					}
					if ( true_score-1 < max_other_scores ) //slack var is > 0
                    {
						sum_slacks += max_other_scores - true_score + 1;
                    }
				}
				m_performanceMeasures( target_index, i) = 0.5 * w2 + regC * sum_slacks;
			}
			break;
		}
		default: throw SHARKEXCEPTION("[SvmStatesCollection::makeHistorySparseEbCs]. Not yet supported (please define for your SVM).");
	}
//    std::cout << "\tdone storing primal " << std::endl;
}

void SvmStatesCollection::storeTestErr( unsigned int svm_variant, unsigned int target_index, 
										const Array<double>& test_inputs, const Array<double>& test_targets )
{
	ASSERT( m_wasInitialized );
	ASSERT( m_wasMadeSparse );
	eSvmMode emode = static_cast< eSvmMode > (svm_variant);
	
	switch ( emode )
	{
		case eEBCS:
		{
			// tmp helper vars:
			unsigned int noof_test_ex = test_inputs.dim(0);
			unsigned int cur_ex;
			unsigned int cur_class;
			unsigned int mistakes;
			unsigned int max_index;
			double cur_kernel_entry;
			double cur_val;
			double max_score;
			double scores [m_noofClasses];
			
			for (unsigned int i=0; i<m_noofShots; i++)
			{
				mistakes = 0;
				for (unsigned int j=0; j<noof_test_ex; j++) //loop over ALL test examples
				{
					for (unsigned int k=0; k<m_noofClasses; k++) //initialize to zero
						scores[k] = 0.0;
					// compute scores: loop over support patterns of snapshot i
					for ( tSupPatCollection::iterator it = m_snapshots[i].begin(); it != m_snapshots[i].end(); it++)
					{
						cur_ex = it->first;
						cur_kernel_entry = mep_kernel->eval( (*mep_trainData)[cur_ex], test_inputs[j] );
						// iterate over all support classes
						for ( tSupClassCollection::iterator jt = it->second.begin(); jt != it->second.end(); jt++)
						{
							cur_class = jt->first;
							cur_val = jt->second;
							scores[cur_class] += cur_val * cur_kernel_entry;
						}
					}
					//get prediction for current test example
					max_score = -1e100;
					for (unsigned int k=0; k<m_noofClasses; k++)
					{
						if ( scores[k] > max_score )
						{
							max_score = scores[k];
							max_index = k;
						}
					}
					if ( max_index != test_targets(j, 0) )
						++mistakes;
				}
				m_performanceMeasures( target_index, i) = mistakes / (double) noof_test_ex;
			}
			break;
		}
		default: throw SHARKEXCEPTION("[SvmStatesCollection::makeHistorySparseEbCs]. Not yet supported (please define for your SVM).");
	}
}

void SvmStatesCollection::storeDirectly( double what, unsigned int target_index, unsigned int shot_number )
{
	ASSERT( m_wasInitialized );
	m_performanceMeasures( target_index, shot_number ) = what;
}

double SvmStatesCollection::accessDirectly( unsigned int target_index, unsigned int shot_number )
{
	ASSERT( m_wasInitialized );
	return m_performanceMeasures( target_index, shot_number );
}

void SvmStatesCollection::printPerformanceMeasureAsNumPyArray( unsigned int target_index )
{
//    std::cout << "\tstarting print " << std::endl;
	ASSERT( m_wasInitialized );
	std::cout << "[ ";
	for (unsigned int i=0; i<m_noofShots; i++)
	{
		std::cout << m_performanceMeasures( target_index, i );
		if ( i != m_noofShots-1 )
			std::cout << ", ";
	}
	std::cout << " ]" << std::endl;
//    std::cout << "\tdone printing " << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
