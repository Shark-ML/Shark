/*!
 * 
 * \brief       Objective function for double non-Markov pole with special fitness functions
 *
 * Class for balancing two poles on a cart using a fitness function
 * that punishes oscillating, i.e. quickly moving the cart back and
 * forth to balance the poles. 
 * 
 * Based on code written by Verena Heidrich-Meisner for the paper
 *
 * V. Heidrich-Meisner and
 * C. Igel. Neuroevolution strategies for episodic reinforcement
 * learning. Journal of Algorithms, 64(4):152–168, 2009.  
 * 
 * Special fitness function from the paper 
 *
 * F. Gruau, D. Whitley, L. Pyeatt, A comparison between cellular
 * encoding and direct encoding for genetic neural networks, in:
 * J.R. Koza, D.E. Goldberg, D.B. Fogel, R.L. Riol (Eds.), Genetic
 * Programming 1996: Proceedings of the First Annual Conference, MIT
 * Press, 1996, pp. 81–89.
 *
 * \author      Johan Valentin Damgaard
 * \date        -
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_POLE_GRUAU_OBJECTIVE_FUNCTION
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_POLE_GRUAU_OBJECTIVE_FUNCTION

#include <iostream>
#include <exception>

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Models/OnlineRNNet.h>
#include <shark/LinAlg/Base.h>

#include <shark/ObjectiveFunctions/Benchmarks/PoleSimulators/DoublePole.h>

namespace shark {

//! \brief 
//! 
//! Class for balancing two poles on a cart using a fitness function that punishes
//! oscillating, i.e. quickly moving the cart back and forth to balance the poles.
//! Based on code written by Verena Heidrich-Meisner for the paper
//! 
//! V. Heidrich-Meisner and C. Igel. Neuroevolution strategies for episodic reinforcement learn-ing. Journal of Algorithms, 64(4):152–168, 2009.
//! 
//!
//! Special fitness function from the paper
//! 
//! F. Gruau, D. Whitley, L. Pyeatt, A comparison between cellular encoding and direct encoding for genetic neural networks, in: J.R. Koza, D.E. Goldberg,
//! D.B. Fogel, R.L. Riol (Eds.), Genetic Programming 1996: Proceedings of the First Annual Conference, MIT Press, 1996, pp. 81–89.
class GruauPole : public SingleObjectiveFunction {

public:
  //! \param hidden Number of hidden neurons in underlying neural network
  //! \param bias Whether to use bias in neural network
  //! \param sigmoidType Activation sigmoid function for neural network
  //! \param normalize Whether to normalize input before use in neural network
  //! \param max_pole_evaluations Balance goal of the function, i.e. number of steps that pole should be able to balance without failure
  GruauPole(std::size_t hidden, bool bias,
	    RecurrentStructure::SigmoidType sigmoidType = RecurrentStructure::FastSigmoid,
	    bool normalize = true,
	    std::size_t max_pole_evaluations = 1000)
    : m_maxPoleEvals(max_pole_evaluations),
      m_normalize(normalize) {
    
    if (sigmoidType == RecurrentStructure::Linear) {
      std::cerr << "Cannot use linear activation function for pole balancing."
	   << std::endl;
      exit(EXIT_FAILURE);
    }    

    std::size_t inputs = 3;
    
    // set features
    m_features |= CAN_PROPOSE_STARTING_POINT;
    
    // set number of variables/weights.
    // number of outputs is always 1. 
    // dimensions depend on whether we use bias
    if (bias){
      m_dimensions = (hidden + 1) * (hidden + 1) +
	inputs * (hidden + 1) + hidden + 1;
    }
    else {
      m_dimensions = (hidden + 1) * (hidden + 1) + inputs * (hidden + 1);
    }    
    
    // make RNNet
    mp_struct = new RecurrentStructure();
    mp_struct->setStructure(inputs, hidden, 1, bias, sigmoidType);
    mp_net = new PoleRNNet(mp_struct);
    
    // check dimensions match
    if(m_dimensions != mp_net->numberOfParameters()) {
      std::cerr << "Gruau pole RNNet: Dimensions do not match, "
	   << m_dimensions
	   << " != " <<  mp_net->numberOfParameters() << std::endl;
      exit(EXIT_FAILURE);
    }
    
    // set eval count
    m_evaluationCounter = 0;
  }

  ~GruauPole(){
    delete mp_struct;
    delete mp_net;
  }


  std::string name() {
    return "Objective Function for non-Markov double pole with Gruau fitness.";
  }

  //! \brief Returns degrees of freedom
  std::size_t numberOfVariables()const{
    return m_dimensions;
  }

  //! \brief Always proposes to start in a zero vector with appropriate degrees of freedom
  SearchPointType proposeStartingPoint() const{
    SearchPointType startingPoint(m_dimensions);
    for(std::size_t i = 0; i != m_dimensions; i++) {
      startingPoint(i) = 0.0; // starting mean = 0
    }
    return startingPoint;
  }

  //! \brief Evaluates weight vector on special fitness function from Gruau paper
  //! \param input Vector to be evaluated.
  //! \return Fitness of vector  
  ResultType eval(const SearchPointType &input) const{
    SIZE_CHECK(input.size() == m_dimensions );
    m_evaluationCounter++;
    return gruauFit(input);
  }

  //! \brief Evaluates weight vector on special fitness function from Gruau paper
  //! \param input Vector to be evaluated.
  //! \return Fitness of vector
  ResultType gruauFit(const SearchPointType &input) const{
    SIZE_CHECK(input.size() == m_dimensions );
    double init_angle = 0.07;
    DoublePole pole(false, m_normalize);
    RealVector state(3);
    RealMatrix output(1,1);
    double totalJiggle = 0;
    std::size_t jiggleSize = 100;
    double jiggle[jiggleSize]; // jiggle for last 100 evals
    std::size_t eval_count = 0;
    bool failed = false;
    
    pole.init(init_angle);
    mp_net->resetInternalState();
    mp_net->setParameterVector(input);
    
    while(!failed && eval_count < m_maxPoleEvals) {
      pole.getState(state);
      RealMatrix newState(1,3);
      row(newState,0) = state;
      mp_net->eval(newState,output);
      pole.move(convertToPoleMovement(output(0,0)));      
      jiggle[eval_count % jiggleSize] = pole.getJiggle();
      failed = pole.failure();
      eval_count++;
    }

    if(eval_count >= jiggleSize){
      for(std::size_t i = 0; i < jiggleSize;i++){
	totalJiggle += jiggle[i];
      }
      totalJiggle = .75 / totalJiggle;
    }

    // fitness gets lower the less the pole jiggles and the more it balances.
    // no theoretical minimum, but usually close to maxFit
    // maxFit set to 1 for precision reasons
    // never seen the fitness result go below 0.2 in experiments
    std::size_t maxFit = 1.0;
    double result = maxFit - (0.1 * eval_count / 1000
			      + 0.9 * totalJiggle);

    return result;
  }

  //! \brief Evaluates weight vector on normal balancing function
  //! \param input Vector to be evaluated.
  //! \param maxEvals Balance goal of the function, i.e. number of steps that pole should be able to balance without failure
  //! \return Fitness of vector
  ResultType balanceFit(const SearchPointType &input,
			std::size_t maxEvals = 100000) {
    SIZE_CHECK(input.size() == m_dimensions);
    double init_angle = 0.07;
    DoublePole pole(false, m_normalize);
    RealVector state(3);
    RealMatrix output(1,1); 
    RealMatrix inState(1,3);
    std::size_t eval_count = 0;
    bool failed = false;
    
    pole.init(init_angle);
    mp_net->resetInternalState();
    mp_net->setParameterVector(input);
    
    while(!failed && eval_count < maxEvals) {
      pole.getState(state);
      row(inState,0) = state;
      mp_net->eval(inState,output);
      pole.move(convertToPoleMovement(output(0,0)));
      failed = pole.failure();
      eval_count++;
    }
    
    // gets lower as number of evaluations grows. min = 0
    return maxEvals - eval_count;
  }

  //! \brief Evaluates weight vector on normal balancing function using 256 different starting positions
  //! \param input Vector to be evaluated.
  //! \param maxEvals Balance goal of the function, i.e. number of steps that pole should be able to balance without failure
  //! \return Number of trials (out of 256) that did not fail
  ResultType generalFit(const SearchPointType &input,
			std::size_t maxEvals = 1000){
    SIZE_CHECK(input.size() == m_dimensions);
    double init_angle = 0.07;
    DoublePole pole(false, m_normalize);
    RealVector state(3);
    RealMatrix output(1,1); 
    bool failed = false;
    const double statevals[5] = {0.05, 0.25, 0.5, 0.75, 0.95};
    std::size_t successes = 0;
    
    for (std::size_t s0c = 0; s0c < 5; ++s0c){
      for (std::size_t s1c = 0; s1c < 5; ++s1c){
	for (std::size_t s2c = 0; s2c < 5; ++s2c){
	  for (std::size_t s3c = 0; s3c < 5; ++s3c) {
	    std::size_t eval_count = 0; // reset counter
	    pole.init(statevals[s0c] * 4.32 - 2.16,
		      statevals[s1c] * 2.70 - 1.35,
		      statevals[s2c] * 0.12566304 - 0.06283152,
		      statevals[s3c] * 0.30019504 - 0.15009752);
	    mp_net->resetInternalState(); // reset t-1 activations
	    mp_net->setParameterVector(input);
	    
	    while(!failed && eval_count < maxEvals) {
	      pole.getState(state);
	      RealMatrix newState(1,3);
	      row(newState,0) = state;
	      mp_net->eval(newState,output);
	      pole.move(convertToPoleMovement(output(0,0)));
	      failed = pole.failure();
	      eval_count++;
	    }
	    
	    if(eval_count == maxEvals){
	      successes++;
	    }
	  }
	}
      }
    }
    return successes;
  }
  
private:

  // private class for recurrent neural network. not be used outside main class.
  class PoleRNNet : public OnlineRNNet {
  public:
    PoleRNNet(RecurrentStructure* structure) : OnlineRNNet(structure){}
    boost::shared_ptr<State> createState()const{
	    throw std::logic_error("State not available for PoleRNNet.");
    }
    void eval(BatchInputType const & patterns, BatchOutputType &outputs,
	      State& state) const{
	    throw std::logic_error("Batch not available for PoleRNNet.");
    }
  };

  //! \brief Converts neural network output for use with pole simulator
  //! \param output Output of the neural network.
  //! \return double precision floating point between 0 and 1.
  double convertToPoleMovement(double output) const{
    switch(mp_struct->sigmoidType())
      {
      case RecurrentStructure::Logistic:
	return output;
      case RecurrentStructure::FastSigmoid:
	return (output + 1.) / 2.;
      case RecurrentStructure::Tanh:
	return (output + 1.) / 2.;
      default:
	std::cerr << "Unsupported activation function for pole balancing." << std::endl;
	exit(EXIT_FAILURE);
      }
    

  }

  //! True if neural network input is normalized, false otherwise
  bool m_normalize;
  //! Degrees of freedom
  std::size_t m_dimensions;
  //! Balance goal
  std::size_t m_maxPoleEvals;

  //! Neural network
  RecurrentStructure *mp_struct;
  OnlineRNNet *mp_net;

};

} 
#endif
