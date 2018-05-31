/*!
 * 
 * \brief       Objective function for single and double poles with partial state information (non-Markovian task) 
 *
 * 
 * Class for balancing one or two poles on a cart using a fitness
 * function that decreases the longer the pole(s) balance(s).  Based
 * on code written by Verena Heidrich-Meisner for the paper
 * 
 * V. Heidrich-Meisner and C. Igel. Neuroevolution strategies for
 * episodic reinforcement learning. Journal of Algorithms,
 * 64(4):152–168, 2009.
 *
 * \author      Johan Valentin Damgaard
 * \date        -
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_POLE_NONMARKOV_OBJECTIVE_FUNCTION
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_POLE_NONMARKOV_OBJECTIVE_FUNCTION

#include <iostream>
#include <exception>

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Models/OnlineRNNet.h>
#include <shark/LinAlg/Base.h>

#include <shark/ObjectiveFunctions/Benchmarks/PoleSimulators/SinglePole.h>
#include <shark/ObjectiveFunctions/Benchmarks/PoleSimulators/DoublePole.h>

namespace shark {namespace benchmarks{

/// \brief Objective function for single and double non-Markov poles
/// 
/// Class for balancing one or two poles on a cart using a fitness function
/// that decreases the longer the pole(s) balance(s).
/// Based on code written by Verena Heidrich-Meisner for the paper
/// 
/// V. Heidrich-Meisner and C. Igel. Neuroevolution strategies for episodic reinforcement learn-ing. Journal of Algorithms, 64(4):152–168, 2009.
/// \ingroup benchmarks
class NonMarkovPole : public SingleObjectiveFunction {

public:
  /// \param single Is this an instance of the single pole problem?
  /// \param hidden Number of hidden neurons in underlying neural network
  /// \param bias Whether to use bias in neural network
  /// \param sigmoidType Activation sigmoid function for neural network
  /// \param normalize Whether to normalize input before use in neural network
  /// \param max_pole_evaluations Balance goal of the function, i.e. number of steps that pole should be able to balance without failure
  NonMarkovPole(bool single, std::size_t hidden, bool bias, 
		RecurrentStructure::SigmoidType sigmoidType = RecurrentStructure::FastSigmoid,
		bool normalize = true,
		std::size_t max_pole_evaluations = 100000)
    : m_single(single),
      m_maxPoleEvals(max_pole_evaluations),
      m_normalize(normalize) {
    if (sigmoidType == RecurrentStructure::Linear) {
      std::cerr << "Cannot use linear activation function for pole balancing."
	   << std::endl;
      exit(EXIT_FAILURE);
    }

    // number of inputs should be 2 for single pole, 3 for double.
    std::size_t inputs = 0;
    if (single) {
      inputs = 2;
    }
    else {
      inputs = 3;
    }
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
      std::cerr << "Non-Markov pole RNNet: Dimensions do not match, "
	   << m_dimensions << " != " <<  mp_net->numberOfParameters() << std::endl;
      exit(EXIT_FAILURE);
    }
    
    // set eval count
    m_evaluationCounter = 0;
    
  }

  ~NonMarkovPole(){
    delete mp_struct;
    delete mp_net;
  }
  
  std::string name() {
    return "Objective Function for Non-Markovian pole balancing.";
  }
  
  /// \brief Returns degrees of freedom
  std::size_t numberOfVariables()const{
    return m_dimensions;
  }
  
  /// \brief Always proposes to start in a zero vector with appropriate degrees of freedom
  SearchPointType proposeStartingPoint() const{
    SearchPointType startingPoint(m_dimensions);
    for(std::size_t i = 0; i != m_dimensions; i++) {
      startingPoint(i) = 0.0;
    }
    return startingPoint;
  }
  
  /// \brief Evaluates weight vector on fitness function
  /// \param input Vector to be evaluated.
  /// \return Fitness of vector
  ResultType eval(const SearchPointType &input) const{
    SIZE_CHECK(input.size() == m_dimensions);
    
    m_evaluationCounter++;
    
    if(m_single) {
      return evalSingle(input);
    }
    else {
      return evalDouble(input);
    }
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

  /// \brief Converts neural network output for use with pole simulator
  /// \param output Output of the neural network.
  /// \return double precision floating point between 0 and 1.
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
  
  /// \brief Fitness function for single poles. Gets lower as pole balances for longer.
  /// \param input Vector to be evaluated.
  /// \return Fitness of vector
  ResultType evalSingle(const SearchPointType &input) const{
    double init_angle = 0.07;
    SinglePole pole(false, m_normalize);
    RealVector state(2);
    RealMatrix output(1,1); 
    RealMatrix inState(1,2);
    std::size_t eval_count = 0;
    bool failed = false;
    
    pole.init(init_angle);
    mp_net->resetInternalState();
    mp_net->setParameterVector(input);
    
    while(!failed && eval_count < m_maxPoleEvals) {
      pole.getState(state);
      row(inState,0) = state;
      mp_net->eval(inState,output);
      pole.move(convertToPoleMovement(output(0,0)));
      failed = pole.failure();
      eval_count++;
    }
    
    // gets lower as number of evaluations grows. min = 0
    return m_maxPoleEvals - eval_count;
  }

  /// \brief Fitness function for double poles. Gets lower as poles balance for longer.
  /// \param input Vector to be evaluated.
  /// \return Fitness of vector
  ResultType evalDouble(const SearchPointType &input) const{
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
    
    while(!failed && eval_count < m_maxPoleEvals) {
      pole.getState(state);
      row(inState,0) = state;
      mp_net->eval(inState,output);
      pole.move(convertToPoleMovement(output(0,0)));
      failed = pole.failure();
      eval_count++;
    }
    // gets lower as number of evaluations grows. min = 0
    return m_maxPoleEvals - eval_count;
  }

  /// True if this is a single pole, false if double pole.
  bool m_single;
  /// True if neural network input is normalized, false otherwise
  bool m_normalize;
  /// Degrees of freedom
  std::size_t m_dimensions;
  /// Balance goal
  std::size_t m_maxPoleEvals;
  
  /// Neural network
  RecurrentStructure *mp_struct;
  OnlineRNNet *mp_net;
  
};

}}
#endif
