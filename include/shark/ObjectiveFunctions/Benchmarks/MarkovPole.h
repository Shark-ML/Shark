/*!
 * 
 * \brief       Objective function for single and double poles with full state information (Markovian task) 
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_POLE_MARKOV_OBJECTIVE_FUNCTION
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_POLE_MARKOV_OBJECTIVE_FUNCTION

#include <iostream>
#include <typeinfo>

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/LinAlg/Base.h>
#include <shark/Models/FFNet.h>

#include <shark/ObjectiveFunctions/Benchmarks/PoleSimulators/SinglePole.h>
#include <shark/ObjectiveFunctions/Benchmarks/PoleSimulators/DoublePole.h>

namespace shark {namespace benchmarks{



/// Class for balancing one or two poles on a cart using a fitness function
/// that decreases the longer the pole(s) balance(s).
/// Based on code written by Verena Heidrich-Meisner for the paper
/// 
/// V. Heidrich-Meisner and C. Igel. Neuroevolution strategies for episodic reinforcement learn-ing. Journal of Algorithms, 64(4):152–168, 2009.
/// \ingroup benchmarks
template<class HiddenNeuron,class OutputNeuron>
class MarkovPole : public SingleObjectiveFunction {
public:
  /// \param single_pole Indicates whether the cast has a single pole (true) or two poles (false)
  /// \param hidden Number of hidden neurons in underlying neural network
  /// \param shortcuts Whether to use shortcuts in neural network
  /// \param bias Whether to use bias in neural network
  /// \param normalize Whether to normalize input before use in neural network
  /// \param max_pole_evaluations Balance goal of the function, i.e. number of steps that pole should be able to balance without failure
  MarkovPole(bool single_pole, std::size_t hidden, bool shortcuts, bool bias,
	     bool normalize = true, std::size_t max_pole_evaluations = 100000)
    : m_single(single_pole),
      m_maxPoleEvals(max_pole_evaluations),
      m_normalize(normalize) {
    // number of inputs should be 4 for single pole, 6 for double.
    std::size_t inputs = 0;
    if (single_pole) {
      inputs = 4;
    }
    else {
      inputs = 6;
    }
    // set features
    m_features |= CAN_PROPOSE_STARTING_POINT;
    
    // set number of variables/weights.
    // number of outputs is always 1. 
    // dimensions depend on whether we use bias and/or shortcuts
    if (bias && shortcuts){
      m_dimensions = hidden * (inputs + 1) + inputs + hidden + 1;
    }
    else if (shortcuts) {
      m_dimensions = hidden * (inputs + 1) + inputs;
    }
    else if (bias) {
      m_dimensions = hidden * (inputs + 1) + hidden + 1;
    }
    else {
      m_dimensions = hidden * (inputs + 1);
    }    
    
    // make FFNet
    mp_net = new FFNet<HiddenNeuron, OutputNeuron>();
    FFNetStructures::ConnectionType type = shortcuts ?
      FFNetStructures::InputOutputShortcut : FFNetStructures::Normal;
    mp_net->setStructure(inputs, hidden, 1, type, bias);  

    // check dimensions match
    if(m_dimensions != mp_net->numberOfParameters()) {
      std::cerr << "Markov pole FFNet: Dimensions do not match, " << m_dimensions
	   << " != " <<  mp_net->numberOfParameters() << std::endl;
      exit(EXIT_FAILURE);
    }
    
    // set eval count
    m_evaluationCounter = 0;    
  }

  ~MarkovPole() {
    delete mp_net;
  }

  
  std::string name() {
    return "Objective Function for Markovian pole balancing.";
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

  /// \brief Converts neural network output for use with pole simulator
  /// \param output Output of the neural network.
  /// \return double precision floating point between 0 and 1.
  double convertToPoleMovement(double output) const{
    if (typeid(mp_net->outputActivationFunction())
	== typeid(LogisticNeuron)) {
      return output;
    }
    else if (typeid(mp_net->outputActivationFunction())
	     == typeid(FastSigmoidNeuron)) {
      return (output + 1.) / 2.;      
    }
    else if (typeid(mp_net->outputActivationFunction()) == typeid(TanhNeuron)) {
      return (output + 1.) / 2.;
    }
    else {
      std::cerr << "Unsupported neuron type in Markov pole FFNet." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  
  /// \brief Fitness function for single poles. Gets lower as pole balances for longer.
  /// \param input Vector to be evaluated.
  /// \return Fitness of vector
  ResultType evalSingle(const SearchPointType &input) const{
    double init_angle = 0.07;
    SinglePole pole(true, m_normalize);
    RealVector state(4);
    RealVector output(1); 
    std::size_t eval_count = 0;
    bool failed = false;
    
    pole.init(init_angle);

    mp_net->setParameterVector(input);
    
    while(!failed && eval_count < m_maxPoleEvals) {
      pole.getState(state);
      mp_net->eval(state,output);
      pole.move(convertToPoleMovement(output(0)));
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
    DoublePole pole(true, m_normalize);
    RealVector state(6);
    RealVector output(1); 
    std::size_t eval_count = 0;
    bool failed = false;

    pole.init(init_angle);
    mp_net->setParameterVector(input);

    while(!failed && eval_count < m_maxPoleEvals) {
      pole.getState(state);
      mp_net->eval(state,output);
      pole.move(convertToPoleMovement(output(0)));
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
  FFNet<HiddenNeuron, OutputNeuron> *mp_net;
  HiddenNeuron m_hiddenNeuron;
  OutputNeuron m_outputNeuron;

};

}}
#endif
