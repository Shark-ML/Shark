/*!
 * 
 * \file        OptimizerTraits.h
 *
 * \brief       OptimizerTraits
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
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
#ifndef SHARK_CORE_TRAITS_OPTIMIZERTRAITS_H
#define SHARK_CORE_TRAITS_OPTIMIZERTRAITS_H

#include <string>

namespace shark {

/**
 * \brief Abstract traits specific to an optimizer type that 
 * are not modeled in interface AbstractOptimizer.
 *
 * The traits defined here contain reporting facilities to store
 * an algorithm's state for later evaluation purposes.
 */
template<typename Optimizer>
struct OptimizerTraits {

  /**
   * \brief Reports the state of the algorithm for later evaluation purposes.
   *
   * \tparam Function The type of the objective function.
   *
   * \param [in] generation The generation for which the report shall be assembled.
   * \param [in] trial The trial for which the report shall be assembled.
   * \param [in] o Instance of the optimizer.
   * \param [in] optimizerName The experiment-specific name of the optimizer.
   * \param [in] f The objective function instance.
   * \param [in] functionName the experiment-specific name of the function.
   *
   */
  template<typename Function>
  static void report( unsigned int generation, 
                      unsigned int trial, 
                      const Optimizer & o,
                      const std::string & optimizerName,
                      const Function & f,
                      const std::string & functionName ) {
    (void) generation;
    (void) trial;
    (void) o;
    (void) optimizerName;
    (void) f;
    (void) functionName;
  }

  /**
   * \brief Prints out the configuration options and usage remarks of the algorithm to the supplied stream.
   * \tparam Stream The type of the stream to output to.
   * \param [in,out] s The stream to print usage information to.
   */
  template<typename Stream>
  static void usage( Stream & s ) {
    (void) s;
  }

  /**
   * \brief Assembles a default configuration for the algorithm.
   * \tparam Tree structures modelling the boost::ptree concept.
   * \param [in,out] t The stream to be filled with default key-value pairs.
   */
  template<typename Tree>
  static void defaultConfig( Tree & t ) {
    (void) t;
  }

};

}

#endif
