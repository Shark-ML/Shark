/**
 *
 *  \brief OptimizerTraits
 *
 *  \author T.Voss, T. Glasmachers, O.Krause
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
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
