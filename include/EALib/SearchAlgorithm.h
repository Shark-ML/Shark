//===========================================================================
/*!
 *  \file SearchAlgorithm.h
 *
 *  \brief General search algorithm class
 *
 *  \author  Tobias Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *
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
 *
 *
 */
//===========================================================================


#ifndef _SearchAlgorithm_H_
#define _SearchAlgorithm_H_


#include <EALib/ObjectiveFunction.h>


//! \brief Super class of all (direct) search algorithms
class SearchAlgorithmBase
{
protected:
	//! Constructor
	SearchAlgorithmBase() { }

	//! Destructor
	virtual ~SearchAlgorithmBase() { }

public:
	//! return the name of the algorithm
	inline const std::string& name() const { return m_name; }

	//! return the number of iterations performed so far
	inline unsigned int iterations() const { return m_iter; }

	//! reset the number of iterations to zero
	inline void resetIterations() { m_iter = 0; }

protected:
	//! name of this algorithm
	std::string m_name;

	//! current number of iterations
	unsigned int m_iter;
};


//! \brief Search algorithm template with flexible search space
//!
//! The SearchAlgorithm template class represents an abstract
//! search algorithm for search on the space represented by the
//! template type T. The search is guided by a single- or multi-
//! objective fitness function, defined on this space.
template <class T>
class SearchAlgorithm : public SearchAlgorithmBase
{
public:
	//! Constructor
	SearchAlgorithm() { }

	//! Destructor
	~SearchAlgorithm() { }


	//! main interface: perform one iteration of the search algorithm
	virtual void run() { m_iter++; }

	//! perform #iter iterations by calling Run() #iter times
	inline void runN(unsigned int iter)
	{
		unsigned int i; for (i=0; i<iter; i++) run();
	}

	//! return the current set of pareto-optimal solutions
	virtual void bestSolutions(std::vector<T>& points) = 0;

	//! Return a two-dimensional array of fitness values.
	//! The first dimension corresponds to the solution,
	//! while the second dimension corresponds to the objective.
	virtual void bestSolutionsFitness(Array<double>& fitness) = 0;

	//! return the best solution (for a single-objective task)
	inline T bestSolution() {
		std::vector<T> p;
		bestSolutions(p);
		RANGE_CHECK(p.size() > 0);
		return p[0];
	}

	//! return the best fitness (for a single-objective task)
	inline double bestSolutionFitness() {
		Array<double> f;
		bestSolutionsFitness(f);
		return f(0, 0);
	}
};

//! \brief Evolutionary algorithm template with flexible search space
//!
//! The EvolutionaryAlgorithm template class represents an abstract
//! evolutionary algorithm for search on the space represented by the
//! template type T. The search is guided by a single- or multi-
//! objective fitness function, defined on this space. Additionally, the 
//! number of parent individuals $\mu$ and the number of offspring individuals $\lambda$ are
//! modelled.
template <class T>
class EvolutionaryAlgorithm : public SearchAlgorithm<T>
{
public:
	EvolutionaryAlgorithm() { }
	~EvolutionaryAlgorithm() { }

	inline unsigned int mu() const { return m_mu; }
	inline unsigned int lambda() const { return m_lambda; }

protected:
	unsigned int m_mu;
	unsigned int m_lambda;
};


#endif
