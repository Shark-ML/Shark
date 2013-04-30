/*
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
*/
#ifndef SHARK_UNSUPERVISED_RBM_GAUSSIANBINARYRBM_H
#define SHARK_UNSUPERVISED_RBM_GAUSSIANBINARYRBM_H

#include <shark/Unsupervised/RBM/RBM.h>
#include <shark/Unsupervised/RBM/Energy.h>

#include <shark/Unsupervised/RBM/Neuronlayers/BinaryLayer.h>
#include <shark/Unsupervised/RBM/Neuronlayers/GaussianLayer.h>
#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
#include <shark/Unsupervised/RBM/Sampling/TemperedMarkovChain.h>
#include <shark/Unsupervised/RBM/Sampling/MarkovChain.h>

#include <shark/Unsupervised/RBM/GradientApproximations/ContrastiveDivergence.h>
#include <shark/Unsupervised/RBM/GradientApproximations/MultiChainApproximator.h>
#include <shark/Unsupervised/RBM/GradientApproximations/SingleChainApproximator.h>
#include <shark/Rng/GlobalRng.h>
namespace shark{

typedef RBM<GaussianLayer,BinaryLayer, Rng::rng_type> GaussianBinaryRBM;
typedef GibbsOperator<GaussianBinaryRBM> GaussianBinaryGibbsOperator;
typedef MarkovChain<GaussianBinaryGibbsOperator> GaussianBinaryGibbsChain;
typedef TemperedMarkovChain<GaussianBinaryGibbsOperator> GaussianBinaryPTChain;

typedef MultiChainApproximator<GaussianBinaryGibbsChain> GaussianBinaryPCD;
typedef ContrastiveDivergence<GaussianBinaryGibbsOperator> GaussianBinaryCD;
typedef SingleChainApproximator<GaussianBinaryPTChain> GaussianBinaryParallelTempering;
}

#endif
