/*!
*  \file CrossValidation.cpp
*
*  \brief Cross Validation
*
*  The cross validation procedure is designed to adapt so-called
*  hyperparameters of a model, which is based upon another model.
*  In every hyperparameter evaluation step, the base model is
*  trained. This inner training procedure depends on the
*  hyperparameters. Thus, the hyperparameters can be evaluated by
*  simply evaluating the trained base model. To avoid empirical
*  risk minimization, the cross validation procedure splits the
*  available data into training and validation subsets, such that
*  all data points appear in training and validation subsets
*  equally often.
*
*
*  \author  T. Glasmachers
*  \date    2006
*
*  \par
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


#include <ReClaM/CrossValidation.h>


Partitioning::Partitioning()
{
	partitions = 0;
}

Partitioning::~Partitioning()
{
	Clear();
}


void Partitioning::CreateIID(int numberOfPartitions, const Array<double>& input, const Array<double>& target)
{
	unsigned int e, ec = input.dim(0);

	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	SIZE_CHECK(target.dim(0) == ec);

	Array<int> index(ec);
	for (e=0; e<ec; e++) index(e) = Rng::discrete(0, numberOfPartitions - 1);
	CreateIndexed(numberOfPartitions, input, target, index);
}

void Partitioning::CreateSameSize(int numberOfPartitions, const Array<double>& input, const Array<double>& target)
{
	Clear();

	partitions = numberOfPartitions;
	unsigned int ec = input.dim(0);
	unsigned int dimension = input.dim(1);
	unsigned int targetdim = target.dim(1);
	unsigned int e, d;
	int i;

	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	SIZE_CHECK(target.dim(0) == ec);

	part_index.resize(ec, false);
	inverse_part_index.resize(partitions);
	part_train_input.resize(partitions);
	part_train_target.resize(partitions);
	part_validation_input.resize(partitions);
	part_validation_target.resize(partitions);

	Array<int> t(partitions);
	Array<int> v(partitions);
	int nn = ec / partitions;
	int c = ec - nn * partitions;
	for (i = 0; i < partitions; i++)
	{
		v(i) = nn;
		if (i < c) v(i)++;
		t(i) = ec - v(i);
		part_train_input[i] = new Array<double>(t(i), dimension);
		part_train_target[i] = new Array<double>(t(i), targetdim);
		part_validation_input[i] = new Array<double>(v(i), dimension);
		part_validation_target[i] = new Array<double>(v(i), targetdim);
		inverse_part_index[i] = new Array<int>(v(i));
	}
	for (e = 0; e < ec; e++)
	{
		do
		{
			c = Rng::discrete(0, partitions - 1);
		}
		while (v(c) == 0);
		v(c)--;
		part_index(e) = c;

		for (d = 0; d < dimension; d++) part_validation_input[c]->operator()(v(c), d) = input(e, d);
		for (d = 0; d < targetdim; d++) part_validation_target[c]->operator()(v(c), d) = target(e, d);

		for (i = 0; i < partitions; i++)
		{
			if (i == c) continue;
			t(i)--;
			for (d = 0; d < dimension; d++) part_train_input[i]->operator()(t(i), d) = input(e, d);
			for (d = 0; d < targetdim; d++) part_train_target[i]->operator()(t(i), d) = target(e, d);
		}

		inverse_part_index[c]->operator()(v(c)) = e;
	}
}

void Partitioning::CreateSameSizeBalanced(int numberOfPartitions, const Array<double>& input, const Array<double>& target)
{
	Clear();

	partitions = numberOfPartitions;
	unsigned int ec = input.dim(0);
	unsigned int dimension = input.dim(1);
	unsigned int e, d;
	int i;

	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	SIZE_CHECK(target.dim(0) == ec);

	part_index.resize(ec, false);
	inverse_part_index.resize(partitions);
	part_train_input.resize(partitions);
	part_train_target.resize(partitions);
	part_validation_input.resize(partitions);
	part_validation_target.resize(partitions);

	SIZE_CHECK(target.dim(1) == 1);
	int l_plus = 0;
	int l_minus = 0;
	for (e = 0; e < ec; e++) if (target(e, 0) > 0.0) l_plus++; else l_minus++;

	Array<int> t(partitions);
	Array<int> v_plus(partitions);
	Array<int> v_minus(partitions);
	int nn_plus = l_plus / partitions;
	int c_plus = l_plus - nn_plus * partitions;
	int nn_minus = l_minus / partitions;
	int c_minus = l_minus - nn_minus * partitions;
	for (i = 0; i < partitions; i++)
	{
		v_plus(i) = nn_plus;
		v_minus(i) = nn_minus;
		if (i < c_plus) v_plus(i)++;
		if (i < c_minus) v_minus(i)++;
		t(i) = ec - v_plus(i) - v_minus(i);
		part_train_input[i] = new Array<double>(t(i), dimension);
		part_train_target[i] = new Array<double>(t(i), 1);
		part_validation_input[i] = new Array<double>(v_plus(i) + v_minus(i), dimension);
		part_validation_target[i] = new Array<double>(v_plus(i) + v_minus(i), 1);
		inverse_part_index[i] = new Array<int>(v_plus(i) + v_minus(i));
	}
	int c;
	for (e = 0; e < ec; e++)
	{
		if (target(e, 0) > 0.0)
		{
			do
			{
				c = Rng::discrete(0, partitions - 1);
			}
			while (v_plus(c) == 0);
			v_plus(c)--;
		}
		else
		{
			do
			{
				c = Rng::discrete(0, partitions - 1);
			}
			while (v_minus(c) == 0);
			v_minus(c)--;
		}
		part_index(e) = c;

		for (d = 0; d < dimension; d++) part_validation_input[c]->operator()(v_plus(c) + v_minus(c), d) = input(e, d);
		part_validation_target[c]->operator()(v_plus(c) + v_minus(c), 0) = target(e, 0);

		for (i = 0; i < partitions; i++)
		{
			if (i == c) continue;
			t(i)--;
			for (d = 0; d < dimension; d++) part_train_input[i]->operator()(t(i), d) = input(e, d);
			part_train_target[i]->operator()(t(i), 0) = target(e, 0);
		}

		inverse_part_index[c]->operator()(v_plus(c) + v_minus(c)) = e;
	}
}

void Partitioning::CreateIndexed(int numberOfPartitions, const Array<double>& input, const Array<double>& target, const Array<int>& index)
{
	Clear();

	partitions = numberOfPartitions;
	part_index = index;

	unsigned int ec = input.dim(0);
	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	SIZE_CHECK(target.dim(0) == ec);

	unsigned int dimension = input.dim(1);
	unsigned int targetdim = target.dim(1);
	unsigned int e, d;
	int i;

	Array<int> part_size(partitions);
	Array<int> t(partitions);
	Array<int> v(partitions);
	part_size = 0;
	t = 0;
	v = 0;
	for (e = 0; e < ec; e++)
	{
		int i = index(e);
		part_size(i)++;
	}

	inverse_part_index.resize(partitions);
	part_train_input.resize(partitions);
	part_train_target.resize(partitions);
	part_validation_input.resize(partitions);
	part_validation_target.resize(partitions);
	for (i = 0; i < partitions; i++)
	{
		part_train_input[i] = new Array<double>(ec - part_size(i), dimension);
		part_train_target[i] = new Array<double>(ec - part_size(i), targetdim);
		part_validation_input[i] = new Array<double>(part_size(i), dimension);
		part_validation_target[i] = new Array<double>(part_size(i), targetdim);
		inverse_part_index[i] = new Array<int>(part_size(i));
	}
	for (e = 0; e < ec; e++)
	{
		for (i = 0; i < partitions; i++)
		{
			if (index(e) == i)
			{
				for (d = 0; d < dimension; d++) (*part_validation_input[i])(v(i), d) = input(e, d);
				for (d = 0; d < targetdim; d++) (*part_validation_target[i])(v(i), d) = target(e, d);
				inverse_part_index[i]->operator()(v(i)) = e;
				v(i)++;
			}
			else
			{
				for (d = 0; d < dimension; d++) (*part_train_input[i])(t(i), d) = input(e, d);
				for (d = 0; d < targetdim; d++) (*part_train_target[i])(t(i), d) = target(e, d);
				t(i)++;
			}
		}
	}
}

void Partitioning::CreateSameSizeMultiClassBalanced(int numberOfPartitions, const Array<double>& input, const Array<double>& target) {
	Clear();
	
	partitions = numberOfPartitions;
	
	unsigned int n = input.dim(0);
	unsigned int dimension = input.dim(1);
	unsigned int targetdim = target.dim(1);
	
	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	SIZE_CHECK(target.dim(0) == n);
	
	part_index.resize(n, false);
	inverse_part_index.resize(partitions);
	part_train_input.resize(partitions);
	part_train_target.resize(partitions);
	part_validation_input.resize(partitions);
	part_validation_target.resize(partitions);
	
	
	Array<int> t(partitions); // training samples in partition i
	Array<int> v(partitions); // validation samples in partition i
	int nn = n / partitions;
	int c = n - nn * partitions;
	for (int i = 0; i < partitions; i++)
	{
		v(i) = nn;
		if (i < c) v(i)++;
		t(i) = n - v(i);
		part_train_input[i] = new Array<double>(t(i), dimension);
		part_train_target[i] = new Array<double>(t(i), targetdim);
		part_validation_input[i] = new Array<double>(v(i), dimension);
		part_validation_target[i] = new Array<double>(v(i), targetdim);
		inverse_part_index[i] = new Array<int>(v(i));
	}
	
	std::vector< std::vector<unsigned> > vv;
	
	int nc = 0;
	for(unsigned i=0; i<n; i++) {
		if(target(i,0)>nc - 1)
			nc = target(i,0) + 1;
	}
	
	vv.resize(nc);
	for(unsigned i=0; i<n; i++) {
		vv[target(i,0)].push_back(i);
	}
	
	unsigned d;
	int vf = 0; // the training fold
	for(int i=0; i<nc; i++) {
		for(unsigned j=0; j<vv[i].size(); j++) {
			unsigned e = vv[i][j];
			
			v(vf)--;
			part_index(e) = vf;
			
			for (d = 0; d < dimension; d++) part_validation_input[vf]->operator()(v(vf), d) = input(e, d);
			for (d = 0; d < targetdim; d++) part_validation_target[vf]->operator()(v(vf), d) = target(e, d);
			
			
			for (int k = 0; k < partitions; k++)
			{
				if (k == vf) continue;
				t(k)--;
				for (d = 0; d < dimension; d++) part_train_input[k]->operator()(t(k), d) = input(e, d);
				for (d = 0; d < targetdim; d++) part_train_target[k]->operator()(t(k), d) = target(e, d);
			}
			
			inverse_part_index[vf]->operator()(v(vf)) = e;
			vf = (vf + 1) % partitions;
		}
	}
}


void Partitioning::Clear()
{
	int i;
	for (i = 0; i < partitions; i++)
	{
		delete part_train_input[i];
		delete part_train_target[i];
		delete part_validation_input[i];
		delete part_validation_target[i];
		delete inverse_part_index[i];
	}
	partitions = 0;
}

////////////////////////////////////////////////////////////


CVModel::CVModel(Array<Model*>& models)
{
	SIZE_CHECK(models.ndim() == 1);
	SIZE_CHECK(models.dim(0) > 1);

	unsigned int p, pc = models(0)->getParameterDimension();

#ifdef DEBUG
	unsigned int i, ic = models.dim(0);
	for (i = 1; i < ic; i++) SIZE_CHECK(models(i)->getParameterDimension() == pc);
#endif

	baseModel = models;
	baseModelIndex = 0;

	parameter.resize(pc, false);
	for (p = 0; p < pc; p++) parameter(p) = models(0)->getParameter(p);
}

CVModel::CVModel(unsigned int folds, Model* basemodel)
{
	SIZE_CHECK( folds > 1 );
	unsigned int p, pc = basemodel->getParameterDimension();

	baseModel.resize(folds, false);
	baseModel = basemodel;
	baseModelIndex = 0;

	parameter.resize(pc, false);
	for (p = 0; p < pc; p++) parameter(p) = basemodel->getParameter(p);
}

CVModel::~CVModel()
{
}


void CVModel::setParameter(unsigned int index, double value)
{
	Model::setParameter(index, value);

	int i, ic = baseModel.dim(0);
	for (i = 0; i < ic; i++) baseModel(i)->setParameter(index, value);
}

void CVModel::setBaseModel(int index)
{
	RANGE_CHECK(index >= 0 && index < (int)baseModel.dim(0));
	baseModelIndex = index;
}

void CVModel::model(const Array<double>& input, Array<double>& output)
{
	baseModel(baseModelIndex)->model(input, output);
}

void CVModel::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	baseModel(baseModelIndex)->modelDerivative(input, derivative);
}

void CVModel::modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative)
{
	baseModel(baseModelIndex)->modelDerivative(input, output, derivative);
}

bool CVModel::isFeasible()
{
	return getBaseModel().isFeasible();
}


////////////////////////////////////////////////////////////


CVError::CVError(Partitioning& part, ErrorFunction& error, Optimizer& optimizer, int iter)
		: partitioning(part)
		, baseError(error)
		, baseOptimizer(optimizer)
{
	iterations = iter;
}

CVError::~CVError()
{
}


double CVError::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	CVModel* pCVM = dynamic_cast<CVModel*>(&model);
	if (pCVM == NULL) throw SHARKEXCEPTION("[CVError::error] invalid model");

	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	SIZE_CHECK(target.dim(0) == input.dim(0));

	double err, ret = 0.0;
	unsigned int t, parts = partitioning.getNumberOfPartitions();
	int it;

	// loop through the partitions
	for (t = 0; t < parts; t++)
	{
		// activate the t-th submodel
		pCVM->setBaseModel(t);

		// do the training
		baseOptimizer.init(pCVM->getBaseModel());
		for (it = 0; it < iterations; it++)
		{
			baseOptimizer.optimize(pCVM->getBaseModel(), baseError, partitioning.train_input(t), partitioning.train_target(t));
		}

		// compute the validation error
		err = baseError.error(pCVM->getBaseModel(), partitioning.validation_input(t), partitioning.validation_target(t));
		ret += err;
	}

	// return the mean error
	return ret / parts;
}

double CVError::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	CVModel* pCVM = dynamic_cast<CVModel*>(&model);
	if (pCVM == NULL) throw SHARKEXCEPTION("[CVError::errorDerivative] invalid model");

	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	SIZE_CHECK(target.dim(0) == input.dim(0));

	double err, ret = 0.0;
	unsigned int t, parts = partitioning.getNumberOfPartitions();
	int it;
	unsigned int p, pc = model.getParameterDimension();
// 	Array<double> train_input;
// 	Array<double> train_target;
// 	Array<double> validation_input;
// 	Array<double> validation_target;
// 	Array<double> initial_param(pc);
	Array<double> innerDerivative;

	derivative.resize(pc, false);
	derivative = 0.0;

	// loop through the partitions
	for (t = 0; t < parts; t++)
	{
		// activate the t-th submodel
		pCVM->setBaseModel(t);

		// do the training
		baseOptimizer.init(pCVM->getBaseModel());
		for (it = 0; it < iterations; it++)
		{
			baseOptimizer.optimize(pCVM->getBaseModel(), baseError, partitioning.train_input(t), partitioning.train_target(t));
		}

		// compute the validation error
		err = baseError.errorDerivative(pCVM->getBaseModel(), partitioning.validation_input(t), partitioning.validation_target(t), innerDerivative);
		for (p = 0; p < pc; p++) derivative(p) += innerDerivative(p);
		ret += err;
	}

	// return the mean derivative
	for (p = 0; p < pc; p++) derivative(p) = derivative(p) / parts;

	// return the mean error
	return ret / parts;
}

