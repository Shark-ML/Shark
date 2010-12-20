//===========================================================================
/*!
 *  \file Svm.cpp
 *
 *  \brief Support Vector Machine implementation
 *
 *  \author  T. Glasmachers
 *  \date	2005-2010
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

#define _SCL_SECURE_NO_WARNINGS

#include <math.h>
#include <fstream>
#include <iostream>
#include <LinAlg/LinAlg.h>
#include <LinAlg/VecMat.h>
#include <ReClaM/Svm.h>
#include <ReClaM/QuadraticProgram.h>
#include <ReClaM/GaussianProcess.h>


using namespace std;


////////////////////////////////////////////////////////////////////////////////


SVM::SVM(KernelFunction* pKernel, bool bSignOutput)
{
	kernel = pKernel;
	signOutput = bSignOutput;

	x = NULL;
	bOwnMemory = false;

	examples = 0;
	inputDimension = 0;
	outputDimension = 1;

	parameter.resize(1, false);
	parameter = 0.0;
}

SVM::SVM(KernelFunction* pKernel, const Array<double>& input, bool bSignOutput)
{
	kernel = pKernel;
	signOutput = bSignOutput;

	x = NULL;
	bOwnMemory = false;

	examples = 0;
	inputDimension = 0;
	outputDimension = 1;

	SetTrainingData(input);
}

SVM::~SVM()
{
	if (bOwnMemory) delete x;
}


void SVM::SetTrainingData(const Array<double>& input, bool copy)
{
	if (bOwnMemory) delete x;

	if (copy)
	{
		x = new Array<double> (input);
		bOwnMemory = true;
	}
	else
	{
		x = &input;
		bOwnMemory = false;
	}

	examples = input.dim(0);
	inputDimension = input.dim(1);

	parameter.resize(examples + 1, false);
	parameter = 0.0;
}

void SVM::model(const Array<double>& input, Array<double>& output)
{
	unsigned int i;
	double a;
	double alpha;
	if (input.ndim() == 1)
	{
		output.resize(1, false);
		a = parameter(examples);
		for (i = 0; i < examples; i++)
		{
			alpha = parameter(i);
			if (alpha != 0.0)
			{
				a += alpha * kernel->eval((*x)[i], input);
			}
		}
		if (signOutput) output(0) = (a > 0.0) ? 1.0 : -1.0;
		else output(0) = a;
	}
	else if (input.ndim() == 2)
	{
		int j, jc = input.dim(0);
		output.resize(jc, 1, false);
		for (j = 0; j < jc; j++)
		{
			a = parameter(examples);
			const ArrayReference<double> ii = input[j];
			for (i = 0; i < examples; i++)
			{
				alpha = parameter(i);
				if (alpha != 0.0)
				{
					a += alpha * kernel->eval((*x)[i], ii);
				}
			}
			if (signOutput) output(j, 0) = (a > 0.0) ? 1.0 : -1.0;
			else output(j, 0) = a;
		}
	}
	else throw SHARKEXCEPTION("[SVM::model] invalid dimension");
}

double SVM::model(const Array<double>& input)
{
	unsigned int i;
	double a;
	double alpha;
	if (input.ndim() == 1)
	{
		a = parameter(examples);
		for (i = 0; i < examples; i++)
		{
			alpha = parameter(i);
			if (alpha != 0.0)
			{
				a += alpha * kernel->eval((*x)[i], input);
			}
		}
	}
	else throw SHARKEXCEPTION("[SVM::model] invalid dimension");
	if (signOutput) return(a > 0.0) ? 1.0 : -1.0;
	else return a;
}

void SVM::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	unsigned int i;
	double v;
	if (input.ndim() == 1)
	{
		derivative.resize(1, examples + 1, false);
		derivative = 0.0;
		for (i = 0; i < examples; i++)
		{
			v = kernel->eval((*x)[i], input);
			derivative(0, i) = v;
		}
		derivative(0, examples) = 1.0;
	}
	else throw SHARKEXCEPTION("[SVM::modelDerivative] invalid dimension");
}

void SVM::modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative)
{
	unsigned int i;
	double a, v;
	if (input.ndim() == 1)
	{
		derivative.resize(1, examples + 1, false);
		derivative = 0.0;
		output.resize(1, false);
		a = parameter(examples);
		for (i = 0; i < examples; i++)
		{
			v = kernel->eval((*x)[i], input);
			a += parameter(i) * v;
			derivative(0, i) = v;
		}
		output(0) = a;
		derivative(0, examples) = 1.0;
	}
	else throw SHARKEXCEPTION("[SVM::modelDerivative] invalid dimension");
}

bool SVM::LoadSVMModel(std::istream& is)
{
	char buffer[50];
	unsigned int t, d;

	// read the header line
	is.read(buffer, 21);

	buffer[21] = 0;
	if (strcmp(buffer, "Shark SVM model\r\nSV: ") != 0) return false;
	if (! is.good()) return false;

	// read the number of support vectors
	is >> examples;

	is.read(buffer, 7); buffer[7] = 0;
	if (strcmp(buffer, "\r\nDIM: ") != 0) return false;
	if (! is.good()) return false;

	// read the input space dimension
	is >> inputDimension;

	is.read(buffer, 10); buffer[10] = 0;
	if (strcmp(buffer, "\r\nkernel: ") != 0) return false;
	if (! is.good()) return false;

	// read the kernel parameters
	is >> (*kernel);

	is.read(buffer, 14); buffer[14] = 0;
	if (strcmp(buffer, "coefficients: ") != 0) return false;
	if (! is.good()) return false;

	// read alpha and b
	parameter.resize(examples + 1, false);
	for (t = 0; t < examples; t++)
	{
		is >> parameter(t);
		is.read(buffer, 1);
		if (buffer[0] != ' ') return false;
	}

	is >> parameter(examples);

	is.read(buffer, 2); buffer[2] = 0;
	if (strcmp(buffer, "\r\n") != 0) return false;
	if (! is.good()) return false;

	// read the support vectors
	Array<double>* sv = new Array<double> (examples, inputDimension);
	for (t = 0; t < examples; t++)
	{
		for (d = 0; d < inputDimension; d++)
		{
			is >> (*sv)(t, d);
			if (d < inputDimension - 1)
			{
				is.read(buffer, 1);
				if (buffer[0] != ' ') return false;
			}
		}

		is.read(buffer, 2); buffer[2] = 0;
		if (strcmp(buffer, "\r\n") != 0) return false;

		if (! is.good()) return false;
	}

	bOwnMemory = true;
	x = sv;

	return true;
}

bool SVM::SaveSVMModel(std::ostream& os)
{
	unsigned int t, d;
	unsigned int T = 0;
	for (t = 0; t < examples; t++) if (parameter(t) != 0.0) T++;

	// write the header line
	os.write("Shark SVM model\r\nSV: ", 21);

	// write the number of support vectors
	os << T;

	os.write("\r\nDIM: ", 7);
	if (! os.good()) return false;

	// write the input space dimension
	os << inputDimension;

	os.write("\r\nkernel: ", 10);
	if (! os.good()) return false;

	// write the kernel parameters
	os << (*kernel);

	os.write("coefficients: ", 14);
	if (! os.good()) return false;

	// write alpha and b
	for (t = 0; t < examples; t++) if (parameter(t) != 0.0) os << parameter(t) << " ";
	os << parameter(examples) << "\r\n";
	if (! os.good()) return false;

	// write the support vectors
	for (t = 0; t < examples; t++)
	{
		if (parameter(t) != 0.0)
		{
			for (d = 0; d < inputDimension; d++)
			{
				os << (*x)(t, d);
				if (d < inputDimension - 1) os << " ";
			}
			os.write("\r\n", 2);
			if (! os.good()) return false;
		}
	}

	return true;
}

// static
SVM* SVM::ImportLibsvmModel(std::istream& is)
{
	char buffer[256];
	int status;
	int i;

	LinearKernel* linear = NULL;
	PolynomialKernel* poly = NULL;
	RBFKernel* rbf = NULL;
	KernelFunction* pKernel = NULL;
	double b = 0.0;
	int examples = -1;
	int dimension = -1;
	while (true)
	{
		status = ReadToken(is, buffer, sizeof(buffer), "\n");
		if (status > 1000) return NULL;

		if (memcmp(buffer, "svm_type ", 9) == 0)
		{
			if (strcmp(buffer + 9, "c_svc") != 0) return NULL;
		}
		else if (memcmp(buffer, "kernel_type ", 12) == 0)
		{
			if (strcmp(buffer + 12, "linear") == 0)
			{
				linear = new LinearKernel();
				pKernel = linear;
			}
			else if (strcmp(buffer + 12, "rbf") == 0)
			{
				rbf = new RBFKernel(1.0);
				pKernel = rbf;
			}
			else if (strcmp(buffer + 12, "polynomial") == 0)
			{
				poly = new PolynomialKernel(1, 1.0);
				pKernel = poly;
			}
			else return NULL;
		}
		else if (memcmp(buffer, "degree ", 7) == 0)
		{
			if (poly != NULL)
			{
				poly->setParameter(0, atof(buffer + 7));
			}
		}
		else if (memcmp(buffer, "gamma ", 6) == 0)
		{
			if (rbf != NULL)
			{
				rbf->setParameter(0, atof(buffer + 6));
			}
		}
		else if (memcmp(buffer, "coef0 ", 6) == 0)
		{
			if (poly != NULL)
			{
				poly->setParameter(1, atof(buffer + 6));
			}
		}
		else if (memcmp(buffer, "nr_class ", 9) == 0)
		{
			if (strcmp(buffer + 9, "2") != 0) return NULL;
		}
		else if (memcmp(buffer, "total_sv ", 9) == 0)
		{
			examples = atoi(buffer + 9);
		}
		else if (memcmp(buffer, "rho ", 4) == 0)
		{
			b = atof(buffer + 4);
		}
		else if (memcmp(buffer, "label ", 6) == 0)
		{
			if (strcmp(buffer + 6, "1 -1") != 0) return NULL;
		}
		else if (strcmp(buffer, "SV") == 0)
		{
			break;
		}
		else
		{
			// just ignore unknown tags
		}
	}
	if (examples == -1) return NULL;
	if (pKernel == NULL) return NULL;

	// first pass: determine the data dimension
	int start = is.tellg();
	for (i = 0; i < examples; i++)
	{
		// read the coefficient
		status = ReadToken(is, buffer, sizeof(buffer), " ");
		if (status != ' ') return NULL;

		while (true)
		{
			// read the index
			status = ReadToken(is, buffer, sizeof(buffer), ":\n");
			if (status > 1000) return NULL;
			if (status == '\n') break;
			int index = atoi(buffer);
			if (index >= dimension) dimension = index + 1;

			// read the value
			status = ReadToken(is, buffer, sizeof(buffer), " \n");
			if (status > 1000) return NULL;
			if (status == '\n') break;
		}
	}
	if (dimension == -1) return NULL;

	// create the objects
	SVM* pSVM = new SVM(pKernel);
	Array<double>* sv = new Array<double> (examples, dimension);
	*sv = 0.0;
	pSVM->parameter.resize(examples + 1, false);
	pSVM->parameter(examples) = b;
	pSVM->examples = examples;
	pSVM->inputDimension = dimension;

	// second pass: read the data
	is.seekg(start, ios::beg);
	for (i = 0; i < examples; i++)
	{
		// read the coefficient
		status = ReadToken(is, buffer, sizeof(buffer), " ");
		if (status != ' ') return NULL;
		pSVM->parameter(i) = atof(buffer);

		while (true)
		{
			// read the index
			status = ReadToken(is, buffer, sizeof(buffer), ":\n");
			if (status > 1000) return NULL;
			if (status == '\n') break;
			int index = atoi(buffer);
			if (index >= dimension) dimension = index + 1;

			// read the value
			status = ReadToken(is, buffer, sizeof(buffer), " \n");
			if (status > 1000) return NULL;
			(*sv)(i, index) = atof(buffer);
			if (status == '\n') break;
		}
	}

	pSVM->x = sv;
	pSVM->bOwnMemory = true;

	return pSVM;
}

// static
SVM* SVM::ImportSvmlightModel(std::istream& is)
{
	char buffer[256];
	int status;
	KernelFunction* pKernel;

	// first line is a comment
	status = DiscardUntil(is, "\n");
	if (status != '\n') return NULL;

	// kernel type
	status = ReadToken(is, buffer, sizeof(buffer), "#\n");
	if (status > 1000) return NULL;
	if (status == '#') DiscardUntil(is, "\n");
	int kernel = atoi(buffer);

	// degree
	status = ReadToken(is, buffer, sizeof(buffer), "#\n");
	if (status > 1000) return NULL;
	if (status == '#') DiscardUntil(is, "\n");
	int degree = atoi(buffer);

	// gamma
	status = ReadToken(is, buffer, sizeof(buffer), "#\n");
	if (status > 1000) return NULL;
	if (status == '#') DiscardUntil(is, "\n");
	double gamma = atof(buffer);

	// poly coeff s
	status = ReadToken(is, buffer, sizeof(buffer), "#\n");
	if (status > 1000) return NULL;
	if (status == '#') DiscardUntil(is, "\n");
	double s = atof(buffer);

	// poly coeff c/r
	status = ReadToken(is, buffer, sizeof(buffer), "#\n");
	if (status > 1000) return NULL;
	if (status == '#') DiscardUntil(is, "\n");
	double r = atof(buffer);

	// string parameter u
	status = DiscardUntil(is, "\n");
	if (status > 1000) return NULL;

	// highest feature index
	status = ReadToken(is, buffer, sizeof(buffer), "#\n");
	if (status > 1000) return NULL;
	if (status == '#') DiscardUntil(is, "\n");
	int dimension = atoi(buffer) + 1;

	// number of training documents
	status = DiscardUntil(is, "\n");
	if (status > 1000) return NULL;

	// number of support vectors plus 1
	status = ReadToken(is, buffer, sizeof(buffer), "#\n");
	if (status > 1000) return NULL;
	if (status == '#') DiscardUntil(is, "\n");
	int examples = atoi(buffer) - 1;

	// threshold b
	status = ReadToken(is, buffer, sizeof(buffer), "#\n");
	if (status > 1000) return NULL;
	if (status == '#') DiscardUntil(is, "\n");
	double b = atof(buffer);

	// create the objects
	if (kernel == 0) pKernel = new LinearKernel();
	else if (kernel == 1)
	{
		if (s == 0.0) return NULL;
		pKernel = new PolynomialKernel(degree, r / s);
	}
	else if (kernel == 2) pKernel = new RBFKernel(gamma);
	else return NULL;
	SVM* pSVM = new SVM(pKernel);
	Array<double>* sv = new Array<double> (examples, dimension);
	*sv = 0.0;
	pSVM->parameter.resize(examples + 1, false);
	pSVM->parameter(examples) = b;
	pSVM->examples = examples;
	pSVM->inputDimension = dimension;

	// read the data
	int i, index;
	for (i = 0; i < examples; i++)
	{
		// read the coefficient
		status = ReadToken(is, buffer, sizeof(buffer), " ");
		if (status > 1000) return NULL;
		pSVM->parameter(i) = atof(buffer);

		while (true)
		{
			// read the index
			status = ReadToken(is, buffer, sizeof(buffer), ":#\n");
			if (status > 1000) return NULL;
			if (status == '#')
			{
				if (DiscardUntil(is, "\n") > 1000) return NULL;
				break;
			}
			else if (status == '\n') break;
			index = atoi(buffer);

			// read the value
			status = ReadToken(is, buffer, sizeof(buffer), " #\n");
			if (status > 1000) return NULL;
			(*sv)(i, index) = atof(buffer);
			if (status == '#')
			{
				if (DiscardUntil(is, "\n") > 1000) return NULL;
				break;
			}
			else if (status == '\n') break;
		}
	}

	pSVM->x = sv;
	pSVM->bOwnMemory = true;

	return pSVM;
}

void SVM::MakeSparse()
{
	if (! bOwnMemory) return;

	// count support vectors and compute a map
	int i, ic = getExamples();
	std::vector<int> map;
	for (i=0; i<ic; i++)
	{
		if (parameter(i) != 0.0)
		{
			map.push_back(i);
		}
	}
	int s, sc = map.size();

	// copy relevant coefficients and training data
	int dim = x->dim(1);
	Array<double> p(sc + 1);
	Array<double>* xx = new Array<double>(sc, dim);

	for (s=0; s<sc; s++)
	{
		i = map[s];
		(*xx)[s] = (*x)[i];
		p(s) = parameter(i);
	}
	p(sc) = parameter(ic);

	parameter = p;
	examples = sc;
	delete x;
	x = xx;
}

// static
int SVM::ReadToken(std::istream& is, char* buffer, int maxlength, const char* separators)
{
	int i;
	int s, sc = strlen(separators);
	char c;
	for (i = 0; i < maxlength - 1; i++)
	{
		if (is.readsome(&c, 1) == 0) return 1001;
		if (is.bad()) return 1002;
		for (s = 0; s < sc; s++) if (separators[s] == c) break;
		if (s < sc)
		{
			buffer[i] = 0;
			return separators[s];
		}
		buffer[i] = c;
	}
	buffer[i] = 0;
	return 1003;
}

// static
int SVM::DiscardUntil(std::istream& is, const char* separators)
{
	int s, sc = strlen(separators);
	char c;
	while (true)
	{
		if (is.readsome(&c, 1) == 0) return 1001;
		if (is.bad()) return 1002;
		for (s = 0; s < sc; s++) if (separators[s] == c) return c;
	}
}


////////////////////////////////////////////////////////////////////////////////


MultiClassSVM::MultiClassSVM(KernelFunction* pKernel, unsigned int numberOfClasses, bool bNumberOutput)
{
	this->kernel = pKernel;
	this->classes = numberOfClasses;
	this->examples = 0;
	this->bOwnMemory = false;
	this->numberOutput = bNumberOutput;

	inputDimension = 0;
	outputDimension = (numberOutput ? 1 : classes);
	x = NULL;
	y = NULL;
}

MultiClassSVM::~MultiClassSVM()
{
	if (bOwnMemory) { delete x; delete y; }
}


void MultiClassSVM::SetTrainingData(const Array<double>& input, const Array<double>& target, bool copy)
{
	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	if (bOwnMemory) { delete x; delete y; }

	examples = input.dim(0);
	inputDimension = input.dim(1);
	SIZE_CHECK(target.dim(0) == examples);

	parameter.resize(examples * classes + classes, false);
	parameter = 0.0;

	if (copy)
	{
		x = new Array<double>(input);
		y = new Array<double>(target);
		bOwnMemory = true;
	}
	else
	{
		x = &input;
		y = &target;
		bOwnMemory = false;
	}
}

void MultiClassSVM::SetTrainingData(const Array<double>& input, bool copy)
{
	SIZE_CHECK(input.ndim() == 2);
	if (bOwnMemory) { delete x; }

	examples = input.dim(0);
	inputDimension = input.dim(1);

	parameter.resize(examples * classes + classes, false);
	parameter = 0.0;

	if (copy)
	{
		x = new Array<double>(input);
		bOwnMemory = true;
	}
	else
	{
		x = &input;
		bOwnMemory = false;
	}
}

void MultiClassSVM::model(const Array<double>& input, Array<double>& output)
{
	if (numberOutput)
	{
		Array<double> tmp(classes);
		if (input.ndim() == 1)
		{
			output.resize(1, false);
			Predict(input, tmp);
			output(0) = VectorToClass(tmp);
		}
		else if (input.ndim() == 2)
		{
			unsigned int i, ic = input.dim(0);
			output.resize(ic, 1, false);
			for (i=0; i<ic; i++)
			{
				Predict(input[i], tmp);
				output(i, 0) = VectorToClass(tmp);
			}
		}
		else throw SHARKEXCEPTION("[MultiClassSVM::model] invalid dimension");
	}
	else
	{
		if (input.ndim() == 1)
		{
			output.resize(classes, false);
			Predict(input, output);
		}
		else if (input.ndim() == 2)
		{
			unsigned int i, ic = input.dim(0);
			output.resize(ic, classes, false);
			for (i=0; i<ic; i++)
			{
				Predict(input[i], output[i]);
			}
		}
		else throw SHARKEXCEPTION("[MultiClassSVM::model] invalid dimension");
	}
}

unsigned int MultiClassSVM::model(const Array<double>& input)
{
	SIZE_CHECK(input.ndim() == 1);
	SIZE_CHECK(input.dim(0) == inputDimension);

	Array<double> tmp(classes);
	Predict(input, tmp);
	return VectorToClass(tmp);
}


bool MultiClassSVM::isSupportVector( unsigned exampleIndex ) 
{
	for ( unsigned c = 0; c < classes; c++ )
		if( getAlpha( exampleIndex, c ) != 0 )
			return true;
	return false;
}


unsigned MultiClassSVM::getNumberOfSupportVectors() 
{
	unsigned numberOfSVs = 0;
	for (unsigned i = 0; i < examples; i++)
		if (isSupportVector(i))
			numberOfSVs++;
	return numberOfSVs;
}


unsigned MultiClassSVM::getNumberOfSupportVectors(unsigned c) 
{
	unsigned numberOfSVs = 0;
	for (unsigned i = 0; i < examples; i++)
		if(getAlpha(i,c) != 0.0)
			numberOfSVs++;
	return numberOfSVs;
}


unsigned int MultiClassSVM::VectorToClass(const Array<double>& v)
{
	SIZE_CHECK(v.ndim() == 1);
	SIZE_CHECK(v.dim(0) == classes);

	unsigned int c;
	unsigned int best = 0;
	double bestval = v(0);
	for (c=1; c<classes; c++)
	{
		if (v(c) > bestval)
		{
			bestval = v(c);
			best = c;
		}
	}

	return best;
}

void MultiClassSVM::MakeSparse()
{
	if (! bOwnMemory) return;

	// count support vectors and compute a map
	unsigned int e, c, i = 0;
	unsigned int s, sc = 0;
	std::vector<unsigned int> map(examples);
	for (e=0; e<examples; e++)
	{
		for (c=0; c<classes; c++) if (parameter(i + c) != 0.0) break;
		if (c < classes)
		{
			map[sc] = e;
			sc++;
		}
		i += classes;
	}

	// copy relevant coefficients and training data
	unsigned int dim = x->dim(1);
	Array<double> p(classes * (sc + 1));
	Array<double>* xx = new Array<double>(sc, dim);
	Array<double>* yy = new Array<double>(sc, 1);

	unsigned int j = 0;
	for (s=0; s<sc; s++)
	{
		e = map[s];
		(*xx)[s] = (*x)[e];
		(*yy)[s] = (*y)[e];
		i = e * classes;
		for (c=0; c<classes; c++) p(j + c) = parameter(i + c);
		j += classes;
	}
	j = sc * classes;
	i = examples * classes;
	for (c=0; c<classes; c++) p(j + c) = parameter(i + c);

	parameter = p;
	examples = sc;
	delete x;
	delete y;
	x = xx;
	y = yy;
}

// We assume that the output array has correct size.
void MultiClassSVM::Predict(const Array<double>& input, Array<double>& output)
{
	unsigned int c, e, i = 0;
	for (c=0; c<classes; c++) output(c) = getOffset(c);
	for (e=0; e<examples; e++)
	{
		double k = kernel->eval((*x)[e], input);
		for (c=0; c<classes; c++)
		{
			output(c) += parameter(i) * k;
			i++;
		}
	}
}

// We assume that the output array has correct size.
void MultiClassSVM::Predict(const Array<double>& input, ArrayReference<double> output)
{
	unsigned int c, e, i = 0;
	for (c=0; c<classes; c++) output(c) = getOffset(c);
	for (e=0; e<examples; e++)
	{
		double k = kernel->eval((*x)[e], input);
		for (c=0; c<classes; c++)
		{
			output(c) += parameter(i) * k;
			i++;
		}
	}
}


////////////////////////////////////////////////////////////////////////////////


MetaSVM::MetaSVM(SVM* pSVM, unsigned int numberOfHyperParameters)
{
	svm = pSVM;
	kernel = pSVM->getKernel();
	hyperparameters = numberOfHyperParameters;

	unsigned int k, kc = kernel->getParameterDimension();
	parameter.resize(hyperparameters + kc, false);
	parameter = 0.0;
	for (k = 0; k < kc; k++) parameter(k + hyperparameters) = kernel->getParameter(k);
}

MetaSVM::MetaSVM(MultiClassSVM* pSVM, unsigned int numberOfHyperParameters)
{
	svm = pSVM;
	kernel = pSVM->getKernel();
	hyperparameters = numberOfHyperParameters;

	unsigned int k, kc = kernel->getParameterDimension();
	parameter.resize(hyperparameters + kc, false);
	parameter = 0.0;
	for (k = 0; k < kc; k++) parameter(k + hyperparameters) = kernel->getParameter(k);
}

MetaSVM::~MetaSVM()
{
}


void MetaSVM::model(const Array<double>& input, Array<double>& output)
{
	svm->model(input, output);
}

void MetaSVM::setParameter(unsigned int index, double value)
{
	Model::setParameter(index, value);
	if (index >= hyperparameters)
	{
		// set a kernel parameter
		kernel->setParameter(index - hyperparameters, value);
	}
}

bool MetaSVM::isFeasible()
{
	return (kernel->isFeasible());
}


////////////////////////////////////////////////////////////////////////////////


C_SVM::C_SVM(SVM* pSVM, double Cplus, double Cminus, bool norm2, bool unconst)
: MetaSVM(pSVM, 1)
{
	C_ratio = Cminus / Cplus;
	norm2penalty = norm2;
	exponential = unconst;

	setParameter(0, (exponential) ? log(Cplus) : Cplus);
}

C_SVM::~C_SVM()
{
}


const Array<double>& C_SVM::PrepareDerivative()
{
	if (norm2penalty) throw SHARKEXCEPTION("[C_SVM::PrepareDerivative] PrepareDerivative works only in the 1-norm case.");

	SVM* svm = getSVM();
	const Array<double>& pt = svm->getPoints();
	int a, ac = svm->getExamples();
	int s, sc = svm->getParameterDimension();
	int k, kc = kernel->getParameterDimension();
	int pc = getParameterDimension();
	Array<double> der;

	alpha_b_Derivative.resize(sc, pc, false);
	alpha_b_Derivative = 0.0;

	// determine free and bounded support vectors
	std::vector<int> fi;
	std::vector<int> ri;
	for (a=0; a<ac; a++)
	{
		double alpha = svm->getAlpha(a);
		if (alpha != 0.0)
		{
			if ((alpha == -C_minus) || (alpha == C_plus)) ri.push_back(a);
			else fi.push_back(a);
		}
	}
	int f, fc = fi.size();
	int r, rc = ri.size();
	fi.push_back(ac);

	Array2D<double> H(fc+1, fc+1);
	Array2D<double> H_inv(fc+1, fc+1);
	Array2D<double> R(fc+1, rc);
	Array<double> dH(kc, fc+1, fc+1);
	Array<double> dR(kc, fc+1, rc);
	Array<double> alpha_f(fc+1);
	Array<double> alpha_r(rc);
	for (f=0; f<fc; f++) alpha_f(f) = svm->getAlpha(fi[f]);
	alpha_f(fc) = svm->getOffset();
	for (r=0; r<rc; r++) alpha_r(r) = svm->getAlpha(ri[r]);

	// initialize H and dH
	for (f=0; f<fc; f++)
	{
		int f2;
		for (f2=0; f2<f; f2++)
		{
			H(f, f2) = H(f2, f) = kernel->evalDerivative(pt[fi[f]], pt[fi[f2]], der);
			for (k=0; k<kc; k++)
			{
				dH(k, f, f2) = dH(k, f2, f) = der(k);
			}
		}
		H(f, f) = kernel->evalDerivative(pt[fi[f]], pt[fi[f]], der);
		for (k=0; k<kc; k++)
		{
			dH(k, f, f) = der(k);
		}
		H(fc, f) = H(f, fc) = 1.0;
		for (k=0; k<kc; k++) dH(k, fc, f) = dH(k, f, fc) = 0.0;
	}
	H(fc, fc) = 0.0;
	for (k=0; k<kc; k++) dH(k, fc, fc) = 0.0;

	// compute H_inv
// 	invertSymm(H_inv, H);
	g_inverse(H, H_inv, 200, 1e-10, true);

	// initialize R and dR
	for (r=0; r<rc; r++)
	{
		for (f=0; f<fc; f++)
		{
			R(f, r) = kernel->evalDerivative(pt[fi[f]], pt[ri[r]], der);
			for (k=0; k<kc; k++)
			{
				dR(k, f, r) = der(k);
			}
		}
		R(fc, r) = 1.0;
		for (k=0; k<kc; k++) dR(k, fc, r) = 0.0;
	}

	der.resize(fc+1, false);

	// compute the derivative of (\alpha, b) w.r.t. C
	if (rc > 0)
	{
		Array<double> y(rc);
		for (r=0; r<rc; r++) y(r) = ((alpha_r(r) > 0.0) ? 1.0 : -1.0);
		Array<double> Ry(fc+1);
		matColVec(Ry, R, y);
		matColVec(der, H_inv, Ry);
		for (f=0; f<=fc; f++) alpha_b_Derivative(fi[f], 0) = -der(f);
		for (r=0; r<rc; r++) alpha_b_Derivative(ri[r], 0) = ((alpha_r(r) > 0.0) ? 1.0 : -C_ratio);
		if (exponential)
		{
			for (s=0; s<sc; s++) alpha_b_Derivative(s, 0) *= C_plus;
		}
	}

	// compute the derivative of (\alpha, b) w.r.t. the kernel parameters
	for (k=0; k<kc; k++)
	{
		Array<double> sum(fc+1);
		Array<double> tmp(fc+1);
		matColVec(sum, dH[k], alpha_f);
		matColVec(tmp, dR[k], alpha_r);
		for (f=0; f<=fc; f++) sum(f) += tmp(f);
		matColVec(der, H_inv, sum);
		for (f=0; f<=fc; f++) alpha_b_Derivative(fi[f], k+1) = -der(f);
	}

	return alpha_b_Derivative;
}

void C_SVM::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	SVM* svm = getSVM();
	const Array<double>& pt = svm->getPoints();
	int k, kc = kernel->getParameterDimension();
	int a, ac = svm->getExamples();
	int pc = getParameterDimension();

	derivative.resize(1, pc, false);
	for (k=0; k<=kc; k++) derivative(0, k) = alpha_b_Derivative(ac, k);

	Array<double> der(pc);

	for (a=0; a<ac; a++)
	{
		double alpha = svm->getAlpha(a);
		double ker = kernel->evalDerivative(input, pt[a], der);
		derivative(0, 0) += ker * alpha_b_Derivative(a, 0);
		for (k=0; k<kc; k++)
		{
			derivative(0, k+1) += alpha * der(k) + ker * alpha_b_Derivative(a, k+1);
		}
	}
}

void C_SVM::setParameter(unsigned int index, double value)
{
	MetaSVM::setParameter(index, value);
	if (index == 0)
	{
		// set C_+ and C_-
		if (exponential)
		{
			value = exp(value);
		}
		C_plus = value;
		C_minus = C_ratio * value;
	}
}

bool C_SVM::isFeasible()
{
	return (C_plus > 0.0 && MetaSVM::isFeasible());
}


////////////////////////////////////////////////////////////////////////////////


OneClassSVM::OneClassSVM(SVM* pSVM, double fractionNu)
: MetaSVM(pSVM, 1)
{
	setParameter(0, fractionNu);
}

OneClassSVM::~OneClassSVM()
{
}


void OneClassSVM::setParameter(unsigned int index, double value)
{
	MetaSVM::setParameter(index, value);
	if (index == 0)
	{
		// There is no way to set this value correctly if this function is calles by the standard constructor.
		// That is, because of the dependency on the number of training examples.

		// The correct initialization takes place in SVM_Optimizer::optimize(...)
		nu = value;
	}
}

bool OneClassSVM::isFeasible()
{
	return (nu > 0.0 && MetaSVM::isFeasible());
}


////////////////////////////////////////////////////////////////////////////////


Epsilon_SVM::Epsilon_SVM(SVM* pSVM, double C, double epsilon, bool unconst)
: MetaSVM(pSVM, 2)
{
	this->C = C;
	this->epsilon = epsilon;

	exponential = unconst;

	setParameter(0, (exponential) ? log(C) : C);
	setParameter(1, (exponential) ? log(epsilon) : epsilon);
}

Epsilon_SVM::~Epsilon_SVM()
{
}


void Epsilon_SVM::setParameter(unsigned int index, double value)
{
	if (index < hyperparameters)
	{
		Model::setParameter(index, value);
		if (index == 0)
		{
			if (exponential) C = exp(value); else C = value;
		}
		else
		{
			if (exponential) epsilon = exp(value); else epsilon = value;
		}
	}
	else
	{
		MetaSVM::setParameter(index, value);
	}
}

bool Epsilon_SVM::isFeasible()
{
	return (C > 0.0 && epsilon > 0.0 && MetaSVM::isFeasible());
}


////////////////////////////////////////////////////////////////////////////////


RegularizationNetwork::RegularizationNetwork(SVM* pSVM, double gamma)
: MetaSVM(pSVM, 1)
{
	parameter(0) = gamma;
}

RegularizationNetwork::~RegularizationNetwork()
{
}


bool RegularizationNetwork::isFeasible()
{
	return (parameter(0) > 0.0 && MetaSVM::isFeasible());
}


////////////////////////////////////////////////////////////////////////////////


AllInOneMcSVM::AllInOneMcSVM(MultiClassSVM* pSVM, double C, bool unconstrained)
: MetaSVM(pSVM, 1)
{
	exponential = unconstrained;
	set_C(C);
}

AllInOneMcSVM::~AllInOneMcSVM()
{
}


bool AllInOneMcSVM::isFeasible()
{
	if (! MetaSVM::isFeasible()) return false;
	if (exponential && parameter(0) <= 0.0) return false;
	return true;
}

const Array<double>& AllInOneMcSVM::PrepareDerivative()
{
	MultiClassSVM* svm = getMultiClassSVM();
	KernelFunction* kernel = svm->getKernel();
	const Array<double>& points = svm->getPoints();
	const Array<double>& target = svm->getTargets();
	int x, x2, xc = svm->getExamples();
	int c, cc = svm->getClasses();
	int a, ac = xc * cc;
	int k, kc = kernel->getParameterDimension();
	int pc = getParameterDimension();
	Array<double> der;
	double C = get_C();
	SIZE_CHECK(kc + 1 == pc);

	alphaDerivative.resize(ac, pc, false);
	alphaDerivative = 0.0;

	std::vector<unsigned int> fI;			// indices of free SVs
	std::vector<unsigned int> bI;			// indices of bounded SVs
	Array<bool> sv(xc);						// is the EXAMPLE a SV?
	sv = false;
	a = 0;
	for (x=0; x<xc; x++)
	{
		for (c=0; c<cc; c++)
		{
			double alpha = svm->getParameter(a);
			if (alpha > 0.0)
			{
				if (alpha < C)
				{
					fI.push_back(a);
				}
				else
				{
					bI.push_back(a);
				}
				sv(x) = true;
			}
			a++;
		}
	}
	unsigned int f, f2, fc = fI.size();
	unsigned int b, bc = bI.size();

	// compute required parts of K and dK
	Array<double> K(xc, xc);					// kernel matrix
	Array<double> dK(xc, xc, kc);				// kernel derivatives
	for (x=0; x<xc; x++)
	{
		if (! sv(x)) continue;
		for (x2=0; x2<x; x2++)
		{
			if (! sv(x2)) continue;
			K(x, x2) = kernel->evalDerivative(points[x], points[x2], der);
			for (k=0; k<kc; k++) dK(x, x2, k) = der(k);
		}
		K(x, x) = kernel->evalDerivative(points[x], points[x], der);
		for (k=0; k<kc; k++) dK(x, x, k) = der(k);
	}

	// build matrices F and B
	Matrix F(fc, fc);
	Vector B1(fc);
	for (f=0; f<fc; f++)
	{
		unsigned int x = fI[f] / cc;
		unsigned int v = fI[f] % cc;
		unsigned int y = (unsigned int)target(x, 0);
		for (f2=0; f2<0; f2++)
		{
			unsigned int x2 = fI[f2] / cc;
			unsigned int v2 = fI[f2] % cc;
			unsigned int y2 = (unsigned int)target(x2, 0);
			double mod = 0.0;
			if (y == y2) mod += 0.5;
			if (y == v2) mod -= 0.5;
			if (y2 == v) mod -= 0.5;
			if (v == v2) mod += 0.5;

			F(f, f2) = F(f2, f) = mod * K(x, x2);
		}
		if (y != v) F(f, f) = K(x, x);

		double sum = 0.0;
		for (b=0; b<bc; b++)
		{
			unsigned int x2 = bI[b] / cc;
			unsigned int v2 = bI[b] % cc;
			unsigned int y2 = (unsigned int)target(x2, 0);
			double mod = 0.0;
			if (y == y2) mod += 0.5;
			if (y == v2) mod -= 0.5;
			if (y2 == v) mod -= 0.5;
			if (v == v2) mod += 0.5;

			sum += mod * K(x, x2);
		}
		B1(f) = sum;
	}

	// invert F
	Matrix invF = F.inverse();

	// compute derivative of alpha w.r.t. C
	Vector tmp = invF * B1;
	for (f=0; f<fc; f++) alphaDerivative(fI[f], 0) = -tmp(f);
	for (b=0; b<bc; b++) alphaDerivative(bI[b], 0) = 1.0;

	// compute derivative of alpha w.r.t. all kernel parameters
	for (k=0; k<kc; k++)
	{
		Matrix dF(fc, fc);
		Vector dB1(fc);
		for (f=0; f<fc; f++)
		{
			unsigned int x = fI[f] / cc;
			unsigned int v = fI[f] % cc;
			unsigned int y = (unsigned int)target(x, 0);
			for (f2=0; f2<0; f2++)
			{
				unsigned int x2 = fI[f2] / cc;
				unsigned int v2 = fI[f2] % cc;
				unsigned int y2 = (unsigned int)target(x2, 0);
				double mod = 0.0;
				if (y == y2) mod += 0.5;
				if (y == v2) mod -= 0.5;
				if (y2 == v) mod -= 0.5;
				if (v == v2) mod += 0.5;
	
				dF(f, f2) = dF(f2, f) = mod * dK(x, x2, k);
			}
			if (y != v) dF(f, f) = dK(x, x, k);

			double sum = 0.0;
			for (b=0; b<bc; b++)
			{
				unsigned int x2 = bI[b] / cc;
				unsigned int v2 = bI[b] % cc;
				unsigned int y2 = (unsigned int)target(x2, 0);
				double mod = 0.0;
				if (y == y2) mod += 0.5;
				if (y == v2) mod -= 0.5;
				if (y2 == v) mod -= 0.5;
				if (v == v2) mod += 0.5;

				sum += mod * dK(x, x2, k);
			}
			dB1(f) = sum;
		}

		Vector inner(fc);
		for (f=0; f<fc; f++)
		{
			double sum = B1(f);
			for (f2=0; f2<0; f2++) sum += svm->getParameter(fI[f]) * dF(f, f2);
			inner(f) = sum;
		}
		Vector tmp = invF * inner;
		for (f=0; f<fc; f++) alphaDerivative(fI[f], k + 1) = -tmp(f);
	}

	return alphaDerivative;


// 	// determine free and bounded support vectors
// 	std::vector<int> fi;
// 	std::vector<int> ri;
// 	for (a=0; a<ac; a++)
// 	{
// 		double alpha = svm->getAlpha(a);
// 		if (alpha != 0.0)
// 		{
// 			if ((alpha == -C_minus) || (alpha == C_plus)) ri.push_back(a);
// 			else fi.push_back(a);
// 		}
// 	}
// 	int f, fc = fi.size();
// 	int r, rc = ri.size();
// 	fi.push_back(ac);
// 
// 	Array2D<double> H(fc+1, fc+1);
// 	Array2D<double> H_inv(fc+1, fc+1);
// 	Array2D<double> R(fc+1, rc);
// 	Array<double> dH(kc, fc+1, fc+1);
// 	Array<double> dR(kc, fc+1, rc);
// 	Array<double> alpha_f(fc+1);
// 	Array<double> alpha_r(rc);
// 	for (f=0; f<fc; f++) alpha_f(f) = svm->getAlpha(fi[f]);
// 	alpha_f(fc) = svm->getOffset();
// 	for (r=0; r<rc; r++) alpha_r(r) = svm->getAlpha(ri[r]);
// 
// 	// initialize H and dH
// 	for (f=0; f<fc; f++)
// 	{
// 		int f2;
// 		for (f2=0; f2<f; f2++)
// 		{
// 			H(f, f2) = H(f2, f) = kernel->evalDerivative(pt[fi[f]], pt[fi[f2]], der);
// 			for (k=0; k<kc; k++)
// 			{
// 				dH(k, f, f2) = dH(k, f2, f) = der(k);
// 			}
// 		}
// 		H(f, f) = kernel->evalDerivative(pt[fi[f]], pt[fi[f]], der);
// 		for (k=0; k<kc; k++)
// 		{
// 			dH(k, f, f) = der(k);
// 		}
// 		H(fc, f) = H(f, fc) = 1.0;
// 		for (k=0; k<kc; k++) dH(k, fc, f) = dH(k, f, fc) = 0.0;
// 	}
// 	H(fc, fc) = 0.0;
// 	for (k=0; k<kc; k++) dH(k, fc, fc) = 0.0;
// 
// 	// compute H_inv
// // 	invertSymm(H_inv, H);
// 	g_inverse(H, H_inv, 200, 1e-10, true);
// 
// 	// initialize R and dR
// 	for (r=0; r<rc; r++)
// 	{
// 		for (f=0; f<fc; f++)
// 		{
// 			R(f, r) = kernel->evalDerivative(pt[fi[f]], pt[ri[r]], der);
// 			for (k=0; k<kc; k++)
// 			{
// 				dR(k, f, r) = der(k);
// 			}
// 		}
// 		R(fc, r) = 1.0;
// 		for (k=0; k<kc; k++) dR(k, fc, r) = 0.0;
// 	}
// 
// 	der.resize(fc+1, false);
// 
// 	// compute the derivative of (\alpha, b) w.r.t. C
// 	if (rc > 0)
// 	{
// 		Array<double> y(rc);
// 		for (r=0; r<rc; r++) y(r) = ((alpha_r(r) > 0.0) ? 1.0 : -1.0);
// 		Array<double> Ry(fc+1);
// 		matColVec(Ry, R, y);
// 		matColVec(der, H_inv, Ry);
// 		for (f=0; f<=fc; f++) alpha_b_Derivative(fi[f], 0) = -der(f);
// 		for (r=0; r<rc; r++) alpha_b_Derivative(ri[r], 0) = ((alpha_r(r) > 0.0) ? 1.0 : -C_ratio);
// 		if (exponential)
// 		{
// 			for (s=0; s<sc; s++) alpha_b_Derivative(s, 0) *= C_plus;
// 		}
// 	}
// 
// 	// compute the derivative of (\alpha, b) w.r.t. the kernel parameters
// 	for (k=0; k<kc; k++)
// 	{
// 		Array<double> sum(fc+1);
// 		Array<double> tmp(fc+1);
// 		matColVec(sum, dH[k], alpha_f);
// 		matColVec(tmp, dR[k], alpha_r);
// 		for (f=0; f<=fc; f++) sum(f) += tmp(f);
// 		matColVec(der, H_inv, sum);
// 		for (f=0; f<=fc; f++) alphaDerivative(fi[f], k+1) = -der(f);
// 	}
// 
// 	return alphaDerivative;
}

void AllInOneMcSVM::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	MultiClassSVM* svm = getMultiClassSVM();
	const Array<double>& pt = svm->getPoints();
	unsigned int k, kc = kernel->getParameterDimension();
	unsigned int x, xc = svm->getExamples();
	unsigned int c, cc = svm->getClasses();
	unsigned int pc = getParameterDimension();
	Array<double> der(kc);
	SIZE_CHECK(kc + 1 == pc);

	derivative.resize(cc, pc, false);
	derivative = 0.0;

	unsigned int a = 0;
	for (x=0; x<xc; x++)
	{
		double ker = kernel->evalDerivative(input, pt[x], der);
		for (c=0; c<cc; c++)
		{
			double alpha = svm->getAlpha(x, c);

			// exclude dummies and exploit sparseness
			if (alpha != 0.0)
			{
				derivative(c, 0) += ker * alphaDerivative(a, 0);
				for (k=0; k<kc; k++)
				{
					derivative(c, k+1) += alpha * der(k) + ker * alphaDerivative(a, k+1);
				}
			}
			a++;
		}
	}
}


////////////////////////////////////////////////////////////////////////////////


CrammerSingerMcSVM::CrammerSingerMcSVM(MultiClassSVM* pSVM, double C, bool unconstrained)
: MetaSVM(pSVM, 1)
{
	exponential = unconstrained;
	set_C(C);
}

CrammerSingerMcSVM::~CrammerSingerMcSVM()
{
}


bool CrammerSingerMcSVM::isFeasible()
{
	if (! exponential && parameter(0) <= 0.0) return false;
	return MetaSVM::isFeasible();
}


////////////////////////////////////////////////////////////////////////////////


LLWMcSVM::LLWMcSVM(MultiClassSVM* pSVM, double C, bool unconstrained)
: MetaSVM(pSVM, 1)
{
	exponential = unconstrained;
	set_C(C);
}

LLWMcSVM::~LLWMcSVM()
{
}


bool LLWMcSVM::isFeasible()
{
	if (! exponential && parameter(0) <= 0.0) return false;
	return MetaSVM::isFeasible();
}


////////////////////////////////////////////////////////////////////////////////


DGIMcSVM::DGIMcSVM(MultiClassSVM* pSVM, double C, bool unconstrained)
: MetaSVM(pSVM, 1)
{
	exponential = unconstrained;
	set_C(C);
}

DGIMcSVM::~DGIMcSVM()
{
}


bool DGIMcSVM::isFeasible()
{
	if (! exponential && parameter(0) <= 0.0) return false;
	return MetaSVM::isFeasible();
}


////////////////////////////////////////////////////////////////////////////////


OVAMcSVM::OVAMcSVM(MultiClassSVM* pSVM, double C)
: MetaSVM(pSVM, 1)
{
	parameter(0) = C;
}

OVAMcSVM::~OVAMcSVM()
{
}


bool OVAMcSVM::isFeasible()
{
	return (parameter(0) > 0.0 && MetaSVM::isFeasible());
}


////////////////////////////////////////////////////////////////////////////////


OCCMcSVM::OCCMcSVM(MultiClassSVM* pSVM, double C)
: MetaSVM(pSVM, 1)
{
	parameter(0) = C;
}

OCCMcSVM::~OCCMcSVM()
{
}


bool OCCMcSVM::isFeasible()
{
	return (parameter(0) > 0.0 && MetaSVM::isFeasible());
}


////////////////////////////////////////////////////////////////////////////////


EpochBasedCsMcSvm::EpochBasedCsMcSvm(MultiClassSVM* pSVM, double C, bool unconst, bool countKernels,
									 unsigned int wss, unsigned int reru, SvmStatesCollection * states,
									 int epochs, double dual)
: MetaSVM(pSVM, 1)
{
	exponential = unconst;
	set_C(C);
	wssMode = wss;
	reruMode = reru;
	count = countKernels;
	epochLim = epochs;
	dualAim = dual;
	mep_historian = states;
}

EpochBasedCsMcSvm::~EpochBasedCsMcSvm()
{ }

bool EpochBasedCsMcSvm::isFeasible()
{
	if ( !exponential && parameter(0) <= 0.0) return false;
	return MetaSVM::isFeasible();
}


////////////////////////////////////////////////////////////////////////////////

SVM_Optimizer::SVM_Optimizer()
{
	solver = NULL;
	matrix = NULL;
	cache = NULL;
	precomputedMatrix = false;

	printInfo = false;
	accuracy = 0.001;
	cacheMB = 256;
	maxIter = -1;
	maxSeconds = -1;

	optimal = true;
}

SVM_Optimizer::~SVM_Optimizer()
{
	if (solver != NULL)
	{
		delete solver;
		solver = NULL;
	}
	if (matrix != NULL)
	{
		delete matrix;
		matrix = NULL;
	}
	if (cache != NULL)
	{
		delete cache;
		cache = NULL;
	}
}


void SVM_Optimizer::init(Model& model)
{
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	Epsilon_SVM* esvm = dynamic_cast<Epsilon_SVM*>(&model);
	OneClassSVM* OCsvm = dynamic_cast<OneClassSVM*>(&model);
	RegularizationNetwork* rn = dynamic_cast<RegularizationNetwork*>(&model);
	GaussianProcess* gaussproc = dynamic_cast<GaussianProcess*>(&model);
	AllInOneMcSVM* aio = dynamic_cast<AllInOneMcSVM*>(&model);
	CrammerSingerMcSVM* cs = dynamic_cast<CrammerSingerMcSVM*>(&model);
	LLWMcSVM* llw = dynamic_cast<LLWMcSVM*>(&model);
	DGIMcSVM* dgi = dynamic_cast<DGIMcSVM*>(&model);
	OVAMcSVM* ova = dynamic_cast<OVAMcSVM*>(&model);
	OCCMcSVM* bc = dynamic_cast<OCCMcSVM*>(&model);
	EpochBasedCsMcSvm* ebcs = dynamic_cast<EpochBasedCsMcSvm*>(&model);

	if (csvm != NULL)
	{
		if (csvm->is2norm()) mode = eC2;
		else mode = eC1;
		Cplus = csvm->get_Cplus();
		Cminus = csvm->get_Cminus();
	}
	else if (esvm != NULL)
	{
		mode = eEpsilon;
		C = esvm->get_C();
		epsilon = esvm->get_epsilon();
	}
	else if (OCsvm != NULL)
	{
		mode = e1Class;
		fractionOfOutliers = OCsvm->getNu();
	}
	else if (rn != NULL)
	{
		mode = eRegularizationNetwork;
		gamma = rn->get_gamma();
	}
	else if (gaussproc != NULL)
	{
		mode = eGaussianProcess;
	}
	else if (aio != NULL)
	{
		mode = eAllInOne;
		C = aio->get_C();
	}
	else if (cs != NULL)
	{
		mode = eCrammerSinger;
		C = cs->get_C();
	}
	else if (llw != NULL)
	{
		mode = eLLW;
		C = llw->get_C();
	}
	else if (dgi != NULL)
	{
		mode = eDGI;
		C = dgi->get_C();
	}
	else if (ova != NULL)
	{
		mode = eOVA;
		C = ova->get_C();
	}
	else if (bc != NULL)
	{
		mode = eOCC;
		C = bc->get_C();
	}
	else if (ebcs != NULL)
	{
		mode = eEBCS;
		C = ebcs->get_C();
		countKernels = ebcs->get_countPreference();
		epochLimit = ebcs->get_epochLim();
		dualLimit = ebcs->get_dualAim();
		wssPref = ebcs->get_wssMode();
		reruPref = ebcs->get_reruMode();
		mep_historian = ebcs->get_historian();
	}
	else throw SHARKEXCEPTION("[SVM_Optimizer::init] The model is not a valid support vector machine meta model.");

	if (solver != NULL)
	{
		delete solver;
		solver = NULL;
	}
	if (matrix != NULL)
	{
		delete matrix;
		matrix = NULL;
	}
	if (cache != NULL)
	{
		delete cache;
		cache = NULL;
	}
}

double SVM_Optimizer::optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target)
{
	switch (mode)
	{
		case eC1:
		case eC2:
		case eEpsilon:
		case eNu:
		case eRegularizationNetwork:
		case e1Class:
		{
			SVM* svm = dynamic_cast<SVM*>(&model);
			if (svm == NULL)
			{
				MetaSVM* meta = dynamic_cast<MetaSVM*>(&model);;
				if (meta != NULL) svm = meta->getSVM();
			}
			if (svm == NULL) throw SHARKEXCEPTION("[SVM_Optimizer::optimize] The model is not a valid SVM model or SVM meta model.");

			return optimize(*svm, input, target);
		}

		case eGaussianProcess:
		{
			GaussianProcess* gaussproc = dynamic_cast<GaussianProcess*>(&model);
			gaussproc->train( input, target );
			return 0.0;
		}

		case eAllInOne:
		case eCrammerSinger:
		case eLLW:
		case eDGI:
		case eOVA:
		case eOCC:
		case eEBCS:
		{
			MultiClassSVM* svm = dynamic_cast<MultiClassSVM*>(&model);
			if (svm == NULL)
			{
				MetaSVM* meta = dynamic_cast<MetaSVM*>(&model);;
				if (meta != NULL) svm = meta->getMultiClassSVM();
			}
			if (svm == NULL) throw SHARKEXCEPTION("[SVM_Optimizer::optimize] The model is not a valid MultiClassSVM model or SVM meta model.");

			optimize(*svm, input, target);
			return 0.0;
		}

		default:
		{
			throw SHARKEXCEPTION("[SVM_Optimizer::optimize] call init(...) before optimize(...)");
			return 0.0;
		}
	}
}

double SVM_Optimizer::optimize(SVM& model, const Array<double>& input, const Array<double>& target, bool copy)
{
	if (solver != NULL)
	{
		delete solver;
		solver = NULL;
	}
	if (matrix != NULL)
	{
		delete matrix;
		matrix = NULL;
	}
	if (cache != NULL)
	{
		delete cache;
		cache = NULL;
	}

	unsigned int e, examples = input.dim(0);

	model.SetTrainingData(input, copy);
	KernelFunction* kernel = model.getKernel();

	Array<double> alpha(examples);
	double b = 0.0;
	double ret = 0.0;

	if (mode == eC1)
	{
		// C-SVM with 1-norm penalty
		Array<double> linear(examples);
		Array<double> lower(examples);
		Array<double> upper(examples);
		for (e = 0; e < examples; e++)
		{
			alpha(e) = model.getParameter(e);
			linear(e) = target(e, 0);
			if (target(e, 0) > 0.0)
			{
				lower(e) = 0.0;
				upper(e) = Cplus;
			}
			else
			{
				lower(e) = -Cminus;
				upper(e) = 0.0;
			}
		}

		if (precomputedMatrix) matrix = new PrecomputedKernelMatrix(kernel, input);
		else matrix = new KernelMatrix(kernel, input);
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));

		QpSvmDecomp* solver = new QpSvmDecomp(*cache);
		this->solver = solver;
		solver->setVerbose(printInfo);
		solver->setMaxIterations(maxIter);
		solver->setMaxSeconds(maxSeconds);

		ret = solver->Solve(linear, lower, upper, alpha, accuracy);

		// computation of b
		Array<double> gradient;
		solver->getGradient(gradient);
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		unsigned int freeVars = 0;
		double value;
		for (e = 0; e < examples; e++)
		{
			value = gradient(e);
			if (alpha(e) == lower(e))
			{
				if (value > lowerBound) lowerBound = value;
			}
			else if (alpha(e) == upper(e))
			{
				if (value < upperBound) upperBound = value;
			}
			else
			{
				sum += value;
				freeVars++;
			}
		}

		if (freeVars > 0)
			b = sum / freeVars;						// stabilized exact value
		else
			b = 0.5 * (lowerBound + upperBound);	// best estimate
	}
	else if (mode == eC2)
	{
		// C-SVM with 2-norm penalty
		Array<double> linear(examples);
		Array<double> diag(examples);
		Array<double> lower(examples);
		Array<double> upper(examples);
		for (e = 0; e < examples; e++)
		{
			alpha(e) = model.getParameter(e);
			linear(e) = target(e, 0);
			if (target(e, 0) > 0.0)
			{
				diag(e) = 1.0 / Cplus;
				lower(e) = 0.0;
				upper(e) = 1e100;
			}
			else
			{
				diag(e) = 1.0 / Cminus;
				lower(e) = -1e100;
				upper(e) = 0.0;
			}
		}

		matrix = new RegularizedKernelMatrix(kernel, input, diag);
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));

		QpSvmDecomp* solver = new QpSvmDecomp(*cache);
		this->solver = solver;
		solver->setVerbose(printInfo);
		solver->setMaxIterations(maxIter);
		solver->setMaxSeconds(maxSeconds);

		ret = solver->Solve(linear, lower, upper, alpha, accuracy);

		// computation of b
		Array<double> gradient;
		solver->getGradient(gradient);
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		unsigned int freeVars = 0;
		double value;
		for (e = 0; e < examples; e++)
		{
			value = gradient(e);
			if (alpha(e) == lower(e))
			{
				if (value > lowerBound) lowerBound = value;
			}
			else if (alpha(e) == upper(e))
			{
				if (value < upperBound) upperBound = value;
			}
			else
			{
				sum += value;
				freeVars++;
			}
		}

		if (freeVars > 0)
			b = sum / freeVars;						// stabilized exact value
		else
			b = 0.5 * (lowerBound + upperBound);	// best estimate
	}
	else if (mode == eEpsilon)
	{
		// epsilon-SVM
		Array<double> solution(2*examples);
		Array<double> linear(2*examples);
		Array<double> lower(2*examples);
		Array<double> upper(2*examples);
		for (e = 0; e < examples; e++)
		{
			double p = model.getParameter(e);
			if (p >= 0.0)
			{
				solution(e) = p;
				solution(e + examples) = 0.0;
			}
			else
			{
				solution(e) = 0.0;
				solution(e + examples) = -p;
			}
			linear(e) = target(e, 0) - epsilon;
			linear(e + examples) = target(e, 0) + epsilon;
			lower(e) = 0.0;
			lower(e + examples) = -C;
			upper(e) = C;
			upper(e + examples) = 0.0;
		}

		if (precomputedMatrix) matrix = new QPMatrix2(new PrecomputedKernelMatrix(kernel, input));
		else matrix = new QPMatrix2(new KernelMatrix(kernel, input));
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));

		QpSvmDecomp* solver = new QpSvmDecomp(*cache);
		this->solver = solver;
		solver->setVerbose(printInfo);
		solver->setMaxIterations(maxIter);
		solver->setMaxSeconds(maxSeconds);

		ret = solver->Solve(linear, lower, upper, solution);
		for (e = 0; e < examples; e++)
		{
			alpha(e) = solution(e) + solution(e + examples);
		}

		// computation of b
		Array<double> gradient;
		solver->getGradient(gradient);
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		unsigned int freeVars = 0;
		double value;
		for (e = 0; e < examples; e++)
		{
			if (solution(e) > 0.0)
			{
				value = gradient(e);
				if (solution(e) < C)
				{
					sum += value;
					freeVars++;
				}
				else
				{
					if (value > lowerBound) lowerBound = value;
				}
			}
			if (solution(e + examples) < 0.0)
			{
				value = gradient(e + examples);
				if (solution(e + examples) > -C)
				{
					sum += value;
					freeVars++;
				}
				else
				{
					if (value < upperBound) upperBound = value;
				}
			}
		}

		if (freeVars > 0)
			b = sum / freeVars;						// stabilized exact value
		else
			b = 0.5 * (lowerBound + upperBound);	// best estimate
	}
	else if (mode == e1Class)
	{
		Array<double> linear(examples);
		Array<double> lower(examples);
		Array<double> upper(examples);

		unsigned int NrInitialSVs = (unsigned int)(examples * fractionOfOutliers);
		double sumAlpha = 0;
		double upperBox;

		// NOT NORMALIZED
//		double upperBox = 1 / (examples * fractionOfOutliers);

		// NORMALIZED VERSION, that is \sum_{i}^\ell \alpha_i = NrInitialSVs
		upperBox  = 1.0;

		// initialize all patterns
		for (e = 0; e < examples; e++)
		{
			// linear part of Quadratic Program = 0
			linear(e) = 0.0;

			// both Box constraints (aka. Cplus and Cminus)
			lower(e)  = 0.0;
			upper(e)  = upperBox;

			// coefficients for optimization
			alpha(e)  = upperBox;
		}

		// INITIALIZATION: Check if too many SVs are at bounds
		if (NrInitialSVs != examples)
		{
			int index;
			int diff = examples - NrInitialSVs;

			for (int s = 0; s < diff; s++)
			{
				// choose randomly a example
				index = Rng::discrete(0, examples-1);

				while(alpha(index) == 0)
				{
					// try again if the example has already been used
					index = Rng::discrete(0, examples-1);
				}
				// and free the SV
				alpha(index) = 0;
			}

			// Check if sumAlpha < e*nu
			sumAlpha = 0.0;
			for (unsigned int e = 0; e < examples; e++)
				sumAlpha += alpha(e);

			if (sumAlpha < (examples * fractionOfOutliers))
			{
				// if yes, find a free example
				index = Rng::discrete(0, examples-1);
				while(alpha(index) != 0)
				{
					// try again if the example already has been used
					index = Rng::discrete(0, examples-1);
				}
				// and now we get sumAlpha = 1
				alpha(index) = (examples * fractionOfOutliers) - sumAlpha;
			}
		}

		if (precomputedMatrix) matrix = new PrecomputedKernelMatrix(kernel, input);
		else matrix = new KernelMatrix(kernel, input);
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));

		QpSvmDecomp* solver = new QpSvmDecomp(*cache);
		this->solver = solver;
		solver->setVerbose(printInfo);
		solver->setMaxIterations(maxIter);
		solver->setMaxSeconds(maxSeconds);

		// solve the QP problem
		ret = solver->Solve(linear, lower, upper, alpha, accuracy);

		// computation of b
		Array<double> gradient;
		solver->getGradient(gradient);
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		unsigned int freeVars = 0;
		double value;

		for (e = 0; e < examples; e++)
		{
			value = gradient(e);
			if (alpha(e) == lower(e))	// no SVs
			{
				if (value > lowerBound) lowerBound = value;
			}
			else if (alpha(e) == upper(e))	// bounded SVs
			{
				if (value < upperBound) upperBound = value;
			}
			else	// free SVs
			{
				sum += value;
				freeVars++;
			}
		}

		if (freeVars > 0)
			b = sum / freeVars;	// stabilized exact value
		else
			b = 0.5 * (lowerBound + upperBound);	// best estimate
	}
	else if (mode == eRegularizationNetwork)
	{
		unsigned int f;
		Array2D<double> K(examples, examples);
		Array2D<double> invK(examples, examples);
		for (e = 0; e < examples; e++)
		{
			for (f = 0; f < e; f++)
			{
				double k = kernel->eval(input[e], input[f]);
				K(e, f) = K(f, e) = k;
			}
			double k = kernel->eval(input[e], input[e]);
			K(e, e) = k + gamma;
		}
		invertSymm(invK, K);
		matColVec(alpha, invK, target, 0);

		b = 0.0;
		ret = 0.0;
	}

	for (e = 0; e < examples; e++) model.setParameter(e, alpha(e));
	model.setParameter(examples, b);

	if (copy) model.MakeSparse();

	return ret;
}

void SVM_Optimizer::optimize(MultiClassSVM& model, const Array<double>& input, const Array<double>& target, bool copy)
{
	if (solver != NULL)
	{
		delete solver;
		solver = NULL;
	}
	if (matrix != NULL)
	{
		delete matrix;
		matrix = NULL;
	}
	if (cache != NULL)
	{
		delete cache;
		cache = NULL;
	}

	unsigned int i;
	unsigned int examples = input.dim(0);
	unsigned int classes = model.getClasses();
	unsigned int variables = examples * classes;

	model.SetTrainingData(input, target, copy);
	KernelFunction* kernel = model.getKernel();

	if (mode == eAllInOne)
	{
		Array<double> alpha(variables);
		Array<double> linear(variables);
		Array<double> lower(variables);
		Array<double> upper(variables);

		if (precomputedMatrix) matrix = new PrecomputedKernelMatrix(kernel, input);
		else matrix = new KernelMatrix(kernel, input);
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));

		unsigned int e, c;
		unsigned int index = 0;
		for (e=0; e<examples; e++)
		{
			for (c=0; c<classes; c++)
			{
				alpha(index) = model.getParameter(index);
				unsigned int y = (unsigned int)target(e, 0);
				linear(index) = 1.0;
				lower(index) = 0.0;
				if (y == c) upper(index) = 0.0;
				else upper(index) = C;
				index++;
			}
		}

		double mod[64] = {
			0.0, 0.5, 0.5, 1.0, -0.5, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, -1.0, -0.5, -0.5, 0.0,
			0.0, 0.5, 0.5, 1.0, -0.5, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, -1.0, -0.5, -0.5, 0.0,
			0.0, 0.5, 0.5, 1.0, -0.5, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, -1.0, -0.5, -0.5, 0.0,
			0.0, 0.5, 0.5, 1.0, -0.5, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, -1.0, -0.5, -0.5, 0.0
// TODO: Above are the constants for the Weston&Watkins approach, below
//	   for Vapnik's formulation. They differ only by a factor of two.
//	   Change to the latter one for improved consistency of the code?
// 			0.0, 1.0, 1.0, 2.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -2.0, -1.0, -1.0, 0.0,
// 			0.0, 1.0, 1.0, 2.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -2.0, -1.0, -1.0, 0.0,
// 			0.0, 1.0, 1.0, 2.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -2.0, -1.0, -1.0, 0.0,
// 			0.0, 1.0, 1.0, 2.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -2.0, -1.0, -1.0, 0.0,
		};

		QpMcDecomp* solver = new QpMcDecomp(*cache);
		this->solver = solver;
		solver->setMaxIterations(maxIter);
		solver->Set_WSS_Strategy(2);
		solver->Solve(classes, mod, target, linear, lower, upper, alpha, accuracy);
		optimal = solver->isOptimal();

		i = 0;
		for (e=0; e<examples; e++)
		{
			unsigned int t = (unsigned int)target(e, 0);
			double sum = 0.0;
			for (c=0; c<classes; c++) sum += alpha(i + c);
			for (c=0; c<classes; c++)
			{
				double value = -alpha(i + c);
				if (c == t) value += sum;
				model.setParameter(i + c, value);
			}
			i += classes;
		}
	}
	else if (mode == eCrammerSinger)
	{
		Array<double> alpha(variables);
		Array<double> linear(variables);
		Array<double> lower(variables);
		Array<double> upper(variables);

		if (precomputedMatrix) matrix = new PrecomputedKernelMatrix(kernel, input);
		else matrix = new KernelMatrix(kernel, input);
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));

		unsigned int e, c;
		unsigned int index = 0;
		for (e=0; e<examples; e++)
		{
			for (c=0; c<classes; c++)
			{
				alpha(index) = -model.getParameter(index);
				unsigned int y = (unsigned int)target(e, 0);
				if (y == c)
				{
					linear(index) = 0.0;
					lower(index) = -C;
					upper(index) = 0.0;
				}
				else
				{
					linear(index) = 1.0;
					lower(index) = 0.0;
					upper(index) = C;
				}
				index++;
			}
		}

		double mod[64] = {
			0.0, 1.0, 1.0, 2.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -2.0, -1.0, -1.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		};

		QpMcStzDecomp* solver = new QpMcStzDecomp(*cache);
		this->solver = solver;
		solver->setMaxIterations(maxIter);
		solver->Solve(classes, mod, target, linear, lower, upper, alpha, accuracy);
		optimal = solver->isOptimal();

		for (i=0; i < variables; i++) model.setParameter(i, -alpha(i));
	}
	else if (mode == eLLW)
	{
		Array<double> alpha(variables);
		Array<double> linear(variables);
		Array<double> lower(variables);
		Array<double> upper(variables);

		if (precomputedMatrix) matrix = new PrecomputedKernelMatrix(kernel, input);
		else matrix = new KernelMatrix(kernel, input);
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));

		double cm1bc = (classes - 1.0) / classes;
		double m1bc = -1.0 / classes;

		unsigned int e, c;
		unsigned int index = 0;
		for (e=0; e<examples; e++)
		{
			for (c=0; c<classes; c++)
			{
				alpha(index) = model.getParameter(index);
				unsigned int y = (unsigned int)target(e, 0);
				if (y == c)
				{
					linear(index) = cm1bc;
					lower(index) = 0.0;
				}
				else
				{
					linear(index) = m1bc;
					lower(index) = -C;
				}
				upper(index) = 0.0;
				index++;
			}
		}

		double mod[64];
		for (c=0; c<64; c++) mod[c] = (c & 2) ? cm1bc : m1bc;

		QpMcDecomp* solver = new QpMcDecomp(*cache);
		this->solver = solver;
		solver->setMaxIterations(maxIter);
		solver->Set_WSS_Strategy(2);
		solver->Solve(classes, mod, target, linear, lower, upper, alpha, accuracy);
		optimal = solver->isOptimal();

		// Wahba's (unnecessary) sum-to-zero constraint:
		// subtract the example-wise mean. However, this
		// reduces sparsity in some cases.
		// The following lines can be commented-out without
		// changing any results.
// 		index = 0;
// 		for (e=0; e<examples; e++)
// 		{
// 			double sum = 0.0;
// 			for (c=0; c<classes; c++) sum += alpha(index + c);
// 			double mean = sum / (double)classes;
// 			for (c=0; c<classes; c++) alpha(index + c) -= mean;
// 			index += classes;
// 		}

		for (i=0; i < variables; i++) model.setParameter(i, alpha(i));
	}
	else if (mode == eDGI)
	{
		Array<double> alpha(variables);
		Array<double> linear(variables);
		Array<double> lower(variables);
		Array<double> upper(variables);

		double cm1bc = (classes - 1.0) / classes;
		double m1bc = -1.0 / classes;

		if (precomputedMatrix) matrix = new PrecomputedKernelMatrix(kernel, input);
		else matrix = new KernelMatrix(kernel, input);
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));

		unsigned int e, c;
		unsigned int index = 0;
		for (e=0; e<examples; e++)
		{
			for (c=0; c<classes; c++)
			{
				alpha(index) = -model.getParameter(index);
				unsigned int y = (unsigned int)target(e, 0);
				if (y == c)
				{
					linear(index) = 0.0;
					lower(index) = -C;
					upper(index) = 0.0;
				}
				else
				{
					linear(index) = 1.0;
					lower(index) = 0.0;
					upper(index) = C;
				}
				index++;
			}
		}

		double mod[64];
		for (c=0; c<16; c++) mod[c] = (c & 2) ? cm1bc : m1bc;
		for (c=16; c<64; c++) mod[c] = 0.0;

		QpMcStzDecomp* solver = new QpMcStzDecomp(*cache);
		this->solver = solver;
		solver->setMaxIterations(maxIter);
		solver->Solve(classes, mod, target, linear, lower, upper, alpha, accuracy);
		optimal = solver->isOptimal();

		for (i=0; i < variables; i++) model.setParameter(i, -alpha(i));
	}
	else if (mode == eOVA)
	{
		Array<double> alpha(examples);
		Array<double> linear(examples);
		Array<double> lower(examples);
		Array<double> upper(examples);

		// train a set of binary classifiers
		unsigned int c, e;
		if (precomputedMatrix) matrix = new PrecomputedKernelMatrix(kernel, input);
		else matrix = new KernelMatrix(kernel, input);
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));

		QpBoxDecomp* solver = new QpBoxDecomp(*cache);
		solver->setMaxIterations(maxIter);
		solver->Set_WSS_Strategy(2);
		this->solver = solver;

		for (c=0; c<classes; c++)
		{
			for (e=0; e<examples; e++)
			{
				alpha(e) = model.getParameter(e * classes + c);
				if (target(e, 0) == c)
				{
					linear(e) = 1.0;
					lower(e) = 0.0;
					upper(e) = C;
				}
				else
				{
					linear(e) = -1.0;
					lower(e) = -C;
					upper(e) = 0.0;
				}
			}
			solver->Solve(linear, lower, upper, alpha, accuracy);
			optimal = solver->isOptimal();

			for (e=0; e<examples; e++)
			{
				model.setParameter(e * classes + c, alpha(e));
			}
		}
	}
	else if (mode == eOCC)
	{
		Array<double> alpha(examples);
		Array<double> linear(examples);
		Array<double> lower(examples);
		Array<double> upper(examples);

		if (precomputedMatrix) matrix = new PrecomputedKernelMatrix(kernel, input);
		else matrix = new KernelMatrix(kernel, input);
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));

		unsigned int e, c;
		for (e=0; e<examples; e++)
		{
			alpha(e) = model.getParameter(e);
			linear(e) = 1.0;
			lower(e) = 0.0;
			upper(e) = C;
		}

		double mod[64];

		// (standard) orthogonal label prototypes
		for (c=0; c<64; c++) mod[c] = (c & 1) ? 1.0 : 0.0;

		// simplex label prototypes
// 		for (c=0; c<64; c++) mod[c] = (c & 1) ? 1.0 : -1.0 / classes;

		QpMcDecomp* solver = new QpMcDecomp(*cache);
		this->solver = solver;
		solver->setMaxIterations(maxIter);
		solver->Set_WSS_Strategy(2);
		solver->Solve(1, mod, target, linear, lower, upper, alpha, accuracy);
		optimal = solver->isOptimal();

		unsigned int index = 0;
		for (e=0; e<examples; e++)
		{
			unsigned int label = (unsigned int)target(e, 0);
			for (c=0; c<classes; c++)
			{
				double value = (label == c) ? alpha(e) : 0.0;
				model.setParameter(index, value);
				index++;
			}
		}
	}
	else if (mode == eEBCS)
	{
		Array<double> alpha(variables);
		alpha = 0; //redundant, but make extra clear that we won't accept custom inital values
		
		if (precomputedMatrix) 
			matrix = new PrecomputedKernelMatrix( kernel, input );
		else
			matrix = new KernelMatrix(kernel, input, countKernels);
			
		cache = new CachedMatrix(matrix, 1048576 * cacheMB / sizeof(float));
		QpEbCsDecomp* solver = new QpEbCsDecomp( *cache, target, classes, wssPref, reruPref, mep_historian );
		this->solver = solver;
		solver->setVerbose(printInfo);
		solver->setStoppingConditions(0.0001, epochLimit, dualLimit); //0.0001 is default in original paper
		solver->Solve(alpha, C);
		for (i=0; i < variables; i++) model.setParameter(i, alpha(i));
	}

	// set the bias term to zero
	for (i=0; i < classes; i++) model.setParameter(variables + i, 0.0);

	if (copy) model.MakeSparse();
}

// static dummy member
ErrorFunction& SVM_Optimizer::dummyError = * ((ErrorFunction*) NULL);
