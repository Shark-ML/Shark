//===========================================================================
/*!
*  \file Model.cpp
*
*  \brief Base class of all models.
*
*  \author  T. Glasmachers, C. Igel
*  \date    2005, 2009
*
*  \par Copyright (c) 1999-2005:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*  \par Project:
*      ReClaM
*
*
*  <BR>
*
*
*  <BR><HR>
*  This file is part of ReClaM. This library is free software;
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

#include <ReClaM/Model.h>


Model::Model()
{
	epsilon = 1e-8;
}

Model::~Model()
{
}


void Model::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	if (input.ndim() == 1)
	{
		double old;
		int p, pc = parameter.dim(0);
		Array<double> output;
		Array<double> perturbed_output;
		model(input, output);
		int o, oc = output.dim(0);
		derivative.resize(oc, pc, false);
		for (p = 0; p < pc; p++)
		{
			old = getParameter(p);
			setParameter(p, old + epsilon);
			model(input, perturbed_output);
			for (o = 0; o < oc; o++)
			{
				derivative(o, p) = (perturbed_output(o) - output(o)) / epsilon;
			}
			setParameter(p, old);
		}
	}
	else throw SHARKEXCEPTION("[Model::modelDerivative] invalid number of dimensions.");
}

void Model::modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative)
{
	model(input, output);
	modelDerivative(input, derivative);
}

void Model::generalDerivative(const Array<double>& input, const Array<double>& coefficient, Array<double>& derivative)
{
	Array<double> md;
	modelDerivative(input, md);

	int o, oc = getOutputDimension();
	int p, pc = getParameterDimension();

	derivative.resize(pc, false);

	for (p = 0; p < pc; p++)
	{
		double td = 0.0;
		for (o = 0; o < oc; o++)
		{
			td += coefficient(o) * md(o, p);
		}
		derivative(p) = td;
	}
}

bool Model::isFeasible()
{
	return true;
}

double Model::getParameter(unsigned int index) const
{
	return parameter(index);
}

void Model::setParameter(unsigned int index, double value)
{
	parameter(index) = value;
}

void Model::read(std::istream& is)
{
	char c;
	double value;
	std::vector<double> v;
	int p, pc;
	while (true)
	{
		is >> value;
		v.push_back(value);
		is.read(&c, 1);
		if (c == '\r')
		{
			is.read(&c, 1);
			if (! is.good() || c != '\n') throw SHARKEXCEPTION("[Model::operator >>] invalid data format");
			break;
		}
		if (! is.good() || c != ' ') throw SHARKEXCEPTION("[Model::operator >>] invalid data format");
	}
	pc = v.size();
	parameter.resize(pc, false);
	for (p = 0; p < pc; p++) setParameter(p, v[p]);
}

// friend of Model
void Model::write(std::ostream& os) const
{
	int oldprec = os.precision();
	os.precision(16);
	int p, pc = parameter.dim(0);
	for (p = 0; p < pc; p++)
	{
		if (p != 0) os << " ";
		os << parameter(p);
	}
	os.write("\r\n", 2);
	os.precision(oldprec);
}

bool Model::load(const char* filename)
{
	std::ifstream is;
	is.open(filename);
	if (! is.is_open()) return false;
	read(is);
	is.close();
	return true;
}

bool Model::save(const char* filename)
{
	std::ofstream os;
	os.open(filename);
	if (! os.is_open()) return false;
	write(os);
	os.close();
	return true;
}


// friend of Model
std::istream& operator >> (std::istream& is, Model& model)
{
	if(is) model.read(is);
	return is;
}

// friend of Model
std::ostream& operator << (std::ostream& os, const Model& model)
{
	if(os) model.write(os);
	return os;
}


