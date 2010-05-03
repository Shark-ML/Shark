//===========================================================================
/*!
 *  \file createConnectionMatrix.h
 *
 *  \brief Offers methods for creating connection matrices
 *         for neural networks.
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    2002
 *
 *  \par Copyright (c) 1995, 1999, 2002:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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


#ifndef CREATE_CONNECTION_MATRIX_H
#define CREATE_CONNECTION_MATRIX_H

#include <Array/ArrayIo.h>
#include <fstream>


//===========================================================================
/*!
 *  \brief Creates a connection matrix for a network.
 *
 *  Automatically creates a connection matrix with several layers, with
 *  the numbers of neurons for each layer defined by \em layers and
 *  (standard) connections defined by \em ff_layer, \em ff_in_out,
 *  \em ff_all and \em bias.
 *
 *      \param  con        the resulting connection matrix.
 *      \param  layers     contains the numbers of neurons for each
 *                         layer of the network.
 *      \param  ff_layer   if set to \em true, connections from
 *                         each neuron of layer \f$i\f$ to each neuron
 *                         of layer \f$i+1\f$ will be set for all layers.
 *      \param  ff_in_out  if set to \em true, connections from
 *                         all input neurons to all output neurons
 *                         will be set.
 *      \param  ff_all     if set to \em true, connections from all
 *                         neurons of layer \f$i\f$ to all neurons of
 *                         layers \f$j\f$ with \f$j > i\f$ will be set
 *                         for all layers \f$i\f$.
 *      \param  bias       if set to \em true, connections from
 *                         all neurons (except the input neurons)
 *                         to the bias will be set.
 *      \return none
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    2002
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void createConnectionMatrix(Array<int> &con,
							Array<unsigned> &layers,
							bool ff_layer  = true,   // all connections
							// between layers?
							bool ff_in_out = true,   // shortcuts from in to
							// out?
							bool ff_all    = true,   // all shortcuts?
							bool bias      = true)   // bias?
{
	unsigned N = 0;  // total number of neurons:
	unsigned row,    // connection matrix row    (target neuron)
	column, // connection matrix column (start neuron)
	k,      // counter.
	z_pos,  // first target neuron in next layer
	s_pos;  // first start neuron in current layer


	//
	// Calculate total number of neurons from the
	// number of neurons per layer:
	//
	for (k = 0; k < layers.dim(0); k++) N += layers(k);
	con.resize(N, N + 1);
	con = 0;

	//
	// set connections from each neuron of layer i to each
	// neuron of layer i + 1 for all layers:
	//
	if (ff_layer)
	{
		z_pos = layers(0);
		s_pos = 0;
		for (k = 0; k < layers.dim(0) - 1; k++)
		{
			for (row = z_pos; row < z_pos + layers(k + 1); row++)
				for (column = s_pos; column < s_pos + layers(k); column++)
					con(row, column) = 1;
			s_pos += layers(k);
			z_pos += layers(k + 1);
		}
	}

	//
	// set connections from all input neurons to all output neurons:
	//
	if (ff_in_out)
	{
		for (row = N - layers(layers.dim(0) - 1); row < N; row++)
			for (column = 0; column < layers(0); column++)
				con(row, column) = 1;
	}

	//
	// set connections from all neurons of layer i to
	// all neurons of layers j with j > i for all layers i:
	//
	if (ff_all)
	{
		z_pos = layers(0);
		s_pos = 0;
		for (k = 0; k < layers.dim(0) - 1; k++)
		{
			for (row = z_pos; row < z_pos + layers(k + 1); row++)
				for (column = 0; column < s_pos + layers(k); column++)
					con(row, column) = 1;
			s_pos += layers(k);
			z_pos += layers(k + 1);
		}
	}

	//
	// set connections from all neurons (except the input neurons)
	// to the bias values:
	//
	if (bias)
		for (k = layers(0); k < N; k++) con(k, N) = 1;

}


//===========================================================================
/*!
 *  \brief Creates a connection matrix for a network with a single
 *         hidden layer.
 *
 *  Automatically creates a connection matrix for a network with
 *  three different layers: An input layer with \em in input neurons,
 *  an output layer with \em out output neurons and a single hidden layer
 *  with \em hidden hidden neurons.
 *  (Standard) connections can be defined by \em ff_layer,
 *  \em ff_in_out, \em ff_all and \em bias.
 *
 *      \param  con        the resulting connection matrix.
 *      \param  in         number of input neurons.
 *      \param  hidden     number of neurons of the single hidden layer.
 *      \param  out        number of output neurons.
 *      \param  ff_layer   if set to \em true, connections from
 *                         each neuron of layer \f$i\f$ to each neuron
 *                         of layer \f$i+1\f$ will be set for all layers.
 *      \param  ff_in_out  if set to \em true, connections from
 *                         all input neurons to all output neurons
 *                         will be set.
 *      \param  ff_all     if set to \em true, connections from all
 *                         neurons of layer \f$i\f$ to all neurons of
 *                         layers \f$j\f$ with \f$j > i\f$ will be set
 *                         for all layers \f$i\f$.
 *      \param  bias       if set to \em true, connections from
 *                         all neurons (except the input neurons)
 *                         to the bias will be set.
 *      \return none
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    2002
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void createConnectionMatrix(Array<int> &con,
							unsigned in, unsigned hidden, unsigned out,
							bool ff_layer  = true,  // all connections beteween layers?
							bool ff_in_out = true,  // shortcuts from in to out?
							bool ff_all    = true,  // all shortcuts?
							bool bias      = true)
{ // bias?
	Array<unsigned> layer(3);
	layer(0) = in;
	layer(1) = hidden;
	layer(2) = out;
	createConnectionMatrix(con, layer, ff_layer, ff_in_out, ff_all, bias);
}


//===========================================================================
/*!
 *  \brief Creates a connection matrix for a network with two
 *         hidden layers.
 *
 *  Automatically creates a connection matrix for a network with
 *  four different layers: An input layer with \em in input neurons,
 *  an output layer with \em out output neurons and two hidden layers
 *  with \em hidden1 and \em hidden2 hidden neurons, respectively.
 *  (Standard) connections can be defined by \em ff_layer,
 *  \em ff_in_out, \em ff_all and \em bias.
 *
 *      \param  con        the resulting connection matrix.
 *      \param  in         number of input neurons.
 *      \param  hidden1    number of neurons of the first hidden layer.
 *      \param  hidden2    number of neurons of the second hidden layer.
 *      \param  out        number of output neurons.
 *      \param  ff_layer   if set to \em true, connections from
 *                         each neuron of layer \f$i\f$ to each neuron
 *                         of layer \f$i+1\f$ will be set for all layers.
 *      \param  ff_in_out  if set to \em true, connections from
 *                         all input neurons to all output neurons
 *                         will be set.
 *      \param  ff_all     if set to \em true, connections from all
 *                         neurons of layer \f$i\f$ to all neurons of
 *                         layers \f$j\f$ with \f$j > i\f$ will be set
 *                         for all layers \f$i\f$.
 *      \param  bias       if set to \em true, connections from
 *                         all neurons (except the input neurons)
 *                         to the bias will be set.
 *      \return none
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    2002
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void createConnectionMatrix(Array<int> &con,
							unsigned in, unsigned hidden1, unsigned hidden2, unsigned out,
							bool ff_layer  = true,  // all connections beteween layers?
							bool ff_in_out = true,  // shortcuts from in to out?
							bool ff_all    = true,  // all shortcuts?
							bool bias      = true)
{ // bias?
	Array<unsigned> layer(4);
	layer(0) = in;
	layer(1) = hidden1;
	layer(2) = hidden2;
	layer(3) = out;
	createConnectionMatrix(con, layer, ff_layer, ff_in_out, ff_all, bias);
}

//===========================================================================
/*!
 *  \brief Creates a connection matrix for a recurrent network.
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void createConnectionMatrixRNN(Array<int> &con,
							   unsigned in, unsigned hidden, unsigned out,
							   unsigned memory = 1, bool layered = false,
							   bool recurrentInputs = true, bool bias = true, bool elman = false, bool previousInputs = false)
{
	unsigned N;  // total number of neurons:
	unsigned k, i, j;

	//
	// Calculate total number of neurons from the
	// number of neurons per layer:
	//
	N = in + hidden + out;
	con.resize(1 + memory, N, N);
	con[0] = 0;

	if (!layered)
	{
		for (i = in; i < N; i++)
		{
			for (j = 0; j < i; j++)
				con(0, i, j) = 1;
			if (bias) con(0, i, i) = 1;
		}
	}
	else
	{
		for (i = in; i < N - out; i++)
		{
			for (j = 0; j < in; j++)
				con(0, i, j) = 1;
			if (bias) con(0, i, i) = 1;
		}
		if (elman)
			for (i = N - out; i < N; i++)
			{
				for (j = 0; j < N - out; j++)
					con(0, i, j) = 1;
				if (bias) con(0, i, i) = 1;
			}
		else
			for (i = N - out; i < N; i++)
			{
				for (j = 0; j < in; j++)
					con(0, i, j) = 1;
				if (bias) con(0, i, i) = 1;
			}
	}
	if (recurrentInputs)
		for (i = 1; i <= memory; i++) con[i] = 1;
	else
	{
		for (i = 1; i <= memory; i++)
		{
			con[i] = 1;
			for (j = 0; j < in; j++)
			{
				(con[i])[j] = 0;
				if (!previousInputs) for (k = in; k < N; k++) con(i, k, j) = 0;
			}
		}
	}
}

#endif



