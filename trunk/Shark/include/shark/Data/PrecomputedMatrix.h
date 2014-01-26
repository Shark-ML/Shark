//===========================================================================
/*!
 * 
 *
 * \brief       Read and write precomputed kernel matrices (using libsvm format)
 * 
 * 
 *
 * \author      M. Tuma
 * \date        2012
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
//===========================================================================

#ifndef SHARK_DATA_PRECOMPUTEDMATRIX_H
#define SHARK_DATA_PRECOMPUTEDMATRIX_H



#include <fstream> 
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Models/Kernels/ScaledKernel.h>
#include <shark/Algorithms/Trainers/NormalizeKernelUnitVariance.h>


namespace shark {

/**
 * \ingroup shark_globals
 *
 * @{
 */

enum KernelMatrixNormalizationType {
	NONE,                               // no normalization. output regular Gram kernel matrix
	MULTIPLICATIVE_TRACE_ONE,           // determine the trace, and devide each entry by it
	MULTIPLICATIVE_TRACE_N,             // determine the trace, devide each entry by it, then multiply by the number of samples
	MULTIPLICATIVE_VARIANCE_ONE,        // normalize to unit variance in feature space. see kloft in jmlr 2012.
	CENTER_ONLY,                        // center the kernel in feature space. see cortes in jmlr 2012 and in icml 2010.
	CENTER_AND_MULTIPLICATIVE_TRACE_ONE // first center the kernel in featrue space. then devide each entry by the centered kernel's trace.
};

/// \brief Write a kernel Gram matrix to stream.
///
/// \param  dataset    data basis for the Gram matrix
/// \param  kernel     pointer to kernel function to be used
/// \param  out         The stream to be written to
/// \param  normalizer what kind of normalization to apply. see enum declaration for details.
/// \param  scientific        should the output be in scientific notation?
/// \param  fieldwidth      field width for pretty printing
template<typename InputType, typename LabelType>
void export_kernel_matrix(
	LabeledData<InputType, LabelType> const &dataset,
        AbstractKernelFunction<InputType> &kernel,           // kernel function (can't be const b/c of ScaledKernel later)
        std::ostream &out,                                     // The stream to be written to
        KernelMatrixNormalizationType normalizer = NONE, // what kind of normalization to apply. see enum declaration for details.
        bool scientific = false,                         // scientific notation?
        unsigned int fieldwidth = 0                      // for pretty-printing
) {
	//get access to the range of elements
	typedef typename LabeledData<InputType, LabelType>::const_element_range Elements;
	typedef typename Elements::reference ElementRef;
	DataView<LabeledData<InputType, LabelType> const> points(dataset);
	std::size_t size = points.size();
	
	SIZE_CHECK(size != 0);
	// check outstream status
	if (!out) {
		throw(std::invalid_argument("[export_kernel_matrix] Can't write to stream."));
	}

	// COMPUTE MODIFIERS

	// if multiplicative trace normalization: determine trace
	double trace = 0.0;
	double trace_factor = 1.0;
	if (normalizer == MULTIPLICATIVE_TRACE_ONE || normalizer == MULTIPLICATIVE_TRACE_N) {
		BOOST_FOREACH(ElementRef point, points){
			trace += kernel.eval(point.input, point.input);
		}
		SHARK_ASSERT(trace > 0);
		trace_factor = 1.0/trace;
		if (normalizer == MULTIPLICATIVE_TRACE_N) {
			trace_factor *= size;
		}
	}

	// if multiplicative variance normalization: determine factor
	double variance_factor = 0.0;
	if (normalizer == MULTIPLICATIVE_VARIANCE_ONE) {
		ScaledKernel<InputType> scaled(&kernel);
		NormalizeKernelUnitVariance<InputType> normalizer;
		normalizer.train(scaled, dataset.inputs());
		variance_factor = scaled.factor();
	}

	// if centering: determine matrix- and row-wise means;
	double mean = 0;
	RealVector rowmeans(size,0.0);
	if (normalizer == CENTER_ONLY || normalizer == CENTER_AND_MULTIPLICATIVE_TRACE_ONE) {
		// initialization: calculate mean and rowmeans
		for (std::size_t i=0; i<size; i++) {
			double k = kernel.eval(points[i].input,points[i].input);
			mean += k; //add diagonal value to mean once
			rowmeans(i) += k; //add diagonal to its rowmean
			for (std::size_t j=0; j<i; j++) {
				double k = kernel.eval(points[i].input,points[j].input);
				mean += 2.0 * k; //add off-diagonals to mean twice
				rowmeans(i) += k; //add to mean of row
				rowmeans(j) += k; //add to mean of transposed row
			}
		}
		mean = mean / (double) size / (double) size;
		rowmeans /= size;
		// get trace if necessary
		if (normalizer == CENTER_AND_MULTIPLICATIVE_TRACE_ONE) {
			trace = 0.0;
			for (std::size_t i=0; i<size; i++) {
				trace += kernel.eval(points[i].input, points[i].input) - 2*rowmeans(i) + mean;
			}
			SHARK_ASSERT(trace > 0);
			trace_factor = 1.0/trace;
		}
	}

	// FIX OUTPUT FORMAT

	// set output format
	if (scientific)
		out.setf(std::ios_base::scientific);
	std::streamsize ss = out.precision();
	out.precision(10);

	// determine dataset type
	double max_label = -1e100;
	double min_label = -max_label;
	bool binary = false;
	bool regression = false;
	BOOST_FOREACH(double cur_label, dataset.labels().elements()){
		if (cur_label > max_label)
			max_label = cur_label;
		if (cur_label < min_label)
			min_label = cur_label;
		if ((cur_label != (int)cur_label) || cur_label < 0)
			regression = true;
	}
	if (!regression && (min_label == 0) && (max_label == 1))
		binary = true;

	// WRITE OUT

	// write to file:
	// loop through examples (rows)
	for (std::size_t i=0; i< size; i++) {

		// write label
		if (regression) {
			out << std::setw(fieldwidth) << std::left << points[i].label << " ";
		} else if (binary) {
			out << std::setw(fieldwidth) << std::left << (int)(points[i].label*2-1) << " ";
		} else {
			out << std::setw(fieldwidth) << std::left << (unsigned int)(points[i].label+1) << " ";
		}

		out << "0:"<< std::setw(fieldwidth) << std::left << i+1; //write index

		// loop through examples (columns)
		// CASE DISTINCTION:
		if (normalizer == NONE) {
			for (std::size_t j=0; j<size; j++) {
				out  << " " << j+1 << ":" << std::setw(fieldwidth) << std::left << kernel.eval(points[i].input, points[j].input);
			}
			out << "\n";
		} else if (normalizer == MULTIPLICATIVE_TRACE_ONE || normalizer == MULTIPLICATIVE_TRACE_N) {
			for (std::size_t j=0; j<size; j++) {
				out  << " " << j+1 << ":" << std::setw(fieldwidth) << std::left << trace_factor * kernel.eval(points[i].input, points[j].input);
			}
			out << "\n";
		} else if (normalizer == MULTIPLICATIVE_VARIANCE_ONE) {
			for (std::size_t j=0; j<size; j++) {
				out  << " " << j+1 << ":" << std::setw(fieldwidth) << std::left <<  variance_factor *kernel.eval(points[i].input, points[j].input);
			}
			out << "\n";
		} else if (normalizer == CENTER_ONLY) {
			for (std::size_t j=0; j<size; j++) {
				double tmp = kernel.eval(points[i].input, points[j].input) - rowmeans(i) - rowmeans(j) + mean;
				out  << " " << j+1 << ":" << std::setw(fieldwidth) << std::left << tmp;
			}
			out << "\n";
		} else if (normalizer == CENTER_AND_MULTIPLICATIVE_TRACE_ONE) {
			for (std::size_t j=0; j<size; j++) {
				double tmp = kernel.eval(points[i].input, points[j].input) - rowmeans(i) - rowmeans(j) + mean;
				out  << " " << j+1 << ":" << std::setw(fieldwidth) << std::left << trace_factor *tmp;
			}
			out << "\n";
		} else {
			throw SHARKEXCEPTION("[detail::export_kernel_matrix] Unknown normalization type.");
		}

	}

	// clean up
	out.precision(ss);
}

/// \brief Write a kernel Gram matrix to file.
///
/// \param  dataset    data basis for the Gram matrix
/// \param  kernel     pointer to kernel function to be used
/// \param  fn         The filename of the file to be written to
/// \param  normalizer what kind of normalization to apply. see enum declaration for details.
/// \param  sci        should the output be in scientific notation?
/// \param  width      field width for pretty printing
template<typename InputType, typename LabelType>
void export_kernel_matrix(
	LabeledData<InputType, LabelType> const &dataset,
	AbstractKernelFunction<InputType> &kernel,
	std::string fn,
	KernelMatrixNormalizationType normalizer = NONE,
	bool sci = false,
	unsigned int width = 0
) {
	std::ofstream ofs(fn.c_str());
	if (ofs) {
		export_kernel_matrix(dataset, kernel, ofs, normalizer, sci, width);
	}
	else
		throw(std::invalid_argument("[detail::export_kernel_matrix] Stream cannot be opened for writing."));
	
}


// TODO: import functionality is still missing.
//		 when that is done, add tutorial


/** @}*/

} // namespace shark



#endif // SHARK_DATA_PRECOMPUTEDMATRIX_H
