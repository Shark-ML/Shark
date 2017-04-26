//===========================================================================
/*!
 * 
 *
 * \brief   Support for downloading data sets from online sources.
 * 
 * 
 * \par
 * The methods in this file allow to download data sets from the
 * mldata.org repository and other sources.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
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
//===========================================================================

#ifndef SHARK_DATA_DOWNLOAD_H
#define SHARK_DATA_DOWNLOAD_H

#include <shark/Core/DLLSupport.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/SparseData.h>
#include <shark/Data/Csv.h>
#include <sstream>

namespace shark {

/**
 * \ingroup shark_globals
 *
 * @{
 */
	
	
/// \brief Split a URL into its domain and resource parts.
///
/// Returns a std::pair where the first element is the domain and the second is
/// the resource. With std::tie you can do pattern-matching:
/// std::tie(domain, resource) = splitUrl(url);
/// will fill the std::string variables domain and resource.
SHARK_EXPORT_SYMBOL std::pair<std::string, std::string> splitUrl(std::string const & url);
	
/// \brief Download a document with the HTTP protocol.
///
/// \param  url       download URL, for example "www.shark-ml.org/index.html"
/// \param  port      TCP/IP port, defaults to 80
///
/// The function requests the document with a HTTP request and returns
/// the body of the corresponding HTTP reply. In case of success this
/// is the requested document. In case of an error the function throws
/// an exception. Note that the function does not perform standard
/// actions of web browsers, e.g., execute javascript or follow http
/// redirects. All HTTP response status codes other than 200 are
/// reported as failure to download the document and trigger an
/// exception.
SHARK_EXPORT_SYMBOL std::string download(std::string const& url, unsigned short port = 80);

/// \brief Download and import a sparse data (libSVM) file.
///
/// \param  dataset       container storing the loaded data
/// \param  url           http URL
/// \param  port          TCP/IP port, default is 80
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
template <class InputType, class LabelType> void downloadSparseData(
	LabeledData<InputType, LabelType>& dataset,
	std::string const& url,
	unsigned short port = 80,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
)
{
	std::string content = download(url, port);
	std::stringstream ss(content);
	importSparseData(dataset, ss, highestIndex, batchSize);
}


/// \brief Download a data set from mldata.org.
///
/// \param  dataset       container storing the loaded data
/// \param  name          data set name
/// \param  batchSize     size of batch
template <class InputType, class LabelType> void downloadFromMLData(
	LabeledData<InputType, LabelType>& dataset,
	std::string const& name,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
)
{
	std::string filename;
	for (char c : name)
	{
		if (c == ' ') c = '-';
		else if (c >= 'A' && c <= 'Z') c += 32;
		else if (c == '[' || c == '(' || c == ')' || c == '.' || c == ']') continue;
		filename += c;
	}
	downloadSparseData(dataset, "mldata.org/repository/data/download/libsvm/" + filename + "/", 80, 0, batchSize);
}


/// \brief Download and import a dense data (CSV) file for classification.
///
/// \param  dataset       container storing the loaded data
/// \param  url           http URL
/// \param  lp            Position of the label in the record, either first or last column
/// \param  separator     Optional separator between entries, typically a comma, spaces ar automatically ignored
/// \param  comment       Trailing character indicating comment line. By dfault it is '#'
/// \param  port          TCP/IP port, default is 80
/// \param  maximumBatchSize   size of batches in the dataset
template <class InputType> void downloadCsvData(
	LabeledData<InputType, unsigned int>& dataset,
	std::string const& url,
	LabelPosition lp,
	char separator = ',',
	char comment = '#',
	unsigned short port = 80,
	std::size_t maximumBatchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
)
{
	std::string content = download(url, port);
	csvStringToData(dataset, content, lp, separator, comment, maximumBatchSize);
}


/// \brief Download and import a dense data (CSV) file for regression.
///
/// \param  dataset       container storing the loaded data
/// \param  url           http URL
/// \param  lp            Position of the label in the record, either first or last column
/// \param  numberOfOutputs   dimensionality of the labels
/// \param  separator     Optional separator between entries, typically a comma, spaces ar automatically ignored
/// \param  comment       Trailing character indicating comment line. By dfault it is '#'
/// \param  port          TCP/IP port, default is 80
/// \param  maximumBatchSize   size of batches in the dataset
template <class InputType> void downloadCsvData(
	LabeledData<InputType, RealVector>& dataset,
	std::string const& url,
	LabelPosition lp,
	std::size_t numberOfOutputs = 1,
	char separator = ',',
	char comment = '#',
	unsigned short port = 80,
	std::size_t maximumBatchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
)
{
	std::string content = download(url, port);
	csvStringToData(dataset, content, lp, numberOfOutputs, separator, comment, maximumBatchSize);
}


/** @}*/

}
#endif
