//===========================================================================
/*!
 * 
 *
 * \brief       Support for importing and exporting data from and to (Weka) ARFF files.
 *
 * The import and export functions in the file aim to comply to the
 * specification at https://weka.wikispaces.com/ARFF+%28stable+version%29 ,
 * however, the spec is inaccurate and far from complete.
 * 
 * 
 * \par
 * The most important application of the methods provided in this
 * file is the import of data from ARFF files into Shark data
 * containers.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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

#ifndef SHARK_DATA_ARFF_H
#define SHARK_DATA_ARFF_H

#include <shark/Core/DLLSupport.h>
#include <shark/Data/Dataset.h>
#include <shark/Core/Exception.h>

#include <boost/lexical_cast.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <map>


namespace shark {

/**
 * \ingroup shark_globals
 *
 * @{
 */



namespace arff {

/// \brief Helper type for defining how to handle nominal attributes during import.
SHARK_EXPORT_SYMBOL enum HandleNominal
{
	NominalIgnore,                     ///< silently ignore nominal attributes
	NominalThrow,                      ///< throw an exception when encountering a nominal attribute
	NominalOneHotEncoding,             ///< convert the nominal into a vector with a one-hot encoding
};

/// \brief Helper type for defining how to handle string attributes during import.
SHARK_EXPORT_SYMBOL enum HandleString
{
	StringIgnore,                      ///< silently ignore string attributes
	StringThrow,                       ///< throw an exception when encountering a string attribute
};

/// \brief Helper type for defining how to handle date attributes during import.
SHARK_EXPORT_SYMBOL enum HandleDate
{
	DateIgnore,                        ///< silently ignore date attributes
	DateThrow,                         ///< throw an exception when encountering a date attribute
	DateConvertToUnixTimestamp,        ///< convert date to a double number, with 0 indicating the start of the epoch
};

/// \brief Helper type for defining how to handle relational attributes during import.
SHARK_EXPORT_SYMBOL enum HandleRelational
{
	RelationalIgnore,                  ///< silently ignore relational attributes
	RelationalThrow,                   ///< throw an exception when encountering a relational attribute
};

/// \brief Helper type for defining how to handle missing values during import.
SHARK_EXPORT_SYMBOL enum HandleMissingValue
{
	MissingValueNaN,                   ///< use silent NaN to denote missing values, and all-zero for one-hot encoded nominal attributes
	MissingValueThrow,                 ///< throw an exception when encountering a missing value
};

/// \brief Helper structure for holding (optional) import options.
SHARK_EXPORT_SYMBOL struct ImportOptions
{
	ImportOptions()
	: handleNominal(NominalOneHotEncoding)
	, handleString(StringThrow)
	, handleDate(DateConvertToUnixTimestamp)
	, handleRelational(RelationalThrow)
	, handleMissingValue(MissingValueThrow)
	{ }

	HandleNominal handleNominal;       ///< definition of how to handle nominal attributes
	HandleString handleString;         ///< definition of how to handle string attributes
	HandleDate handleDate;             ///< definition of how to handle date attributes
	HandleRelational handleRelational;   ///< definition of how to handle relational attributes
	HandleMissingValue handleMissingValue;  ///< definition of how to handle missing values
};


namespace detail {

enum AttributeType
{
	Nominal,
	Numeric,
	Date,
	Ignore,
};

SHARK_EXPORT_SYMBOL inline std::size_t parseString(std::string const& line, std::size_t start, std::string& token, std::string const& delimitors)
{
	while (line[start] == ' ') start++;
	std::size_t end = start;
	if (line[start] == '\"' || line[start] == '\'')
	{
		char quote = line[start];
		start++;
		end = line.find(quote, start);
		if (end == std::string::npos) throw SHARKEXCEPTION("[importARFF] improper string (missing trailing quote)");
		token = line.substr(start, end - start);
		end++;
		if (end == line.size()) return end;
		if (delimitors.find(line[end]) == std::string::npos)
		{
			while (line[end] == ' ') end++;
			if (end == line.size()) return end;
			if (delimitors.find(line[end]) == std::string::npos) throw SHARKEXCEPTION("[importARFF] missing delimitor");
		}
	}
	else
	{
		end = line.find_first_of(delimitors, start);
		if (end == std::string::npos) end = line.size();
		token = line.substr(start, end - start);
		if (end == line.size()) return end;
	}
	while (line[end] == ' ') end++;
	return end;
}

SHARK_EXPORT_SYMBOL inline std::size_t parseToken(std::string const& line, std::size_t start, std::string& token, char delimitor = ' ')
{
	while (line[start] == ' ') start++;
	std::size_t end = start;
	while (end < line.size() && line[end] != delimitor) end++;
	token = line.substr(start, end - start);
	while (line[end] == ' ') end++;
	return end;
}

SHARK_EXPORT_SYMBOL inline double parseDate(std::string const& date, std::string const& format)
{
	static const boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));

	try
	{
		// ignore the format for now and rely on the boost date parser
		boost::posix_time::ptime t;
		std::stringstream ss(date);
		ss >> t;
		boost::posix_time::time_duration diff = t - epoch;
		return (double)(diff.ticks() / diff.ticks_per_second());
	}
	catch (std::exception const& ex)
	{
		throw SHARKEXCEPTION(std::string("[importARFF] failed to parse date; ") + ex.what());
	}
}

template <typename T>
void setLabel(T& container, double value)
{
	SHARK_ASSERT(container.size() == 1);
	container[0] = value;
}

template <>
inline void setLabel<unsigned int>(unsigned int& label, double value)
{
	label = (unsigned int)value;
	SHARK_ASSERT((double)label == std::floor(value));
}

template <typename T>
std::string label2string(T const& label)
{
	throw SHARKEXCEPTION("[exportARFF] cannot to convert label to ARFF format");
}

template <>
inline std::string label2string<RealVector>(RealVector const& label)
{
	SHARK_ASSERT(label.size() == 1);
	return boost::lexical_cast<std::string>(label[0]);
}

template <>
inline std::string label2string<unsigned int>(unsigned int const& label)
{
	return boost::lexical_cast<std::string>(label);
}

};  // namespace detail
};  // namespace arff


/// \brief Import a labeled Dataset from an ARFF file
///
/// \param  stream             the stream to be read from
/// \param  labelname          name of the attribute acting as a label
/// \param  dataset            container storing the loaded data
/// \param  options            options, mostly for handling ARFF oddities
/// \param  maximumBatchSize   size of batches in the dataset
SHARK_EXPORT_SYMBOL template<typename InputT, typename LabelT> void importARFF(
	std::istream& stream,
	std::string const& labelname,
	LabeledData<InputT, LabelT>& dataset,
	arff::ImportOptions options = arff::ImportOptions(),
	std::size_t maximumBatchSize = LabeledData<InputT, LabelT>::DefaultBatchSize)
{
	std::vector<arff::detail::AttributeType> attributeType;
	std::map< std::size_t, std::map<std::string, std::size_t> > attributeNominalValue;
	std::map< std::size_t, std::string> dateFormat;
	std::vector<int> attributeStart;
	int labelIndex = -1;
	std::size_t dimension = 0;

	// read header section
	std::string line;
	while (std::getline(stream, line))
	{
		if (line.empty()) continue;
		if (line[line.size()-1] == '\r') line.erase(line.size() - 1);
		if (line.empty()) continue;
		if (line[0] == '%') continue;
		if (line[0] == '@')
		{
			std::size_t pos = line.find(' ');
			if (pos == std::string::npos) pos = line.size();
			// convert to ASCII lower case (i.e., don't rely on the locale)
			for (std::size_t i=0; i<pos; i++) if (line[i] >= 65 && line[i] <= 90) line[i] += 32;
			std::string type = line.substr(0, pos);
			if (type == "@data") break;
			else if (type == "@relation") continue;   // ignore
			else if (type == "@attribute")
			{
				std::size_t attributeIndex = attributeType.size();
				std::string s;
				pos = arff::detail::parseString(line, pos, s, " ");
				if (s == labelname) labelIndex = attributeIndex;
				if (line[pos] == '{')
				{
					pos++;
					if (options.handleNominal == arff::NominalThrow) throw SHARKEXCEPTION("[importARFF] cannot handle nominal attribute");
					while (true)
					{
						pos = arff::detail::parseString(line, pos, s, ",}");
						std::size_t attributeValue = attributeNominalValue[attributeIndex].size();
						attributeNominalValue[attributeIndex][s] = attributeValue;
						if (line[pos] == ',') pos++;
						else if (line[pos] == '}') { pos++; break; }
						else throw SHARKEXCEPTION("[importARFF] format error in nominal attribute declaration");
					}
					if (options.handleNominal == arff::NominalIgnore)
					{
						attributeType.push_back(arff::detail::Ignore);
						attributeStart.push_back(-1);
					}
					else if (options.handleNominal == arff::NominalOneHotEncoding)
					{
						attributeType.push_back(arff::detail::Nominal);
						attributeStart.push_back(dimension);
						if (labelIndex != attributeIndex) dimension += attributeNominalValue[attributeIndex].size();
					}
				}
				else
				{
					pos = arff::detail::parseToken(line, pos, s);
					if (s == "binary" || s == "numeric" || s == "integer" || s == "real")
					{
						attributeType.push_back(arff::detail::Numeric);
						attributeStart.push_back(dimension);
						if (labelIndex != attributeIndex) dimension++;
					}
					else if (s == "string")
					{
						if (options.handleString == arff::StringThrow) throw SHARKEXCEPTION("[importARFF] cannot handle string-valued attribute");
						attributeType.push_back(arff::detail::Ignore);
						attributeStart.push_back(-1);
					}
					else if (s == "date")
					{
						if (options.handleDate == arff::DateThrow) throw SHARKEXCEPTION("[importARFF] cannot handle date-valued attribute");
						attributeType.push_back(arff::detail::Date);
						if (options.handleDate == arff::DateIgnore)
						{
							attributeType.push_back(arff::detail::Ignore);
							attributeStart.push_back(-1);
						}
						else if (options.handleDate == arff::DateConvertToUnixTimestamp)
						{
							dateFormat[attributeIndex] = line.substr(pos);
							attributeType.push_back(arff::detail::Date);
							attributeStart.push_back(dimension);
							if (labelIndex != attributeIndex) dimension++;
						}
						else throw SHARKEXCEPTION("[importARFF] unknown date handling method");
					}
					else
					{
						throw SHARKEXCEPTION("[importARFF] unsupported attribute type: " + s);
					}
				}
			}
			else throw SHARKEXCEPTION("[importARFF] unsupported header field: " + type);
		}
		else throw SHARKEXCEPTION("[importARFF] invalid line in ARFF header");
	}
	if (labelIndex < 0) throw SHARKEXCEPTION("[importARFF] label attribute (dependent variable) not found");
	if (dimension == 0) throw SHARKEXCEPTION("[importARFF] no (valid) attributes found");
	SHARK_ASSERT(attributeType.size() == attributeStart.size());

	// read (sparse or dense) data
	std::vector<InputT> inputs;
	std::vector<LabelT> labels;
	while (std::getline(stream, line))
	{
		if (line.empty()) continue;
		if (line[line.size()-1] == '\r') line.erase(line.size() - 1);
		if (line.empty()) continue;
		if (line[0] == '%') continue;

		InputT input(dimension);
		LabelT label(1);               // works for unsigned int and for RealVector
		std::size_t pos = 0;
		std::string s;
		if (line[0] == '{')
		{
			// sparse format
			while (pos < line.size())
			{
				pos = arff::detail::parseString(line, pos, s, " ");
				std::size_t i = boost::lexical_cast<std::size_t>(s);
				pos = arff::detail::parseString(line, pos, s, ",}");
				if (attributeType[i] == arff::detail::Nominal)
				{
					if (s == "?")
					{
						if (options.handleMissingValue == arff::MissingValueThrow) throw SHARKEXCEPTION("[importARFF] cannot handle missing value");
						// nothing to do, represented as all-zeros
					}
					else
					{
						std::map<std::string, std::size_t>::const_iterator it = attributeNominalValue[i].find(s);
						if (it == attributeNominalValue[i].end()) throw SHARKEXCEPTION("[importARFF] undeclared value in nominal field");
						std::size_t value = it->second;
						if (i == labelIndex) arff::detail::setLabel(label, value);
						else input(attributeStart[i + value]) = 1.0;
					}
				}
				else if (attributeType[i] == arff::detail::Numeric)
				{
					if (s == "?")
					{
						if (options.handleMissingValue == arff::MissingValueThrow) throw SHARKEXCEPTION("[importARFF] cannot handle missing value");
						else input(attributeStart[i]) = std::nan("");
					}
					else
					{
						double value = boost::lexical_cast<double>(s);
						if (i == labelIndex) arff::detail::setLabel(label, value);
						else input(attributeStart[i]) = value;
					}
				}
				else if (attributeType[i] == arff::detail::Date)
				{
					if (s == "?")
					{
						if (options.handleMissingValue == arff::MissingValueThrow) throw SHARKEXCEPTION("[importARFF] cannot handle missing value");
						else input(attributeStart[i]) = std::nan("");
					}
					else
					{
						double value = arff::detail::parseDate(s, dateFormat[i]);
						if (i == labelIndex) arff::detail::setLabel(label, value);
						else input(attributeStart[i]) = value;
					}
				}
				else
				{
					SHARK_ASSERT(attributeType[i] == arff::detail::Ignore);
					continue;
				}
				if (line[pos] == '}') break;
				else pos++;
			}
		}
		else
		{
			// dense format
			for (std::size_t i=0; i<attributeType.size(); i++)
			{
				pos = arff::detail::parseString(line, pos, s, ",");
				if (attributeType[i] == arff::detail::Nominal)
				{
					if (s == "?")
					{
						if (options.handleMissingValue == arff::MissingValueThrow) throw SHARKEXCEPTION("[importARFF] cannot handle missing value");
						// nothing to do, represented as all-zeros
					}
					else
					{
						std::map<std::string, std::size_t>::const_iterator it = attributeNominalValue[i].find(s);
						if (it == attributeNominalValue[i].end()) throw SHARKEXCEPTION("[importARFF] undeclared value in nominal field");
						std::size_t value = it->second;
						if (i == labelIndex) arff::detail::setLabel(label, value);
						else input(attributeStart[i + value]) = 1.0;
					}
				}
				else if (attributeType[i] == arff::detail::Numeric)
				{
					if (s == "?")
					{
						if (options.handleMissingValue == arff::MissingValueThrow) throw SHARKEXCEPTION("[importARFF] cannot handle missing value");
						else input(attributeStart[i]) = std::nan("");
					}
					else
					{
						double value = boost::lexical_cast<double>(s);
						if (i == labelIndex) arff::detail::setLabel(label, value);
						else input(attributeStart[i]) = value;
					}
				}
				else if (attributeType[i] == arff::detail::Date)
				{
					if (s == "?")
					{
						if (options.handleMissingValue == arff::MissingValueThrow) throw SHARKEXCEPTION("[importARFF] cannot handle missing value");
						else input(attributeStart[i]) = std::nan("");
					}
					else
					{
						double value = arff::detail::parseDate(s, dateFormat[i]);
						if (i == labelIndex) arff::detail::setLabel(label, value);
						else input(attributeStart[i]) = value;
					}
				}
				else
				{
					SHARK_ASSERT(attributeType[i] == arff::detail::Ignore);
					continue;
				}
				if (line[pos] == ',') pos++;
			}
		}
		inputs.push_back(input);
		labels.push_back(label);
	}

	// create actual data set
	dataset = createLabeledDataFromRange(inputs, labels);
}

/// \brief Import a labeled Dataset from an ARFF file
///
/// \param  filename           name of the file to be imported
/// \param  labelname          name of the attribute acting as a label
/// \param  dataset            container storing the loaded data
/// \param  options            import options, mostly for handling ARFF oddities
/// \param  maximumBatchSize   Size of batches in the dataset
SHARK_EXPORT_SYMBOL template<typename InputT, typename LabelT> void importARFF(
	std::string const& filename,
	std::string const& labelname,
	LabeledData<InputT, LabelT>& dataset,
	arff::ImportOptions options = arff::ImportOptions(),
	std::size_t maximumBatchSize = LabeledData<InputT, LabelT>::DefaultBatchSize)
{
	std::ifstream stream(filename.c_str());
	importARFF(stream, labelname, dataset, options, maximumBatchSize);
}

/// \brief Write labeled data to an ARFF stream.
///
/// \param  stream         the stream to be written to
/// \param  dataset        container to be exported
/// \param  datasetname    name of the data set, stored as the ARFF @relation
/// \param  featurenames   feature names, stored as ARFF @attribute-s
/// \param  width          argument to std::setw when writing the output
SHARK_EXPORT_SYMBOL template<typename LabelT> void exportARFF(
	std::ostream& stream,
	LabeledData<RealVector, LabelT> const& dataset,
	std::string const& datasetname,
	std::vector<std::string> const& featurenames = std::vector<std::string>(),
	int width = 0)
{
	stream << "@relation " << datasetname << std::endl;
	std::size_t batches = dataset.numberOfBatches();
	std::size_t dimension = inputDimension(dataset);

	// check whether features are binary or numeric
	std::vector<bool> is_binary(dimension, true);
	for (std::size_t b=0; b<batches; b++)
	{
		typename LabeledData<RealVector, LabelT>::const_batch_reference batch = dataset.batch(b);
		std::size_t size = boost::size(batch);
		for (std::size_t i=0; i<size; i++)
		{
			typename LabeledData<RealVector, LabelT>::const_element_reference element = shark::get(batch, i);
			for (std::size_t j=0; j<dimension; j++)
			{
				double value = element.input(j);
				if (value != 0.0 && value != 1.0) is_binary[j] = false;
			}
		}
	}

	// write the attributes
	for (std::size_t i=0; i<dimension; i++)
	{
		stream << "@attribute "
				<< ((featurenames.size() > i) ? featurenames[i] : "attribute" + boost::lexical_cast<std::string>(i))
				<< " "
				<< (is_binary[i] ? "binary" : "numerical")
				<< std::endl;
	}
	stream << "@attribute label " << ((boost::is_integral<LabelT>::value) ? "integer" : "numeric") << std::endl;

	// write the data
	stream << "@data" << std::endl;
	for (std::size_t b=0; b<batches; b++)
	{
		typename LabeledData<RealVector, LabelT>::const_batch_reference batch = dataset.batch(b);
		std::size_t size = boost::size(batch);
		for (std::size_t i=0; i<size; i++)
		{
			typename LabeledData<RealVector, LabelT>::const_element_reference element = shark::get(batch, i);
			for (std::size_t j=0; j<dimension; j++)
			{
				stream << element.input(j) << ",";
			}
			LabelT l = element.label;
			stream << arff::detail::label2string<LabelT>(l) << std::endl;
		}
	}
}

/// \brief Write labeled data to an ARFF stream.
///
/// \param  stream         the stream to be written to
/// \param  dataset        container to be exported
/// \param  datasetname    name of the data set, stored as the ARFF @relation
/// \param  featurenames   feature names, stored as ARFF @attribute-s
/// \param  width          argument to std::setw when writing the output
SHARK_EXPORT_SYMBOL template<typename LabelT> void exportARFF(
	std::ostream& stream,
	LabeledData<CompressedRealVector, LabelT> const& dataset,
	std::string const& datasetname,
	std::vector<std::string> const& featurenames = std::vector<std::string>(),
	int width = 0)
{
	stream << "@relation " << datasetname << std::endl;
	std::size_t batches = dataset.numberOfBatches();
	std::size_t dimension = inputDimension(dataset);

	// check whether features are binary or numeric
	std::vector<bool> is_binary(dimension, true);
	std::vector<bool> is_integer(dimension, true);
	for (std::size_t b=0; b<batches; b++)
	{
		typename LabeledData<CompressedRealVector, LabelT>::const_batch_reference batch = dataset.batch(b);
		std::size_t size = boost::size(batch);
		for (std::size_t i=0; i<size; i++)
		{
			typename LabeledData<CompressedRealVector, LabelT>::const_element_reference element = shark::get(batch, i);
			typename Data<CompressedRealVector>::const_element_reference input = element.input;

			typedef typename Data<CompressedRealVector>::const_element_reference::const_iterator sparse_iterator;
			for (sparse_iterator it = input.begin(); it != input.end(); ++it)
			{
				std::size_t j = it.index();
				double value = *it;
				if (value != std::floor(value))
				{
					is_binary[j] = false;
					is_integer[j] = false;
				}
				else if (value != 0.0 && value != 1.0) is_binary[j] = false;
			}
		}
	}

	// write the attributes
	for (std::size_t i=0; i<dimension; i++)
	{
		std::string type = "numerical";
		if (is_integer[i]) type = "integer";
		if (is_binary[i]) type = "binary";
		stream << "@attribute "
				<< ((featurenames.size() > i) ? featurenames[i] : "attribute" + boost::lexical_cast<std::string>(i))
				<< " "
				<< type
				<< std::endl;
	}
	stream << "@attribute label " << ((boost::is_integral<LabelT>::value) ? "integer" : "numeric") << std::endl;

	// write the data
	stream << "@data" << std::endl;
	for (std::size_t b=0; b<batches; b++)
	{
		typename LabeledData<CompressedRealVector, LabelT>::const_batch_reference batch = dataset.batch(b);
		std::size_t size = boost::size(batch);
		for (std::size_t i=0; i<size; i++)
		{
			typename LabeledData<CompressedRealVector, LabelT>::const_element_reference element = shark::get(batch, i);
			typename Data<CompressedRealVector>::const_element_reference input = element.input;

			stream << "{";
			typedef typename Data<CompressedRealVector>::const_element_reference::const_iterator sparse_iterator;
			for (sparse_iterator it = input.begin(); it != input.end(); ++it)
			{
				std::size_t j = it.index();
				double value = *it;
				if (std::isnan(value)) stream << "?";
				else stream << j;
				stream << " " << value << ",";
			}
			LabelT l = element.label;
			stream << dimension << " " << arff::detail::label2string<LabelT>(l) << "}" << std::endl;
		}
	}
}

/// \brief Write labeled data to an ARFF file.
///
/// \param  filename      the file to be written to
/// \param  dataset       container to be exported
/// \param  datasetname   name of the data set, stored as the ARFF @relation
/// \param  featurenames  feature names, stored as ARFF @attribute-s
/// \param  width         argument to std::setw when writing the output
SHARK_EXPORT_SYMBOL template<typename InputT, typename LabelT> void exportARFF(
	std::string const& filename,
	LabeledData<InputT, LabelT> const& dataset,
	std::string const& datasetname,
	std::vector<std::string> const& featurenames = std::vector<std::string>(),
	int width = 0)
{
	std::ofstream stream(filename.c_str());
	exportARFF(stream, dataset, datasetname, featurenames, width);
}


/** @}*/

} // namespace shark
#endif // SHARK_ML_CSV_H
