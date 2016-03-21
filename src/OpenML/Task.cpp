//===========================================================================
/*!
 * 
 *
 * \brief       Implementation of an OpenML Task.
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


#include <shark/OpenML/OpenML.h>
#include <shark/OpenML/detail/Tools.h>
#include <shark/Core/Exception.h>

#include <boost/lexical_cast.hpp>

#include <fstream>


namespace shark {
namespace openML {


Task::Task(IDType id, bool downloadSplits)
: PooledEntity<Task>(id)
, m_file("task_" + boost::lexical_cast<std::string>(id) + "_splits.arff")
{
	detail::Json result = connection.get("/task/" + boost::lexical_cast<std::string>(id));

	detail::Json desc = result["task"];
	detail::Json input = desc["input"];
	detail::Json output = desc["output"];
	std::string s;

	s = desc["task_type"].asString();
	if (s == "Supervised Classification") m_tasktype = SupervisedClassification;
	else if (s == "Supervised Regression") m_tasktype = SupervisedRegression;
	else throw SHARKEXCEPTION("unknown task type: " + s);

	IDType datasetID = detail::json2number<IDType>(input[0]["data_set"]["data_set_id"]);
	m_dataset = Dataset::get(datasetID);
	m_targetFeature = input[0]["data_set"]["target_feature"].asString();

	detail::Json ep = input[1]["estimation_procedure"];
	s = ep["type"].asString();
	if (s == "crossvalidation") m_estimationProcedure = CrossValidation;
	else throw SHARKEXCEPTION("unknown estimation procedure: " + s);

	m_file.setUrl(ep["data_splits_url"].asString());

	m_folds = 1;
	m_repetitions = 1;

	detail::Json param = ep["parameter"];
	for (std::size_t i=0; i<param.size(); i++)
	{
		std::string name = param[i]["name"].asString();
		if (name == "number_repeats") m_repetitions = detail::json2number<std::size_t>(param[i]["value"]);
		else if (name == "number_folds") m_folds = detail::json2number<std::size_t>(param[i]["value"]);
	}

	s = input[3]["evaluation_measures"]["evaluation_measure"].asString();
	if (s == "") m_evaluationMeasure = UnspecifiedMeasure;
	else if (s == "predictive_accuracy") m_evaluationMeasure = PredictiveAccuracy;
	else throw SHARKEXCEPTION("unknown evaluation measure: " + s);

	// expected output
	m_outputFormat = output["predictions"]["format"].asString();
	detail::Json features = output["predictions"]["feature"];
	for (std::size_t i=0; i<features.size(); i++)
	{
		detail::Json feature = features[i];
		FeatureDescription fd;
		std::string type = feature["type"].asString();
		detail::ASCIItoLowerCase(type);
		if (type == "binary") fd.type = BINARY;
		else if (type == "integer") fd.type = INTEGER;
		else if (type == "numeric") fd.type = NUMERIC;
		else if (type == "nominal") fd.type = NOMINAL;
		else if (type == "string") fd.type = STRING;
		else if (type == "date") fd.type = DATE;
		else throw SHARKEXCEPTION("unknown feature type in task definition");
		fd.name = feature["name"].asString();
		fd.target = false;
		fd.ignore = false;
		fd.rowIdentifier = false;
		m_outputFeature.push_back(fd);
	}

	if (desc.has("tag")) setTags(desc["tag"]);

	if (downloadSplits) m_file.download();
}

void Task::load()
{
	if (! m_split.empty()) return;

	m_file.download();

	std::ifstream stream(m_file.filename().string().c_str());
	std::string line;

	// read the split file header
	while (std::getline(stream, line))
	{
		if (line.empty()) continue;
		if (line[line.size()-1] == '\r') line.erase(line.size() - 1);
		if (line == "@data") break;
	}

	// read the split file data section
	std::vector<size_t> ix;
	std::vector<size_t> fd;
	std::vector<size_t> rp;
	std::size_t n = 0;
	while (std::getline(stream, line))
	{
		if (line.empty()) continue;
		if (line[line.size()-1] == '\r') line.erase(line.size() - 1);
		if (line.empty()) continue;
		if (line[0] == '%') continue;
		std::size_t c1 = line.find(',');
		std::size_t c2 = line.find(',', c1+1);
		std::size_t c3 = line.find(',', c2+1);
		std::string traintest  = line.substr(0, c1);
		if (traintest == "TEST")
		{
			std::size_t row        = boost::lexical_cast<std::size_t>(line.substr(c1+1, c2-c1-1));
			std::size_t repetition = boost::lexical_cast<std::size_t>(line.substr(c2+1, c3-c2-1));
			std::size_t fold       = boost::lexical_cast<std::size_t>(line.substr(c3+1));
			if (repetition >= m_repetitions) throw SHARKEXCEPTION("invalid repetition index in split file");
			if (fold >= m_folds) throw SHARKEXCEPTION("invalid fold index in split file");
			ix.push_back(row);
			fd.push_back(fold);
			rp.push_back(repetition);
			if (row >= n) n = row + 1;
		}
	}

	// store test fold indices
	m_split.resize(m_repetitions, std::vector<std::size_t>(n));
	for (std::size_t i=0; i<ix.size(); i++) m_split[rp[i]][ix[i]] = fd[i];
}

void Task::tag(std::string const& tagname)
{
	Connection::ParamType param;
	param.push_back(std::make_pair("task_id", boost::lexical_cast<std::string>(id())));
	param.push_back(std::make_pair("tag", tagname));
	detail::Json result = connection.post("/task/tag", param);
	Entity::tag(tagname);
}

void Task::untag(std::string const& tagname)
{
	Connection::ParamType param;
	param.push_back(std::make_pair("task_id", boost::lexical_cast<std::string>(id())));
	param.push_back(std::make_pair("tag", tagname));
	detail::Json result = connection.post("/task/untag");
	Entity::untag(tagname);
}

void Task::print(std::ostream& os) const
{
	os << "Task:" << std::endl;
	Entity::print(os);
	os << " type: " << taskTypeName[m_tasktype] << std::endl;
	os << " data set: " << m_dataset->name() << " [" << m_dataset->id() << "]" << std::endl;
	os << " target feature: " << m_targetFeature << std::endl;
	os << " estimation procedure: " << estimationProcedureName[m_estimationProcedure] << std::endl;
	os << " splits url: " << m_file.url() << std::endl;
	os << " number of repetitions: " << m_repetitions << std::endl;
	os << " number of folds: " << m_folds << std::endl;
	os << " number of data splits: " << m_repetitions * m_folds << std::endl;
	os << " evaluation measure: " << evaluationMeasureName[m_evaluationMeasure] << std::endl;
	os << " output format: " << m_outputFormat << std::endl;
	os << " file status: ";
	if (m_file.downloaded()) os << "in cache at " << m_file.filename().string();
	else os << "not in cache";
	os << std::endl;
	for (std::size_t i=0; i<m_outputFeature.size(); i++)
	{
		FeatureDescription const& fd = m_outputFeature[i];
		os << "  feature " << i << ": " << fd.name << " (" << featureTypeName[(unsigned int)fd.type] << ")";
		if (fd.target) os << " [target]";
		if (fd.ignore) os << " [ignore]";
		if (fd.rowIdentifier) os << " [row-identifier]";
		os << std::endl;
	}
}


};  // namespace openML
};  // namespace shark
