//===========================================================================
/*!
 * 
 *
 * \brief       Implementation of an OpenML Run.
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
#include <shark/OpenML/detail/Json.h>
#include <shark/Data/Arff.h>


namespace shark {
namespace openML {


Run::Run(IDType id)
: Entity(id)
, m_file("run_" + boost::lexical_cast<std::string>(id) + ".arff")
{
	detail::Json result = connection.get("/run/" + boost::lexical_cast<std::string>(id));

	// TODO!
}

Run::Run(std::shared_ptr<Task> task, std::shared_ptr<Flow> flow)
: Entity()
, m_task(task)
, m_flow(flow)
, m_hyperparameterValue(m_flow->numberOfHyperparameters())
, m_predictions(m_task->repetitions(), std::vector< std::vector<double> >(m_task->folds()))
{ }


void Run::tag(std::string const& tagname)
{
	Connection::ParamType param;
	param.push_back(std::make_pair("run_id", boost::lexical_cast<std::string>(id())));
	param.push_back(std::make_pair("tag", tagname));
	detail::Json result = connection.post("/run/tag", param);
	Entity::tag(tagname);
}

void Run::untag(std::string const& tagname)
{
	Connection::ParamType param;
	param.push_back(std::make_pair("run_id", boost::lexical_cast<std::string>(id())));
	param.push_back(std::make_pair("tag", tagname));
	detail::Json result = connection.post("/run/untag", param);
	Entity::untag(tagname);
}

void Run::print(std::ostream& os) const
{
	os << "Run:" << std::endl;
	Entity::print(os);
	os << " task: " << taskTypeName[m_task->tasktype()] << " on " << m_task->dataset()->name() << " [" << m_task->id() << "]" << std::endl;
	os << " flow: " << m_flow->name() << " [" << m_flow->id() << "]" << std::endl;
	for (std::size_t i=0; i<m_hyperparameterValue.size(); i++)
	{
		os << "  hyperparameter " << m_flow->hyperparameter(i).name << " = " << m_hyperparameterValue[i] << std::endl;
	}
}

void Run::commit()
{
	if (id() != invalidID) throw SHARKEXCEPTION("Cannot commit an already existing run.");

	std::shared_ptr<Dataset> dataset = m_task->dataset();

	// If the label is a nominal attribute then we have to obtain the
	// possible values from the ARFF file. This is working around a
	// weakness in the OpenML API.
	std::vector<std::string> nominalValue;
	std::string labelType = "NUMERIC";
	bool targetNominal = false;
	std::string const& targetName = m_task->targetFeature();
	for (std::size_t i=0; i<dataset->numberOfFeatures(); i++)
	{
		FeatureDescription const& fd = dataset->feature(i);
		if (fd.name == targetName)
		{
			if (fd.type == NOMINAL)
			{
				targetNominal = true;
				CachedFile const& f = dataset->datafile();
				f.download();
				std::ifstream stream(f.filename().string().c_str());
				std::string line;
				while (std::getline(stream, line))
				{
					if (line.empty()) continue;
					if (line[line.size()-1] == '\r') line.erase(line.size() - 1);
					if (line.empty()) continue;
					if (line[0] == '%') continue;
					if (line[0] != '@') throw SHARKEXCEPTION("invalid line in ARFF data set file");
					std::size_t pos = line.find(' ');
					if (pos == std::string::npos) pos = line.size();
					std::string type = line.substr(0, pos);
					detail::ASCIItoLowerCase(type);
					if (type == "@attribute")
					{
						std::string name;
						pos = arff::detail::parseString(line, pos, name, " ");
						if (name == targetName)
						{
							if (line[pos] != '{') throw SHARKEXCEPTION("failed to determine nominal label values from ARFF data set file");
							pos++;
							while (pos < line.size())
							{
								std::string value;
								pos = arff::detail::parseString(line, pos, value, ",}");
								if (value.empty()) break;
								nominalValue.push_back(value);
								pos++;
							}
							break;
						}
					}
					if (type == "@data") throw SHARKEXCEPTION("target attribute " + targetName + " not found in ARFF data set file");
				}
				stream.close();

				if (nominalValue.empty()) throw SHARKEXCEPTION("failed to determine nominal label values from ARFF data set file");
				labelType = "{" + nominalValue[0];
				for (std::size_t i=1; i<nominalValue.size(); i++) labelType += "," + nominalValue[i];
				labelType += "}";
			}
		}
	}

	// compile the predictions into an ARFF file
	std::string predictions = "@relation openml_task_" + boost::lexical_cast<std::string>(m_task->id()) + "_predictions\n"
			"@ATTRIBUTE repeat INTEGER\n"
			"@ATTRIBUTE fold INTEGER\n"
			"@ATTRIBUTE rowid INTEGER\n"
			"@ATTRIBUTE prediction " + labelType + "\n";
	if (targetNominal)
	{
		for (std::size_t i=0; i<nominalValue.size(); i++)
		{
			predictions += "@ATTRIBUTE confidence." + nominalValue[i] + " NUMERIC\n";
		}
	}
	predictions += "@data\n";
	for (std::size_t r=0; r<m_task->repetitions(); r++)
	{
		std::vector<std::size_t> const& foldIndices = m_task->splitIndices(r);

		for (std::size_t f=0; f<m_task->folds(); f++)
		{
			std::vector<double> const& p = m_predictions[r][f];
			if (p.empty()) throw SHARKEXCEPTION("predictions for repetition " + boost::lexical_cast<std::string>(r) + " and fold " + boost::lexical_cast<std::string>(f) + " are missing");
			std::size_t row = 0;
			for (std::size_t i=0; i<p.size(); i++)
			{
				while (foldIndices[row] != f) row++;    // find the "row_index" corresponding to the test input
				predictions += boost::lexical_cast<std::string>(r);
				predictions += ",";
				predictions += boost::lexical_cast<std::string>(f);
				predictions += ",";
				predictions += boost::lexical_cast<std::string>(row);
				predictions += ",";
				if (targetNominal)
				{
					unsigned int cls = static_cast<unsigned int>(p[i]);
					if (cls != p[i]) throw SHARKEXCEPTION("prediction of nominal label must be integer valued");
					predictions += nominalValue[cls];
					for (std::size_t j=0; j<nominalValue.size(); j++)
					{
						predictions += (cls == j) ? ",1" : ",0";
					}
				}
				else
				{
					predictions += boost::lexical_cast<std::string>(p[i]);
				}
				predictions += "\n";
				row++;
			}
		}
	}

	// create an XML description for the run
	std::string description = "<oml:run xmlns:oml=\"http://openml.org/openml\">"
				"<oml:task_id>" + boost::lexical_cast<std::string>(m_task->id()) + "</oml:task_id>"
				"<oml:flow_id>" + boost::lexical_cast<std::string>(m_flow->id()) + "</oml:flow_id>";
	for (std::size_t i=0; i<m_flow->numberOfHyperparameters(); i++)
	{
		if (m_hyperparameterValue[i].empty()) throw SHARKEXCEPTION("hyperparameter " + boost::lexical_cast<std::string>(i) + " (" + m_flow->hyperparameter(i).name + ") is not defined");
		Hyperparameter const& p = m_flow->hyperparameter(i);
		description += "<oml:parameter_setting>"
				"<oml:name>" + detail::xmlencode(p.name) + "</oml:name>"
				"<oml:value>" + detail::xmlencode(m_hyperparameterValue[i]) + "</oml:value>"
				"</oml:parameter_setting>";
	}
	description += "</oml:run>";

	Connection::ParamType param;
	param.push_back(std::make_pair("description|application/xml", description));
	param.push_back(std::make_pair("predictions|application/octet-stream", predictions));
//	param.push_back(std::make_pair("model_readable|text/plain", ""));
//	param.push_back(std::make_pair("model_serialized|application/octet-stream", ""));
	detail::Json result = connection.post("/run", param);

	IDType id = detail::json2number<IDType>(result["upload_run"]["run_id"]);
	setID(id);

	// cache the results locally
	m_file.setFilename("run_" + boost::lexical_cast<std::string>(id) + ".arff");
	std::ofstream ofs(m_file.filename().string().c_str());
	ofs << predictions;
}


};  // namespace openML
};  // namespace shark
