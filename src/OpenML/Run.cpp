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


#include <shark/OpenML/OpenML.h>
#include <shark/OpenML/detail/Tools.h>
#include <shark/OpenML/detail/Json.h>
#include <shark/Data/Arff.h>


namespace shark {
namespace openML {


Run::Run(IDType id, bool downloadPredictions)
: Entity(id)
, m_file("run_" + boost::lexical_cast<std::string>(id) + ".arff")
{
	detail::Json result = connection.get("/run/" + boost::lexical_cast<std::string>(id));
	detail::Json desc = result["run"];

	IDType taskID = detail::json2number<IDType>(desc["task_id"]);
	IDType flowID = detail::json2number<IDType>(desc["flow_id"]);
	m_task = Task::get(taskID);
	m_flow = Flow::get(flowID);
	m_hyperparameterValue.resize(m_flow->numberOfHyperparameters());
	if (desc.has("tag")) setTags(desc["tag"]);
	if (desc.has("parameter_setting"))
	{
		detail::Json p = desc["parameter_setting"];
		for (std::size_t i=0; i<p.size(); i++)
		{
			std::string name = p[i]["name"].asString();
			std::size_t index = m_flow->hyperparameterIndex(name);
			m_hyperparameterValue[index] = detail::json2string(p[i]["value"]);
		}
	}
	m_file.setUrl(desc["output_data"]["file"][1]["url"].asString());

	m_predictions.resize(m_task->repetitions());
	for (std::size_t i=0; i<m_predictions.size(); i++) m_predictions[i].resize(m_task->splitIndices(i).size(), invalidValue);

	if (downloadPredictions) m_file.download();
}

Run::Run(std::shared_ptr<Task> task, std::shared_ptr<Flow> flow)
: Entity()
, m_task(task)
, m_flow(flow)
, m_hyperparameterValue(m_flow->numberOfHyperparameters())
{
	m_predictions.resize(m_task->repetitions());
	for (std::size_t i=0; i<m_predictions.size(); i++) m_predictions[i].resize(m_task->splitIndices(i).size(), invalidValue);
}


void Run::load() const
{
	if (m_predictions[0][0] != invalidValue) return;

	m_file.download();

	std::ifstream stream(m_file.filename().string().c_str());
	std::string line;

	// for the load() function we have conceptual (not actual) constness
	std::vector< std::vector< double > >& pred = const_cast< std::vector< std::vector< double > >& >(m_predictions);

	// read the predictions file header and find the positions of the informative columns
	int dim = 0;
	int repeatIndex = -1;
	int foldIndex = -1;
	int rowIndex = -1;
	int predictionIndex = -1;
	std::map<std::string, std::size_t> nominalValue;
	bool nominal = false;
	while (std::getline(stream, line))
	{
		if (line.empty()) continue;
		if (line[line.size()-1] == '\r') line.erase(line.size() - 1);
		if (line[0] == '@')
		{
			std::size_t pos = line.find(' ');
			if (pos == std::string::npos) pos = line.size();
			std::string type = line.substr(0, pos);
			detail::ASCIItoLowerCase(type);
			if (type == "@data") break;
			else if (type == "@relation") continue;   // ignore
			else if (type == "@attribute")
			{
				std::string s;
				pos = arff::detail::parseString(line, pos, s, " ");
				if (s == "repeat") repeatIndex = dim;
				else if (s == "fold") foldIndex = dim;
				else if (s == "rowid") rowIndex = dim;
				else if (s == "prediction")
				{
					predictionIndex = dim;
					while (line[pos] == ' ') pos++;
					if (line[pos] == '{')
					{
						pos++;
						for (std::size_t i=0; ; i++)
						{
							pos = arff::detail::parseString(line, pos, s, ",}");
							nominalValue[s] = i;
							if (line[pos] != ',') break;
							pos++;
						}
						nominal = true;
					}
					else nominal = false;
				}
				dim++;
			}
			else throw SHARKEXCEPTION("[importARFF] unsupported header field: " + type);
		}
		else throw SHARKEXCEPTION("[importARFF] invalid line in ARFF header");
	}
	if (repeatIndex < 0 || foldIndex < 0 || rowIndex < 0 || predictionIndex < 0) throw SHARKEXCEPTION("[Run::load] invalid predictions file");

	// read the predictions file data section
	while (std::getline(stream, line))
	{
		if (line.empty()) continue;
		if (line[line.size()-1] == '\r') line.erase(line.size() - 1);
		if (line.empty()) continue;
		if (line[0] == '%') continue;

		int repetition = -1;
		int fold = -1;
		int row = -1;
		double value = invalidValue;
		std::size_t pos = 0;
		for (std::size_t i=0; i<dim; i++)
		{
			std::string s;
			pos = arff::detail::parseString(line, pos, s, ",");
			if (i == repeatIndex) repetition = boost::lexical_cast<int>(s);
			else if (i == foldIndex) fold = boost::lexical_cast<int>(s);
			else if (i == rowIndex) row = boost::lexical_cast<int>(s);
			else if (i == predictionIndex)
			{
				if (nominal)
				{
					std::map<std::string, std::size_t>::const_iterator it = nominalValue.find(s);
					if (it == nominalValue.end()) throw SHARKEXCEPTION("[Run::load] invalid value in nominal prediction");
					value = it->second;
				}
				else
				{
					value = boost::lexical_cast<double>(s);
				}
			}
			if (line[pos] == ',') pos++;
		}

		if (repetition == -1 || fold == -1 || row == -1 || value == invalidValue) throw SHARKEXCEPTION("[Run::load] error reading ARFF predictions file");
		if (m_task->splitIndices(repetition)[row] != fold) throw SHARKEXCEPTION("[Run::load] predictions are not consistent with the data splits defined in the task");
		pred[repetition][row] = value;
	}
}

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
	os << " task: " << m_task->type() << " on " << m_task->dataset()->name() << " [" << m_task->id() << "]" << std::endl;
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
	// https://github.com/openml/OpenML/issues/261
	std::vector<std::string> nominalValue;
	std::string labelType = "NUMERIC";
	AttributeDescription const& fd = dataset->attribute(dataset->attributeIndex(m_task->targetAttribute()));
	if (fd.type == NOMINAL)
	{
		CachedFile const& f = dataset->datafile();
		f.download();
		nominalValue = importARFFnominalLabel(f.filename().string(), m_task->targetAttribute());
		if (nominalValue.empty()) throw SHARKEXCEPTION("failed to determine nominal label values from ARFF data set file");
		labelType = "{" + nominalValue[0];
		for (std::size_t i=1; i<nominalValue.size(); i++) labelType += "," + nominalValue[i];
		labelType += "}";
	}

	// compile the predictions into an ARFF file
	std::string predictions = "@relation openml_task_" + boost::lexical_cast<std::string>(m_task->id()) + "_predictions\n"
			"@ATTRIBUTE repeat INTEGER\n"
			"@ATTRIBUTE fold INTEGER\n"
			"@ATTRIBUTE rowid INTEGER\n"
			"@ATTRIBUTE prediction " + labelType + "\n";
	if (fd.type == NOMINAL)
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
		std::vector<double> const& p = m_predictions[r];
		SHARK_ASSERT(p.size() == foldIndices.size());
		for (std::size_t i=0; i<p.size(); i++)
		{
			std::size_t f = foldIndices[i];   // fold
			double value = p[i];              // value
			if (value == invalidValue) throw SHARKEXCEPTION("predictions for repetition " + boost::lexical_cast<std::string>(r) + " and fold " + boost::lexical_cast<std::string>(f) + " are missing");
			predictions += boost::lexical_cast<std::string>(r);
			predictions += ",";
			predictions += boost::lexical_cast<std::string>(f);
			predictions += ",";
			predictions += boost::lexical_cast<std::string>(i);
			predictions += ",";
			if (fd.type == NOMINAL)
			{
				unsigned int cls = static_cast<unsigned int>(value);
				if (cls != value) throw SHARKEXCEPTION("prediction of nominal label must be integer valued");
				predictions += nominalValue[cls];
				for (std::size_t j=0; j<nominalValue.size(); j++)
				{
					predictions += (cls == j) ? ",1" : ",0";
				}
			}
			else
			{
				predictions += boost::lexical_cast<std::string>(value);
			}
			predictions += "\n";
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
	param.push_back(std::make_pair("description|application/xml|description.xml", description));
	param.push_back(std::make_pair("predictions|application/octet-stream|predictions.arff", predictions));
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
