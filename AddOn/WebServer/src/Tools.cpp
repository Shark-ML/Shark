
#include <shark/Network/Tools.h>
#include <iostream>
#include <fstream>


namespace shark {
namespace http {


std::string readFile(std::string filename)
{
	std::ifstream ifs(filename.c_str(), std::ios_base::binary);
	std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	return content;
}

std::string cgiEncode(std::string s)
{
	std::string enc;

	unsigned int i, len = s.size();
	for (i = 0; i < len; i++)
	{
		if (!((s[i] >= '0' && s[i] <= '9')
				|| (s[i] >= 'A' && s[i] <= 'Z')
				|| (s[i] >= 'a' && s[i] <= 'z')))
		{
			int l, h;
			char tmp[4];
			tmp[0] = '%';
			tmp[3] = 0;
			l = s[i] & 15;
			h = (s[i] >> 4) & 15;
			if (h >= 10) tmp[1] = 'A' + h - 10; else tmp[1] = '0' + h;
			if (l >= 10) tmp[2] = 'A' + l - 10; else tmp[2] = '0' + l;
			enc += tmp;
		}
		else enc += s[i];
	}

	return enc;
}

std::string cgiDecode(std::string s)
{
	std::string dec;
	std::size_t pos = 0;
	std::size_t perc;
	std::string sub;

	while (true)
	{
		perc = s.find("%", pos);
		if (perc == std::string::npos)
		{
			sub = s.substr(pos);
			for (std::size_t i=0; i<sub.size(); i++) if (sub[i] == '+') sub[i] = ' ';
			dec += sub;
			break;
		}
		else
		{
			if (perc > pos)
			{
				sub = s.substr(pos, perc - pos);
				for (std::size_t i=0; i<sub.size(); i++) if (sub[i] == '+') sub[i] = ' ';
				dec += sub;
			}

			if (s.size() < perc + 2) break;
			char c;
			if (s[perc+1] >= '0' && s[perc+1] <= '9') c = 16 * (s[perc+1] - '0');
			else if (s[perc+1] >= 'A' && s[perc+1] <= 'F') c = 16 * (s[perc+1] - 'A' + 10);
			else if (s[perc+1] >= 'a' && s[perc+1] <= 'f') c = 16 * (s[perc+1] - 'a' + 10);
			else break;
			if (s[perc+2] >= '0' && s[perc+2] <= '9') c += s[perc+2] - '0';
			else if (s[perc+2] >= 'A' && s[perc+2] <= 'F') c += s[perc+2] - 'A' + 10;
			else if (s[perc+2] >= 'a' && s[perc+2] <= 'f') c += s[perc+2] - 'a' + 10;
			else break;
			pos = perc + 3;

			dec += c;
		}
	}

	return dec;
}


}}
