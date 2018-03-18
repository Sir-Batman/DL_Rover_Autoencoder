/* The C++ 11 half of the json test, using nlohman/json as the C++ json library */
#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
using json= nlohmann::json;

void send(const std::string & message, const std::string & filename)
{
	std::ofstream topy(filename);
	topy << message;
	topy.close();
}

std::string recieve(const std::string &filename)
{
	std::string message;
	std::cout << "Opening file... " << std::endl;
	std::ifstream tocpp(filename);
	std::cout << "Opened!" << std::endl;
	tocpp >> message;
	std::cout << "Finished writing" << std::endl;
	tocpp.close();
	std::cout << " and closed" << std::endl;
	return message;
}

