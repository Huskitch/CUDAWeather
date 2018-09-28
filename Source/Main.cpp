#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <CL/cl.hpp>
#include <iomanip>
#include <algorithm>

#include "Utils.h"

void PrintHelp() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

// Record structure to hold the data for each weather station entry, containing year, month, day, time and temp.
// As none of the values in record except for the temperature, they can be stored as unsigned short ints to save memory.

struct Record
{
	unsigned short int year, month, day, time;
	float temp;
};

// Load the temperature data from the text file and map each
// data point to a record entry.

std::unordered_map<string, vector<Record>> LoadTemperatureData()
{
	// An unordered map only stores a pointer to the next element rather than a regular map which stores the previous element, halving the data structure size and saving memory.
	std::unordered_map<string, vector<Record>> entries;

	// Strings only need to be stored once as chars are a large data type.
	std::vector<string> stations = { "BARKSTON_HEATH", "SCAMPTON", "WADDINGTON", "CRANWELL", "CONINGSBY" };

	// Read the temperature file in sequentially line by line to store the data.
	std::ifstream file("temp_lincolnshire_short.txt");
	std::string str;
	while (std::getline(file, str))
	{
		// Create a new data record entry for each line in the input file.
		Record rec = Record();

		string buf;
		stringstream ss(str);
		vector<string> tokens;

		// Split the current line by whitespaces and add each one to a list of tokens.
		while (ss >> buf)
		{
			tokens.push_back(buf);
		}

		// Convert each data point to their respective data structures
		rec.year = (unsigned short int)std::stoi(tokens[1]);
		rec.month = (unsigned short int)std::stoi(tokens[2]);
		rec.day = (unsigned short int)std::stoi(tokens[3]);
		rec.time = (unsigned short int)std::stoi(tokens[4]);
		rec.temp = std::stof(tokens[5]);

		// Loop through the stations array (5)
		for (int i = 0; i < stations.size(); i++)
		{
			// Compare the name of the station in the file with the string array
			if (std::strcmp(tokens[0].c_str(), stations[i].c_str()) == 0)
			{
				// If it matches, add a new record to the map.
				entries[stations[i]].push_back(rec);
			}
		}
	}
	// After each line has been looped from the input file, return the map.
	return entries;
}


// By polymorphising the kernel execution code, it can be made easy to change which program is running.

vector<float> RunKernel(cl::Context context, cl::CommandQueue queue, cl::Program program, const char* name, vector<float> A)
{
	try
	{
		size_t local_size = 10;
		size_t padding_size = A.size() % local_size;
		int originalSize = A.size();

		if (padding_size) {
			std::vector<int> A_ext(local_size - padding_size, 0);
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();
		size_t input_size = A.size() * sizeof(float);
		size_t nr_groups = input_elements / local_size;

		std::vector<float> B(input_elements);
		size_t output_size = B.size() * sizeof(float);

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

		cl::Kernel kernel_1 = cl::Kernel(program, name);
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		return B;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;

		int x;
		cin >> x;
	}
}

// This struct holds the data for individual years for each station.
struct YearData {
	unsigned short int dataPoints;
    unsigned short int year;
	float minTemp, maxTemp, avgTemp;
};

int main(int argc, char **argv) {
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { PrintHelp(); }
	}

	cl::Context context = GetContext(platform_id, device_id);
	std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

	cl::CommandQueue queue(context);

	cl::Program::Sources sources;
	AddSources(sources, "kernels.cl");
	cl::Program program(context, sources);

	try {
		program.build();
	}
	catch (const cl::Error& err)
	{
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		throw err;
	}

	// Sequentially load the data from file into the unordered map structure.
	std::unordered_map<string, vector<Record>> tempData = LoadTemperatureData();

	// Create a vector of floats to store the input array for the kernel
	std::vector<float> A;

	// Loop through each record in the temperature data map
	for (std::pair<string, vector<Record>> record : tempData)
	{
		A = std::vector<float>();
		std::list<short int> years;

		// Output the current station that will be processed by the kernel.
		std::cout << "________________________________________________________________________" << std::endl;
		std::cout << std::endl << "Station: " << record.first << std::endl;
		std::cout << "________________________________________________________________________" << std::endl << std::endl;

		// Create a vector for the first year in the station
		std::vector<YearData> yearData;

		// Get the number of unqiue years in the list for this station
		for (int i = 0; i < record.second.size(); i++)
		{
			years.push_back(record.second[i].year);
		}
		years.unique();

		// Loop through each year in the station
		for (short int year : years)
		{
			int currentYear = year;
			A = std::vector<float>();

			// Add each temperature record to the kernel input array
			for (int i = 0; i < record.second.size(); i++)
			{
				if (record.second[i].year == year)
				{
					A.push_back(record.second[i].temp);
				}

				if (!A.empty() && record.second[i].year != year)
				{
					break;
				}
			}

			// Run the min/max bitonic sort kernel and store it in the B output float array
			std::vector<float> B;
			B = RunKernel(context, queue, program, "MinMaxSort", A);

			// Store the output from the kernel into the YearData structure
			YearData data = YearData();
			data.year = year;
			// Run the reduction kernel and divide by the 0th element by the total number of elements to get the average
			data.avgTemp = RunKernel(context, queue, program, "ReduceFloatArray", A)[0] / A.size();
			data.minTemp = B[0];
			data.maxTemp = B.back();
			data.dataPoints = A.size();
			yearData.push_back(data);
		}

		// Output the data for each year in the station
		for (YearData data : yearData)
		{
			std::cout << data.year << " | " << "Data points: " << data.dataPoints << " Avg: " << data.avgTemp;
			std::cout << " Min: " << data.minTemp;
			std::cout << " Max: " << data.maxTemp << std::endl;
		}

		std::cout << std::endl;

	}



	int x;
	cin >> x;

	return 0;
}