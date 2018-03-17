DL Autoencoder for Rover Domain
===============================

Eigen Dependency
--------------------

Simulator uses `Eigen` as a dependency.

Download `Eigen` from the [Eigen Mainpage](http://eigen.tuxfamily.org/index.php). Unzip the folder and place the `Eigen` folder into `/usr/include/eigen3/` or a directory of your choice.

If putting `Eigen` in a different directory, update [CMakeLists.txt] in the home directory so the line:
`set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -I <path to eigen folder here>")` references the correct location.


Building the Simulator
-----------------------

To build the simulator code, run:
```
mkdir build
cd build
cmake ..
make
```

To run the simulator, update the parameters in [generateMultiRoverExperts.cpp], run `make` in the `build` folder and run `./generateMultiRoverExperts`

To generate laser data, update the parameters in [generateLaserData.cpp], run `make` in the `build` folder and run `./generateLaserData`


Running the AutoEncoder
-------------------------

The code to create and train the autoencoder is in [autoencoder/laser_cae.py]. 