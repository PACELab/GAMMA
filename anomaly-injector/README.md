# Performance Anomaly Injector

Performance anomaly injector consists of a set of contentious microbenchmark targeting at different types of shared resources. It can be used to inject performance anomalies caused by shared resource interference and the interference timing, intensity or/and core location are configurable. Types of shared resources include CPU, memory bandwidth, LLC bandwidth, network bandwidth and I/O bandwidth.

## Build

Tested on Ubuntu 18.04.

```
make
cd pmbw-0.6.2
./configure && make
cd ..
sudo apt -y install make automake libtool pkg-config libaio-dev libmysqlclient-dev libssl-dev
# optional
sudo apt -y install libpq-dev
cd sysbench
./autogen.sh
./configure
make -j
sudo make install
cd ..
sudo apt install iproute2
```
