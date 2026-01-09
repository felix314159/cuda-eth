# Overview

* `vanity-address-miner.cu`: Mines customizable EOA vanity addresses
* `solidity_function_selector_miner.cu`: Mines 4 byte (all zeroes) function selector for saving gas on frequently called function (see snappy compression)
* `solidity_create2_miner.cu`: Mines salt for deploying a given contract to a vanity address, aims for up to 10 leading zeroes in default config.

## This is for educational purposes only
