#!/bin/bash

# install TabPFN
git clone https://github.com/PriorLabs/TabPFN.git
pip install -e "TabPFN[dev]"

# install AutoTabPFN
git clone https://github.com/priorlabs/tabpfn-extensions.git
pip install -e tabpfn-extensions
pip install hyperopt

exit 0