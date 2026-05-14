#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sudo apt install -y libgtest-dev libsqlite3-dev
source $SCRIPT_DIR/install_or_tools.sh 24.04
