#!/usr/bin/bash

docker build -t serena .

docker run -it --rm -v "$(pwd)":/workspace serena
