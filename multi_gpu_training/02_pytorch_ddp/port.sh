#!/bin/bash

# Define the range of ports to check
PORT_RANGE_START=10000
PORT_RANGE_END=20000

# Function to check if a port is free
is_port_free() {
    local port=$1
    if ss -lnt | grep -q ":$port "; then
        return 1
    else
        return 0
    fi
}

# Find a free port
for port in $(seq $PORT_RANGE_START $PORT_RANGE_END); do
    if is_port_free $port; then
        echo "Free port found: $port"
        break
    fi
done
