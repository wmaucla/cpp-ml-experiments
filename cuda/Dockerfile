# Use an official CUDA base image
FROM nvidia/cuda:11.2.2-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    cuda-toolkit-11-2 \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy the C++ source code into the container
COPY vector_add.cpp /app/vector_add.cpp

# Compile the C++ CUDA code using the nvcc compiler
RUN nvcc -o vector_add vector_add.cpp

# Set the entrypoint for the container to run the compiled program
CMD ["./vector_add"]
