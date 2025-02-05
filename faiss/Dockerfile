FROM continuumio/miniconda3

# Set up environment variables
ARG ENV_NAME=faiss-cpp-env
ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/$ENV_NAME/lib:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    make \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a Conda environment with only C++ tools
RUN conda create -n $ENV_NAME -y \
    cmake \
    make \
    libgcc-ng \
    libstdcxx-ng \
    gxx_linux-64 \
    libgcc \
    && conda clean --all -y

# Install FAISS from Conda Forge
RUN conda install -n $ENV_NAME pytorch::faiss-cpu

# Set the working directory
WORKDIR /faiss

# Copy the sample C++ file
COPY sample_faiss.cpp /faiss/sample_faiss.cpp

# Compile the C++ program using FAISS
RUN conda run -n $ENV_NAME g++ -o /faiss/sample_faiss /faiss/sample_faiss.cpp \
    -I/opt/conda/envs/$ENV_NAME/include \
    -L/opt/conda/envs/$ENV_NAME/lib -lfaiss

# Run the compiled program
CMD ["/faiss/sample_faiss"]
