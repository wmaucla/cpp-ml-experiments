FROM continuumio/miniconda3

# Set up environment variables
ARG ENV_NAME=xgboost-cpp-env
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
    && conda clean --all -y

# Clone XGBoost with submodules
RUN git clone --recursive https://github.com/dmlc/xgboost /xgboost && \
    mkdir -p /xgboost/build

# Build and install XGBoost in the Conda environment
RUN /bin/bash -c "source activate $ENV_NAME && \
    cd /xgboost/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=\$CONDA_PREFIX && \
    make install"

# Set library path permanently
ENV LD_LIBRARY_PATH=/opt/conda/envs/$ENV_NAME/lib:$LD_LIBRARY_PATH

# Copy the sample C++ file
COPY xgb_rust_versus.cpp /xgboost/xgb_rust_versus.cpp

# Compile the C++ program using the Conda environment's compiler
RUN /bin/bash -c "source activate $ENV_NAME && \
    g++ -o /xgboost/xgb_rust_versus /xgboost/xgb_rust_versus.cpp \
    -I/xgboost/include -L/opt/conda/envs/$ENV_NAME/lib -lxgboost"

# Set working directory
WORKDIR /xgboost

# Run the compiled program
CMD ["/xgboost/xgb_rust_versus"]
