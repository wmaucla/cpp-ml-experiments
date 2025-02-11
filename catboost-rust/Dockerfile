ARG RUST_VERSION=1.82.0
FROM --platform=linux/amd64 rust:${RUST_VERSION}-slim-bookworm AS builder

#
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    lsb-release \
    software-properties-common \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && echo "deb http://apt.llvm.org/$(lsb_release -sc)/ llvm-toolchain-$(lsb_release -sc)-16 main" >> /etc/apt/sources.list.d/llvm.list

#
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    ninja-build \
    python3-pip \
    clang-16 \
    libc++-16-dev \
    libc++abi-16-dev \
    lld-16 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up clang toolchain
RUN ln -sf /usr/bin/clang-16 /usr/bin/clang && \
    ln -sf /usr/bin/clang++-16 /usr/bin/clang++ && \
    ln -sf /usr/bin/lld-16 /usr/bin/lld

# Configure build environment
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++
ENV LIBCLANG_PATH=/usr/lib/llvm-16/lib
ENV LLVM_CONFIG_PATH=/usr/bin/llvm-config-16

# Install specific version of Conan
RUN pip3 install --break-system-packages "conan==2.11.0"

# Verify installations
RUN cmake --version && \
    clang --version && \
    conan --version

# Install dependencies first
COPY ./Cargo.* ./
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    RUSTFLAGS="-C codegen-units=1" cargo build --release && \
    rm -rf src

# Build the application
COPY ./src src

ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

COPY model.cbm ./
RUN cargo build --release && \
    cp target/release/hello_world /usr/local/bin/hello_world
