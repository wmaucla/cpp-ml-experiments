# Stage 1: Base Rust image with cargo-chef
FROM rust:latest as chef
WORKDIR /app
RUN cargo install cargo-chef

# Stage 2: Dependency planning
FROM chef as planner
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo chef prepare --recipe-path recipe.json

# Stage 3: Build dependencies and application
FROM chef as builder
# Install required system dependencies
RUN apt update && apt install -y clang cmake gcc musl-tools 

# Ensure musl for Alpine compatibility
RUN rustup target add x86_64-unknown-linux-musl

COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Copy source and build
COPY . .
RUN ln -s /bin/g++ /bin/musl-g++
RUN cargo build --release --target=x86_64-unknown-linux-musl

# Stage 4: Minimal production image
FROM alpine:latest as prod
RUN apk add --no-cache ca-certificates

# Copy statically built binary
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/hello_world /usr/local/bin/hello_world

ENTRYPOINT ["/usr/local/bin/hello_world"]
