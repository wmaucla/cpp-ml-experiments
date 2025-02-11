FROM rust:latest as chef
WORKDIR /app
RUN cargo install cargo-chef

FROM chef as planner
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo chef prepare --recipe-path recipe.json

FROM chef as builder
RUN rustup target add x86_64-unknown-linux-musl
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --target=x86_64-unknown-linux-musl --recipe-path recipe.json

COPY . .
RUN cargo build --release --target=x86_64-unknown-linux-musl

FROM alpine:latest as prod
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/hello_world /usr/local/bin/hello_world
ENTRYPOINT ["/usr/local/bin/hello_world"]

