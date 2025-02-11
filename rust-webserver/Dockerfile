FROM rust:1.82.0-bookworm

WORKDIR /app

COPY . .

RUN cargo build --release

ENTRYPOINT [ "/app/target/release/rust-webserver" ]