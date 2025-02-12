FROM rust:latest as chef

RUN apt update && apt install -y \
   musl-tools gcc clang cmake libssl-dev libclang-dev make \
   libxgboost-dev

WORKDIR /app
COPY . .

RUN cargo build --release

CMD ["./target/release/hello_world"]
