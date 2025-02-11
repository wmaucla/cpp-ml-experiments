use actix_web::{get, web, App, HttpServer, Responder};

#[get("/{name}")]
async fn hello(name: web::Path<String>) -> impl Responder {
    format!("Hello {}!", &name)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(hello))
        .workers(1)
        .bind(("0.0.0.0", 8080))?
        .run()
        .await
}