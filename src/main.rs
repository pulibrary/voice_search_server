use actix_web::{App, Error, HttpRequest, HttpResponse, HttpServer, rt, web};
use actix_ws::AggregatedMessage;
use futures_util::StreamExt as _;
mod audio;

async fn websocket_server(req: HttpRequest, stream: web::Payload) -> Result<HttpResponse, Error> {
    let (res, mut session, stream) = actix_ws::handle(&req, stream)?;

    let mut stream = stream
        .aggregate_continuations()
        .max_continuation_size(2_usize.pow(20));

    rt::spawn(async move {
        while let Some(msg) = stream.next().await {
            match msg {
                Ok(AggregatedMessage::Text(text)) => {
                    session.text(text).await.unwrap();
                }

                Ok(AggregatedMessage::Binary(bin)) => {
                    session.text("Heya!").await.unwrap();
                }

                _ => {}
            }
        }
    });
    Ok(res)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().route("/", web::get().to(websocket_server)))
        .bind(("127.0.0.1", 7025))?
        .run()
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_http::ws;
    use actix_test;
    use actix_web::{App, web::Bytes};
    use futures_util::SinkExt as _;

    #[actix_web::test]
    async fn test_websocket_can_send_plaintext_message() {
        let mut server =
            actix_test::start(|| App::new().route("/", web::get().to(websocket_server)));
        let mut socket = server.ws().await.unwrap();
        socket
            .send(ws::Message::Text("Hello".into()))
            .await
            .unwrap();
        let item = socket.next().await.unwrap().unwrap();
        assert_eq!(item, ws::Frame::Text(Bytes::from_static(b"Hello")));
    }
}
