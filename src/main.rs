use actix_web::{App, Error, HttpRequest, HttpResponse, HttpServer, middleware::Logger, rt, web};
use actix_ws::AggregatedMessage;
use env_logger::Env;
use futures_util::StreamExt as _;
use std::io::Cursor;
mod audio;
mod feature_extraction;
mod transcription;
mod whisper;

async fn websocket_server(req: HttpRequest, stream: web::Payload) -> Result<HttpResponse, Error> {
    let (res, mut session, stream) = actix_ws::handle(&req, stream)?;

    let mut stream = stream
        .max_frame_size(1024 * 1024)
        .aggregate_continuations()
        .max_continuation_size(2_usize.pow(20));

    rt::spawn(async move {
        while let Some(msg) = stream.next().await {
            match msg {
                Ok(AggregatedMessage::Binary(bin)) => {
                    log::info!("Received binary websocket message");
                    let (samples, rate) = audio::pcm_decode(Cursor::new(bin)).unwrap();
                    let features = feature_extraction::extract_features(samples).unwrap();
                    let files = whisper::download().unwrap();
                    let transcription = transcription::transcribe(features, files).unwrap();
                    log::info!("HI EVERYONE I TRANSCRIBED!!! {}", transcription);
                    session.text(transcription).await.unwrap();
                }
                Err(err) => {
                    log::error!("Received a websocket message that caused error: {:?}", err)
                }
                _ => {}
            }
        }
    });
    Ok(res)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(websocket_server))
            .wrap(Logger::default())
    })
    .bind(("127.0.0.1", 7025))?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use actix_http::ws;
    use actix_test;
    use actix_web::{App, web::Bytes};
    use futures_util::SinkExt as _;

    #[actix_web::test]
    async fn test_websocket_transcribes_binary_message() {
        let mut server =
            actix_test::start(|| App::new().route("/", web::get().to(websocket_server)));
        let mut socket = server.ws().await.unwrap();
        socket
            .send(ws::Message::Binary(
                fs::read("./test_data/english/complete_book_of_cheese_mono.webm")
                    .unwrap()
                    .into(),
            ))
            .await
            .unwrap();
        let item = socket.next().await.unwrap().unwrap();
        assert_eq!(
            item,
            ws::Frame::Text(Bytes::from_static(
                b" The Complete Book of Cheese by Robert Carlton Brown"
            ))
        );
    }
}
