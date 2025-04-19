use kalosm::sound::*;
use rodio::Decoder;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSource;
use symphonia::default::formats::MkvReader;
use std::fs::File;
use std::io::BufReader;

async fn transcribe(blob: impl MediaSource + 'static) -> Result<String, anyhow::Error> {
    // Create a new small whisper model
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::QuantizedLargeV3Turbo)
        .build()
        .await?;

    let audio = Decoder::new(blob).unwrap();
    let mut text = model.transcribe(audio);
    Ok(text.all_text().await.to_lowercase())
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn it_can_transcribe_spanish() {
        let file = File::open("./test_data/microphone-recording.webm").unwrap();
        let transcription = transcribe(file).await.unwrap();
        assert_eq!(transcription, ("very basic test"));
    }
}
