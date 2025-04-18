use kalosm::sound::*;
use rodio::Decoder;
use std::fs::File;
use std::io::BufReader;

async fn transcribe(blob: BufReader<File>) -> Result<String, anyhow::Error> {

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
        let file = BufReader::new(File::open("./test_data/poem.wav").unwrap());
        let transcription = transcribe(file).await.unwrap();
        assert!(transcription.contains("poemas para ser leídos en el tranvía"));
    }
}

