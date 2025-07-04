## Voice Search Server

An experiment of a websockets server that provides transcription of short spoken webm audio
via the [Whisper model](https://openai.com/index/whisper/).

The use case is for a user to speak a search query into their browser
and quickly get back an accurate text transcription of their query in the search box.
We want to keep these spoken queries private, rather than sending them to a third
party, to protect our patrons' privacy.

The code is largely based on [the whisper candle example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper)

### Setup

1. Install rust
1. `brew install cmake pkgconf opus`
1. `cargo run` to start locally

### Tests

The inference tests are quite greedy with
system resources.  You may wish to limit the number of
parallel threads to avoid freezing the rest of your computer
while the test suite is running:

```
RUST_TEST_THREADS=3 cargo test
```

### Basic client for testing

1. Run: `ruby -run -e httpd . -p 7020`
1. In your browser, go to http://localhost:7020/test_client.html

### Creating your own recording

1. Run: `ruby -run -e httpd . -p 7020`
1. In Chrome, go to http://localhost:7020/quick_record.html
1. Start microphone
1. Start recording.  When prompted, put your recording in the test_data directory
1. Stop recording
1. Stop microphone

### Creating a webm recording from an existing recording (e.g. a librivox)

1. Open the recording in audacity
1. Clip/modify the recording as needed
1. File -> Export Audio
1. Format: Custom FFmpeg Export
1. Open custom FFmpeg format options
1. Format: webm
1. Codec: libopus

### Todo
* MPSC channel should close when done
* MPSC channel should send on each chunk, not at the end
* The in-browser tester does not work on firefox?
* Really refactor the transcription
* finish writing transcription tests (at least one mono and one stereo per language).
* extract more config options into config.rs

### Mermaid

```mermaid
sequenceDiagram
  actor Client
  Client->>Server: Websockets handshake
  Client->>Server: Send binary websockets message with ogg-encoded WebM audio
  Server->>Audio module: Send audio for processing
  Audio module->>Audio module: Convert audio to PCM samples
  Audio module->>Audio module: Convert PCM samples to MEL features
  Audio module->>Whisper: Send MEL features
  Whisper->>Server: Send full or partial transcription
  Server->>Client: Send plaintext websocket message with transcription
```
