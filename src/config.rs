// -----------------------
// Audio decoding settings
// -----------------------

// Sample rate must be 8, 12, 16, 24, or 48 kHz.  The libopus documentation recommends 48.
// The whisper paper mentioned that they re-sampled their audio to 16 MHz in training.
// Interestingly: when using the symphonia crate's matroska demuxer, a lower sample rate provided
// more accurate results.  With the matroska-demuxer crate, a higher sample rate seems to work better.
// In either case, if you get it too high, the transcriptions are just "... ... ..."
//
// Note that the sample rate of a file from the browser may not be one of the rates supported
// by libopus.  When using the browser's MediaRecorder API, you can pass in a custom sample
// rate, and the default rate "is adaptive, depending upon the sample rate and the number of channels."
// See:
//  * https://opus-codec.org/docs/opus_api-1.5.pdf
//  * https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/MediaRecorder#audiobitspersecond
//  * https://arxiv.org/pdf/2212.04356 (the whisper paper, their process is described on page 3)
pub const AUDIO_DECODE_SAMPLE_RATE: u32 = 12_000;

// ---------------------------------
// HuggingFace repository settings
// ---------------------------------
pub const REPO_ID: &str = "Demonthos/candle-quantized-whisper-large-v3-turbo";


// -------------------
// Inference settings
// -------------------

// Seed to help provide randomness in a weighted index
pub const SEED: u64 = 299792458;
