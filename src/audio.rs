// This module is responsible for decoding Webm audio into
// a vector of PCM samples (see https://en.wikipedia.org/wiki/Pulse-code_modulation)
//
// Webm uses a Matroska (aka MKV) format and a Vorbis or OPUS encoding.
// This code supports only the OPUS encoding, since that is the encoding
// that Firefox and Chromium use.
//
// Each sample is expressed in a 32 bit float
// This can handle Mono or Stereo, which is kinda cool!

use anyhow::{Result, anyhow};
use opus::{Channels, Decoder};
use symphonia::{
    core::{
        codecs::CODEC_TYPE_OPUS,
        formats::FormatReader,
        io::{MediaSource, MediaSourceStream},
    },
    default::formats::MkvReader,
};

struct Track {
    sample_rate: u32,
    channels: Channels,
    id: u32,
    reader: MkvReader,
}

impl Track {
    pub fn decode(&mut self) -> Result<(Vec<f32>, u32)> {
        // Sample rate must be 8, 12, 16, 24, or 48 kHz.  The libopus documentation recommends 48.
        // Note that the sample rate of a file from the browser may not be one of the rates supported
        // by libopus.  When using the browser's MediaRecorder API, you can pass in a custom sample
        // rate, and the default rate "is adaptive, depending upon the sample rate and the number of channels."
        // See:
        //  * https://opus-codec.org/docs/opus_api-1.5.pdf
        //  * https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/MediaRecorder#audiobitspersecond
        let mut decoder = Decoder::new(48_000, self.channels).unwrap();

        let mut pcm_data = Vec::new();
        while let Ok(packet) = self.reader.next_packet() {
            while !self.reader.metadata().is_latest() {
                self.reader.metadata().pop();
            }
            if packet.track_id() != self.id {
                continue;
            }
            let num_samples = decoder.get_nb_samples(packet.buf())?;
            let mut decoded = vec![0.0; num_samples * self.channels as usize];
            let _ = decoder.decode_float(packet.buf(), &mut decoded, false);
            pcm_data.append(&mut decoded);
        }

        Ok((pcm_data, self.sample_rate))
    }
}

pub fn pcm_decode(original: impl MediaSource + 'static) -> Result<(Vec<f32>, u32)> {
    let mut track = demux(original)?;
    track.decode()
}

fn demux(original: impl MediaSource + 'static) -> Result<Track> {
    let stream = MediaSourceStream::new(Box::new(original), Default::default());
    let reader = MkvReader::try_new(stream, &Default::default()).unwrap();
    let first_track_option = reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec == CODEC_TYPE_OPUS);
    let first_track = match first_track_option {
        Some(track) => track,
        None => return Err(anyhow!("No Opus tracks in this file!")),
    };
    let sample_rate = first_track.codec_params.sample_rate.unwrap_or(0);
    let channel_count = first_track.codec_params.channels.iter().count();
    Ok(Track {
        sample_rate,
        id: first_track.id,
        channels: if channel_count == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        },
        reader,
    })
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;

    #[test]
    fn it_can_pcm_decode() {
        let file = File::open("./test_data/opus.webm").unwrap();
        let (samples, rate) = pcm_decode(file).unwrap();
        assert!(samples.len() > 100_000);
        assert_eq!(rate, 48_000);
    }

    #[test]
    fn it_errors_on_vorbis_encoding() {
        let file = File::open("./test_data/vorbis.webm").unwrap();
        assert!(pcm_decode(file).is_err());
    }
}
