use std::any;

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

// This module is responsible for decoding Webm audio into
// a vector of PCM samples (see https://en.wikipedia.org/wiki/Pulse-code_modulation)
//
// Webm uses a Matroska (aka MKV) format and a Vorbis or OPUS encoding.
// This code supports only the OPUS encoding, since that is the encoding
// that Firefox and Chromium use.
//
// Each sample is expressed in a 32 bit float
// The sample rate is taken from the provided audio
//   Note: MDN says that for the MediaRecorder API: "If bits per second values are not specified for[...] audio,
//   [...] the audio default is adaptive, depending upon the sample rate and the number of channels."
//   See: https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/MediaRecorder#audiobitspersecond
// The bitrate is 32 * the sample rate

struct Track {
    sample_rate: u32,
    channels: Channels,
    id: u32,
    reader: MkvReader,
}

pub fn pcm_decode(original: impl MediaSource + 'static) -> Result<(Vec<f32>, u32)> {
    let mut track = demux(original)?;
    // Sample rate must be 8000, 12000, 16000, 24000, or 48000.
    let mut decoder = Decoder::new(48_000, track.channels).unwrap();

    let mut pcm_data = Vec::new();
    while let Ok(packet) = track.reader.next_packet() {
        // Consume any new metadata that has been read since the last packet.
        while !track.reader.metadata().is_latest() {
            track.reader.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track.id {
            continue;
        }
        let num_samples = decoder.get_nb_samples(packet.buf())?;
        let mut decoded = vec![0.0; num_samples * 2];
        let res = decoder.decode_float(packet.buf(), &mut decoded, false);
        pcm_data.append(&mut decoded);
    }

    Ok((pcm_data, track.sample_rate))
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
