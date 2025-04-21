// This module is responsible for decoding Webm audio into
// a vector of PCM samples (see https://en.wikipedia.org/wiki/Pulse-code_modulation)
//
// Webm uses a Matroska (aka MKV) format and a Vorbis or OPUS encoding.
// This code supports only the OPUS encoding, since that is the encoding
// that Firefox and Chromium use.
//
// Each sample is expressed in a 32 bit float
// This can handle Mono or Stereo, which is kinda cool!

use std::io::{Read, Seek};

use anyhow::{Result, anyhow};
use matroska_demuxer::{Frame, MatroskaFile};
use opus::{Channels, Decoder};

struct Track<R: Seek + Read> {
    sample_rate: f64,
    channels: Channels,
    track: u64,
    reader: MatroskaFile<R>,
}

impl<R: Seek + Read> Track<R> {
    pub fn decode(&mut self) -> Result<(Vec<f32>, f64)> {
        // Sample rate must be 8, 12, 16, 24, or 48 kHz.  The libopus documentation recommends 48.
        // Interestingly: when using the symphonia crate's matroska demuxer, a lower sample rate provided
        // more accurate results.  With the matroska-demuxer crate, a higher sample rate seems to work better.
        // In either case, if you get it too high, the transcriptions are just "... ... ..."
        // Note that the sample rate of a file from the browser may not be one of the rates supported
        // by libopus.  When using the browser's MediaRecorder API, you can pass in a custom sample
        // rate, and the default rate "is adaptive, depending upon the sample rate and the number of channels."
        // See:
        //  * https://opus-codec.org/docs/opus_api-1.5.pdf
        //  * https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/MediaRecorder#audiobitspersecond
        let mut decoder = Decoder::new(12_000, self.channels).unwrap();

        let mut pcm_data = Vec::new();
        let mut packet = Frame::default();
        while self.reader.next_frame(&mut packet).unwrap() {
            if packet.track != self.track {
                continue;
            }
            if packet.is_invisible {
                continue;
            }
            let num_samples = decoder.get_nb_samples(&packet.data)?;
            let mut decoded = vec![0.0; num_samples * self.channels as usize];
            let _ = decoder.decode_float(&packet.data, &mut decoded, false);
            pcm_data.append(&mut decoded);
        }

        Ok((pcm_data, self.sample_rate))
    }
}

pub fn pcm_decode<R: Seek + Read>(original: R) -> Result<(Vec<f32>, f64)> {
    let mut track: Track<R> = demux(original)?;
    track.decode()
}

fn demux<R: Seek + Read>(original: R) -> Result<Track<R>> {
    let mut stream = MatroskaFile::open(original).unwrap();
    let first_track_option = stream
        .tracks()
        .iter()
        .find(|t| t.codec_id() == "A_OPUS");
    let first_track = match first_track_option {
        Some(track) => track,
        None => return Err(anyhow!("No Opus tracks in this file!")),
    };
    let sample_rate = first_track.audio().unwrap().sampling_frequency();
    let channel_count = first_track.audio().unwrap().channels().get();
    Ok(Track {
        sample_rate,
        track: first_track.track_number().get(),
        channels: if channel_count == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        },
        reader: stream,
    })
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Cursor};

    use super::*;

    #[test]
    fn it_can_pcm_decode_mono() {
        let file = File::open("./test_data/portuguese/semana_de_arte_moderna_mono.webm").unwrap();
        let (samples, rate) = pcm_decode(file).unwrap();
        assert!(samples.len() > 50_000);
        assert_eq!(rate, 24_000 as f64);
    }

    #[test]
    fn it_can_pcm_decode_stereo() {
        let file = File::open("./test_data/portuguese/a_filha_do_patrao_stereo.webm").unwrap();
        let (samples, rate) = pcm_decode(file).unwrap();
        assert!(samples.len() > 40_000);
        assert_eq!(rate, 24_000 as f64);
    }

    #[test]
    #[allow(non_snake_case)]
    fn it_can_pcm_decode_sample_rate_of_48_MHz() {
        let file =
            File::open("./test_data/russian/po_nedele_ni_slova_ni_s_kem_ne_skazhu_mono.webm")
                .unwrap();
        let (samples, rate) = pcm_decode(file).unwrap();
        assert!(samples.len() > 40_000);
        assert_eq!(rate, 48_000 as f64);
    }

    #[test]
    #[allow(non_snake_case)]
    fn it_can_pcm_decode_sample_rate_of_8_MHz() {
        let file = File::open("./test_data/russian/voron_mono_8MHz.webm").unwrap();
        let (samples, rate) = pcm_decode(file).unwrap();
        assert_eq!(rate, 8_000 as f64);
    }

    #[test]
    fn it_can_decode_webm_recorded_in_firefox() {
        // This file is in bad shape, `mkvalidator test_data/firefox.webm` returns pages of errors!
        // I got it using the MediaRecorder API in firefox on a mac
        // The matroska_demuxer crate can handle it, the symphonia crate cannot.
        let file = File::open("./test_data/firefox.webm").unwrap();
        let (samples, rate) = pcm_decode(file).unwrap();
        assert_eq!(rate, 44_100 as f64);
    }

    #[test]
    fn it_can_decode_webm_recorded_in_edge() {
        let file = File::open("./test_data/edge.webm").unwrap();
        let (samples, rate) = pcm_decode(file).unwrap();
        assert_eq!(rate, 48_000 as f64);
    }

    #[test]
    fn it_can_pcm_decode_a_cursor() {
        let binary_data =
            std::fs::read("./test_data/english/alexander_the_great_mono.webm").unwrap();
        let cursor = Cursor::new(binary_data);
        let (samples, rate) = pcm_decode(cursor).unwrap();
        assert_eq!(rate, 12_000 as f64);
    }

    #[test]
    fn it_errors_on_vorbis_encoding() {
        let file = File::open("./test_data/vorbis.webm").unwrap();
        assert!(pcm_decode(file).is_err());
    }
}
