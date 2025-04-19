use anyhow::Result;
use symphonia::{core::{audio::{AudioBufferRef, Signal}, codecs::{CodecParameters, CodecType, Decoder, DecoderOptions, CODEC_TYPE_VORBIS}, formats::{FormatOptions, FormatReader}, io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions}}, default::{codecs::VorbisDecoder, formats::MkvReader}};
use symphonia::core::conv::FromSample;

// This module is responsible for decoding Webm audio into
// a vector of PCM samples (see https://en.wikipedia.org/wiki/Pulse-code_modulation)
//
// Webm uses a Matroska (aka MKV) format and a Vorbis or OPUS encoding.
// This code supports only the Vorbis encoding.
//
// Each sample is expressed in a 32 bit float
// The sample rate is taken from the provided audio
//   Note: MDN says that for the MediaRecorder API: "If bits per second values are not specified for[...] audio,
//   [...] the audio default is adaptive, depending upon the sample rate and the number of channels."
//   See: https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/MediaRecorder#audiobitspersecond
// The bitrate is 32 * the sample rate

fn conv<T>(samples: &mut Vec<f32>, data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>)
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
}

fn pcm_decode(original: impl MediaSource + 'static) -> Result<(Vec<f32>, u32)> {
    let stream = MediaSourceStream::new(Box::new(original), Default::default());
    let mut reader = MkvReader::try_new(stream, &Default::default()).unwrap();
    let first_track = reader.tracks().iter().find(|t| t.codec_params.codec == CODEC_TYPE_VORBIS).expect("No Vorbis tracks in this file!");
    let track_id = first_track.id;
    let sample_rate = first_track.codec_params.sample_rate.unwrap_or(0);
    let dec_opts: DecoderOptions = Default::default();
    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&first_track.codec_params, &dec_opts)
        .expect("unsupported codec");
    let mut pcm_data = Vec::new();
    while let Ok(packet) = reader.next_packet() {
        // Consume any new metadata that has been read since the last packet.
        while !reader.metadata().is_latest() {
            reader.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
        }
    }
    Ok((pcm_data, sample_rate))
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;

    #[test]
    fn it_can_pcm_decode() {
        let file = File::open("./test_data/audacity-vorbis.webm").unwrap();
        let (samples, rate) = pcm_decode(file).unwrap();
        assert!(samples.len() > 100_000);
        assert_eq!(rate, 44100);
    }
}
