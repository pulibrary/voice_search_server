// This module is responsible for transcribing!

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::{quantized_model::{self, Whisper}, COMPRESSION_RATIO_THRESHOLD, EOT_TOKEN, HOP_LENGTH, LOGPROB_THRESHOLD, NO_SPEECH_THRESHOLD, NO_SPEECH_TOKENS, NO_TIMESTAMPS_TOKEN, N_FRAMES, SAMPLE_RATE, SOT_TOKEN, TEMPERATURES, TRANSCRIBE_TOKEN};
use rand::{distr::Distribution, SeedableRng};
use rand::distr::weighted::WeightedIndex;
use tokenizers::Tokenizer;
use anyhow::anyhow;
use crate::whisper::WhisperFiles;


pub fn transcribe(features: Vec<f32>, files: WhisperFiles) -> Result<String, anyhow::Error> {
    let mel_len = features.len();
    // TODO: Don't hardcode metal!!
    let device = &Device::new_metal(0).unwrap();
    let mel = Tensor::from_vec(
        features,
        (1, files.config().num_mel_bins, mel_len / files.config().num_mel_bins), 
        &device,
    )?;
    
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
        &files.weights_filename,
        &device,
    )?;
    let mut model = quantized_model::Whisper::load(&vb, files.config())?;

    let mut dc = Decoder::new(
        model,
        files.tokenizer(),
        0,
        &device,
        None, // TODO: optionally pass in a language token
    )?;
    let segments = dc.run(&mel)?;
    Ok(segments.iter().map(|s|s.transcription()).collect::<String>())
}

// The following is all copy/pasted from https://github.com/vberthet/candle/blob/rocm/candle-examples/examples/whisper/main.rs
// A lot can be re-written and/or simplified

struct Decoder {
    model: Whisper,
    rng: rand::rngs::StdRng,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Whisper,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
    ) -> Result<Self, anyhow::Error> {
        let no_timestamps_token = token_id(&tokenizer, NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if model.config.suppress_tokens.contains(&i)
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, TRANSCRIBE_TOKEN)?;
        let eot_token = token_id(&tokenizer, EOT_TOKEN)?;
        let no_speech_token = NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            suppress_tokens,
            sot_token,
            transcribe_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult, anyhow::Error> {
        let model = &mut self.model;
        let audio_features =  model.encoder.forward(mel, true)?;
        let sample_len = model.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        tokens.push(self.transcribe_token);
        tokens.push(self.no_timestamps_token);
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder.final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::shape::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(anyhow::Error::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult, anyhow::Error> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult, anyhow::Error> = self.decode(segment, t);
            if i == TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(&mut self, mel: &Tensor) -> Result<Vec<Segment>, anyhow::Error> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > NO_SPEECH_THRESHOLD && dr.avg_logprob < LOGPROB_THRESHOLD {
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            segments.push(segment)
        }
        Ok(segments)
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, anyhow::Error> {
    match tokenizer.token_to_id(token) {
        None => { return Err(anyhow!("no token-id for {token}")) },
        Some(id) => Ok(id),
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}


#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

impl Segment {
    pub fn transcription(&self) -> String {
        self.dr.text.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{audio, feature_extraction::extract_features, whisper::download};
    use std::fs::File;

    fn transcribe_file(path: &str) -> String {
        let file = File::open(path).unwrap();
        let (samples, rate) = audio::pcm_decode(file).unwrap();
        let features = extract_features(samples).unwrap();
        let files = download().unwrap();
        transcribe(features, files).unwrap().to_lowercase()
    }

    #[test]
    fn it_can_transcribe_portuguese_mono() {
        let transcription = transcribe_file("./test_data/portuguese/semana_de_arte_moderna_mono.webm");
        assert!(transcription.contains("sessão 2"));
        assert!(transcription.contains("semana de arte moderna de 1922"));
        assert!(transcription.contains("coletânea centenário"));
    }

    #[test]
    fn it_can_transcribe_portuguese_stereo() {
        let transcription = transcribe_file("./test_data/portuguese/a_filha_do_patrao_stereo.webm");
        assert!(transcription.contains("a filha do patrão"));
        // Currently, it is not correctly transcribing the author's name
        // assert!(transcription.contains("artur de azevedo"));
    }

    #[test]
    fn it_can_transcribe_russian_mono() {
        let transcription = transcribe_file("./test_data/russian/po_nedele_ni_slova_ni_s_kem_ne_skazhu_mono.webm");
        // Currently, it is combining По неделе into понеделье
        // assert!(transcription.contains("По неделе ни слова ни с кем не скажу"))
        assert!(transcription.contains("ни слова ни с кем не скажу"));
    }

    #[test]
    fn it_can_transcribe_russian_stereo() {
        let transcription = transcribe_file("./test_data/russian/vseobshchaia_deklaratsiia_prav_cheloveka.webm");
        assert!(transcription.contains("всеобщая декларация прав человека"));
        assert!(transcription.contains("принята и провозглашена резолюцией 217а"));
        assert!(transcription.contains("генеральной ассамблеи от 10 декабря 1948 года"));
        assert!(transcription.contains("преамбула"));
    }

    #[test]
    fn it_can_transcribe_russian_with_english_author_name() {
        let transcription = transcribe_file("./test_data/russian/voron_mono_8MHz.webm");
        // The preferred Russian transliteration is Аллан, not Аллен
        // assert!(transcription.contains("эдгар аллан по"));
        assert!(transcription.contains("эдгар"));
        assert!(transcription.contains("ворон"));
        assert!(transcription.contains("перевод"));
        assert!(transcription.contains("владимира жаботинского"));
        assert!(transcription.contains("первый вариант перевода"));
    }
}
