use crate::config;
use anyhow::Error;
use candle_transformers::models::whisper::Config;
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::{path::PathBuf, sync::OnceLock};
use tokenizers::Tokenizer;

pub static WHISPER_REPO: OnceLock<WhisperRepo> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct WhisperRepo {
    pub config_file: PathBuf,
    pub tokenizer_file: PathBuf,
    pub weights_file: PathBuf,
}

impl WhisperRepo {
    pub fn config(&self) -> Config {
        serde_json::from_str(&std::fs::read_to_string(&self.config_file).unwrap()).unwrap()
    }

    pub fn tokenizer(&self) -> Tokenizer {
        Tokenizer::from_file(&self.tokenizer_file).unwrap()
    }

    pub fn get() -> &'static WhisperRepo {
        WHISPER_REPO.get_or_init(|| download().unwrap())
    }
}

fn download() -> Result<WhisperRepo, Error> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        config::REPO_ID.to_owned(),
        RepoType::Model,
        "main".to_owned(),
    ));
    Ok(WhisperRepo {
        config_file: repo.get(config::MODEL_CONFIG_FILENAME).unwrap(),
        tokenizer_file: repo.get(config::TOKENIZER_FILENAME).unwrap(),
        weights_file: repo.get(config::MODEL_FILENAME).unwrap(),
    })
}
