use anyhow::Error;
use candle_transformers::models::whisper::Config;
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use crate::config;

#[derive(Debug, Clone)]
pub struct WhisperRepo {
    pub config_filename: PathBuf,
    pub tokenizer_filename: PathBuf,
    pub weights_filename: PathBuf,
}

impl WhisperRepo {
    pub fn config(&self) -> Config {
        serde_json::from_str(&std::fs::read_to_string(&self.config_filename).unwrap()).unwrap()
    }

    pub fn tokenizer(&self) -> Tokenizer {
        Tokenizer::from_file(&self.tokenizer_filename).unwrap()
    }
}

pub fn download() -> Result<WhisperRepo, Error> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        config::REPO_ID.to_owned(),
        RepoType::Model,
        "main".to_owned(),
    ));
    Ok(WhisperRepo {
        config_filename: repo.get("config.json").unwrap(),
        tokenizer_filename: repo.get("tokenizer.json").unwrap(),
        weights_filename: repo.get("model.gguf").unwrap(),
    })
}
