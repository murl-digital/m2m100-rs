use std::path::PathBuf;

use rust_bert::{
    m2m_100::{M2M100Config, M2M100Model, M2M100SourceLanguages, M2M100TargetLanguages},
    pipelines::{
        common::ModelResource,
        translation::{TranslationConfig, TranslationModel},
    },
    resources::{LocalResource, ResourceProvider},
    Config,
};
use rust_tokenizers::tokenizer::M2M100Tokenizer;
use tch::{nn::VarStore, Device};

fn main() -> anyhow::Result<()> {
    let config = LocalResource {
        local_path: PathBuf::from("/home/draconium/Desktop/repos/m2m-rs/m2m100_418M/config.json"),
    };
    let vocab = LocalResource {
        local_path: PathBuf::from("/home/draconium/Desktop/repos/m2m-rs/m2m100_418M/vocab.json"),
    };
    let merges = LocalResource {
        local_path: PathBuf::from(
            "/home/draconium/Desktop/repos/m2m-rs/m2m100_418M/sentencepiece.bpe.model",
        ),
    };
    let weights = LocalResource {
        local_path: PathBuf::from("/home/draconium/Desktop/repos/m2m-rs/m2m100_418M/rust_model.ot"),
    };

    let device = Device::cuda_if_available();

    let pipeline = TranslationModel::new(TranslationConfig::new(
        rust_bert::pipelines::common::ModelType::M2M100,
        ModelResource::Torch(Box::new(weights)),
        config,
        vocab,
        Some(merges),
        M2M100SourceLanguages::M2M100_418M,
        M2M100TargetLanguages::M2M100_418M,
        device,
    ))?;

    let translated = pipeline.translate(
        &["Oh mein Gott, diese Höhle ist voller bluntsmoken!"],
        Some(rust_bert::pipelines::translation::Language::German),
        Some(rust_bert::pipelines::translation::Language::English),
    )?;

    println!("{:?}", translated);

    Ok(())
}
