use std::error::Error;
use std::fs;
use std::path::Path;
use opencv::prelude::*;
use opencv::core::Vector;
use opencv::dnn_superres::DnnSuperResImpl;

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let output_path = Path::new(&config.output_folder);
    let remaining_images = fs::read_dir(&config.input_folder)?;

    let mut upscaler = DnnSuperResImpl::new("fsrcnn", 4)?;
    upscaler.read_model(&config.upscaler_path)?;

    for (i, path) in remaining_images.enumerate() {
        let read_path = path?.path();
        let read_path = read_path.to_str().expect("somehow got an empty path? shouldn't be possible!");
        let image : Mat = opencv::imgcodecs::imread(read_path, opencv::imgcodecs::IMREAD_COLOR)?;
        let mut upscaled_image : Mat = Default::default();
        upscaler.upsample(&image, &mut upscaled_image)?;

        let write_path = output_path.join(format!("{}.jpg",  &i));
        let write_path = write_path.to_str().expect("cannot write to empty path");
        opencv::imgcodecs::imwrite(write_path, &upscaled_image, &Vector::with_capacity(0))?;
    }
    Ok(())
}

pub struct Config {
    input_folder : String,
    output_folder : String,
    upscaler_path : String,
}

impl Config {
    pub fn new (args: &[String]) -> Result<Config, Box<dyn Error>> {
        if args.len() < 4 {
            return Err("Not enough arguments!")?;
        }
        let input_folder = args[1].clone();
        let output_folder = args[2].clone();
        let upscaler_path = args[3].clone();
        Ok(Config { input_folder, output_folder, upscaler_path })
    }
}
