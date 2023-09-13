use std::error::Error;
use std::fs;
use std::path::Path;
use opencv::prelude::{*};
use opencv::core::Vector;
use opencv::dnn_superres::DnnSuperResImpl;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use clap::{Parser,ValueHint};
use colored::Colorize;

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let output_path = Path::new(&config.output_path);
    let remaining_images = fs::read_dir(&config.input_path)?;
    let desired_aspect = (config.desired_ratio_w as f32) / (config.desired_ratio_h as f32);

    let upscaler_path = Path::new(&config.upscaler_path);
    let mut upscaler_x4 = DnnSuperResImpl::new("fsrcnn", 4)?;
    {
        let fsrcnn_x4 = upscaler_path.join("FSRCNN_x4.pb");
        let fsrcnn_x4 = fsrcnn_x4.to_str().unwrap_or_default();
        upscaler_x4.read_model(fsrcnn_x4)?;
    }

    let mut upscaler_x2 = DnnSuperResImpl::new("fsrcnn", 2)?;
    {
        let fsrcnn_x2 = upscaler_path.join("FSRCNN_x2.pb");
        let fsrcnn_x2 = fsrcnn_x2.to_str().unwrap_or_default();
        upscaler_x2.read_model(fsrcnn_x2)?;
    }

    let mut remaining_images : Vec<_> = remaining_images.collect();
    remaining_images.shuffle(&mut thread_rng());

    let mut image_database : Vec<Image> = Vec::new();
    for path in remaining_images {
        if path.is_err() {
            println!("Skipping image due to path read error {}", path.unwrap_err());
            continue;
        }
        let read_path = path.unwrap().path();
        let read_path = read_path.to_str().unwrap_or_default();
        match Image::new(read_path) {
            Ok(img) => image_database.push(img),
            Err(err) => println!("Skipping image {} due to error {}", read_path, err),
        }
    }

    let mut current_image_number = 0;
    while image_database.len() != 0 {
        let current_image = match image_database.pop() {
            Some(img) => img,
            None => break,
        };
        let mut current_mat : Mat = current_image.get_mat()?;

        println!("{} \t{} {}",
            "󰷊".blue(),
            format!("[{:>5}/{:>5}]", current_mat.cols(), current_mat.rows()).yellow(),
            current_image.filename);

        Image::fit(&mut current_mat, config.minimum_height, config.maximum_height, &mut upscaler_x2, &mut upscaler_x4);
        let mut working_aspect : f32 = (current_mat.cols() as f32) / (current_mat.rows() as f32);

        println!("{} \t{}",
            "".blue(),
            format!("[{:>5}/{:>5}]", current_mat.cols(), current_mat.rows()).yellow());

        println!("{}", "┊".blue());
        while image_database.len() > 0 && working_aspect < desired_aspect {
            let add_image_index = match image_database.iter().position(|test| test.ratio < desired_aspect-working_aspect) {
                Some(i) => i,
                None => break,
            };

            let add_image = image_database.get(add_image_index).unwrap();
            let mut add_mat : Mat = add_image.get_mat()?;
            println!("{} {} \t{} {}",
                "┊".blue(),
                "󰷊".bright_blue(),
                format!("[{:>5}/{:>5}]", add_mat.cols(), add_mat.rows()).bright_blue(),
                add_image.filename);
            Image::fit(&mut add_mat, current_mat.rows().try_into().unwrap(), current_mat.rows().try_into().unwrap(), &mut upscaler_x2, &mut upscaler_x4);

            println!("{} {} \t{}",
                "┊".blue(),
                "".bright_blue(),
                format!("[{:>5}/{:>5}]", add_mat.cols(), add_mat.rows()).bright_blue());

            let mut new_image : Mat = Default::default();
            opencv::core::hconcat2(&add_mat, &current_mat, &mut new_image)?;
            current_mat = new_image;

            println!("{} {} \t{}",
                "┊".blue(),
                "+".bright_blue(),
                format!("[{:>5}/{:>5}]", current_mat.cols(), current_mat.rows()).yellow());

            println!("{}", "┊".blue());
            working_aspect = (current_mat.cols() as f32) / (current_mat.rows() as f32);
            image_database.remove(add_image_index);
        }
        let write_path = output_path.join(format!("{}.jpg",  &current_image_number));
        let write_path = write_path.to_str().unwrap();
        current_image_number += 1;
        opencv::imgcodecs::imwrite(&write_path, &current_mat, &Vector::with_capacity(0))?;

        println!("{} \t{} {}",
            "󱣫 ->".green(),
            format!("[{:>5}/{:>5}]", current_mat.cols(), current_mat.rows()).green(),
            &write_path);

        println!("");
    }
    Ok(())
}

pub struct Image {
    filepath : String,
    filename : String,
    ratio : f32,
}

impl Image {
    pub fn new(filepath : &str) -> Result<Image, Box<dyn Error>> {
        let filepath : String = String::from(filepath);
        let filename : String = Path::new(&filepath).file_name().unwrap().to_str().unwrap().to_string();
        let cookie = magic::Cookie::open(magic::CookieFlags::ERROR)?;
        cookie.load::<&str>(&[])?;
        let analysis = cookie.file(&filepath)?;
        let read_analysis = analysis.split(",");
        let mut width : u32 = 0;
        let mut height : u32 = 0;
        for token in read_analysis {
            if !token.contains('x') {
                continue;
            }
            let possible_size : Vec<&str> = token.split('x').collect();
            if possible_size.len() == 2 {
                width = possible_size.get(0).unwrap_or(&"0").trim().parse().unwrap_or(0);
                height = possible_size.get(1).unwrap_or(&"0").trim().parse().unwrap_or(0);
                if width != 0 && height != 0 {
                    break;
                }
            }
        }
        if width == 0 || height == 0 {
            let mat = opencv::imgcodecs::imread(&filepath, opencv::imgcodecs::IMREAD_COLOR)?;
            width = mat.cols().try_into()?;
            height = mat.rows().try_into()?;
        }
        if width == 0 || height == 0 {
            Err("0 width, or 0 height image!")?;
        }
        let ratio = (width as f32)/(height as f32);
        println!("󰷊 \t[{:>5}/{:>5}] {}", width, height, filename);
        Ok(Image { filepath, ratio, filename })
    }
    pub fn get_mat(&self) -> Result<Mat, Box<dyn Error>> {
        Ok(opencv::imgcodecs::imread(&self.filepath, opencv::imgcodecs::IMREAD_COLOR)?)
    }
    fn fit_upscale(a : &mut Mat, resizer_x2 : &mut DnnSuperResImpl, resizer_x4 : &mut DnnSuperResImpl, min_height : u32) {
        if (a.rows() as u32)*2 < min_height {
            let mut upscaled_image : Mat = Default::default();
            resizer_x4.upsample(a, &mut upscaled_image).expect("Failed to resize image!");
            *a = upscaled_image;
        } else if (a.rows() as u32) < min_height {
            let mut upscaled_image : Mat = Default::default();
            resizer_x2.upsample(a, &mut upscaled_image).expect("Failed to resize image!");
            *a = upscaled_image;
        }
    }
    pub fn fit(a : &mut Mat, min_height : u32, max_height : u32, resizer_x2 : &mut DnnSuperResImpl, resizer_x4 : &mut DnnSuperResImpl) {
        Self::fit_upscale(a, resizer_x2, resizer_x4, min_height);
        if (a.rows() as u32) < min_height || (a.rows() as u32) > max_height {
            let mut scaled_image : Mat = Default::default();
            let percentage_change_mul : f32 = (max_height as f32)/(a.rows() as f32);
            let new_width : i32 = ((a.cols() as f32)*percentage_change_mul) as i32;
            let mode = if (a.rows() as u32) > max_height { opencv::imgproc::INTER_AREA } else { opencv::imgproc::INTER_LINEAR };
            opencv::imgproc::resize(a, &mut scaled_image, opencv::core::Size::new(new_width, max_height as i32), 0.0, 0.0, mode).expect("failed to resize!");
            *a = scaled_image;
        }
    }
}



// Program that takes a list of random images and makes them desktop-worthy.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Config {
    // The input folder that contains all the images you want to make desktop-worthy.
    #[arg(short, long, required(true), value_hint(ValueHint::DirPath))]
    input_path : String,
    // Where collaged images will be saved. As 1.jpg, 2.jpg, ...
    #[arg(short, long, required(true), value_hint(ValueHint::DirPath))]
    output_path : String,
    // Where to find FRNSRNN_x4.pb and FRNSRNN_x2.pb
    #[arg(short, long, required(true), value_hint(ValueHint::DirPath))]
    upscaler_path : String,
    // The minimum output height
    #[arg(long, default_value_t=1440)]
    minimum_height : u32,
    // The maximum output height
    #[arg(long, default_value_t=2880)]
    maximum_height : u32,
    // The desired width in the output ratio w/h
    #[arg(long, default_value_t=16)]
    desired_ratio_w : u32,
    // The desired height in the output ratio w/h
    #[arg(long, default_value_t=9)]
    desired_ratio_h : u32,
}

impl Config {
    pub fn new () -> Config {
        let mut output = Config::parse();
        output.input_path = shellexpand::full(&output.input_path).unwrap().to_string();
        output.output_path = shellexpand::full(&output.output_path).unwrap().to_string();
        output.upscaler_path = shellexpand::full(&output.upscaler_path).unwrap().to_string();
        output
    }
}
