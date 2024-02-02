use std::error::Error;
use std::fs;
use std::path::{Path,PathBuf};
use std::fs::File;
use opencv::prelude::{*};
use opencv::core::Vector;
use opencv::dnn_superres::DnnSuperResImpl;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use clap::{Parser,ValueHint};
use colored::Colorize;
use serde::{Serialize, Deserialize};
use std::io::BufWriter;

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

    let database_file_path = String::from(&config.output_path) + "/database.json";
    let mut database = WallpaperDatabase::new(database_file_path.as_str())?;
    database.delete_missing(&mut image_database);
    let mut image_database : Vec<Image> = database.get_regen_image_paths(image_database);

    let mut current_image_number = 0;
    while image_database.len() != 0 {
        let mut wallpaper_images : Vec<Image> = vec![];
        let current_image = match image_database.pop() {
            Some(img) => img,
            None => break,
        };
        let mut current_mat : Mat = current_image.get_mat()?;
        wallpaper_images.push(current_image.clone());

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
        while working_aspect < desired_aspect {
            let mut wallpaper_match : Option<Wallpaper> = None;
            let add_image = match image_database.iter().position(|test| test.ratio < desired_aspect-working_aspect) {
                Some(index) => {
                    let image_match = image_database.get(index).unwrap().to_owned();
                    image_database.remove(index);
                    image_match
                },
                None => match database.original_wallpapers.iter().position(|wallpaper| wallpaper.wallpaper_image.ratio < desired_aspect-working_aspect) {
                    Some(wallpaper_index) => {
                        wallpaper_match = Some(database.original_wallpapers.get(wallpaper_index).unwrap().clone());
                        let wallpaper_work = wallpaper_match.clone().unwrap();
                        database.original_wallpapers.remove(wallpaper_index);
                        for image in &wallpaper_work.original_images {
                            wallpaper_images.push(image.clone());
                        }
                        wallpaper_work.wallpaper_image
                    },
                    None => break,
                },
            };
            wallpaper_images.push(add_image.clone());
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

            if wallpaper_match.is_some() {
                wallpaper_match.unwrap().delete();
            }
        }

        let write_path = output_path.join(format!("{}.jpg",  &current_image_number));
        let mut write_path = String::from(write_path.to_str().unwrap());
        while Path::new(write_path.as_str()).exists() {
            current_image_number += 1;
            let path_text = output_path.join(format!("{}.jpg",  &current_image_number));
            write_path = String::from(path_text.to_str().unwrap());
        }
        opencv::imgcodecs::imwrite(&write_path, &current_mat, &Vector::with_capacity(0))?;
        current_image_number += 1;

        println!("{} \t{} {}",
            "󱣫 ->".green(),
            format!("[{:>5}/{:>5}]", current_mat.cols(), current_mat.rows()).green(),
            &write_path);

        database.original_wallpapers.push(Wallpaper {
            original_images : wallpaper_images.clone(),
            wallpaper_image : Image::new(&write_path)?,
        });

        println!("");
    }
    let file = File::create(database_file_path.as_str()).unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &database)?;
    Ok(())
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Image {
    filepath : String,
    filename : String,
    ratio : f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Wallpaper {
    original_images : Vec<Image>,
    wallpaper_image : Image,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WallpaperDatabase {
    original_wallpapers : Vec<Wallpaper>,
}

impl Wallpaper {
    pub fn new(original_images : Vec<Image>, filepath : String) -> Wallpaper {
        Wallpaper { 
            original_images,
            wallpaper_image : Image::new(filepath.as_str()).unwrap(),
        }
    }
    pub fn contains_image(&self, image : &Image) -> bool {
        return self.original_images.iter().any(|wimage| wimage.filename == image.filename);
    }
    pub fn delete(&self) {
        println!("{} \t{}", "󰮘".red(), &self.wallpaper_image.filepath);
        fs::remove_file(&self.wallpaper_image.filepath).unwrap();
    }
    pub fn invalid(&self, images : &Vec<Image>) -> bool {
        return self.original_images.iter().any(|image| {
            for other in images {
                if image.filename == other.filename {
                    return false;
                }
            }
            return true;
        });
    }
}

impl WallpaperDatabase {
    pub fn new(filepath : &str) -> Result<WallpaperDatabase, Box<dyn Error>> {
        match File::open(filepath) {
            Ok(file) => {
                let deserialized_database : WallpaperDatabase = serde_json::from_reader(file)?;
                Ok(deserialized_database)
            },
            Err(_err) => Ok(WallpaperDatabase {
                original_wallpapers : vec![],
            }),
        }
    }
    pub fn delete_missing(&mut self, images : &Vec<Image>) {
        let remove_wallpapers = self.original_wallpapers.clone().into_iter().filter(|wallpaper| wallpaper.invalid(&images)).collect::<Vec<Wallpaper>>();
        for wallpaper in remove_wallpapers {
            wallpaper.delete();
        }
        self.original_wallpapers = self.original_wallpapers.clone().into_iter().filter(|wallpaper| !wallpaper.invalid(&images)).collect::<Vec<Wallpaper>>();
    }
    pub fn get_regen_image_paths(&self, images : Vec<Image>) -> Vec<Image> {
        images.into_iter().filter(|image| {
            for wallpaper in &self.original_wallpapers {
                if wallpaper.contains_image(&image) {
                    return false;
                }
            }
            return true;
        }).collect::<Vec<Image>>()
    }
}

impl Image {
    pub fn new(filepath : &str) -> Result<Image, Box<dyn Error>> {
        let read_path = fs::canonicalize(PathBuf::from(filepath))?;
        let read_path = read_path.as_path();
        let read_path = read_path.to_str().unwrap_or_default();

        let filepath : String = String::from(read_path);
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
