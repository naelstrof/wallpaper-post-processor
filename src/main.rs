use std::process;
use wallpp::Config;

fn main() {
    let config : Config = Config::new();
    if let Err(e) = wallpp::run(config) {
        println!("Failed to run for reason: {}", e);
        process::exit(1);
    }
}

