use std::env;
use std::process;
use wallpp::Config;

fn main() {
    let args: Vec<String> = env::args().collect();
    let config : Config = Config::new(&args).unwrap_or_else(|err| {
        println!("Problem parsing arguments: {err}");
        process::exit(1);
    });
    if let Err(e) = wallpp::run(config) {
        println!("Failed to run for reason: {}", e);
        process::exit(1);
    }
}

