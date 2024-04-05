fn main() {
    //vcpkg::find_package("libmagic").unwrap();
    println!("cargo:rustc-link-search=native=C:/Users/naelstrof/vcpkg/installed/x64-windows-static/lib");
    println!("cargo:rustc-link-lib=tre");
    println!("cargo:rustc-link-lib=getopt");
}