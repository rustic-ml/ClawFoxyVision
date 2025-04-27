use std::env;

fn main() {
    let _src = env::var("CARGO_MANIFEST_DIR").unwrap();
    let _dst = format!("{}/src/build_info.rs", env::var("OUT_DIR").unwrap());
    built::write_built_file().expect("Failed to generate build info");
}
