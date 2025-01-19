#[macro_export]
// Copied from https://stackoverflow.com/a/74550371
macro_rules! test_ressource {
    ($fname:expr) => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/ressources/tests/", $fname) // assumes Linux ('/')!
    };
}
