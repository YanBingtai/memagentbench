#[derive(Debug, Default)]
pub(in crate::tools::filesystem::search) struct SearchReport {
    pub(in crate::tools::filesystem::search) matches: Vec<String>,
    pub(in crate::tools::filesystem::search) skipped_files: usize,
    pub(in crate::tools::filesystem::search) truncated: bool,
    pub(in crate::tools::filesystem::search) file_limit_reached: bool,
}
