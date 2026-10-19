use super::*;

#[test]
fn truncate_output_keeps_utf8_and_limit() {
    let result = truncate_output("你好，世界".to_string(), 7);

    assert!(result.is_char_boundary(result.len()));
    assert!(result.len() <= 7);
    assert!(result.ends_with('…'));
}
