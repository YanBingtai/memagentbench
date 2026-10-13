//! Deterministic digests for JSON action arguments.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// Returns a `sha256:` digest of recursively canonicalized JSON.
pub fn canonical_json_digest(value: &Value) -> String {
    let canonical = canonicalize(value);
    let encoded = serde_json::to_vec(&canonical).expect("serde_json::Value is serializable");
    bytes_digest(&encoded)
}

/// Returns a `sha256:` digest of arbitrary bytes.
pub fn bytes_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(7 + digest.len() * 2);
    encoded.push_str("sha256:");
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
    }
    encoded
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonicalize).collect()),
        Value::Object(values) => {
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            let mut canonical = Map::with_capacity(entries.len());
            for (key, value) in entries {
                canonical.insert(key.clone(), canonicalize(value));
            }
            Value::Object(canonical)
        }
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_key_order_does_not_change_digest() {
        let first = serde_json::json!({"b": 2, "a": 1});
        let second = serde_json::json!({"a": 1, "b": 2});

        assert_eq!(
            canonical_json_digest(&first),
            canonical_json_digest(&second)
        );
    }

    #[test]
    fn arrays_keep_their_order() {
        let first = serde_json::json!([1, 2]);
        let second = serde_json::json!([2, 1]);

        assert_ne!(
            canonical_json_digest(&first),
            canonical_json_digest(&second)
        );
    }

    #[test]
    fn byte_digest_has_a_stable_prefix_and_length() {
        let digest = bytes_digest(b"runtime");

        assert!(digest.starts_with("sha256:"));
        assert_eq!(digest.len(), 71);
    }
}
