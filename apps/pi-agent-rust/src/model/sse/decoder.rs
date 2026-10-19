use super::super::{error::ModelError, limits::*};

#[derive(Debug, Default)]
pub(crate) struct SseDecoder {
    buffer: Vec<u8>,
    data_lines: Vec<String>,
    event_bytes: usize,
}

impl SseDecoder {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, ModelError> {
        self.buffer.extend_from_slice(bytes);

        let mut payloads = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..newline).collect();
            self.buffer.drain(..1);
            if line.last() == Some(&b'\r') {
                line.pop();
            }

            let line = std::str::from_utf8(&line).map_err(|_| {
                ModelError::StreamProtocol("SSE line is not valid UTF-8".to_string())
            })?;
            if line.is_empty() {
                if !self.data_lines.is_empty() {
                    payloads.push(self.data_lines.join("\n"));
                    self.data_lines.clear();
                    self.event_bytes = 0;
                }
                continue;
            }
            if line.starts_with(':') {
                continue;
            }
            if let Some(value) = line.strip_prefix("data:") {
                let value = value.strip_prefix(' ').unwrap_or(value);
                self.event_bytes = self
                    .event_bytes
                    .checked_add(value.len() + 1)
                    .ok_or(ModelError::StreamLimit { limit: "SSE event" })?;
                if self.event_bytes > MAX_STREAM_EVENT_BYTES {
                    return Err(ModelError::StreamLimit { limit: "SSE event" });
                }
                self.data_lines.push(value.to_string());
            }
        }
        if self.buffer.len() > MAX_STREAM_EVENT_BYTES {
            return Err(ModelError::StreamLimit { limit: "SSE event" });
        }
        Ok(payloads)
    }

    pub(crate) fn finish(&self) -> Result<(), ModelError> {
        if self.buffer.is_empty() && self.data_lines.is_empty() {
            Ok(())
        } else {
            Err(ModelError::StreamIncomplete)
        }
    }
}
