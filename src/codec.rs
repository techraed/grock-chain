// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

//! Encoding and decoding functionalities for Grok Chain.

#![allow(dead_code)]

use crate::errors::CodecError;
use serde::{Deserialize, Serialize};

/// Encodes the given data into a byte vector using `postcard` serialization.
pub fn encode<T: Serialize + ?Sized>(data: &T) -> Result<Vec<u8>, CodecError> {
    postcard::to_allocvec(data).map_err(CodecError::SerializationFailed)
}

/// Decodes the given byte slice into the specified type using `postcard` deserialization.
pub fn decode<'a, T: Deserialize<'a>>(s: &'a [u8]) -> Result<T, CodecError> {
    postcard::from_bytes(s).map_err(CodecError::DeserializationFailed)
}
