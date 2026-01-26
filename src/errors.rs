// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

//! Error definitions for Grok Chain.

// TODO: temporarily
#![allow(dead_code)]

/// General grok chain error.
#[derive(Debug, thiserror::Error)]
pub enum GrokChainError {
    #[error("Cryptography error: {0}")]
    Crypto(#[from] CryptoError),
    #[error("Codec error: {0}")]
    Codec(#[from] CodecError),
}

/// Cryptography related errors.
#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    #[error("Failed to sign message with the private key: {0}")]
    MessageSigningFailed(k256::ecdsa::Error),
    #[error("Failed to verify message signature: {0}")]
    MessageVerificationFailed(k256::ecdsa::Error),
}

/// Transaction related errors.
#[derive(Debug, thiserror::Error)]
pub enum TransactionError {
    #[error("Failed to serialize transaction: {0}")]
    FailedTransactionSerialization(#[from] CodecError),
}

/// Codec related errors
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("Failed to serialize data: {0}")]
    SerializationFailed(postcard::Error),
    #[error("Failed to deserialize data: {0}")]
    DeserializationFailed(postcard::Error),
}
