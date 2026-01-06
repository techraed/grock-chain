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
}

/// Cryptography related errors.
#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    #[error("Failed to sign message with the private key: {0}")]
    MessageSigningFailed(#[from] k256::ecdsa::Error),
    #[error("Failed to verify message signature: {0}")]
    MessageVerificationFailed(#[from] k256::ecdsa::Error),
}
