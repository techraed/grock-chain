// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

//! Error definitions for Grok Chain.

// TODO [sab]:
// 1. Huge design mistake - every error depends too much on underlying error type
// 2. Database errors aren't unique. But uniting them makes it hard to identify the exact error.

// TODO: temporarily
#![allow(dead_code)]

use crate::crypto::Hash256;

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
    #[error("Failed to serialize transaction inputs and outputs {0:?}")]
    FailedRawTransactionSerialization(CodecError),
    #[error("Empty transaction: transaction ID {0:?} has either no inputs or no outputs")]
    EmptyTransaction(Hash256),
    #[error("Invalid transaction {0:?} signature {1:?}")]
    InvalidSignature(Hash256, CryptoError),
    #[error("Insufficient funds in transaction {0:?}")]
    InsufficientFunds(Hash256),
    #[error("Transaction output with ID {0:?} and idx {1} not found: {2:?}")]
    TransactionOutputNotFound(Hash256, usize, DatabaseError),
    #[error("Failed to insert transaction output with ID {0:?} and idx {1}: {2:?}")]
    FailedToInsertTransactionOutput(Hash256, usize, DatabaseError),
}

/// Codec related errors
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("Failed to serialize data: {0}")]
    SerializationFailed(postcard::Error),
    #[error("Failed to deserialize data: {0}")]
    DeserializationFailed(postcard::Error),
}

/// Database related errors.
#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    #[error("Failed to open database: {0}")]
    OpenFailed(sled::Error),

    // Block related errors
    #[error("Failed to get the block {0}")]
    CannotGetBlock(sled::Error),
    #[error("Block with hash {0:?} not found")]
    BlockNotFound(Hash256),
    #[error("Failed to deserialize block with hash {0:?}: {1}")]
    FailedBlockDeserialization(Hash256, CodecError),
    #[error("Failed to serialize block with hash {0:?}: {1}")]
    FailedBlockSerialization(Hash256, CodecError),
    #[error("Failed to insert block into database: {0}")]
    FailedBlockInsertion(sled::Error),

    // Transaction output related errors
    #[error("Failed to get the transaction output {0}")]
    CannotGetTxOutput(sled::Error),
    #[error("Transaction output with ID {0:?} not found")]
    TransactionOutputNotFound(Hash256),
    #[error("Failed to deserialize transaction output with ID {0:?}: {1}")]
    FailedTxOutputDeserialization(Hash256, CodecError),
    #[error("Failed to serialize transaction output with ID {0:?}: {1}")]
    FailedTxOutputSerialization(Hash256, CodecError),
    #[error("Failed to insert transaction output into database: {0}")]
    FailedTxOutputInsertion(sled::Error),
}

/// Block chain related errors.
#[derive(Debug, thiserror::Error)]
pub enum BlockChainError {
    #[error("Failed to add block with hash {0:?}: {1}")]
    FailedToAddBlock(Hash256, DatabaseError),
    #[error("Failed to get block with hash {0:?}: {1}")]
    CannotGetBlock(Hash256, DatabaseError),
    #[error("Orphan block received with hash {0:?}")]
    OrphanBlockReceived(Hash256),
    #[error("Invalid transaction {0:?}")]
    InvalidTransaction(Box<(Hash256, TransactionError)>),
    #[error("Failed to serialize block: {0}")]
    FailedBlockSerialization(CodecError),
    #[error("Failed to store block to db: {0}")]
    FailedToStoreBlock(DatabaseError),
    #[error("Failed to remove block from db: {0}")]
    FailedToRemoveBlock(DatabaseError),
}
