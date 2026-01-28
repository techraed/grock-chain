// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

//! Block structure and related functionalities.

// TODO : temporarily
#![allow(dead_code)]

use crate::{
    codec,
    crypto::Hash256,
    db::Database,
    errors::{BlockChainError, DatabaseError},
    transaction::{self, Transaction},
};
use serde::{Deserialize, Serialize};

/// Applies a valid block to the blockchain.
pub fn apply_block(block: Block, db: &Database) -> Result<(), BlockChainError> {
    for tx in &block.transactions {
        // todo [sab]: get rid of clone
        transaction::execute_tx(tx.clone(), db).map_err(|tx_err| {
            BlockChainError::InvalidTransaction(Box::new((tx.id.inner(), tx_err)))
        })?;
    }

    let block_hash = Block::block_hash(&block)?;

    db.insert_block(block_hash, block)
        .map_err(BlockChainError::FailedToStoreBlock)
}

pub fn revert_block(block_hash: Hash256, db: &Database) -> Result<Block, BlockChainError> {
    let block = db
        .remove_block(block_hash)
        .map_err(BlockChainError::FailedToRemoveBlock)?;

    for tx in block.transactions.iter() {
        // todo [sab]: get rid of clone
        transaction::revert_tx(tx.clone(), db).map_err(|tx_err| {
            BlockChainError::InvalidTransaction(Box::new((tx.id.inner(), tx_err)))
        })?;
    }

    let block_hash = Block::block_hash(&block)?;

    db.remove_block(block_hash)
        .map_err(BlockChainError::FailedToStoreBlock)
}

/// Validates a block.
///
/// Basically checks:
/// 1. Previous block exists in the database.
/// 2. All transactions in the block are valid.
pub fn validate_block(block: &Block, db: &Database) -> Result<(), BlockChainError> {
    match db.block(block.previous_block_hash) {
        Err(DatabaseError::BlockNotFound(_)) => {
            return Err(BlockChainError::OrphanBlockReceived(
                block.previous_block_hash,
            ));
        }
        Err(db_err) => {
            return Err(BlockChainError::CannotGetBlock(
                block.previous_block_hash,
                db_err,
            ));
        }
        _ => {}
    };

    for tx in &block.transactions {
        transaction::validate_tx(tx, db).map_err(|tx_err| {
            BlockChainError::InvalidTransaction(Box::new((tx.id.inner(), tx_err)))
        })?;
    }

    Ok(())
}

/// Block structure representing a single block in the blockchain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    pub previous_block_hash: Hash256,
    pub block_number: u32,
    pub block_timestamp: u64,
    pub transactions: Vec<Transaction>,
}

impl Block {
    pub fn new(
        previous_block_hash: Hash256,
        db: &Database,
        tx_mempool: &mut Vec<Transaction>,
    ) -> Result<Self, BlockChainError> {
        let previous_block = db
            .block(previous_block_hash)
            .map_err(|db_err| BlockChainError::CannotGetBlock(previous_block_hash, db_err))?;
        let block_number = previous_block.block_number.saturating_add(1);

        let transactions = if tx_mempool.is_empty() {
            Vec::new()
        } else {
            // First 10 transactions from mempool
            let end = tx_mempool.len().min(10);
            tx_mempool.drain(0..end).collect()
        };

        Ok(Self {
            previous_block_hash,
            block_number,
            block_timestamp: now_secs(),
            transactions,
        })
    }

    pub fn block_hash(block: &Self) -> Result<Hash256, BlockChainError> {
        let block_bytes =
            codec::encode(block).map_err(BlockChainError::FailedBlockSerialization)?;

        Ok(Hash256::new(block_bytes))
    }
}

fn now_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}
