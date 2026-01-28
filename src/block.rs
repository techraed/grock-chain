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
    transaction::{self, Transaction, TransactionId, TransactionOutput},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Applies a valid block to the blockchain.
pub fn apply_block(
    block: Block,
    db: &Database,
    recovery_store: &mut BTreeMap<Hash256, BlockRecoveryData>,
) -> Result<(), BlockChainError> {
    let block_hash = Block::block_hash(&block)?;

    for tx in &block.transactions {
        apply_tx(block_hash, tx, db, recovery_store)?;
    }

    db.insert_block(block_hash, block)
        .map_err(BlockChainError::FailedToStoreBlock)
}

fn apply_tx(
    block_hash: Hash256,
    tx: &Transaction,
    db: &Database,
    recovery_store: &mut BTreeMap<Hash256, BlockRecoveryData>,
) -> Result<(), BlockChainError> {
    for input in &tx.inputs {
        let output_id = input.tx_id;
        let output_index = input.idx;

        let output = db
            .remove_tx_output(output_id.inner(), output_index)
            .map_err(BlockChainError::FailedToRemoveTransactionOutput)?;

        let data = recovery_store
            .entry(block_hash)
            .or_default();

        data.outputs.push((output_id, output, output_index));
    }

    for (idx, output) in tx.outputs.iter().enumerate() {
        db.insert_tx_output(tx.id.inner(), idx, output)
            .map_err(BlockChainError::FailedToInsertTransactionOutput)?;
    }

    Ok(())
}

pub fn revert_block(
    block_hash: Hash256,
    db: &Database,
    recovery_store: &mut BTreeMap<Hash256, BlockRecoveryData>,
) -> Result<Block, BlockChainError> {
    let block = db
        .remove_block(block_hash)
        .map_err(BlockChainError::FailedToRemoveBlock)?;

    for tx in &block.transactions {
        revert_tx(tx, db)?;
    }

    let block_recovery = recovery_store
        .remove(&block_hash)
        .unwrap_or_else(|| unreachable!("Block recovery data not found"));

    for (tx_id, output, idx) in block_recovery.outputs {
        db.insert_tx_output(tx_id.inner(), idx, &output)
            .map_err(BlockChainError::FailedToInsertTransactionOutput)?;
    }

    Ok(block)
}

fn revert_tx(tx: &Transaction, db: &Database) -> Result<(), BlockChainError> {
    for idx in 0..tx.outputs.len() {
        db.remove_tx_output(tx.id.inner(), idx)
            .map_err(BlockChainError::FailedToRemoveTransactionOutput)?;
    }

    Ok(())
}

/// Validates a block.
///
/// Basically checks:
/// 1. Previous block exists in the database.
/// 2. All transactions in the block are valid.
pub fn validate_block(block: &Block, db: &Database) -> Result<(), BlockChainError> {
    match db.block(block.previous_block_hash) {
        Err(DatabaseError::BlockNotFound(_)) => Err(BlockChainError::OrphanBlockReceived(
            block.previous_block_hash,
        )),
        Err(db_err) => Err(BlockChainError::CannotGetBlock(
            block.previous_block_hash,
            db_err,
        )),
        _ => {
            for tx in &block.transactions {
                transaction::validate_tx(tx, db).map_err(|tx_err| {
                    BlockChainError::InvalidTransaction(Box::new((tx.id.inner(), tx_err)))
                })?;
            }

            Ok(())
        }
    }
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

/// Block data for block reverting.
///
/// Whenever a block is applied we store a removed
/// from the state data to recover it in case of a reorg.
#[derive(Debug, Clone, Default)]
pub struct BlockRecoveryData {
    outputs: Vec<(TransactionId, TransactionOutput, usize)>,
}
