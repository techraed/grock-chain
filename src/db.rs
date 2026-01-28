// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

//! Database functionalities for Grok Chain.

// TODO:
// 1. DB is dependant on sled right now. Abstract it away later.

// TODO: temporarily
#![allow(dead_code)]

use crate::{
    block::Block, codec, crypto::Hash256, errors::DatabaseError, transaction::TransactionOutput,
};

pub struct Database {
    blocks_tree: sled::Tree,
    txs_tree: sled::Tree,
}

impl Database {
    pub fn new(path: &str) -> Result<Self, DatabaseError> {
        let db = sled::open(path).map_err(DatabaseError::OpenFailed)?;

        let blocks_tree = db.open_tree("blocks").map_err(DatabaseError::OpenFailed)?;
        let txs_tree = db
            .open_tree("tx_outputs")
            .map_err(DatabaseError::OpenFailed)?;

        Ok(Self {
            blocks_tree,
            txs_tree,
        })
    }

    pub fn block(&self, block_hash: Hash256) -> Result<Block, DatabaseError> {
        let key = block_hash.to_bytes();
        let block_bytes = self
            .blocks_tree
            .get(key)
            .map_err(DatabaseError::CannotGetBlock)?
            .ok_or(DatabaseError::BlockNotFound(block_hash))?;

        codec::decode::<Block>(block_bytes.as_ref())
            .map_err(|codec_err| DatabaseError::FailedBlockDeserialization(block_hash, codec_err))
    }

    pub fn insert_block(&self, block_hash: Hash256, block: Block) -> Result<(), DatabaseError> {
        let key = block_hash.to_bytes();
        let block_bytes = codec::encode(&block)
            .map_err(|codec_err| DatabaseError::FailedBlockSerialization(block_hash, codec_err))?;

        self.blocks_tree
            .insert(key, block_bytes)
            .map_err(DatabaseError::FailedBlockInsertion)?;

        Ok(())
    }

    pub fn tx_output(
        &self,
        tx_id: Hash256,
        idx: usize,
    ) -> Result<TransactionOutput, DatabaseError> {
        let key = self.tx_key(tx_id, idx);
        let tx_output_bytes = self
            .txs_tree
            .get(key)
            .map_err(DatabaseError::CannotGetTxOutput)?
            .ok_or(DatabaseError::TransactionOutputNotFound(tx_id))?;

        codec::decode::<TransactionOutput>(tx_output_bytes.as_ref())
            .map_err(|codec_err| DatabaseError::FailedTxOutputDeserialization(tx_id, codec_err))
    }

    pub fn insert_tx_output(
        &self,
        tx_id: Hash256,
        idx: usize,
        output: &TransactionOutput,
    ) -> Result<(), DatabaseError> {
        let key = self.tx_key(tx_id, idx);
        let output_bytes = codec::encode(output)
            .map_err(|codec_err| DatabaseError::FailedTxOutputSerialization(tx_id, codec_err))?;

        self.txs_tree
            .insert(key, output_bytes)
            .map_err(DatabaseError::FailedTxOutputInsertion)?;
        Ok(())
    }

    pub fn remove_tx_output(
        &self,
        tx_id: Hash256,
        idx: usize,
    ) -> Result<TransactionOutput, DatabaseError> {
        let key = self.tx_key(tx_id, idx);
        let output_bytes = self
            .txs_tree
            .remove(key)
            .map_err(DatabaseError::CannotGetTxOutput)?
            .ok_or(DatabaseError::TransactionOutputNotFound(tx_id))?;

        codec::decode::<TransactionOutput>(output_bytes.as_ref())
            .map_err(|codec_err| DatabaseError::FailedTxOutputDeserialization(tx_id, codec_err))
    }

    pub fn remove_block(&self, block_hash: Hash256) -> Result<Block, DatabaseError> {
        let key = block_hash.to_bytes();
        let v = self
            .blocks_tree
            .remove(key)
            .map_err(DatabaseError::CannotGetBlock)?
            .ok_or(DatabaseError::BlockNotFound(block_hash))?;

        codec::decode::<Block>(v.as_ref())
            .map_err(|codec_err| DatabaseError::FailedBlockDeserialization(block_hash, codec_err))
    }

    fn tx_key(&self, tx_id: Hash256, idx: usize) -> Vec<u8> {
        [tx_id.to_bytes().as_ref(), &idx.to_be_bytes()].concat()
    }
}

#[cfg(test)]
impl Database {
    pub fn create_test_db() -> Database {
        let db = sled::Config::new()
            .temporary(true)
            .open()
            .expect("Failed to create temp db");

        let blocks_tree = db.open_tree("blocks").expect("Failed to open blocks tree");
        let txs_tree = db
            .open_tree("tx_outputs")
            .expect("Failed to open tx_outputs tree");

        Database {
            blocks_tree,
            txs_tree,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{PrivateKey, PublicKey};

    // Helper to create a dummy block
    fn dummy_block(bn: u32, timestamp: u64) -> Block {
        Block {
            previous_block_hash: Hash256::new(b"prev"),
            block_number: bn,
            block_timestamp: timestamp,
            transactions: vec![],
        }
    }

    // Helper to create a dummy transaction output with custom params
    fn dummy_tx_output(amount: u64, key_seed: u8) -> TransactionOutput {
        let private_key = PrivateKey::from_bytes(&[key_seed; 32]).unwrap();
        TransactionOutput {
            amount,
            ownership: PublicKey::from(&private_key),
            challenge: [0u8; 32],
        }
    }

    #[test]
    fn block_access() {
        let db = Database::create_test_db();

        // Test 1: non-existent block
        let hash = Hash256::new(b"nonexistent");
        let result = db.block(hash);
        assert!(matches!(result, Err(DatabaseError::BlockNotFound(_))));

        // Test 2: invalid bytes
        let hash = Hash256::new(b"invalid");
        db.blocks_tree
            .insert(hash.to_bytes(), b"invalid_data")
            .expect("Failed to insert");
        let result = db.block(hash);
        assert!(result.is_err());

        // Test 3: existent block
        let hash = Hash256::new(b"block1");
        let block = dummy_block(1, 100);
        db.insert_block(hash, block.clone())
            .expect("Failed to insert block");
        let retrieved = db.block(hash).expect("Failed to retrieve block");
        assert_eq!(retrieved.block_number, block.block_number);
        assert_eq!(retrieved.block_timestamp, block.block_timestamp);
    }

    #[test]
    fn block_remove_and_inaccessible() {
        let db = Database::create_test_db();
        let hash = Hash256::new(b"block2");
        let block = dummy_block(1, 100);

        db.insert_block(hash, block)
            .expect("Failed to insert block");
        db.remove_block(hash).expect("Failed to remove block");

        let result = db.block(hash);
        assert!(matches!(result, Err(DatabaseError::BlockNotFound(_))));
    }

    #[test]
    fn block_reset_on_overwrite() {
        let db = Database::create_test_db();
        let hash = Hash256::new(b"block3");

        let block1 = dummy_block(1, 100);
        let block2 = dummy_block(2, 100);

        db.insert_block(hash, block1)
            .expect("Failed to insert block1");
        db.insert_block(hash, block2)
            .expect("Failed to insert block2");

        let retrieved = db.block(hash).expect("Failed to retrieve block");
        assert_eq!(retrieved.block_number, 2);
    }

    #[test]
    fn tx_access() {
        let db = Database::create_test_db();

        // Test 1: non-existent tx hash
        let tx_id = Hash256::new(b"nonexistent_tx");
        let result = db.tx_output(tx_id, 0);
        assert!(matches!(
            result,
            Err(DatabaseError::TransactionOutputNotFound(_))
        ));

        // Test 2: invalid bytes
        let tx_id = Hash256::new(b"tx_invalid");
        let key = [tx_id.to_bytes().as_ref(), &0usize.to_be_bytes()].concat();
        db.txs_tree
            .insert(key, b"invalid_tx_data")
            .expect("Failed to insert");
        let result = db.tx_output(tx_id, 0);
        assert!(result.is_err());

        // Test 3: valid existent tx
        let tx_id = Hash256::new(b"tx_valid");
        let output = dummy_tx_output(1000, 1);
        db.insert_tx_output(tx_id, 0, &output)
            .expect("Failed to insert tx output");
        let retrieved = db.tx_output(tx_id, 0).expect("Failed to retrieve tx");
        assert_eq!(retrieved.amount, output.amount);
        assert_eq!(retrieved.challenge, output.challenge);

        // Test 4: non-existent index (correct tx_id, wrong idx)
        let result = db.tx_output(tx_id, 1);
        assert!(matches!(
            result,
            Err(DatabaseError::TransactionOutputNotFound(_))
        ));
    }

    #[test]
    fn tx_remove_and_inaccessible() {
        let db = Database::create_test_db();
        let tx_id = Hash256::new(b"tx4");
        let output = dummy_tx_output(1000, 1);

        db.insert_tx_output(tx_id, 0, &output)
            .expect("Failed to insert tx output");
        assert!(db.remove_tx_output(tx_id, 1).is_err());
        assert!(db.tx_output(tx_id, 0).is_ok());

        db.remove_tx_output(tx_id, 0)
            .expect("Failed to remove tx output");

        let result = db.tx_output(tx_id, 0);
        assert!(matches!(
            result,
            Err(DatabaseError::TransactionOutputNotFound(_))
        ));
    }

    #[test]
    fn tx_reset_on_overwrite() {
        let db = Database::create_test_db();
        let tx_id = Hash256::new(b"tx5");

        let output1 = dummy_tx_output(1000, 1);
        let output2 = dummy_tx_output(2000, 2);

        db.insert_tx_output(tx_id, 0, &output1)
            .expect("Failed to insert tx output1");
        db.insert_tx_output(tx_id, 0, &output2)
            .expect("Failed to insert tx output2");

        let retrieved = db.tx_output(tx_id, 0).expect("Failed to retrieve tx");
        assert_eq!(retrieved.amount, 2000);
    }
}
