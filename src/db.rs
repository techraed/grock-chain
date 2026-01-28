// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

//! Database functionalities for Grok Chain.

// todo [sab]:
// - txs - separate tree
// db dependant on sled

// TODO: temporarily
#![allow(dead_code)]

use crate::{
    block::Block, codec, crypto::Hash256, errors::DatabaseError, transaction::TransactionOutput,
};

const TX_OUTPUT_KEY_PREFIX: &str = "tx_output";
const BLOCK_KEY_PREFIX: &str = "block";

pub enum DatabaseKeys {
    Block(Hash256),
    TxOutput(Hash256, usize),
}

impl DatabaseKeys {
    pub fn into_db_key(self) -> Vec<u8> {
        match self {
            // "block:{block_hash}"
            DatabaseKeys::Block(block_hash) => [
                BLOCK_KEY_PREFIX.as_bytes(),
                b":",
                block_hash.to_bytes().as_ref(),
            ]
            .concat(),
            // "tx_output:{tx_id}:{idx}"
            DatabaseKeys::TxOutput(tx_id, idx) => [
                TX_OUTPUT_KEY_PREFIX.as_bytes(),
                b":",
                tx_id.to_bytes().as_ref(),
                b":",
                &idx.to_be_bytes(),
            ]
            .concat(),
        }
    }
}

pub struct Database {
    db: sled::Db,
}

impl Database {
    pub fn new(path: &str) -> Result<Self, DatabaseError> {
        let db = sled::open(path).map_err(DatabaseError::OpenFailed)?;

        Ok(Self { db })
    }

    pub fn block(&self, block_hash: Hash256) -> Result<Block, DatabaseError> {
        let key = DatabaseKeys::Block(block_hash).into_db_key();
        let block_bytes = self
            .db
            .get(key)
            .map_err(DatabaseError::CannotGetBlock)?
            .ok_or(DatabaseError::BlockNotFound(block_hash))?;

        codec::decode::<Block>(block_bytes.as_ref())
            .map_err(|codec_err| DatabaseError::FailedBlockDeserialization(block_hash, codec_err))
    }

    pub fn insert_block(&self, block_hash: Hash256, block: Block) -> Result<(), DatabaseError> {
        let key = DatabaseKeys::Block(block_hash).into_db_key();
        let block_bytes = codec::encode(&block)
            .map_err(|codec_err| DatabaseError::FailedBlockSerialization(block_hash, codec_err))?;

        self.db
            .insert(key, block_bytes)
            .map_err(DatabaseError::FailedBlockInsertion)?;

        Ok(())
    }

    pub fn tx_output(
        &self,
        tx_id: Hash256,
        idx: Option<usize>,
    ) -> Result<TransactionOutput, DatabaseError> {
        let idx = idx.unwrap_or(0);
        let key = DatabaseKeys::TxOutput(tx_id, idx).into_db_key();
        let tx_output_bytes = self
            .db
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
        let key = DatabaseKeys::TxOutput(tx_id, idx).into_db_key();
        let output_bytes = codec::encode(output)
            .map_err(|codec_err| DatabaseError::FailedTxOutputSerialization(tx_id, codec_err))?;

        self.db
            .insert(key, output_bytes)
            .map_err(DatabaseError::FailedTxOutputInsertion)?;
        Ok(())
    }

    pub fn remove_tx_output(&self, tx_id: Hash256, idx: usize) -> Result<(), DatabaseError> {
        let key = DatabaseKeys::TxOutput(tx_id, idx).into_db_key();
        self.db
            .remove(key)
            .map_err(DatabaseError::CannotGetTxOutput)?;

        Ok(())
    }

    pub fn remove_block(&self, block_hash: Hash256) -> Result<Block, DatabaseError> {
        let key = DatabaseKeys::Block(block_hash).into_db_key();
        let v = self
            .db
            .remove(key)
            .map_err(DatabaseError::CannotGetBlock)?
            .ok_or(DatabaseError::BlockNotFound(block_hash))?;

        codec::decode::<Block>(v.as_ref())
            .map_err(|codec_err| DatabaseError::FailedBlockDeserialization(block_hash, codec_err))
    }
}
