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
use sled::{
    IVec, Transactional, Tree,
    transaction::{ConflictableTransactionError, TransactionalTree, UnabortableTransactionError},
};

/// A trait to abstract over `sled::Tree` and `sled::transaction::TransactionalTree`.
///
/// This allows writing generic code that works with both transactional and
/// non-transactional tree operations, avoiding duplication of serialization,
/// deserialization, and key handling logic.
trait DbTreeOps {
    type Error;

    fn insert<K, V>(&self, key: K, value: V) -> Result<Option<IVec>, Self::Error>
    where
        K: AsRef<[u8]>,
        V: Into<IVec>;

    fn remove<K>(&self, key: K) -> Result<Option<IVec>, Self::Error>
    where
        K: AsRef<[u8]>;

    fn get<K>(&self, key: K) -> Result<Option<IVec>, Self::Error>
    where
        K: AsRef<[u8]>;
}

impl DbTreeOps for Tree {
    type Error = sled::Error;

    fn insert<K, V>(&self, key: K, value: V) -> Result<Option<IVec>, Self::Error>
    where
        K: AsRef<[u8]>,
        V: Into<IVec>,
    {
        Tree::insert(self, key, value)
    }

    fn remove<K>(&self, key: K) -> Result<Option<IVec>, Self::Error>
    where
        K: AsRef<[u8]>,
    {
        Tree::remove(self, key)
    }

    fn get<K>(&self, key: K) -> Result<Option<IVec>, Self::Error>
    where
        K: AsRef<[u8]>,
    {
        Tree::get(self, key)
    }
}

impl DbTreeOps for TransactionalTree {
    type Error = UnabortableTransactionError;

    fn insert<K, V>(&self, key: K, value: V) -> Result<Option<IVec>, Self::Error>
    where
        K: AsRef<[u8]>,
        V: Into<IVec>,
    {
        // TransactionalTree requires K: Into<IVec>, but our key types ([u8; 32], Vec<u8>)
        // satisfy this bound, so we convert to IVec first
        let key_ivec: IVec = IVec::from(key.as_ref());
        TransactionalTree::insert(self, key_ivec, value)
    }

    fn remove<K>(&self, key: K) -> Result<Option<IVec>, Self::Error>
    where
        K: AsRef<[u8]>,
    {
        // TransactionalTree requires K: Into<IVec>, but our key types ([u8; 32], Vec<u8>)
        // satisfy this bound, so we convert to IVec first
        let key_ivec: IVec = IVec::from(key.as_ref());
        TransactionalTree::remove(self, key_ivec)
    }

    fn get<K>(&self, key: K) -> Result<Option<IVec>, Self::Error>
    where
        K: AsRef<[u8]>,
    {
        TransactionalTree::get(self, key)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatabaseOperation {
    InsertBlock {
        block_hash: Hash256,
        block: Block,
    },
    RemoveBlock {
        block_hash: Hash256,
    },
    InsertTxOutput {
        tx_id: Hash256,
        idx: usize,
        output: TransactionOutput,
    },
    RemoveTxOutput {
        tx_id: Hash256,
        idx: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatabaseOperationOutcome {
    InsertBlock(Hash256),
    RemoveBlock(Block),
    InsertTxOutput(Hash256, usize),
    RemoveTxOutput {
        tx_id: Hash256,
        idx: usize,
        output: TransactionOutput,
    },
}

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

    pub fn blocks_count(&self) -> usize {
        self.blocks_tree.iter().count()
    }

    pub fn transactions_count(&self) -> usize {
        self.txs_tree.iter().count()
    }

    pub fn transactional_ops(
        &self,
        ops: Vec<DatabaseOperation>,
    ) -> Result<Vec<DatabaseOperationOutcome>, DatabaseError> {
        let Database {
            blocks_tree,
            txs_tree,
        } = &self;

        if ops.is_empty() {
            return Ok(Vec::new());
        }

        let run_transaction = |trees: (&Tree, &Tree), ops: Vec<DatabaseOperation>| {
            <(&Tree, &Tree) as Transactional<DatabaseError>>::transaction(
                &trees,
                |(blocks_tree, txs_tree): &(TransactionalTree, TransactionalTree)| {
                    let mut outcomes = Vec::with_capacity(ops.len());
                    for op in &ops {
                        match op {
                            DatabaseOperation::InsertBlock { block_hash, block } => {
                                let block_hash = *block_hash;
                                Self::insert_block_inner(
                                    blocks_tree,
                                    block_hash,
                                    block,
                                    |codec_err| {
                                        DatabaseError::FailedBlockSerialization(
                                            block_hash, codec_err,
                                        )
                                    },
                                    |_| {
                                        DatabaseError::FailedBlockInsertion(
                                            sled::Error::Unsupported("transactional insert".into()),
                                        )
                                    },
                                )?;
                                outcomes.push(DatabaseOperationOutcome::InsertBlock(block_hash));
                            }
                            DatabaseOperation::RemoveBlock { block_hash } => {
                                let block_hash = *block_hash;
                                let block = Self::remove_block_inner(
                                    blocks_tree,
                                    block_hash,
                                    |_| {
                                        DatabaseError::CannotGetBlock(sled::Error::Unsupported(
                                            "transactional remove".into(),
                                        ))
                                    },
                                    || DatabaseError::BlockNotFound(block_hash),
                                    |codec_err| {
                                        DatabaseError::FailedBlockDeserialization(
                                            block_hash, codec_err,
                                        )
                                    },
                                )?;
                                outcomes.push(DatabaseOperationOutcome::RemoveBlock(block));
                            }
                            DatabaseOperation::InsertTxOutput { tx_id, idx, output } => {
                                let (tx_id, idx) = (*tx_id, *idx);
                                Self::insert_tx_output_inner(
                                    txs_tree,
                                    tx_id,
                                    idx,
                                    output,
                                    |codec_err| {
                                        DatabaseError::FailedTxOutputSerialization(tx_id, codec_err)
                                    },
                                    |_| {
                                        DatabaseError::FailedTxOutputInsertion(
                                            sled::Error::Unsupported("transactional insert".into()),
                                        )
                                    },
                                )?;
                                outcomes.push(DatabaseOperationOutcome::InsertTxOutput(tx_id, idx));
                            }
                            DatabaseOperation::RemoveTxOutput { tx_id, idx } => {
                                let (tx_id, idx) = (*tx_id, *idx);
                                let output = Self::remove_tx_output_inner(
                                    txs_tree,
                                    tx_id,
                                    idx,
                                    |_| {
                                        DatabaseError::CannotGetTxOutput(sled::Error::Unsupported(
                                            "transactional remove".into(),
                                        ))
                                    },
                                    || DatabaseError::TransactionOutputNotFound(tx_id),
                                    |codec_err| {
                                        DatabaseError::FailedTxOutputDeserialization(
                                            tx_id, codec_err,
                                        )
                                    },
                                )?;
                                outcomes.push(DatabaseOperationOutcome::RemoveTxOutput {
                                    output,
                                    tx_id,
                                    idx,
                                });
                            }
                        }
                    }

                    Ok(outcomes)
                },
            )
        };

        run_transaction((blocks_tree, txs_tree), ops)
            .map_err(|tx_err| DatabaseError::TransactionFailed(Box::new(tx_err)))
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
        Self::insert_block_inner(
            &self.blocks_tree,
            block_hash,
            &block,
            |codec_err| DatabaseError::FailedBlockSerialization(block_hash, codec_err),
            DatabaseError::FailedBlockInsertion,
        )
    }

    pub fn tx_output(
        &self,
        tx_id: Hash256,
        idx: usize,
    ) -> Result<TransactionOutput, DatabaseError> {
        let key = Self::tx_key(tx_id, idx);
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
        Self::insert_tx_output_inner(
            &self.txs_tree,
            tx_id,
            idx,
            output,
            |codec_err| DatabaseError::FailedTxOutputSerialization(tx_id, codec_err),
            DatabaseError::FailedTxOutputInsertion,
        )
    }

    pub fn remove_tx_output(
        &self,
        tx_id: Hash256,
        idx: usize,
    ) -> Result<TransactionOutput, DatabaseError> {
        Self::remove_tx_output_inner(
            &self.txs_tree,
            tx_id,
            idx,
            DatabaseError::CannotGetTxOutput,
            || DatabaseError::TransactionOutputNotFound(tx_id),
            |codec_err| DatabaseError::FailedTxOutputDeserialization(tx_id, codec_err),
        )
    }

    pub fn remove_block(&self, block_hash: Hash256) -> Result<Block, DatabaseError> {
        Self::remove_block_inner(
            &self.blocks_tree,
            block_hash,
            DatabaseError::CannotGetBlock,
            || DatabaseError::BlockNotFound(block_hash),
            |codec_err| DatabaseError::FailedBlockDeserialization(block_hash, codec_err),
        )
    }

    fn tx_key(tx_id: Hash256, idx: usize) -> Vec<u8> {
        [tx_id.to_bytes().as_ref(), &idx.to_be_bytes()].concat()
    }

    /// Helper to insert a block into any tree implementing `DbTreeOps`.
    ///
    /// This extracts the common logic of:
    /// 1. Deriving the key from block_hash
    /// 2. Serializing the block
    /// 3. Inserting into the tree
    ///
    /// Error mapping is delegated to the caller via closures to preserve
    /// context-specific error handling.
    fn insert_block_inner<T, E, SerErr, InsertErr>(
        tree: &T,
        block_hash: Hash256,
        block: &Block,
        map_serialization_err: SerErr,
        map_insert_err: InsertErr,
    ) -> Result<(), E>
    where
        T: DbTreeOps,
        SerErr: FnOnce(crate::errors::CodecError) -> E,
        InsertErr: FnOnce(T::Error) -> E,
    {
        let key = block_hash.to_bytes();
        let block_bytes = codec::encode(block).map_err(map_serialization_err)?;
        let _ = tree.insert(key, block_bytes).map_err(map_insert_err)?;
        Ok(())
    }

    /// Helper to remove a block from any tree implementing `DbTreeOps`.
    ///
    /// This extracts the common logic of:
    /// 1. Deriving the key from block_hash
    /// 2. Removing from the tree
    /// 3. Checking if the block existed
    /// 4. Deserializing the block bytes
    ///
    /// Error mapping is delegated to the caller via closures to preserve
    /// context-specific error handling.
    fn remove_block_inner<T, E, RemoveErr, NotFoundErr, DeserErr>(
        tree: &T,
        block_hash: Hash256,
        map_remove_err: RemoveErr,
        map_not_found_err: NotFoundErr,
        map_deserialization_err: DeserErr,
    ) -> Result<Block, E>
    where
        T: DbTreeOps,
        RemoveErr: FnOnce(T::Error) -> E,
        NotFoundErr: FnOnce() -> E,
        DeserErr: FnOnce(crate::errors::CodecError) -> E,
    {
        let key = block_hash.to_bytes();
        let block_bytes = tree
            .remove(key)
            .map_err(map_remove_err)?
            .ok_or_else(map_not_found_err)?;
        codec::decode::<Block>(block_bytes.as_ref()).map_err(map_deserialization_err)
    }

    /// Helper to insert a transaction output into any tree implementing `DbTreeOps`.
    ///
    /// This extracts the common logic of:
    /// 1. Deriving the key from tx_id and idx
    /// 2. Serializing the output
    /// 3. Inserting into the tree
    ///
    /// Error mapping is delegated to the caller via closures to preserve
    /// context-specific error handling.
    fn insert_tx_output_inner<T, E, SerErr, InsertErr>(
        tree: &T,
        tx_id: Hash256,
        idx: usize,
        output: &TransactionOutput,
        map_serialization_err: SerErr,
        map_insert_err: InsertErr,
    ) -> Result<(), E>
    where
        T: DbTreeOps,
        SerErr: FnOnce(crate::errors::CodecError) -> E,
        InsertErr: FnOnce(T::Error) -> E,
    {
        let key = Self::tx_key(tx_id, idx);
        let output_bytes = codec::encode(output).map_err(map_serialization_err)?;
        let _ = tree.insert(key, output_bytes).map_err(map_insert_err)?;
        Ok(())
    }

    /// Helper to remove a transaction output from any tree implementing `DbTreeOps`.
    ///
    /// This extracts the common logic of:
    /// 1. Deriving the key from tx_id and idx
    /// 2. Removing from the tree
    /// 3. Checking if the output existed
    /// 4. Deserializing the output bytes
    ///
    /// Error mapping is delegated to the caller via closures to preserve
    /// context-specific error handling.
    fn remove_tx_output_inner<T, E, RemoveErr, NotFoundErr, DeserErr>(
        tree: &T,
        tx_id: Hash256,
        idx: usize,
        map_remove_err: RemoveErr,
        map_not_found_err: NotFoundErr,
        map_deserialization_err: DeserErr,
    ) -> Result<TransactionOutput, E>
    where
        T: DbTreeOps,
        RemoveErr: FnOnce(T::Error) -> E,
        NotFoundErr: FnOnce() -> E,
        DeserErr: FnOnce(crate::errors::CodecError) -> E,
    {
        let key = Self::tx_key(tx_id, idx);
        let output_bytes = tree
            .remove(key)
            .map_err(map_remove_err)?
            .ok_or_else(map_not_found_err)?;
        codec::decode::<TransactionOutput>(output_bytes.as_ref()).map_err(map_deserialization_err)
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

impl From<DatabaseError> for ConflictableTransactionError<DatabaseError> {
    fn from(error: DatabaseError) -> Self {
        ConflictableTransactionError::Abort(error)
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

    // Checks transactional operations with 4 scenarios:
    // 1. Successful transaction with multiple operations (changed state)
    // 2. Failed transaction with rollback (no state change)
    // 3. Successful transaction with same operations as 1 (no state change)
    // 4. Successful transaction with only one tree changed (partially changed state)
    #[test]
    fn transactional_ops() {
        let db = Database::create_test_db();

        // Transaction 1: Insert block and tx output (should succeed)
        let block_hash_1 = Hash256::new(b"block_tx_1");
        let block_1 = dummy_block(1, 100);
        let tx_id_1 = Hash256::new(b"tx_txops_1");
        let output_1 = dummy_tx_output(500, 5);

        let ops_1 = vec![
            DatabaseOperation::InsertBlock {
                block_hash: block_hash_1,
                block: block_1.clone(),
            },
            DatabaseOperation::InsertTxOutput {
                tx_id: tx_id_1,
                idx: 0,
                output: output_1.clone(),
            },
        ];

        let res = db.transactional_ops(ops_1.clone());
        assert!(res.is_ok());
        assert_eq!(
            res.expect("checked"),
            vec![
                DatabaseOperationOutcome::InsertBlock(block_hash_1),
                DatabaseOperationOutcome::InsertTxOutput(tx_id_1, 0)
            ]
        );
        assert!(db.block(block_hash_1).is_ok());
        assert!(db.tx_output(tx_id_1, 0).is_ok());
        assert_eq!(db.txs_tree.len(), 1);
        assert_eq!(db.blocks_tree.len(), 1);

        let tx_state_after_tx1 = db.txs_tree.checksum().expect("failed to get checksum");
        let block_state_after_tx1 = db.blocks_tree.checksum().expect("failed to get checksum");

        // Transaction 2: Try to remove non-existent block (should fail and rollback)
        let non_existent_block_hash = Hash256::new(b"nonexistent");
        let tx_id_2 = Hash256::new(b"tx_txops_2");
        let output_2 = dummy_tx_output(1000, 6);

        let ops_2 = vec![
            DatabaseOperation::InsertTxOutput {
                tx_id: tx_id_2,
                idx: 1,
                output: output_2,
            },
            DatabaseOperation::RemoveBlock {
                block_hash: non_existent_block_hash,
            },
        ];

        let res = db.transactional_ops(ops_2);
        assert!(res.is_err());
        let res_err = res.expect_err("checked");
        let DatabaseError::BlockNotFound(block_hash) = res_err
            .transaction_inner_error()
            .expect("Expected getting transaction inner error")
        else {
            panic!("Expected BlockNotFound error");
        };
        assert_eq!(block_hash, &non_existent_block_hash);
        // Check previous state
        let tx_state_after_tx2 = db.txs_tree.checksum().expect("failed to get checksum");
        let block_state_after_tx2 = db.blocks_tree.checksum().expect("failed to get checksum");
        assert_eq!(tx_state_after_tx1, tx_state_after_tx2);
        assert_eq!(block_state_after_tx1, block_state_after_tx2);
        assert_eq!(db.txs_tree.len(), 1);
        assert_eq!(db.blocks_tree.len(), 1);

        assert!(db.block(block_hash_1).is_ok());
        assert!(db.tx_output(tx_id_1, 0).is_ok());

        // Check no new tx output was inserted
        assert!(db.tx_output(tx_id_2, 1).is_err());

        // Transaction 3: Run same transaction as 1 again (should succeed)
        let res = db.transactional_ops(ops_1);
        assert!(res.is_ok());
        assert_eq!(
            res.expect("checked"),
            vec![
                DatabaseOperationOutcome::InsertBlock(block_hash_1),
                DatabaseOperationOutcome::InsertTxOutput(tx_id_1, 0)
            ]
        );

        // Check nothings has changed
        let tx_state_after_tx3 = db.txs_tree.checksum().expect("failed to get checksum");
        let block_state_after_tx3 = db.blocks_tree.checksum().expect("failed to get checksum");
        assert_eq!(tx_state_after_tx1, tx_state_after_tx3);
        assert_eq!(block_state_after_tx1, block_state_after_tx3);
        assert_eq!(db.txs_tree.len(), 1);
        assert_eq!(db.blocks_tree.len(), 1);
        assert!(db.block(block_hash_1).is_ok());
        assert!(db.tx_output(tx_id_1, 0).is_ok());

        // Transaction 4: Insert another block (should succeed)
        let block_4 = dummy_block(3, 300);
        let block_hash_4 = Hash256::new(b"block_hash_3");
        let ops_4 = vec![DatabaseOperation::InsertBlock {
            block_hash: block_hash_4,
            block: block_4,
        }];

        let res = db.transactional_ops(ops_4);
        assert!(res.is_ok());
        assert_eq!(
            res.expect("checked"),
            vec![DatabaseOperationOutcome::InsertBlock(block_hash_4)]
        );

        // Check the state: block state changed, tx state is the same
        let tx_state_after_tx4 = db.txs_tree.checksum().expect("failed to get checksum");
        let block_state_after_tx4 = db.blocks_tree.checksum().expect("failed to get checksum");

        assert_eq!(tx_state_after_tx1, tx_state_after_tx4);
        assert_ne!(block_state_after_tx1, block_state_after_tx4);
        assert!(db.block(block_hash_4).is_ok());
        assert!(db.tx_output(tx_id_1, 0).is_ok());
        assert_eq!(db.blocks_tree.len(), 2);
        assert_eq!(db.txs_tree.len(), 1);
    }

    #[test]
    fn transactional_ops_empty_operations() {
        let db = Database::create_test_db();

        let ops = vec![];
        let result = db.transactional_ops(ops);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }
}
