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
    Transactional, Tree,
    transaction::{ConflictableTransactionError, TransactionalTree},
};

/// Represents a single database operation in the Grok Chain blockchain.
///
/// This enum defines all possible state mutations that can be performed on the blockchain
/// database. Operations are designed to be composable - multiple operations can be batched
/// and executed atomically via `Database::transactional_ops`.
///
/// # Design Philosophy
///
/// Each variant represents a **pure data transformation** without side effects. Operations
/// are first-class values that can be collected, inspected, and transformed before execution.
/// This enables powerful patterns like:
/// - Batching multiple operations for atomic execution
/// - Dry-running operations without database mutations
/// - Building complex multi-step transactions with rollback semantics
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

/// Represents the outcome of a successfully executed database operation.
///
/// Each variant corresponds to a `DatabaseOperation` variant and captures relevant
/// information about what was changed in the database. Outcomes enable:
/// - Recovery mechanisms (storing removed data for potential rollback)
/// - Audit trails (tracking what changed in each transaction)
/// - State verification (confirming operations had expected effects)
///
/// For operations that remove data (like `RemoveTxOutput`), the outcome includes the
/// removed data itself, which is critical for blockchain reorganizations.
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

/// Database for storing blockchain state in the Grok Chain.
///
/// The database uses a key-value store (sled) with two separate trees:
/// - `blocks_tree`: Stores blocks indexed by their hash
/// - `txs_tree`: Stores unspent transaction outputs (UTXO set) indexed by (tx_id, output_index)
///
/// # UTXO Model
///
/// The database implements a UTXO (Unspent Transaction Output) model where:
/// - Transaction outputs are added when blocks are applied
/// - Outputs are removed when spent by subsequent transactions
/// - Only unspent outputs remain in `txs_tree`, forming the current UTXO set
///
/// # Transactional Guarantees
///
/// All multi-operation updates use database transactions to ensure atomicity. Either all
/// operations in a batch succeed, or none do. This is critical for maintaining blockchain
/// consistency during block application, reversion, and reorganizations.
pub struct Database {
    blocks_tree: sled::Tree,
    txs_tree: sled::Tree,
}

impl Database {
    /// Creates a new database instance at the specified path.
    ///
    /// Opens a sled database at the given filesystem path and initializes two trees:
    /// - "blocks": For storing blockchain blocks
    /// - "tx_outputs": For storing the UTXO set
    ///
    /// If the path doesn't exist, a new database is created. If it exists, the existing
    /// database is opened, preserving all previously stored data.
    ///
    /// # Arguments
    ///
    /// * `path` - Filesystem path where the database should be stored
    ///
    /// # Returns
    ///
    /// * `Ok(Database)` - Successfully opened/created database
    /// * `Err(DatabaseError::OpenFailed)` - Failed to open database or trees
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

    /// Returns the total number of blocks stored in the database.
    ///
    /// This count represents all blocks in the blockchain, including the genesis block.
    pub fn blocks_count(&self) -> usize {
        self.blocks_tree.iter().count()
    }

    /// Returns the total number of unspent transaction outputs (UTXOs) in the database.
    ///
    /// This count represents the current UTXO set - all outputs that have been created
    /// but not yet spent. It does not include spent outputs, which are removed from the database.
    pub fn transactions_count(&self) -> usize {
        self.txs_tree.iter().count()
    }

    /// Executes multiple database operations atomically in a single transaction.
    ///
    /// This is the **core primitive** for blockchain state mutations. All operations are
    /// executed within a database transaction, ensuring that either all succeed or none do.
    /// This atomic behavior is essential for maintaining blockchain consistency.
    ///
    /// # Why Transactional Operations?
    ///
    /// Blockchains require **all-or-nothing semantics** for state changes:
    /// - A block is either fully applied or not applied at all
    /// - A reorganization either fully succeeds or leaves the chain unchanged
    /// - Partial state changes would corrupt the blockchain
    ///
    /// By batching operations and executing them transactionally, we achieve:
    ///
    /// **Atomicity**: All operations succeed together or fail together. No partial updates.
    ///
    /// **Consistency**: The blockchain state is always valid at transaction boundaries.
    /// Invalid intermediate states never persist.
    ///
    /// **Isolation**: Concurrent access (future extension) won't see partial updates.
    ///
    /// **Durability**: Once a transaction commits, changes are permanent even if the system crashes.
    ///
    /// # Functional Design Philosophy
    ///
    /// This method embodies a **functional approach** to state management:
    ///
    /// 1. **Separate description from execution**: Operations are data structures describing
    ///    desired changes, not imperative commands that execute immediately.
    ///
    /// 2. **Batch before commit**: Collect all operations first, then execute atomically.
    ///    This enables inspection, validation, and optimization before any mutations occur.
    ///
    /// 3. **Explicit outcomes**: Return structured data about what changed, enabling callers
    ///    to maintain recovery data, logs, or other derived state.
    ///
    /// 4. **Composability**: Operations can be built incrementally from smaller pieces,
    ///    combined, and executed as a single unit of work.
    ///
    /// This design makes blockchain reorganizations tractable - we can collect operations
    /// for reverting blocks and applying new ones, then execute everything atomically.
    ///
    /// # Arguments
    ///
    /// * `ops` - Vector of operations to execute atomically. Can be empty (returns empty outcomes).
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<DatabaseOperationOutcome>)` - All operations succeeded. Outcomes correspond
    ///   to input operations in order.
    /// * `Err(DatabaseError::TransactionFailed)` - Transaction was aborted due to an error.
    ///   Database remains unchanged.
    ///
    /// # Example
    ///
    /// ```text
    /// let ops = vec![
    ///     DatabaseOperation::RemoveTxOutput { tx_id: old_tx, idx: 0 },
    ///     DatabaseOperation::InsertTxOutput { tx_id: new_tx, idx: 0, output: new_output },
    ///     DatabaseOperation::InsertBlock { block_hash: hash, block: block },
    /// ];
    ///
    /// // All three operations execute atomically
    /// let outcomes = db.transactional_ops(ops)?;
    ///
    /// // Extract removed output from outcomes for recovery data
    /// if let DatabaseOperationOutcome::RemoveTxOutput { output, .. } = &outcomes[0] {
    ///     recovery_store.push(output.clone());
    /// }
    /// ```
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

        // TODO: issue #13
        let run_transaction = |trees: (&Tree, &Tree), ops: Vec<DatabaseOperation>| {
            <(&Tree, &Tree) as Transactional<DatabaseError>>::transaction(
                &trees,
                |(blocks_tree, txs_tree): &(TransactionalTree, TransactionalTree)| {
                    let mut outcomes = Vec::with_capacity(ops.len());
                    for op in &ops {
                        match op {
                            DatabaseOperation::InsertBlock { block_hash, block } => {
                                let block_hash = *block_hash;
                                let key = block_hash.to_bytes();
                                let block_bytes = codec::encode(&block).map_err(|codec_err| {
                                    DatabaseError::FailedBlockSerialization(block_hash, codec_err)
                                })?;
                                let _ = blocks_tree.insert(&key, block_bytes)?;

                                outcomes.push(DatabaseOperationOutcome::InsertBlock(block_hash));
                            }
                            DatabaseOperation::RemoveBlock { block_hash } => {
                                let block_hash = *block_hash;
                                let key = block_hash.to_bytes();
                                let block_bytes = blocks_tree
                                    .remove(&key)?
                                    .ok_or(DatabaseError::BlockNotFound(block_hash))?;
                                let block = codec::decode::<Block>(block_bytes.as_ref()).map_err(
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
                                let key = Self::tx_key(tx_id, idx);
                                let output_bytes = codec::encode(&output).map_err(|codec_err| {
                                    DatabaseError::FailedTxOutputSerialization(tx_id, codec_err)
                                })?;

                                let _ = txs_tree.insert(key, output_bytes)?;

                                outcomes.push(DatabaseOperationOutcome::InsertTxOutput(tx_id, idx));
                            }
                            DatabaseOperation::RemoveTxOutput { tx_id, idx } => {
                                let (tx_id, idx) = (*tx_id, *idx);
                                let key = Self::tx_key(tx_id, idx);
                                let output_bytes = txs_tree
                                    .remove(key)?
                                    .ok_or(DatabaseError::TransactionOutputNotFound(tx_id))?;

                                let output =
                                    codec::decode::<TransactionOutput>(output_bytes.as_ref())
                                        .map_err(|codec_err| {
                                            DatabaseError::FailedTxOutputDeserialization(
                                                tx_id, codec_err,
                                            )
                                        })?;
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

    /// Retrieves a block from the database by its hash.
    ///
    /// # Arguments
    ///
    /// * `block_hash` - The hash of the block to retrieve
    ///
    /// # Returns
    ///
    /// * `Ok(Block)` - Block successfully retrieved and deserialized
    /// * `Err(DatabaseError::BlockNotFound)` - No block exists with this hash
    /// * `Err(DatabaseError::CannotGetBlock)` - Database read error
    /// * `Err(DatabaseError::FailedBlockDeserialization)` - Block data is corrupted
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

    /// Inserts a block into the database.
    ///
    /// If a block with the same hash already exists, it is overwritten.
    ///
    /// # Arguments
    ///
    /// * `block_hash` - The hash of the block (used as the database key)
    /// * `block` - The block to store
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Block successfully serialized and stored
    /// * `Err(DatabaseError::FailedBlockSerialization)` - Block serialization failed
    /// * `Err(DatabaseError::FailedBlockInsertion)` - Database write error
    pub fn insert_block(&self, block_hash: Hash256, block: Block) -> Result<(), DatabaseError> {
        let key = block_hash.to_bytes();
        let block_bytes = codec::encode(&block)
            .map_err(|codec_err| DatabaseError::FailedBlockSerialization(block_hash, codec_err))?;

        self.blocks_tree
            .insert(key, block_bytes)
            .map_err(DatabaseError::FailedBlockInsertion)?;

        Ok(())
    }

    /// Retrieves a transaction output from the UTXO set.
    ///
    /// Outputs are indexed by transaction ID and output index. Only unspent outputs
    /// are stored in the database - spent outputs are removed.
    ///
    /// # Arguments
    ///
    /// * `tx_id` - The transaction ID that created this output
    /// * `idx` - The index of the output within the transaction (0-based)
    ///
    /// # Returns
    ///
    /// * `Ok(TransactionOutput)` - Output successfully retrieved and deserialized
    /// * `Err(DatabaseError::TransactionOutputNotFound)` - Output doesn't exist (never created or already spent)
    /// * `Err(DatabaseError::CannotGetTxOutput)` - Database read error
    /// * `Err(DatabaseError::FailedTxOutputDeserialization)` - Output data is corrupted
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

    /// Inserts a transaction output into the UTXO set.
    ///
    /// If an output with the same (tx_id, idx) already exists, it is overwritten.
    ///
    /// # Arguments
    ///
    /// * `tx_id` - The transaction ID that created this output
    /// * `idx` - The index of the output within the transaction
    /// * `output` - The transaction output to store
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Output successfully serialized and stored
    /// * `Err(DatabaseError::FailedTxOutputSerialization)` - Output serialization failed
    /// * `Err(DatabaseError::FailedTxOutputInsertion)` - Database write error
    pub fn insert_tx_output(
        &self,
        tx_id: Hash256,
        idx: usize,
        output: &TransactionOutput,
    ) -> Result<(), DatabaseError> {
        let key = Self::tx_key(tx_id, idx);
        let output_bytes = codec::encode(output)
            .map_err(|codec_err| DatabaseError::FailedTxOutputSerialization(tx_id, codec_err))?;

        self.txs_tree
            .insert(key, output_bytes)
            .map_err(DatabaseError::FailedTxOutputInsertion)?;
        Ok(())
    }

    /// Removes a transaction output from the UTXO set and returns it.
    ///
    /// This is typically used when an output is spent by a transaction. The removed
    /// output is returned so it can be stored in recovery data for potential block reversions.
    ///
    /// # Arguments
    ///
    /// * `tx_id` - The transaction ID that created this output
    /// * `idx` - The index of the output within the transaction
    ///
    /// # Returns
    ///
    /// * `Ok(TransactionOutput)` - Output successfully removed and deserialized
    /// * `Err(DatabaseError::TransactionOutputNotFound)` - Output doesn't exist
    /// * `Err(DatabaseError::CannotGetTxOutput)` - Database operation error
    /// * `Err(DatabaseError::FailedTxOutputDeserialization)` - Output data is corrupted
    pub fn remove_tx_output(
        &self,
        tx_id: Hash256,
        idx: usize,
    ) -> Result<TransactionOutput, DatabaseError> {
        let key = Self::tx_key(tx_id, idx);
        let output_bytes = self
            .txs_tree
            .remove(key)
            .map_err(DatabaseError::CannotGetTxOutput)?
            .ok_or(DatabaseError::TransactionOutputNotFound(tx_id))?;

        codec::decode::<TransactionOutput>(output_bytes.as_ref())
            .map_err(|codec_err| DatabaseError::FailedTxOutputDeserialization(tx_id, codec_err))
    }

    /// Removes a block from the database and returns it.
    ///
    /// This is typically used during blockchain reorganizations when reverting blocks.
    ///
    /// # Arguments
    ///
    /// * `block_hash` - The hash of the block to remove
    ///
    /// # Returns
    ///
    /// * `Ok(Block)` - Block successfully removed and deserialized
    /// * `Err(DatabaseError::BlockNotFound)` - No block exists with this hash
    /// * `Err(DatabaseError::CannotGetBlock)` - Database operation error
    /// * `Err(DatabaseError::FailedBlockDeserialization)` - Block data is corrupted
    pub fn remove_block(&self, block_hash: Hash256) -> Result<Block, DatabaseError> {
        let key = block_hash.to_bytes();
        let block_bytes = self
            .blocks_tree
            .remove(key)
            .map_err(DatabaseError::CannotGetBlock)?
            .ok_or(DatabaseError::BlockNotFound(block_hash))?;

        codec::decode::<Block>(block_bytes.as_ref())
            .map_err(|codec_err| DatabaseError::FailedBlockDeserialization(block_hash, codec_err))
    }

    /// Constructs a database key for a transaction output.
    ///
    /// The key is formed by concatenating the transaction ID (32 bytes) with the
    /// output index (8 bytes, big-endian). This ensures each output has a unique key.
    ///
    /// # Arguments
    ///
    /// * `tx_id` - The transaction ID
    /// * `idx` - The output index
    ///
    /// # Returns
    ///
    /// A 40-byte key (32 bytes tx_id + 8 bytes idx)
    fn tx_key(tx_id: Hash256, idx: usize) -> Vec<u8> {
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
