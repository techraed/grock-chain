// Copyright 2025-2026 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

//! Block structure and related functionalities.

// TODO : temporarily
#![allow(dead_code)]

use crate::{
    codec,
    crypto::Hash256,
    db::{Database, DatabaseOperation, DatabaseOperationOutcome},
    errors::{BlockChainError, DatabaseError},
    transaction::{self, Transaction, TransactionOutput},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Applies a valid block to the blockchain.
///
/// This function processes a block and updates the blockchain state by:
/// 1. Applying all transactions in the block (removing spent outputs, creating new outputs)
/// 2. Storing the block itself in the database
/// 3. Recording recovery data for potential blockchain reorganizations
///
/// # Functional Design and Transactional Approach
///
/// This function follows a **functional programming pattern** where state mutations are
/// deferred until the very end. Rather than mutating the database incrementally as each
/// transaction is processed, we:
///
/// 1. **Collect operations**: Build a complete list of all database operations needed
/// 2. **Execute atomically**: Apply all operations in a single database transaction
/// 3. **All-or-nothing semantics**: Either all operations succeed or none do
///
/// ## Why This Design?
///
/// **Atomicity Guarantees**: Blockchain state must remain consistent. If any transaction
/// in a block is invalid or fails to apply, the entire block application must be rolled back.
/// By collecting all operations first and executing them in a single transaction, we ensure
/// that partial state changes never persist.
///
/// **Crash Recovery**: If the system crashes during block application, the database transaction
/// ensures no partial state is written. The blockchain remains in a consistent state at block
/// boundaries.
///
/// **Reorganization Support**: When blockchain forks occur, we need to revert blocks and apply
/// alternative chains. The recovery store captures all removed data (spent outputs) during
/// application, enabling precise reversal of state changes during reorganizations.
///
/// **Performance**: Batching operations reduces database round-trips and allows the underlying
/// database (sled) to optimize the transaction execution.
///
/// # Arguments
///
/// * `block` - The block to apply to the blockchain
/// * `db` - Database instance containing blockchain state
/// * `recovery_store` - Map storing recovery data for each applied block, indexed by block hash.
///   When a block is applied, any outputs removed from the UTXO set are stored here to enable
///   block reversal during reorganizations.
///
/// # Returns
///
/// * `Ok(())` - Block successfully applied
/// * `Err(BlockChainError)` - Block application failed (database error, serialization failure)
///
/// # Example Flow
///
/// ```text
/// Block with 2 transactions:
///   TX1: spends output A, creates outputs B and C
///   TX2: spends output D, creates output E
///
/// Operations collected:
///   1. Remove output A (from TX1 input)
///   2. Insert output B (from TX1 output 0)
///   3. Insert output C (from TX1 output 1)
///   4. Remove output D (from TX2 input)
///   5. Insert output E (from TX2 output 0)
///   6. Insert block
///
/// Single database transaction executes all 6 operations atomically.
/// Recovery store records: [A, D] for potential reversion.
/// ```
pub fn apply_block(
    block: Block,
    db: &Database,
    recovery_store: &mut BTreeMap<Hash256, BlockRecoveryData>,
) -> Result<(), BlockChainError> {
    let block_hash = Block::block_hash(&block)?;

    let mut data_operations = Vec::new();

    // Apply all transactions in the block
    for tx in &block.transactions {
        let mut ops = apply_tx(tx)?;
        data_operations.append(&mut ops);
    }

    // Finally, insert the block itself
    data_operations.push(DatabaseOperation::InsertBlock { block_hash, block });

    // Execute all operations in a single transaction
    let outcomes = db
        .transactional_ops(data_operations)
        .map_err(|db_err| BlockChainError::FailedToApplyBlock(block_hash, db_err))?;

    // TODO: issue #11
    // Store removed data for recovery in case of a reorg.
    for outcome in outcomes {
        let DatabaseOperationOutcome::RemoveTxOutput { tx_id, idx, output } = outcome else {
            continue;
        };

        let data = recovery_store.entry(block_hash).or_default();

        data.outputs.push((tx_id, output, idx));
    }

    Ok(())
}

/// Converts a transaction into database operations for block application.
///
/// This function translates the logical transaction (inputs consuming previous outputs,
/// outputs creating new UTXOs) into concrete database operations. It does not perform
/// any database mutations itself - it simply builds the operation list.
///
/// For each input, generates a remove operation for the spent output.
/// For each output, generates an insert operation for the new UTXO.
///
/// # Arguments
///
/// * `tx` - The transaction to process
///
/// # Returns
///
/// * `Ok(Vec<DatabaseOperation>)` - List of database operations to apply
/// * `Err(BlockChainError)` - Currently never fails, reserved for future validation
fn apply_tx(tx: &Transaction) -> Result<Vec<DatabaseOperation>, BlockChainError> {
    let mut data_operations = Vec::with_capacity(tx.inputs.len() + tx.outputs.len());

    for input in &tx.inputs {
        let output_id = input.tx_id.inner();
        let output_index = input.idx;

        data_operations.push(DatabaseOperation::RemoveTxOutput {
            tx_id: output_id,
            idx: output_index,
        });
    }

    for (idx, output) in tx.outputs.iter().enumerate() {
        // TODO: issue #12
        data_operations.push(DatabaseOperation::InsertTxOutput {
            tx_id: tx.id.inner(),
            idx,
            output: output.clone(),
        });
    }

    Ok(data_operations)
}

/// Reverts a previously applied block from the blockchain.
///
/// This function reverses the state changes made by applying a block, restoring the
/// blockchain to its state before the block was applied. It:
/// 1. Removes all outputs created by transactions in the block
/// 2. Restores all outputs that were spent by the block (from recovery store)
/// 3. Removes the block itself from the database
///
/// # Functional Design and Transactional Approach
///
/// Like `apply_block`, this function follows a **functional, batch-and-commit pattern**:
///
/// 1. **Retrieve block**: Fetch the block data to know which transactions to revert
/// 2. **Collect operations**: Build complete list of removal and restoration operations
/// 3. **Execute atomically**: Apply all operations in a single database transaction
///
/// ## Why This Design?
///
/// **Reorganization Atomicity**: During blockchain reorganizations, we may need to revert
/// multiple blocks and apply an alternative chain. Each block reversion must be atomic -
/// we cannot leave the blockchain in a state where a block is partially reverted.
///
/// **Crash Safety**: If the system crashes mid-reversion, the database transaction ensures
/// that either the block is fully reverted or remains fully applied. No intermediate states
/// can persist.
///
/// **Recovery Data Integrity**: The recovery store contains all data needed to reverse
/// block application. By processing all operations in a single transaction, we ensure that
/// recovery data is used consistently and removed from the store only when reversion succeeds.
///
/// **State Consistency**: The blockchain must maintain UTXO set consistency. Atomic reversion
/// guarantees that spent outputs are restored exactly as they existed before block application,
/// preventing double-spends or missing outputs.
///
/// # Arguments
///
/// * `block_hash` - Hash of the block to revert
/// * `db` - Database instance containing blockchain state
/// * `recovery_store` - Map containing recovery data. The entry for `block_hash` is removed
///   after successful reversion.
///
/// # Returns
///
/// * `Ok(Block)` - The reverted block (removed from database)
/// * `Err(BlockChainError)` - Reversion failed (block not found, database error)
///
/// # Panics
///
/// Panics if recovery data for the block is not found in the recovery store. This indicates
/// a critical invariant violation - blocks should only be reverted if they were previously
/// applied (which creates recovery data).
///
/// # Example Flow
///
/// ```text
/// Block being reverted had 1 transaction:
///   TX1: spent output A, created outputs B and C
///
/// Recovery store contains: [(A, original_data)]
///
/// Operations collected:
///   1. Remove block
///   2. Remove output B (TX1 output 0)
///   3. Remove output C (TX1 output 1)
///   4. Insert output A (restore from recovery store)
///
/// Single database transaction executes all 4 operations atomically.
/// Recovery store entry for block_hash is removed.
/// ```
pub fn revert_block(
    block_hash: Hash256,
    db: &Database,
    recovery_store: &mut BTreeMap<Hash256, BlockRecoveryData>,
) -> Result<Block, BlockChainError> {
    let mut data_operations = Vec::new();

    // The retrieval is a compromise, as there is no way to get the block data
    // from the operations only.
    let block = db
        .block(block_hash)
        .map_err(|db_err| BlockChainError::CannotGetBlock(block_hash, db_err))?;

    // Remove the block itself
    data_operations.push(DatabaseOperation::RemoveBlock { block_hash });

    // Revert all transactions (outputs) in the block
    for tx in &block.transactions {
        let mut ops = revert_tx(tx)?;
        data_operations.append(&mut ops);
    }

    let block_recovery = recovery_store
        .remove(&block_hash)
        .unwrap_or_else(|| unreachable!("Block recovery data not found"));

    // Recover all removed transactions (outputs).
    for (tx_id, output, idx) in block_recovery.outputs {
        data_operations.push(DatabaseOperation::InsertTxOutput { tx_id, idx, output });
    }

    db.transactional_ops(data_operations)
        .map_err(|db_err| BlockChainError::FailedToRevertBlock(block_hash, db_err))?;

    Ok(block)
}

/// Converts a transaction into database operations for block reversion.
///
/// This function generates operations to remove all outputs created by a transaction.
/// It does not handle restoring spent inputs - that's done separately using recovery data
/// in `revert_block`.
///
/// # Arguments
///
/// * `tx` - The transaction to revert
///
/// # Returns
///
/// * `Ok(Vec<DatabaseOperation>)` - List of database operations to revert the transaction
/// * `Err(BlockChainError)` - Currently never fails, reserved for future use
fn revert_tx(tx: &Transaction) -> Result<Vec<DatabaseOperation>, BlockChainError> {
    let mut data_operations = Vec::with_capacity(tx.outputs.len());
    for idx in 0..tx.outputs.len() {
        data_operations.push(DatabaseOperation::RemoveTxOutput {
            tx_id: tx.id.inner(),
            idx,
        });
    }

    Ok(data_operations)
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
    // TODO: use `TransactionId`?
    outputs: Vec<(Hash256, TransactionOutput, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        crypto::{PrivateKey, PublicKey},
        transaction::{GROK, TransactionId, TransactionInput},
    };

    fn prepare_chain() -> (Database, Hash256, TransactionId, TestTxCreator) {
        let db = Database::create_test_db();
        let (tx_creator, genesis_tx) = TestTxCreator::new();
        let genesis_tx_id = genesis_tx.id;

        let genesis_block = Block {
            previous_block_hash: Hash256::new([0u8; 32]),
            block_number: 0,
            block_timestamp: now_secs(),
            transactions: vec![genesis_tx],
        };
        let block_hash = Block::block_hash(&genesis_block).expect("failed to hash genesis block");

        apply_block(genesis_block, &db, &mut Default::default())
            .expect("failed to apply genesis block");

        (db, block_hash, genesis_tx_id, tx_creator)
    }

    fn assert_blocks(db: &Database, blocks: Vec<(Hash256, Block)>) {
        for (block_hash, block) in blocks {
            let db_block = db.block(block_hash).expect("failed to get block from db");
            assert_eq!(db_block, block);
        }
    }

    fn assert_output(
        db: &Database,
        tx_id: TransactionId,
        idx: usize,
        expected_owner: &PublicKey,
        amount: u64,
    ) {
        let output = db
            .tx_output(tx_id.inner(), idx)
            .expect("failed to get tx output");

        assert_eq!(output.amount, amount);
        assert_eq!(&output.ownership, expected_owner);
    }

    /// A helper struct to create test transactions.
    ///
    /// It's main purpose is to manage key pairs and output ownerships to easily
    /// create valid transactions.
    struct TestTxCreator {
        owners: BTreeMap<PublicKey, PrivateKey>,
        outputs: BTreeMap<(TransactionId, usize), (PrivateKey, [u8; 32])>,
    }

    impl TestTxCreator {
        /// Creates a new test transaction creator with a random key pair
        /// and a genesis transaction.
        #[allow(clippy::new_without_default)]
        fn new() -> (Self, Transaction) {
            let pk = PrivateKey::random();
            let pub_key = pk.public_key();

            let mut owners = BTreeMap::new();
            owners.insert(pub_key.clone(), pk.clone());
            let outputs = BTreeMap::new();

            let mut this = Self { owners, outputs };
            let tx = this.create_tx((vec![], vec![(1000 * GROK, pub_key.clone())]));

            (this, tx)
        }

        fn create_acc(&mut self) -> PublicKey {
            let pk = PrivateKey::random();
            let pub_key = pk.public_key();

            self.owners.insert(pub_key.clone(), pk);

            pub_key
        }

        #[allow(clippy::type_complexity)]
        fn create_tx(
            &mut self,
            data: (Vec<(TransactionId, usize)>, Vec<(u64, PublicKey)>),
        ) -> Transaction {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();

            let (input_ids, outputs_data) = data;
            for (input_id, idx) in input_ids {
                let input = self.create_input(input_id, idx);
                inputs.push(input);
            }
            for (amount, to) in outputs_data {
                let output = self.create_output(amount, to);
                outputs.push(output);
            }

            let tx = Transaction::new(inputs, outputs).expect("failed to create transaction");

            for (idx, output) in tx.outputs.iter().enumerate() {
                let pk = self
                    .owners
                    .get(&output.ownership)
                    .expect("owner private key not found")
                    .clone();
                self.outputs.insert((tx.id, idx), (pk, output.challenge));
            }

            tx
        }

        fn create_input(&self, output_id: TransactionId, idx: usize) -> TransactionInput {
            let (pk, output_challenge) = self
                .outputs
                .get(&(output_id, idx))
                .expect("output challenge not found");
            let signature = pk
                .sign(output_challenge)
                .expect("failed to sign output challenge");

            TransactionInput {
                tx_id: output_id,
                idx,
                signature,
            }
        }

        fn create_output(&mut self, amount: u64, to: PublicKey) -> TransactionOutput {
            TransactionOutput {
                amount,
                ownership: to,
                challenge: rand::random(),
            }
        }
    }

    #[test]
    fn apply_revert_basic() {
        let (db, genesis_block_hash, genesis_tx_id, mut tx_creator) = prepare_chain();
        let mut recovery_data = BTreeMap::new();

        let pub_key1 = tx_creator.create_acc();
        let pub_key2 = tx_creator.create_acc();

        // Apply block 1
        let transaction_1 = tx_creator.create_tx((
            vec![(genesis_tx_id, 0)],
            vec![
                (200 * GROK, pub_key1.clone()),
                (200 * GROK, pub_key2.clone()),
                (200 * GROK, pub_key1.clone()),
            ],
        ));
        let transaction_1_id = transaction_1.id;

        let block_1 = Block {
            previous_block_hash: genesis_block_hash,
            block_number: 1,
            block_timestamp: now_secs(),
            transactions: vec![transaction_1],
        };
        let block_1_hash = Block::block_hash(&block_1).expect("failed to hash block 1");

        assert!(apply_block(block_1.clone(), &db, &mut recovery_data).is_ok());

        // Check the state
        assert_output(&db, transaction_1_id, 0, &pub_key1, 200 * GROK);
        assert_output(&db, transaction_1_id, 1, &pub_key2, 200 * GROK);
        assert_output(&db, transaction_1_id, 2, &pub_key1, 200 * GROK);
        assert_blocks(&db, vec![(block_1_hash, block_1.clone())]);
        assert_eq!(db.blocks_count(), 2);
        assert_eq!(db.transactions_count(), 3);

        // Apply block 2
        let transaction_2 = tx_creator.create_tx((
            vec![(transaction_1_id, 0)],
            vec![
                (150 * GROK, pub_key2.clone()),
                (50 * GROK, pub_key1.clone()),
            ],
        ));
        let transaction_2_id = transaction_2.id;
        let transaction_3 = tx_creator.create_tx((
            vec![(transaction_1_id, 1)],
            vec![
                (100 * GROK, pub_key1.clone()),
                (100 * GROK, pub_key2.clone()),
            ],
        ));
        let transaction_3_id = transaction_3.id;

        let block_2 = Block {
            previous_block_hash: block_1_hash,
            block_number: 2,
            block_timestamp: now_secs(),
            transactions: vec![transaction_2, transaction_3],
        };
        let block_2_hash = Block::block_hash(&block_2).expect("failed to hash block 2");
        assert!(apply_block(block_2.clone(), &db, &mut recovery_data).is_ok());

        // Check the state
        assert_blocks(
            &db,
            vec![
                (block_1_hash, block_1.clone()),
                (block_2_hash, block_2.clone()),
            ],
        );
        assert_output(&db, transaction_2_id, 0, &pub_key2, 150 * GROK);
        assert_output(&db, transaction_2_id, 1, &pub_key1, 50 * GROK);
        assert_output(&db, transaction_3_id, 0, &pub_key1, 100 * GROK);
        assert_output(&db, transaction_3_id, 1, &pub_key2, 100 * GROK);
        assert_output(&db, transaction_1_id, 2, &pub_key1, 200 * GROK);
        assert_eq!(db.blocks_count(), 3);
        assert_eq!(db.transactions_count(), 5);

        assert!(db.tx_output(transaction_1_id.inner(), 0).is_err());
        assert!(db.tx_output(transaction_1_id.inner(), 1).is_err());

        // Revert block-2
        let reverted_block =
            revert_block(block_2_hash, &db, &mut recovery_data).expect("failed to revert block 2");
        assert_eq!(reverted_block, block_2);

        // Check the state
        assert_eq!(db.blocks_count(), 2);
        assert_eq!(db.transactions_count(), 3);
        assert_blocks(&db, vec![(block_1_hash, block_1.clone())]);
        assert_output(&db, transaction_1_id, 0, &pub_key1, 200 * GROK);
        assert_output(&db, transaction_1_id, 1, &pub_key2, 200 * GROK);
        assert_output(&db, transaction_1_id, 2, &pub_key1, 200 * GROK);
    }

    #[test]
    fn apply_block_fails() {
        let (db, genesis_block_hash, genesis_tx_id, mut tx_creator) = prepare_chain();
        let mut recovery_data = BTreeMap::new();

        let pub_key1 = tx_creator.create_acc();

        // Apply block 1 successfully
        let transaction_1 = tx_creator.create_tx((
            vec![(genesis_tx_id, 0)],
            vec![(500 * GROK, pub_key1.clone())],
        ));
        let transaction_1_id = transaction_1.id;

        let block_1 = Block {
            previous_block_hash: genesis_block_hash,
            block_number: 1,
            block_timestamp: now_secs(),
            transactions: vec![transaction_1],
        };
        let block_1_hash = Block::block_hash(&block_1).expect("failed to hash block 1");

        assert!(apply_block(block_1.clone(), &db, &mut recovery_data).is_ok());

        // Check the state after block 1
        assert_output(&db, transaction_1_id, 0, &pub_key1, 500 * GROK);
        assert_blocks(&db, vec![(block_1_hash, block_1.clone())]);
        assert_eq!(db.blocks_count(), 2);
        assert_eq!(db.transactions_count(), 1);

        // Try to apply block 2 with a transaction that tries to spend an already-spent output
        let invalid_transaction = tx_creator.create_tx((
            vec![(genesis_tx_id, 0)], // This output was already spent in block 1
            vec![(300 * GROK, pub_key1.clone())],
        ));

        let block_2 = Block {
            previous_block_hash: block_1_hash,
            block_number: 2,
            block_timestamp: now_secs(),
            transactions: vec![invalid_transaction],
        };
        let block_2_hash = Block::block_hash(&block_2).expect("failed to hash block 2");

        // This should fail because the output was already spent
        let result = apply_block(block_2.clone(), &db, &mut recovery_data);
        assert!(result.is_err());
        let BlockChainError::FailedToApplyBlock(block_hash, db_err) = result.unwrap_err() else {
            panic!("Expected InvalidTransaction error");
        };
        assert_eq!(block_hash, block_2_hash);
        let DatabaseError::TransactionOutputNotFound(tx_id) =
            db_err.transaction_inner_error().unwrap()
        else {
            panic!("Expected TransactionOutputNotFound error");
        };
        assert_eq!(tx_id, &genesis_tx_id.inner());

        // Verify that the state remains unchanged (block 2 was not applied)
        assert_eq!(db.blocks_count(), 2);
        assert_eq!(db.transactions_count(), 1);
        assert!(db.block(block_2_hash).is_err());
        assert_output(&db, transaction_1_id, 0, &pub_key1, 500 * GROK);
        assert_blocks(&db, vec![(block_1_hash, block_1.clone())]);
    }

    #[test]
    fn revert_block_fails() {
        let (db, genesis_block_hash, genesis_tx_id, mut tx_creator) = prepare_chain();
        let mut recovery_data = BTreeMap::new();

        let pub_key1 = tx_creator.create_acc();
        let pub_key2 = tx_creator.create_acc();

        // Apply block 1 with multiple outputs
        let transaction_1 = tx_creator.create_tx((
            vec![(genesis_tx_id, 0)],
            vec![
                (400 * GROK, pub_key1.clone()),
                (300 * GROK, pub_key2.clone()),
            ],
        ));
        let transaction_1_id = transaction_1.id;

        let block_1 = Block {
            previous_block_hash: genesis_block_hash,
            block_number: 1,
            block_timestamp: now_secs(),
            transactions: vec![transaction_1],
        };
        let block_1_hash = Block::block_hash(&block_1).expect("failed to hash block 1");

        assert!(apply_block(block_1.clone(), &db, &mut recovery_data).is_ok());

        // Check the state after applying block 1
        assert_output(&db, transaction_1_id, 0, &pub_key1, 400 * GROK);
        assert_output(&db, transaction_1_id, 1, &pub_key2, 300 * GROK);
        assert_blocks(&db, vec![(block_1_hash, block_1.clone())]);
        assert_eq!(db.blocks_count(), 2);
        assert_eq!(db.transactions_count(), 2);

        // Manually remove one of the outputs created by block 1
        // This simulates database corruption or concurrent modification
        db.remove_tx_output(transaction_1_id.inner(), 0)
            .expect("failed to remove output");

        // Verify the output was removed
        assert!(db.tx_output(transaction_1_id.inner(), 0).is_err());
        assert_eq!(db.transactions_count(), 1);

        // Attempt to revert block 1
        // This should fail because one of the outputs that needs to be removed is missing
        let result = revert_block(block_1_hash, &db, &mut recovery_data);

        // The revert should fail with a database error
        assert!(result.is_err());
        let BlockChainError::FailedToRevertBlock(hash, db_err) = result.unwrap_err() else {
            panic!("Expected FailedToRevertBlock error");
        };
        assert_eq!(hash, block_1_hash);
        let DatabaseError::TransactionOutputNotFound(tx_id) =
            db_err.transaction_inner_error().unwrap()
        else {
            panic!("Expected TransactionOutputNotFound error");
        };
        assert_eq!(tx_id, &transaction_1_id.inner());
    }

    /*
       Cases:
       1. apply 3 blocks, revert last one, check block states and tx states
       2. apply 1 block, second apply fails
       3. apply 1 block, revert fails
    */
}
