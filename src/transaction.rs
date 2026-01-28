// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

//! Transactions structure and related functionalities.

// TODO : temporarily
#![allow(dead_code)]

use crate::{
    codec,
    crypto::{Hash256, PublicKey, Signature},
    db::Database,
    errors::TransactionError,
};
use serde::{Deserialize, Serialize};

/// Heinlein is the smallest unit of currency.
///
/// The name is inspired by Robert A. Heinlein, a science fiction author, who
/// actually coined the term "grok", which stands for "to understand deeply and
/// intuitively".
const HEINLEIN: u64 = 1;

/// Grok is the main currency unit.
///
/// 1 Grok = 1,000,000,000 Heinleins.
/// Heinlein is needed to avoid floating point precision issues.
/// So 0.000000001 Grok = 1 Heinlein.
const GROK: u64 = 1_000_000_000 * HEINLEIN;

/// Executes a valid transaction
pub fn execute_tx(tx: Transaction, db: &Database) -> Result<(), TransactionError> {
    let Transaction {
        inputs,
        outputs,
        id,
    } = tx;

    for input in inputs {
        let output_id = input.tx_id.inner();
        let output_idx = input.idx;

        db.remove_tx_output(output_id, output_idx)
            .map_err(|db_err| {
                TransactionError::TransactionOutputNotFound(output_id, output_idx, db_err)
            })?;
    }

    let id = id.inner();
    for (idx, output) in outputs.into_iter().enumerate() {
        db.insert_tx_output(id, idx, &output)
            .map_err(|db_err| TransactionError::FailedToInsertTransactionOutput(id, idx, db_err))?;
    }

    Ok(())
}

pub fn revert_tx(tx: &Transaction, db: &Database) -> Result<(), TransactionError> {
    let Transaction { outputs, id, .. } = tx;

    let id = id.inner();
    for idx in 0..outputs.len() {
        db.remove_tx_output(id, idx)
            .map_err(|db_err| TransactionError::TransactionOutputNotFound(id, idx, db_err))?;
    }

    Ok(())
}

/// Validates a transaction.
///
/// Basically checks:
/// 1. Inputs and outputs are non-empty.
/// 2. Each input refers to a valid unspent output in the database.
/// 3. Each input's signature is valid for the corresponding output's challenge data.
/// 4. Total input amount is greater than or equal to total output amount.
pub fn validate_tx(tx: &Transaction, db: &Database) -> Result<(), TransactionError> {
    let tx_id = tx.id.inner();
    if tx.inputs.is_empty() || tx.outputs.is_empty() {
        return Err(TransactionError::EmptyTransaction(tx_id));
    }

    let mut total_input_sum = 0u64;
    for input in &tx.inputs {
        let output_id = input.tx_id.inner();
        let output_idx = input.idx;
        let output = db.tx_output(output_id, output_idx).map_err(|db_err| {
            TransactionError::TransactionOutputNotFound(output_id, output_idx, db_err)
        })?;

        // TODO: not safe verification (open issue)
        input
            .signature
            .verify(output.challenge.as_ref(), &output.ownership)
            .map_err(|ce| TransactionError::InvalidSignature(tx.id.inner(), ce))?;

        total_input_sum = total_input_sum.saturating_add(output.amount);
    }

    let total_output_sum = tx.outputs.iter().map(|output| output.amount).sum::<u64>();

    if total_input_sum < total_output_sum {
        return Err(TransactionError::InsufficientFunds(tx.id.inner()));
    }

    Ok(())
}

/// Grok chain UTXO transaction.
///
/// A UTXO transaction is basically:
/// - A list of inputs ([`TransactionInput`]);
/// - A list of outputs ([`TransactionOutput`]);
/// - An identifier ([`TransactionId`]).
///
/// UTXO-based transactions can be compared to bills of exchange, or, more precisely, checks from
/// the traditional banking system. Each input in a UTXO transaction represents a claim on a specific
/// output from a previous transaction, similar to how a check represents a claim on funds from a
/// bank account. The outputs of the transaction specify new claims on funds, akin to how a check
/// specifies the amount to be paid to the recipient. Outputs are like new checks created by the
/// transaction sender, which can be redeemed by the recipients specified in the outputs. The receipient
/// then can use those outputs as inputs in future transactions.
///
/// The description above looks like we are designing the processing and transferring system, but where
/// is the actual money? The actual money is represented by the amount field in the outputs. The UTXO transaction
/// system can be compared to the process of purchasing a land. The seller doesn't bring the acres you are buying
/// as it's, obviously, impossible. Instead, the seller provides you with a legal document, signing of which
/// creates a transaction of giving you the ownership of the land. That's the whole idea of any electronic transaction
/// system - you just give ownership of some value to someone else.
///
/// *Note*: transaction outputs primary for the system, as inputs always refer to previous outputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transaction {
    pub inputs: Vec<TransactionInput>,
    pub outputs: Vec<TransactionOutput>,
    pub id: TransactionId,
}

impl Transaction {
    /// Creates a new transaction from inputs and outputs.
    ///
    /// Basically encodes the raw transaction, which consists of inputs and outputs, and computes identifier
    /// over the serialized bytes.
    pub fn new(
        inputs: Vec<TransactionInput>,
        outputs: Vec<TransactionOutput>,
    ) -> Result<Self, TransactionError> {
        let tx_bytes = codec::encode(&(&inputs, &outputs))
            .map_err(TransactionError::FailedRawTransactionSerialization)?;
        let id = TransactionId(Hash256::new(tx_bytes));

        Ok(Self {
            inputs,
            outputs,
            id,
        })
    }
}

/// Transaction identifier.
///
/// A transaction identifier is a 32 bytes hash of the inputs and outputs.
///
/// *Note*: The identifier isn't possible to be instantiated directly, only through
/// creating a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionId(Hash256);

impl TransactionId {
    /// Returns the inner hash.
    pub fn inner(&self) -> Hash256 {
        self.0
    }
}

/// Transaction input.
///
/// Refers to a previous transaction output by its identifier and index.
///
/// The index is the position of the output in the previous transaction's outputs list.
/// The signature is used to prove ownership of the output being spent. It's basically a signature
/// over the challenge data in the transaction output being spent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionInput {
    pub tx_id: TransactionId,
    pub idx: usize,
    pub signature: Signature,
}

/// Transaction output.
///
/// Transaction output is a primary component of UTXO-based transaction systems.
/// It represents a specific amount of currency assigned to a public key (ownership).
/// The challenge field is an arbitrary data that a spender must sign to prove ownership of the output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionOutput {
    pub amount: u64,
    pub ownership: PublicKey,
    pub challenge: [u8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;
    use k256::elliptic_curve::rand_core::{OsRng, RngCore};

    // Test that `encode` gives us expected data.
    #[test]
    fn codec_smoke_test() {
        let pk = crate::crypto::PrivateKey::random();
        let pub_key = PublicKey::from(&pk);
        let signature = pk
            .sign(b"test transaction")
            .expect("internal error: signing failed");

        let tx_input = TransactionInput {
            tx_id: TransactionId(Hash256::new(b"transaction id")),
            idx: 1,
            signature,
        };
        let encoded = codec::encode(&tx_input).expect("internal error: encoding failed");
        let decoded =
            codec::decode::<TransactionInput>(&encoded).expect("internal error: decoding failed");
        assert_eq!(tx_input.tx_id.inner(), decoded.tx_id.inner());
        assert_eq!(tx_input.idx, decoded.idx);
        assert_eq!(tx_input.signature.to_bytes(), decoded.signature.to_bytes());

        let mut random_challenge = [0u8; 32];
        OsRng.fill_bytes(&mut random_challenge);
        let tx_output = TransactionOutput {
            amount: 2 * GROK,
            ownership: pub_key,
            challenge: random_challenge,
        };
        let encoded = codec::encode(&tx_output).expect("internal error: encoding failed");
        let decoded =
            codec::decode::<TransactionOutput>(&encoded).expect("internal error: decoding failed");
        assert_eq!(tx_output.amount, decoded.amount);
        assert_eq!(tx_output.ownership.to_bytes(), decoded.ownership.to_bytes());
        assert_eq!(tx_output.challenge, decoded.challenge);
    }
}
