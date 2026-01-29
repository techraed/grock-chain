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
pub const HEINLEIN: u64 = 1;

/// Grok is the main currency unit.
///
/// 1 Grok = 1,000,000,000 Heinleins.
/// Heinlein is needed to avoid floating point precision issues.
/// So 0.000000001 Grok = 1 Heinlein.
pub const GROK: u64 = 1_000_000_000 * HEINLEIN;

/// Validates a transaction.
///
/// Basically checks:
/// 1. Inputs and outputs are non-empty.
/// 2. Each input refers to a valid unspent output in the database.
/// 3. Each input's signature is valid for the corresponding output's challenge data.
/// 4. Total output amount is greater than zero.
/// 5. Total input amount is greater than or equal to total output amount
pub fn validate_tx(tx: &Transaction, db: &Database) -> Result<(), TransactionError> {
    let tx_id = tx.id.inner();
    if tx.inputs.is_empty() || tx.outputs.is_empty() {
        return Err(TransactionError::EmptyTransaction(tx_id));
    }

    let mut total_input_sum = 0u64;
    for input in &tx.inputs {
        let amount = validate_input(input, db)?;
        total_input_sum = total_input_sum.saturating_add(amount);
    }

    let total_output_sum = tx.outputs.iter().map(|output| output.amount).sum::<u64>();

    if total_output_sum == 0 {
        return Err(TransactionError::ZeroOutputTransaction(tx_id));
    }

    if total_input_sum < total_output_sum {
        return Err(TransactionError::InsufficientFunds(tx_id));
    }

    Ok(())
}

/// Validates particular transaction input.
///
/// Basically checks:
/// 1. Input refers to a valid unspent output in the database.
/// 2. Input's signature is valid for the corresponding output's challenge data.
fn validate_input(input: &TransactionInput, db: &Database) -> Result<u64, TransactionError> {
    let output_id = input.tx_id.inner();
    let output_idx = input.idx;
    let output = db.tx_output(output_id, output_idx).map_err(|db_err| {
        TransactionError::TransactionOutputNotFound(output_id, output_idx, db_err)
    })?;

    input
        .signature
        .verify(output.challenge.as_ref(), &output.ownership)
        .map_err(|crypto_err| {
            TransactionError::InvalidSignature(input.tx_id.inner(), crypto_err)
        })?;

    Ok(output.amount)
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
/// transaction sender, which can be redeemed by the recipients specified in the outputs. The recipient
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
    use crate::crypto::{PrivateKey, PublicKey};
    use k256::elliptic_curve::rand_core::{OsRng, RngCore};

    fn create_valid_output(
        amount: u64,
        public_key: Option<PublicKey>,
        challenge: [u8; 32],
    ) -> TransactionOutput {
        let public_key = public_key.unwrap_or_else(|| {
            let pk = PrivateKey::random();
            PublicKey::from(&pk)
        });

        TransactionOutput {
            amount,
            ownership: public_key,
            challenge,
        }
    }

    // Test that `encode` gives us expected data.
    #[test]
    fn codec_smoke_test() {
        let pk = PrivateKey::random();
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

    #[test]
    fn validate_empty() {
        let db = Database::create_test_db();
        let pk = PrivateKey::random();
        let challenge = [1u8; 32];
        let output = create_valid_output(GROK, Some((&pk).into()), challenge);

        // Empty inputs
        let tx = Transaction {
            inputs: vec![],
            outputs: vec![output],
            id: TransactionId(Hash256::new(b"test_tx")),
        };
        let result = validate_tx(&tx, &db);
        assert!(matches!(result, Err(TransactionError::EmptyTransaction(_))));

        // Empty outputs
        let signature = pk.sign(challenge).expect("internal error: signing failed");
        let input = TransactionInput {
            tx_id: TransactionId(Hash256::new(b"prev_tx")),
            idx: 0,
            signature,
        };
        let tx = Transaction {
            inputs: vec![input],
            outputs: vec![],
            id: TransactionId(Hash256::new(b"test_tx")),
        };
        let result = validate_tx(&tx, &db);
        assert!(matches!(result, Err(TransactionError::EmptyTransaction(_))));

        // Both empty
        let tx = Transaction {
            inputs: vec![],
            outputs: vec![],
            id: TransactionId(Hash256::new(b"test_tx")),
        };
        let result = validate_tx(&tx, &db);
        assert!(matches!(result, Err(TransactionError::EmptyTransaction(_))));
    }

    #[test]
    fn validate_missing_output_in_db() {
        let pk = PrivateKey::random();
        let challenge = [3u8; 32];
        let signature = pk.sign(challenge).unwrap();

        let output_idx = TransactionId(Hash256::new(b"prev_tx"));
        let output = create_valid_output(GROK, Some((&pk).into()), challenge);
        let input = TransactionInput {
            tx_id: output_idx,
            idx: 0,
            signature: signature.clone(),
        };

        let tx = Transaction {
            inputs: vec![input],
            outputs: vec![output.clone()],
            id: TransactionId(Hash256::new(b"test_tx")),
        };

        let db = Database::create_test_db();
        let result = validate_tx(&tx, &db);
        assert!(matches!(
            result,
            Err(TransactionError::TransactionOutputNotFound(_, _, _))
        ));

        // Insert the output and test again with wrong index
        db.insert_tx_output(output_idx.inner(), 0, &output)
            .expect("Failed to insert output");
        let input = TransactionInput {
            tx_id: output_idx,
            idx: 1, // wrong index
            signature,
        };
        let tx = Transaction {
            inputs: vec![input],
            outputs: vec![output],
            id: TransactionId(Hash256::new(b"test_tx")),
        };
        let result = validate_tx(&tx, &db);
        assert!(matches!(
            result,
            Err(TransactionError::TransactionOutputNotFound(_, _, _))
        ));
    }

    #[test]
    fn validate_zero_output_amount() {
        let db = Database::create_test_db();
        let pk = PrivateKey::random();
        let challenge = [0u8; 32];
        let signature = pk.sign(challenge).unwrap();

        // Create and store a valid output
        let prev_output = create_valid_output(GROK, Some((&pk).into()), challenge);
        let prev_tx_id = Hash256::new(b"prev_tx");
        db.insert_tx_output(prev_tx_id, 0, &prev_output)
            .expect("Failed to insert output");

        let input = TransactionInput {
            tx_id: TransactionId(prev_tx_id),
            idx: 0,
            signature,
        };

        // Create transaction with zero total output amount
        let tx = Transaction {
            inputs: vec![input],
            outputs: vec![create_valid_output(0, None, [0u8; 32])],
            id: TransactionId(Hash256::new(b"test_tx")),
        };

        let result = validate_tx(&tx, &db);
        assert!(matches!(
            result,
            Err(TransactionError::ZeroOutputTransaction(_))
        ));
    }

    #[test]
    fn validate_insufficient_funds() {
        let db = Database::create_test_db();
        let pk = PrivateKey::random();
        let challenge = [0u8; 32];
        let signature = pk.sign(challenge).unwrap();

        // Create and store an output with 1 GROK
        let prev_output = create_valid_output(GROK, Some((&pk).into()), challenge);
        let prev_tx_id = Hash256::new(b"prev_tx");
        db.insert_tx_output(prev_tx_id, 0, &prev_output)
            .expect("Failed to insert output");

        let input = TransactionInput {
            tx_id: TransactionId(prev_tx_id),
            idx: 0,
            signature,
        };

        // Create transaction trying to spend 2 GROK (only have 1)
        let tx = Transaction {
            inputs: vec![input],
            outputs: vec![create_valid_output(2 * GROK, None, [0u8; 32])],
            id: TransactionId(Hash256::new(b"test_tx")),
        };

        let result = validate_tx(&tx, &db);
        assert!(matches!(
            result,
            Err(TransactionError::InsufficientFunds(_))
        ));
    }

    #[test]
    fn validate_invalid_signature() {
        let db = Database::create_test_db();

        // Create first valid input
        let pk1 = PrivateKey::random();
        let challenge1 = [1u8; 32];
        let signature1 = pk1.sign(challenge1).unwrap();

        let prev_output1 = create_valid_output(GROK, Some((&pk1).into()), challenge1);
        let prev_tx_id1 = Hash256::new(b"prev_tx_1");
        db.insert_tx_output(prev_tx_id1, 0, &prev_output1)
            .expect("Failed to insert output");

        let input1 = TransactionInput {
            tx_id: TransactionId(prev_tx_id1),
            idx: 0,
            signature: signature1,
        };

        // Create second input with wrong signature (pk2 signs challenge1, but stored output has different ownership)
        let pk2 = PrivateKey::random();
        let wrong_signature = pk2.sign([2u8; 32]).unwrap(); // Wrong signature: signed different data

        let prev_output2 = create_valid_output(GROK, Some((&pk2).into()), [10u8; 32]); // Different data
        let prev_tx_id2 = Hash256::new(b"prev_tx_2");
        db.insert_tx_output(prev_tx_id2, 0, &prev_output2)
            .expect("Failed to insert output");

        let input2 = TransactionInput {
            tx_id: TransactionId(prev_tx_id2),
            idx: 0,
            signature: wrong_signature,
        };

        let tx = Transaction {
            inputs: vec![input1, input2],
            outputs: vec![create_valid_output(GROK, None, [0u8; 32])],
            id: TransactionId(Hash256::new(b"test_tx")),
        };

        let result = validate_tx(&tx, &db);
        assert!(matches!(
            result,
            Err(TransactionError::InvalidSignature(_, _))
        ));
    }
}
