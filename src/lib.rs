// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// Id of the network actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Address(u64);

impl Address {
    /// Generates a new random address.
    pub fn generate() -> Self {
        use rand::Rng;

        let mut rand = rand::rng();
        let id: u64 = rand.random();

        Address(id)
    }
}

/// A transaction sending value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Transaction {
    from: Address,
    to: Address,
    amount: u64,
}

impl Transaction {
    /// Creates a new transaction.
    pub fn new(from: Address, to: Address, amount: u64) -> Self {
        Transaction { from, to, amount }
    }
}

/// A block containing multiple transactions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Block {
    current_block: u32,
    transactions: Vec<Transaction>,
    previous_block: u32,
}

impl Block {
    /// Creates an empty block.
    pub fn empty(previous_block: u32) -> Self {
        Block {
            current_block: previous_block.saturating_add(1),
            transactions: Vec::new(),
            previous_block,
        }
    }

    /// Adds a transaction to the block.
    pub fn add_transaction(&mut self, transaction: Transaction) {
        self.transactions.push(transaction);
    }
}
