// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

mod block;
mod codec;
mod crypto;
mod db;
mod errors;
mod transaction;

use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    hash::Hash,
    net::{TcpListener, TcpStream},
};

pub fn main() {
    let mut accs: HashMap<Address, u64> = HashMap::new();
    let mut blocks: HashMap<u32, Block> = HashMap::new();
    let mut network = Network::new();

    loop {
        transit_state(&mut accs, &mut blocks, &mut network);
    }
}

fn transit_state(
    accs: &mut HashMap<Address, u64>,
    blocks: &mut HashMap<u32, Block>,
    network: &mut Network,
) {
    match network.recv() {
        Ok(message) => match message {
            NetworkMessage::SendTransaction(tx) => {
                println!("Received transaction: {tx:?}");

                // Validate the transaction
                if !validate_transaction(accs, &tx) {
                    eprintln!("Invalid transaction: {tx:?}");

                    return;
                }

                // Execute the transaction
                execute_transaction(accs, &tx);

                // Create a new block with the transaction
                let last_block_number = blocks
                    .values()
                    .max_by_key(|block| block.current_block)
                    .map_or(0, |block| block.current_block);
                let new_block = create_new_block(last_block_number, tx);

                // Add the block to the blocks state
                blocks.insert(new_block.current_block, new_block.clone());

                // Populate the block to peers
                network.send_message(NetworkMessage::AddBlock(new_block));
            }
            NetworkMessage::AddBlock(block) => {
                println!("Received block: {block:?}");

                // Validate the block
                if !validate_block(blocks, accs, &block) {
                    eprintln!("Invalid block: {block:?}");
                    return;
                }

                // Execute the transactions in the block
                for tx in &block.transactions {
                    execute_transaction(accs, tx);
                }

                // Add the block to the state
                blocks.insert(block.current_block, block.clone());

                // Populate the block to peers
                network.send_message(NetworkMessage::AddBlock(block));
            }
        },
        Err(e) => {
            eprintln!("Network error: {e}");
        }
    }
}

fn validate_transaction(accs: &HashMap<Address, u64>, tx: &Transaction) -> bool {
    let from_balance = accs.get(&tx.from).copied().unwrap_or(1_000); // Default balance for new addresses

    from_balance >= tx.amount
}

fn execute_transaction(accs: &mut HashMap<Address, u64>, tx: &Transaction) {
    let from_balance = accs.get(&tx.from).copied().unwrap_or(1_000);
    let to_balance = accs.get(&tx.to).copied().unwrap_or(0);

    accs.insert(tx.from, from_balance.saturating_sub(tx.amount));
    accs.insert(tx.to, to_balance.saturating_add(tx.amount));
}

fn create_new_block(last_block_number: u32, transaction: Transaction) -> Block {
    let mut block = Block::empty(last_block_number);
    block.add_transaction(transaction);

    block
}

fn validate_block(
    blocks: &HashMap<u32, Block>,
    accs: &HashMap<Address, u64>,
    block: &Block,
) -> bool {
    let actual_last_block = blocks.values().max_by_key(|b| b.current_block);

    // Check proper block numbering
    if let Some(actual_last_block) = actual_last_block {
        if actual_last_block.current_block != block.previous_block
            && actual_last_block.current_block + 1 != block.current_block
        {
            return false;
        }
    } else if block.current_block != 1 || block.previous_block != 0 {
        return false;
    }

    // Check transactions
    for tx in &block.transactions {
        if !validate_transaction(accs, tx) {
            return false;
        }
    }

    true
}

/// Id of the network actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Address(u64);

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

/// Blockchain node network handler.
pub struct Network {
    listener: TcpListener,
    peers: Vec<TcpStream>,
}

/// Messages sent over the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    SendTransaction(Transaction),
    AddBlock(Block),
}

impl Network {
    /// Create a new network handler
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind TCP listener");

        println!(
            "Network listening on {}",
            listener.local_addr().expect("Failed to get local address")
        );

        Network {
            listener,
            peers: Vec::new(),
        }
    }

    pub fn send_message(&self, message: NetworkMessage) {
        for peer in &self.peers {
            if let Err(e) = serde_json::to_writer(peer, &message) {
                let peer = peer
                    .peer_addr()
                    .expect("internal error: failed to get peer addr");
                eprintln!("Failed to send message to peer {peer}: {e}");
            }
        }
    }

    /// Run the network handler.
    pub fn recv(&mut self) -> Result<NetworkMessage, String> {
        let Some(stream) = self.listener.incoming().next() else {
            unreachable!("`Incoming` iterator is infinite");
        };

        let stream = stream.map_err(|e| format!("Failed to accept connection: {e}"))?;
        let message = serde_json::from_reader(&stream)
            .map_err(|e| format!("Failed to deserialize message: {e}"))?;

        self.peers.push(stream);

        Ok(message)
    }
}
