# Rationale

The core goal of this project is to implement a semi-toy blockchain that is close to real-world blockchains in its practical characteristics, in order to encounter in practice the limitations that have driven the evolution of new technologies and approaches in production-grade blockchains.

Practical understanding is far more valuable than purely theoretical study of this domain. Moreover, a hands-on approach makes it possible to become familiar with Rust crates that are useful for blockchain development. Therefore, the project is not only educational in nature, but also aimed at professional growth.

An important methodological note: the project is developed iteratively, with implementation taking precedence over specification. In other words, we are not going to first read theory (A. Antonopoulos, for example) and only then start writing code. Instead, we will write code, encounter problems, and then look for solutions that are already described in the literature. This approach enables deeper understanding of the material and provides practical skills.

We would also like to note that this plan will be refined as the project progresses.

# Engage with us
Telegram channel - [@grok_chain](https://t.me/grok_chain)

# Project

### Stage 1 – Implementation of a basic toy blockchain

The purpose of this stage is to implement a minimal baseline blockchain with a large number of shortcomings, but containing the core components:

* Transactions and blocks
* Networking
* Storage

Experienced developers will certainly point out that consensus, an execution engine (VM, smart contracts), RPC, and similar components are also required. Yes, we are aware of this. However, at this stage it is important to implement the most fundamental components—the ones that come to mind first and whose purpose does not require lengthy explanations. It is generally understood that any web service requires a storage layer, a networking layer for interaction, and the application logic itself. In Substrate terminology, the latter is referred to as the runtime. At the first stage, this will consist of simple transactions that modify the blockchain state.

#### Core terms:

1. Transaction

```rust
struct Transaction {
    from: Address,
    to: Address,
    amount: u64,
}
```

2. Block

```rust
struct Block {
    current_block: u32,
    transactions: Vec<Transaction>,
    previous_block: u32
}
```

3. Validity check — only basic validation for balance sufficiency.
4. Data encoding — JSON.

### Stage 2 – Iterative improvement of the blockchain with the addition of real-world characteristics

#### Stage 2.1.
1. Cryptography for authentication and integrity (ECDSA signatures, SHA256 hashing).
- [x] Basic private/public key generation and management.
- [x] Signing transactions.
- [x] Verifying signatures.
2. Transactions UTXO model
- [ ] Basic UTXO structure
- [ ] Transaction creation, validation, execution
3. Persistence layer (using `sled` crate) - SECTION TBD
- [ ] Storing blocks
- [ ] Storing UTXOs
4. Block format
- [ ] Merkle tree for transactions
- [ ] Block header with timestamp, nonce, previous block hash
- [ ] Block verification, application and revocation
5. Networking layer
6. Sync and fork handling
7. Performance and observability

### Stage 3 – Implementation of a virtual machine and smart contracts

At this stage, a simple implementation of a minimal virtual machine and a smart contract language for it is planned. In addition to the obvious benefits, this will allow evaluation of architectural decisions in the design of smart contract languages and their execution environments.

### Stage 4 – Iterative improvement of the virtual machine and smart contracts

### Stage 5 – Experiments with cutting-edge ideas and technologies.

# License

This project is licensed under the Apache License, Version 2.0.
See the `LICENSE` file in the repository root for details.
