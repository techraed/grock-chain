// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

//! Crypto module providing cryptographic functionalities.

// TODO: temporarily
#![allow(dead_code)]

use crate::errors::CryptoError;
#[allow(deprecated)]
use k256::{
    PublicKey as K256PublicKey, SecretKey as K256SecretKey,
    ecdsa::{
        Signature as K256Signature, SigningKey, VerifyingKey, signature::hazmat::PrehashVerifier,
    },
    elliptic_curve::{generic_array::GenericArray, rand_core::OsRng, sec1::ToEncodedPoint},
    sha2::{Digest, Sha256},
};
use serde::{Deserialize, Serialize};

/// Private key.
///
/// Basically it's just a number in the range of [0, 2^256 - 1].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrivateKey(K256SecretKey);

impl PrivateKey {
    /// Generates a new random private key.
    pub fn random() -> Self {
        // Cryptographically secure random number generator
        let mut os_rng = OsRng;
        let secret_key = K256SecretKey::random(&mut os_rng);

        PrivateKey(secret_key)
    }

    /// Creates a private key from a 32-byte array.
    pub fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
        #[allow(deprecated)]
        K256SecretKey::from_bytes(GenericArray::from_slice(bytes))
            .ok()
            .map(PrivateKey)
    }

    /// Sign a message with the private key.
    #[allow(deprecated)]
    pub fn sign(&self, message: impl AsRef<[u8]>) -> Result<Signature, CryptoError> {
        let signing_key = SigningKey::from(&self.0);

        let msg_digest = Sha256::digest(message.as_ref());
        signing_key
            .sign_prehash_recoverable(msg_digest.as_slice())
            .map(|(signature, recovery_id)| Signature {
                signature,
                recovery_id: recovery_id.to_byte(),
            })
            .map_err(CryptoError::MessageSigningFailed)
    }
}

/// Public key.
///
/// A public key is a point on the secp256k1 elliptic curve, derived from a private key
/// through scalar multiplication: `PublicKey = PrivateKey × G`, where G is the curve's
/// generator point.
///
/// This is a one-way operation: computing the public key from a private key is easy,
/// but deriving the private key from the public key requires solving the Elliptic Curve
/// Discrete Logarithm Problem (ECDLP), which is computationally infeasible.
///
/// Public keys can be safely shared. They are actively used to verify signatures created by
/// the corresponding private key and to derive blockchain addresses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicKey(K256PublicKey);

impl PublicKey {
    /// Converts the public key to its compressed byte representation.
    ///
    /// Compressed format stores only the x-coordinate (32 bytes) plus a prefix byte (0x02 or 0x03)
    /// indicating whether the y-coordinate is even or odd. This reduces the size from 65 bytes
    /// (uncompressed) to 33 bytes, which is critical for blockchain efficiency where storage
    /// and transmission costs matter.
    ///
    /// The full point can be reconstructed from this compressed form because for any x-coordinate
    /// on the curve, there are at most two valid y-coordinates (differing only in sign).
    pub fn to_bytes(&self) -> [u8; 33] {
        self.0
            .to_encoded_point(true)
            .as_bytes()
            .try_into()
            .expect("secp256k1 compressed public key should be 33 bytes")
    }
}

impl From<&PrivateKey> for PublicKey {
    fn from(private_key: &PrivateKey) -> Self {
        let public_key = private_key.0.public_key();

        PublicKey(public_key)
    }
}

/// ECDSA signature.
///
/// An ECDSA signature proves that the holder of a specific private key signed a message.
/// It consists of two components:
/// - `signature`: Contains the (r, s) values from the ECDSA algorithm
/// - `recovery_id`: A 2-bit value (0-3) that enables recovering the public key from the signature
///
/// The recovery_id encodes:
/// - Bit 0: Whether the y-coordinate of the ephemeral point R was odd (1) or even (0)
/// - Bit 1: Whether the x-coordinate of R exceeded the curve order (extremely rare)
///
/// Using recoverable signatures saves space in blockchain transactions: instead of storing
/// the signature (64 bytes) + public key (33 bytes), we only need signature (64 bytes) +
/// recovery_id (1 byte), saving 32 bytes per transaction.
///
/// The signature is generated deterministically using RFC 6979, which derives the nonce
/// from the private key and message hash, preventing catastrophic nonce reuse vulnerabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    signature: K256Signature,
    recovery_id: u8,
}

impl Signature {
    /// Verifies the signature against the given message and public key.
    ///
    /// This performs ECDSA verification:
    /// 1. Hashes the message using SHA-256
    /// 2. Computes verification points using the signature's (r, s) values
    /// 3. Checks if the computed point matches the signature's r value
    ///
    /// Returns `Ok` if the signature is valid for the given message and public key,
    /// `Err` otherwise. This proves that:
    /// - The message was signed by the holder of the private key corresponding to the public key
    /// - The message has not been tampered with since signing
    ///
    /// Note: This method hashes the message internally.
    #[allow(deprecated)]
    pub fn verify(
        &self,
        message: impl AsRef<[u8]>,
        public_key: &PublicKey,
    ) -> Result<(), CryptoError> {
        let verifying_key = VerifyingKey::from(&public_key.0);

        let msg_digest = Sha256::digest(message);
        verifying_key
            .verify_prehash(msg_digest.as_slice(), &self.signature)
            .map_err(CryptoError::MessageVerificationFailed)
    }
}
