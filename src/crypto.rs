// Copyright 2025 Sabaun Taraki
// SPDX-License-Identifier: Apache-2.0

//! Crypto module providing cryptographic functionalities.

// TODO: temporarily
#![allow(dead_code, deprecated)]

use crate::errors::CryptoError;
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
        K256SecretKey::from_bytes(GenericArray::from_slice(bytes))
            .ok()
            .map(PrivateKey)
    }

    /// Sign a message with the private key.
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

    /// Derives the corresponding public key from the private key.
    pub fn public_key(&self) -> PublicKey {
        PublicKey::from(self)
    }

    /// Converts the private key to a 32-byte array.
    ///
    /// # Safety
    /// This function exposes the raw bytes of the private key.
    /// Use with caution to avoid leaking sensitive information.
    pub(crate) unsafe fn to_bytes_unchecked(&self) -> [u8; 32] {
        self.0.to_bytes().into()
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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

// Serialize public key to 33-bytes compressed format
impl Serialize for PublicKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as a tuple (fixed-size sequence) to avoid length prefix
        use serde::ser::SerializeTuple;
        let bytes = self.to_bytes();
        let mut tuple = serializer.serialize_tuple(33)?;
        for byte in &bytes {
            tuple.serialize_element(byte)?;
        }
        tuple.end()
    }
}

// Deserialize public key from 33-bytes compressed format
impl<'de> Deserialize<'de> for PublicKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Create a visitor to read a fixed-size array of 33 bytes
        struct ArrayVisitor;
        impl<'de> serde::de::Visitor<'de> for ArrayVisitor {
            type Value = [u8; 33];

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("a 33-byte array")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut arr = [0u8; 33];
                for (i, byte) in arr.iter_mut().enumerate() {
                    *byte = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(arr)
            }
        }

        // Reads a tuple of 33 elements, because the public key was serialized as a tuple of 33 bytes.
        let bytes = deserializer.deserialize_tuple(33, ArrayVisitor)?;

        K256PublicKey::from_sec1_bytes(&bytes)
            .map(PublicKey)
            .map_err(serde::de::Error::custom)
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

    /// Converts the signature to a 65-byte array: 64 bytes for (r, s) + 1 byte for recovery_id.
    pub fn to_bytes(&self) -> [u8; 65] {
        let mut ret = [0u8; 65];
        ret.get_mut(..64)
            .expect("slice is 65 bytes; qed.")
            .copy_from_slice(self.signature.to_bytes().as_slice());
        ret[64] = self.recovery_id;

        ret
    }
}

/// SHA-256 hash wrapper.
///
/// Basically a newtype around a 32-byte array representing a SHA-256 hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Hash256([u8; 32]);

impl Hash256 {
    /// Creates a new SHA-256 hash from the given data.
    pub fn new(data: impl AsRef<[u8]>) -> Self {
        let inner = Sha256::digest(data)
            .as_slice()
            .try_into()
            .expect("sha256 returns 32 bytes; qed.");

        Self(inner)
    }

    /// Returns the inner 32-byte array.
    pub fn to_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Allows creating a `Hash256` directly from a 32-byte array.
impl From<[u8; 32]> for Hash256 {
    fn from(data: [u8; 32]) -> Self {
        Self(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec;

    #[test]
    fn codec_smoke_test() {
        let hash = Hash256::new(b"hello world");
        let encoded = codec::encode(&hash).expect("internal error: encoding failed");
        let bytes = hash.to_bytes().to_vec();
        assert_eq!(encoded, bytes);
        let decoded = codec::decode::<Hash256>(&encoded).expect("internal error: decoding failed");
        assert_eq!(hash, decoded);

        let pk = PrivateKey::random();

        let signature = pk
            .sign(b"test message")
            .expect("internal error: signing failed");
        let encoded = codec::encode(&signature).expect("internal error: encoding failed");
        let bytes = signature.to_bytes().to_vec();
        assert_eq!(encoded, bytes);
        let decoded =
            codec::decode::<Signature>(&encoded).expect("internal error: decoding failed");
        assert_eq!(signature, decoded);

        let pub_key = PublicKey::from(&pk);
        let encoded_pub = codec::encode(&pub_key).expect("internal error: encoding failed");
        let bytes = pub_key.to_bytes().to_vec();
        assert_eq!(encoded_pub, bytes);
        let decoded_pub =
            codec::decode::<PublicKey>(&encoded_pub).expect("internal error: decoding failed");
        assert_eq!(pub_key, decoded_pub);
    }
}
