#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AttestPublisher - IPFS upload + EAS on-chain attestation for ChainScope.

Pipeline:
  1. Upload anomaly report to IPFS (web3.storage API or local fallback)
  2. Publish EAS attestation on Sepolia testnet with CID + anomaly metadata
  3. Return verifiable on-chain proof (tx_hash + Etherscan link)
"""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from web3 import Web3

from chainscope.config import (
    SEPOLIA_RPC_URL,
    WALLET_PRIVATE_KEY,
    IPFS_TOKEN,
    DATA_DIR,
)

# ── EAS Contract Addresses (Sepolia) ──
EAS_SEPOLIA = Web3.to_checksum_address("0xC2679fBD37d54388Ce493F1DB75320D236e1815e")
SCHEMA_REGISTRY_SEPOLIA = Web3.to_checksum_address("0x0a7E2Ff54e76B8E6659aedc9103FB21c038050D0")
SEPOLIA_CHAIN_ID = 11155111

# Schema UID for: string address, string cid, uint256 score, bool isAnomalous
SCHEMA_UID = bytes.fromhex("0d012fdb105771ee6d1577f2a7610ee4813259deb939a990f28f7db5624af0fd")

# ── EAS ABI (minimal: attest + getAttestation) ──
EAS_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "bytes32", "name": "schema", "type": "bytes32"},
                    {
                        "components": [
                            {"internalType": "address", "name": "recipient", "type": "address"},
                            {"internalType": "uint64", "name": "expirationTime", "type": "uint64"},
                            {"internalType": "bool", "name": "revocable", "type": "bool"},
                            {"internalType": "bytes32", "name": "refUID", "type": "bytes32"},
                            {"internalType": "bytes", "name": "data", "type": "bytes"},
                            {"internalType": "uint256", "name": "value", "type": "uint256"},
                        ],
                        "internalType": "struct AttestationRequestData",
                        "name": "data",
                        "type": "tuple",
                    },
                ],
                "internalType": "struct AttestationRequest",
                "name": "request",
                "type": "tuple",
            }
        ],
        "name": "attest",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "uid", "type": "bytes32"}],
        "name": "getAttestation",
        "outputs": [
            {
                "components": [
                    {"internalType": "bytes32", "name": "uid", "type": "bytes32"},
                    {"internalType": "bytes32", "name": "schemaId", "type": "bytes32"},
                    {"internalType": "uint64", "name": "time", "type": "uint64"},
                    {"internalType": "uint64", "name": "expirationTime", "type": "uint64"},
                    {"internalType": "uint64", "name": "revocationTime", "type": "uint64"},
                    {"internalType": "bytes32", "name": "refUID", "type": "bytes32"},
                    {"internalType": "address", "name": "recipient", "type": "address"},
                    {"internalType": "address", "name": "attester", "type": "address"},
                    {"internalType": "bool", "name": "revocable", "type": "bool"},
                    {"internalType": "bytes", "name": "data", "type": "bytes"},
                ],
                "internalType": "struct Attestation",
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

# ── Local fallback directory ──
LOCAL_REPORTS_DIR = DATA_DIR / "local_reports"


class AttestPublisher:
    """IPFS upload + EAS on-chain attestation publisher."""

    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
        self.eas = self.w3.eth.contract(address=EAS_SEPOLIA, abi=EAS_ABI)

        if WALLET_PRIVATE_KEY:
            self.account = self.w3.eth.account.from_key(WALLET_PRIVATE_KEY)
        else:
            self.account = None
            print("[WARN] WALLET_PRIVATE_KEY not set, on-chain attestation disabled")

        LOCAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── IPFS Upload ──

    def upload_to_ipfs(self, report_text, metadata=None):
        """Upload report to IPFS via web3.storage API.

        Falls back to local file storage with SHA256 pseudo-CID if no token.

        Args:
            report_text: string content to upload
            metadata: optional dict with extra metadata

        Returns:
            CID string
        """
        payload = {
            "report": report_text,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)

        # Try web3.storage API
        if IPFS_TOKEN:
            try:
                cid = self._upload_web3_storage(content)
                if cid:
                    print(f"[IPFS] Uploaded to web3.storage, CID: {cid}")
                    return cid
            except Exception as e:
                print(f"[IPFS] web3.storage upload failed: {e}, falling back to local")

        # Fallback: save locally, generate deterministic CID-like hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        pseudo_cid = f"bafybei{content_hash[:50]}"  # CIDv0-like format

        local_path = LOCAL_REPORTS_DIR / f"{pseudo_cid}.json"
        local_path.write_text(content, encoding="utf-8")
        print(f"[IPFS] Saved locally: {local_path}")
        print(f"[IPFS] Pseudo-CID: {pseudo_cid}")
        return pseudo_cid

    def _upload_web3_storage(self, content):
        """Upload content to web3.storage via REST API."""
        headers = {
            "Authorization": f"Bearer {IPFS_TOKEN}",
            "Content-Type": "application/octet-stream",
        }
        resp = requests.post(
            "https://api.web3.storage/upload",
            headers=headers,
            data=content.encode("utf-8"),
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("cid")
        else:
            print(f"[IPFS] web3.storage error: {resp.status_code} {resp.text[:200]}")
            return None

    # ── EAS Attestation ──

    def publish_attestation(self, address, cid, anomaly_score, is_anomalous):
        """Publish EAS attestation on Sepolia testnet.

        Args:
            address: Ethereum address being attested
            cid: IPFS CID of the report
            anomaly_score: float anomaly score (0-1)
            is_anomalous: bool whether address is flagged anomalous

        Returns:
            dict with tx_hash, attestation_uid, sepolia_url, status
        """
        if not self.account:
            return {
                "status": "error",
                "message": "WALLET_PRIVATE_KEY not configured",
            }

        # Encode schema data: string address, string cid, uint256 score, bool isAnomalous
        from eth_abi import encode as abi_encode

        score_int = int(anomaly_score * 10000)  # scale to uint256
        schema_data = abi_encode(
            ["string", "string", "uint256", "bool"],
            [address, cid, score_int, is_anomalous],
        )

        # Build attestation request
        recipient = Web3.to_checksum_address(address) if len(address) == 42 else Web3.to_checksum_address("0x0000000000000000000000000000000000000000")
        att_request = (
            SCHEMA_UID,  # schemaId
            (
                recipient,       # recipient
                0,               # expirationTime (0 = no expiration)
                True,            # revocable
                b"\x00" * 32,    # refUID (no reference)
                schema_data,     # data
                0,               # value (no ETH sent)
            ),
        )

        # Build transaction
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        # Use 2x current gas price to ensure inclusion on congested testnet
        base_gas = self.w3.eth.gas_price
        gas_price = max(base_gas * 2, self.w3.to_wei(5, 'gwei'))
        print(f"[EAS] gasPrice={self.w3.from_wei(gas_price, 'gwei'):.1f} gwei")

        tx = self.eas.functions.attest(att_request).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": 500000,
            "gasPrice": gas_price,
            "chainId": SEPOLIA_CHAIN_ID,
            "value": 0,
        })

        # Sign and send
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"[EAS] TX sent: {tx_hash.hex()}")

        # Wait for receipt (Sepolia can be slow)
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        except Exception as e:
            print(f"[EAS] Timeout waiting for receipt: {e}")
            return {
                "status": "pending",
                "tx_hash": f"0x{tx_hash.hex()}",
                "attestation_uid": None,
                "sepolia_url": f"https://sepolia.etherscan.io/tx/0x{tx_hash.hex()}",
                "eas_url": "",
                "gas_used": 0,
                "block_number": 0,
                "message": "TX submitted but not yet mined. Check Etherscan link.",
            }
        status = "success" if receipt.status == 1 else "failed"
        print(f"[EAS] TX status: {status}, gas={receipt.gasUsed}")

        # Extract attestation UID from logs
        # EAS Attested event: UID is in the first 32 bytes of data field
        att_uid = None
        for log in receipt.logs:
            if log.address.lower() == EAS_SEPOLIA.lower() and len(log.data) >= 32:
                att_uid = log.data[:32].hex()
                break

        sepolia_url = f"https://sepolia.etherscan.io/tx/0x{tx_hash.hex()}"
        eas_url = f"https://sepolia.easscan.org/attestation/view/{att_uid}" if att_uid else ""

        result = {
            "status": status,
            "tx_hash": f"0x{tx_hash.hex()}",
            "attestation_uid": f"0x{att_uid}" if att_uid else None,
            "sepolia_url": sepolia_url,
            "eas_url": eas_url,
            "gas_used": receipt.gasUsed,
            "block_number": receipt.blockNumber,
        }
        return result

    # ── Full Pipeline ──

    def publish_report(self, address, report_text, anomaly_score, is_anomalous, metadata=None):
        """Full pipeline: IPFS upload + EAS attestation.

        Args:
            address: Ethereum address under investigation
            report_text: anomaly explanation text
            anomaly_score: float (0-1)
            is_anomalous: bool
            metadata: optional extra metadata dict

        Returns:
            dict with cid, ipfs_url, tx_hash, sepolia_url, status
        """
        # Step 1: Upload to IPFS
        cid = self.upload_to_ipfs(report_text, metadata={
            **(metadata or {}),
            "address": address,
            "anomaly_score": anomaly_score,
            "is_anomalous": is_anomalous,
        })

        # Step 2: Publish on-chain attestation
        att_result = self.publish_attestation(address, cid, anomaly_score, is_anomalous)

        # Combine results. Decide the URL by whether a local fallback file
        # actually exists (robust to web3.storage upload failures), instead of
        # guessing from the CID prefix.
        local_file = LOCAL_REPORTS_DIR / f"{cid}.json"
        ipfs_url = f"local:{local_file}" if local_file.exists() else f"https://dweb.link/ipfs/{cid}"

        return {
            "cid": cid,
            "ipfs_url": ipfs_url,
            **att_result,
        }
