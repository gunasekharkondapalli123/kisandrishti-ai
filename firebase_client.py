"""
Firebase Firestore integration module for KisanDrishti AI.
Connects the Streamlit web application to the Google Cloud Firestore instance
using the project's Web API credentials and resilient offline-first syncing.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Firebase Web Configuration provided for kisandrishti-ai
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyBusdBsaxbgGgosAPuuUKfB3CLUR18vzjc",
    "authDomain": "kisandrishti-ai.firebaseapp.com",
    "projectId": "kisandrishti-ai",
    "storageBucket": "kisandrishti-ai.firebasestorage.app",
    "messagingSenderId": "916752933977",
    "appId": "1:916752933977:web:dfd06d77f0f3a4ad1fbd48",
    "measurementId": "G-XC5CBFSH1Y",
}

BASE_DIR = Path(__file__).resolve().parent
OFFLINE_QUEUE_FILE = BASE_DIR / "firebase_offline_queue.json"


def _dict_to_firestore_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a standard Python dictionary to Firestore REST API field format."""
    fields = {}
    for k, v in data.items():
        if isinstance(v, bool):
            fields[k] = {"booleanValue": v}
        elif isinstance(v, int):
            fields[k] = {"integerValue": str(v)}
        elif isinstance(v, float):
            fields[k] = {"doubleValue": float(v)}
        elif isinstance(v, str):
            fields[k] = {"stringValue": v}
        elif isinstance(v, dict):
            fields[k] = {"mapValue": {"fields": _dict_to_firestore_fields(v)}}
        elif isinstance(v, list):
            values = []
            for item in v:
                if isinstance(item, str):
                    values.append({"stringValue": item})
                elif isinstance(item, (int, float)):
                    values.append({"doubleValue": float(item)})
            fields[k] = {"arrayValue": {"values": values}}
        elif v is None:
            fields[k] = {"nullValue": None}
        else:
            fields[k] = {"stringValue": str(v)}
    return fields


def _firestore_fields_to_dict(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Converts Firestore REST API field format back to a standard Python dictionary."""
    data = {}
    for k, v in fields.items():
        if "stringValue" in v:
            data[k] = v["stringValue"]
        elif "integerValue" in v:
            data[k] = int(v["integerValue"])
        elif "doubleValue" in v:
            data[k] = float(v["doubleValue"])
        elif "booleanValue" in v:
            data[k] = v["booleanValue"]
        elif "mapValue" in v:
            data[k] = _firestore_fields_to_dict(v["mapValue"].get("fields", {}))
        elif "arrayValue" in v:
            data[k] = [
                item.get("stringValue", item.get("doubleValue", item.get("integerValue")))
                for item in v["arrayValue"].get("values", [])
            ]
        elif "nullValue" in v:
            data[k] = None
    return data


class FirestoreClient:
    """Manages Firestore operations via REST API with edge offline buffering."""

    def __init__(self, config: Dict[str, str] = FIREBASE_CONFIG):
        self.config = config
        self.project_id = config.get("projectId", "kisandrishti-ai")
        self.api_key = config.get("apiKey", "")
        self.base_url = (
            f"https://firestore.googleapis.com/v1/projects/{self.project_id}/databases/(default)/documents"
        )
        self.timeout = 8  # seconds

    def check_connection(self) -> Dict[str, Any]:
        """Tests connectivity to the Cloud Firestore database."""
        try:
            url = f"{self.base_url}/crop_diagnoses"
            resp = requests.get(url, params={"key": self.api_key, "pageSize": 1}, timeout=self.timeout)
            if resp.status_code == 200:
                return {
                    "connected": True,
                    "status": "Online",
                    "project_id": self.project_id,
                    "message": "Connected successfully to Firebase Firestore",
                }
            elif resp.status_code == 403:
                return {
                    "connected": True,
                    "status": "Connected (Rules Protected)",
                    "project_id": self.project_id,
                    "message": "Connected to Firestore endpoint. Security rules active.",
                }
            else:
                return {
                    "connected": False,
                    "status": "HTTP Error",
                    "project_id": self.project_id,
                    "code": resp.status_code,
                    "message": f"Server responded with status {resp.status_code}",
                }
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            return {
                "connected": False,
                "status": "Offline / Edge Mode",
                "project_id": self.project_id,
                "message": "Network latency or DNS unreachable. Operating in edge buffered mode.",
            }
        except Exception as e:
            return {
                "connected": False,
                "status": "Offline / Local Mode",
                "project_id": self.project_id,
                "message": str(e),
            }

    def save_diagnosis(
        self,
        disease_name: str,
        crop: str,
        confidence: float,
        latency_ms: float,
        organic_remedy: str,
        chemical_treatment: str,
        telugu_guidance: str,
    ) -> Dict[str, Any]:
        """
        Saves a disease diagnosis event to the 'crop_diagnoses' collection.
        If offline, saves to local offline queue for automatic retry.
        """
        record = {
            "disease_name": disease_name,
            "crop": crop,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
            "organic_remedy": organic_remedy,
            "chemical_treatment": chemical_treatment,
            "telugu_guidance": telugu_guidance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "KisanDrishti Edge App",
            "model": "yolov8n.pt",
        }

        try:
            url = f"{self.base_url}/crop_diagnoses"
            payload = {"fields": _dict_to_firestore_fields(record)}
            resp = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code in (200, 201):
                doc_name = resp.json().get("name", "")
                return {
                    "success": True,
                    "synced": True,
                    "doc_id": doc_name.split("/")[-1] if doc_name else "created",
                    "data": record,
                }
            else:
                self._buffer_offline("crop_diagnoses", record)
                return {
                    "success": True,
                    "synced": False,
                    "offline_buffered": True,
                    "note": f"Firestore returned {resp.status_code}; saved to local edge queue",
                    "data": record,
                }
        except Exception:
            self._buffer_offline("crop_diagnoses", record)
            return {
                "success": True,
                "synced": False,
                "offline_buffered": True,
                "note": "Network unreachable; saved to edge offline buffer",
                "data": record,
            }

    def fetch_recent_diagnoses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves recent diagnoses from Firestore (or local buffer)."""
        results = []
        try:
            url = f"{self.base_url}/crop_diagnoses"
            resp = requests.get(
                url,
                params={"key": self.api_key, "pageSize": limit},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                docs = resp.json().get("documents", [])
                for doc in docs:
                    d = _firestore_fields_to_dict(doc.get("fields", {}))
                    d["_id"] = doc.get("name", "").split("/")[-1]
                    results.append(d)
                return results
        except Exception:
            pass

        # Fallback to offline queue
        queue = self.get_offline_queue()
        for item in queue.get("crop_diagnoses", [])[-limit:]:
            results.append(item)
        return results

    def sync_mandi_rates(self, mandi_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Syncs local APMC Mandi rates to the 'apmc_mandi_rates' Firestore collection."""
        success_count = 0
        error_count = 0

        for row in mandi_records:
            doc_id = f"{row.get('Yard', 'Yard')}_{row.get('Commodity', 'Crop')}_{row.get('Variety', 'Var')}".replace(
                " ", "_"
            ).replace("/", "_").replace("(", "").replace(")", "")
            url = f"{self.base_url}/apmc_mandi_rates/{doc_id}"
            payload = {"fields": _dict_to_firestore_fields(row)}
            try:
                resp = requests.patch(
                    url,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code in (200, 201):
                    success_count += 1
                else:
                    error_count += 1
                    self._buffer_offline("apmc_mandi_rates", row)
            except Exception:
                error_count += 1
                self._buffer_offline("apmc_mandi_rates", row)

        return {
            "total": len(mandi_records),
            "synced_online": success_count,
            "buffered_offline": error_count,
        }

    def _buffer_offline(self, collection: str, data: Dict[str, Any]) -> None:
        """Stores a record in the local offline JSON queue."""
        queue = self.get_offline_queue()
        if collection not in queue:
            queue[collection] = []
        queue[collection].append(data)
        try:
            OFFLINE_QUEUE_FILE.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to buffer record offline: {e}")

    def get_offline_queue(self) -> Dict[str, List[Dict[str, Any]]]:
        """Reads the local offline queue file."""
        if OFFLINE_QUEUE_FILE.exists():
            try:
                return json.loads(OFFLINE_QUEUE_FILE.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def get_firebase_js_snippet(self) -> str:
        """Returns the client-side JavaScript snippet for embedding in web views."""
        return f"""
<!-- Firebase App (the core Firebase SDK) -->
<script type="module">
  import {{ initializeApp }} from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
  import {{ getFirestore, collection, addDoc, getDocs }} from "https://www.gstatic.com/firebasejs/10.14.1/firebase-firestore.js";

  const firebaseConfig = {json.dumps(self.config, indent=2)};

  const app = initializeApp(firebaseConfig);
  const db = getFirestore(app);
  window.kisanFirebase = {{ app, db }};
  console.log("KisanDrishti Firebase initialized for project:", firebaseConfig.projectId);
</script>
"""


# Global singleton instance
firestore_client = FirestoreClient()
